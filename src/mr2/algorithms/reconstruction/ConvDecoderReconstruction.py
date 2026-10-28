"""Scan-specific MRI reconstruction with a convolutional generator."""

import math
from collections.abc import Callable

import torch
from torch import nn

from mr2.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mr2.algorithms.prewhiten_kspace import prewhiten_kspace
from mr2.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mr2.data import CsmData, DcfData, IData, KData, KNoise
from mr2.nn.ComplexAsChannel import ComplexAsChannel
from mr2.nn.nets.ConvDecoder import ConvDecoder
from mr2.operators import DensityCompensationOp, LinearOperator, SensitivityOp


class ConvDecoderReconstruction(DirectReconstruction):
    r"""Reconstruct MRI data with an untrained convolutional decoder.

    Fits generator weights :math:`\theta` by minimizing
    :math:`\|A G_\theta(z)-y\|_2^2`, where :math:`A=FS` combines Fourier
    sampling and coil sensitivities, :math:`y` is acquired complex k-space,
    and :math:`z` is fixed random noise.

    Each call fits fresh `~mr2.nn.nets.ConvDecoder` networks with Adam.
    `latent_shape` selects the spatial dimensions modeled jointly; remaining
    dimensions use independent generators. Two real output channels form one
    complex object image. Fitting uses adjoint RMS scaling and returns an
    image in the input units. The callback and iteration budget control stopping.

    This variant uses InstanceNorm instead of the authors' BatchNorm
    [DARESTANI2021]_. Optimization is not differentiable through the iterations.

    References
    ----------
    .. [DARESTANI2021] Darestani MZ, Heckel R (2021). Accelerated MRI with
       Un-trained Neural Networks. https://doi.org/10.1109/TCI.2021.3097596
    """

    def __init__(
        self,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable[[IData], CsmData] | CsmData | SensitivityOp | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | DensityCompensationOp | None = None,
        *,
        max_iterations: int = 2500,
        learning_rate: float = 0.008,
        n_features: int = 256,
        n_levels: int = 6,
        latent_shape: tuple[int, ...] = (8, 4),
        seed: int = 0,
        callback: Callable[[OptimizerStatus], bool | None] | None = None,
    ) -> None:
        """Initialize the reconstruction.

        Parameters
        ----------
        kdata
            Data used to set up Fourier, density compensation and sensitivity
            operators, as in `DirectReconstruction`.
        fourier_op
            Fourier/sampling operator. If omitted, construct from `kdata`.
        csm
            Sensitivity maps, operator or estimation callable, as in
            `DirectReconstruction`. Required for multicoil object reconstruction;
            `None` is supported for single-coil data. When `noise` is supplied,
            explicit maps must already be in the whitened coil basis. Estimation
            callables receive coil images reconstructed from whitened data.
        noise
            Noise measurement used to prewhiten the k-space data, if provided.
            Explicit sensitivity maps are not transformed; supply maps calibrated
            from equivalently whitened data, as for `DirectReconstruction`.
        dcf
            Density compensation used only for sensitivity estimation, not for
            weighting the fitting loss. If omitted, estimate from `kdata`.
        max_iterations
            Finite number of Adam updates. Zero returns the initial generator output.
        learning_rate
            Positive Adam learning rate.
        n_features
            Number of latent channels and hidden features.
        n_levels
            Number of nearest-neighbor upsampling stages.
        latent_shape
            Shape of the fixed random input, excluding channels. Its length selects
            the generator dimensionality (e.g. two for slices, four for time/volume).
            The coil axis is excluded. Must fit inside the output dimensions.
        seed
            Random initialization seed, reset on each call.
        callback
            Called after each update with zero-based `iteration_number` and
            `solution=(image,)` in mr2 layout. Return `False` to stop.
        """
        if max_iterations < 0:
            raise ValueError('max_iterations must be nonnegative.')
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive.')
        if n_features < 1 or n_levels < 1:
            raise ValueError('n_features and n_levels must be positive.')
        if not latent_shape or min(latent_shape) < 1:
            raise ValueError('latent_shape must contain positive sizes.')
        super().__init__(kdata, fourier_op, csm, noise, dcf)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.n_levels = n_levels
        self.latent_shape = latent_shape
        self.seed = seed
        self.callback = callback

    @torch.enable_grad()
    def forward(self, kdata: KData) -> IData:
        """Fit fresh generators and return complex image data with the input header."""
        if not kdata.data.is_complex():
            raise ValueError('Expected complex k-space data.')
        if self.csm_op is None and kdata.data.shape[-4] != 1:
            raise ValueError('Sensitivity maps are required for multicoil object reconstruction.')
        if self.noise is not None:
            kdata = prewhiten_kspace(kdata, self.noise)
        acquisition = self.fourier_op @ self.csm_op if self.csm_op is not None else self.fourier_op
        data = kdata.data.detach()
        with torch.no_grad():
            (adjoint,) = acquisition.H(data)
            shape = adjoint.shape

        if shape[-4] != 1:
            raise ValueError('The acquisition model must map a single complex object to k-space.')
        n_dim = len(self.latent_shape)
        object_shape = (*shape[:-4], *shape[-3:])
        if n_dim > len(object_shape):
            raise ValueError('latent_shape has more dimensions than the coil-combined image.')
        image_shape = tuple(object_shape[-n_dim:])
        with torch.no_grad():
            scale_dims = tuple(range(-max(3, n_dim), 0))
            scale = adjoint.squeeze(-4).abs().square().mean(dim=scale_dims, keepdim=True).sqrt().unsqueeze(-4)
            scale = scale.clamp_min(torch.finfo(data.real.dtype).eps)
        rng = torch.Generator(device=data.device).manual_seed(self.seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            models = nn.ModuleList(
                ComplexAsChannel(
                    ConvDecoder(n_dim, self.n_features, 2, image_shape, self.n_features, self.n_levels)
                ).to(device=data.device, dtype=data.real.dtype)
                for _ in range(math.prod(object_shape[:-n_dim]))
            )
        latents = [
            torch.rand(
                (1, self.n_features, *self.latent_shape), generator=rng, device=data.device, dtype=data.real.dtype
            )
            for _ in models
        ]

        def image() -> torch.Tensor:
            return (
                torch.cat([model(z) for model, z in zip(models, latents, strict=True)])
                .reshape(object_shape)
                .unsqueeze(-4)
            )

        parameters = tuple(models.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        for iteration in range(self.max_iterations):
            optimizer.zero_grad(set_to_none=True)
            (prediction,) = acquisition(image())
            loss = (prediction - data / scale).abs().square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite ConvDecoder fitting loss.')
            loss.backward(inputs=parameters)
            optimizer.step()
            if self.callback is not None:
                with torch.no_grad():
                    solution = (image() * scale).detach().clone()
                if self.callback({'solution': (solution,), 'iteration_number': iteration}) is False:
                    break
        with torch.no_grad():
            result = (image() * scale).detach()
        return IData.from_tensor_and_kheader(result, kdata.header)
