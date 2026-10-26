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
    r"""Fit an untrained ConvDecoder prior to acquired complex k-space.

    Minimize :math:`\|A G_\theta(z)-y\|_2^2` over generator weights, with
    :math:`A=FS` the existing Fourier/sampling and sensitivity operators. The
    random latent :math:`z` stays fixed. Each call starts fresh generators and
    Adam states; each spatial patch has its own weights. The number of entries
    in `latent_shape` selects 1D, 2D or 3D generators. Remaining image dimensions
    (including leading batch dimensions) are fitted with separate generators.
    No fitted network is retained and this reconstruction is not
    differentiable through the optimization. Calls inside ``torch.no_grad()``
    are supported; ``torch.inference_mode()`` must be disabled while fitting.

    The `~mr2.nn.nets.ConvDecoder` variant uses nearest upsampling, 3-tap
    convolutions, ReLU, stateless affine InstanceNorm, an additional block at
    full resolution, and a linear 1-tap output. The authors' network uses BatchNorm
    [DARESTANI2021]_. Two real channels represent one complex object; sensitivity
    maps enter the forward model rather than generating separate coil images.
    The paper describes both output formulations. This is not a reproduction
    of all of its fitting and postprocessing choices.

    Each acquisition batch entry is scaled by its adjoint image RMS during fitting,
    and the returned image is restored to the input units. The adjoint is not
    used as the network input.

    The iteration budget and callback control stopping. A decreasing acquired
    k-space loss does not protect against noise overfitting.

    References
    ----------
    .. [DARESTANI2021] Darestani MZ, Heckel R (2021). Accelerated MRI with
       Un-trained Neural Networks. IEEE Transactions on Computational Imaging,
       7, 724-733. https://doi.org/10.1109/TCI.2021.3097596
       Architecture and MRI formulations: https://arxiv.org/abs/2007.02471
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
            Spatial shape of the fixed uniform random input, with 1, 2 or 3
            entries selecting the generator dimensionality. Two entries fit each
            2D slice independently; three fit each volume with a 3D generator.
            Must fit inside the corresponding output dimensions. The first upsampling
            stage must contain more than one pixel for InstanceNorm.
        seed
            Seed for a local generator, reset on each call without changing global
            RNG state. Reproducible on the same device with deterministic kernels.
        callback
            Called after each update with zero-based `iteration_number` and
            `solution=(image,)`, a detached copy in physical units and mr2 layout.
            Return `False` to stop. The returned image includes that update.
        """
        if max_iterations < 0:
            raise ValueError('max_iterations must be nonnegative.')
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive.')
        if n_features < 1 or n_levels < 1:
            raise ValueError('n_features and n_levels must be positive.')
        if not 1 <= len(latent_shape) <= 3 or min(latent_shape) < 1:
            raise ValueError('latent_shape must contain 1, 2 or 3 positive sizes.')
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
            scale = adjoint.abs().square().mean(dim=(-3, -2, -1), keepdim=True).sqrt()
            scale = scale.clamp_min(torch.finfo(data.real.dtype).eps)
        if shape[-4] != 1:
            raise ValueError('The acquisition model must map a single complex object to k-space.')
        n_dim = len(self.latent_shape)
        image_shape = tuple(shape[-n_dim:])
        rng = torch.Generator(device=data.device).manual_seed(self.seed)
        models = nn.ModuleList()
        for _ in range(math.prod(shape[:-n_dim])):
            # Avoid consuming global RNG state in the network constructors.
            with torch.device('meta'):
                decoder = ConvDecoder(n_dim, self.n_features, 2, image_shape, self.n_features, self.n_levels)
            decoder.to(dtype=data.real.dtype)
            decoder.to_empty(device=data.device)
            decoder.reset_parameters(generator=rng)
            models.append(ComplexAsChannel(decoder))
        latents = [
            torch.rand(
                (1, self.n_features, *self.latent_shape), generator=rng, device=data.device, dtype=data.real.dtype
            )
            for _ in models
        ]

        def image() -> torch.Tensor:
            return torch.cat([model(z) for model, z in zip(models, latents, strict=True)]).reshape(shape)

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
