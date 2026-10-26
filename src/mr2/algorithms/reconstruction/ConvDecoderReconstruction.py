"""Scan-specific MRI reconstruction with a convolutional generator."""

import math
from collections.abc import Callable

import torch
from torch import nn

from mr2.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mr2.algorithms.prewhiten_kspace import prewhiten_kspace
from mr2.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mr2.data import CsmData, DcfData, IData, KData, KNoise
from mr2.nn.ndmodules import convND
from mr2.nn.Sequential import Sequential
from mr2.operators import DensityCompensationOp, LinearOperator, SensitivityOp
from mr2.utils.interpolate import interpolate


class _ConvDecoder(nn.Module):
    """Decoder with explicit spatial sizes and locally initialized parameters."""

    def __init__(
        self,
        shape: tuple[int, int],
        latent_shape: tuple[int, int],
        n_features: int,
        n_levels: int,
        rng: torch.Generator,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.sizes = [
            tuple(math.ceil(a * (b / a) ** (level / n_levels)) for a, b in zip(latent_shape, shape, strict=True))
            for level in range(1, n_levels + 1)
        ]
        self.sizes[-1] = shape
        # Meta construction avoids consuming the process RNG in module constructors.
        with torch.device('meta'):
            self.blocks = nn.ModuleList(
                Sequential(
                    convND(2)(n_features, n_features, 3, padding=1),
                    nn.ReLU(),
                    nn.InstanceNorm2d(n_features, affine=True, track_running_stats=False),
                )
                for _ in range(n_levels + 1)
            )
            self.output = convND(2)(n_features, 2, 1)
        self.to(dtype=dtype)
        self.to_empty(device=rng.device)
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=rng)
                    if module.bias is not None:
                        bound = 1 / math.sqrt(module.weight[0].numel())
                        nn.init.uniform_(module.bias, -bound, bound, generator=rng)
                elif isinstance(module, nn.InstanceNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Generate one complex slice."""
        x = latent
        for size, block in zip(self.sizes, self.blocks[:-1], strict=True):
            x = block(interpolate(x, size, dim=(-2, -1), mode='nearest'))
        x = self.output(self.blocks[-1](x))
        return torch.complex(x[:, 0], x[:, 1])


class ConvDecoderReconstruction(DirectReconstruction):
    r"""Fit an untrained ConvDecoder prior to acquired complex k-space.

    Minimize :math:`\|A G_\theta(z)-y\|_2^2` over generator weights, with
    :math:`A=FS` the existing Fourier/sampling and sensitivity operators. The
    random latent :math:`z` stays fixed. Each call starts fresh generators and
    Adam states; each 2D slice (including leading batch dimensions) has its own
    weights. No fitted network is retained and this reconstruction is not
    differentiable through the optimization.

    This ConvDecoder variant uses nearest upsampling, 3x3
    convolution, ReLU, stateless affine InstanceNorm, an additional block at
    full resolution, and a linear 1x1 output. The authors' network uses BatchNorm
    [DARESTANI2021]_. Two real channels represent one complex object; sensitivity
    maps enter the forward model rather than generating separate coil images.
    The paper describes both output formulations. This is not a reproduction
    of all of its fitting and postprocessing choices.

    Data are scaled by the RMS magnitude of the adjoint image during fitting,
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
        latent_shape: tuple[int, int] = (8, 4),
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
            `None` is supported for single-coil data.
        noise
            Noise measurement used for prewhitening, if provided.
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
            Spatial shape of the fixed uniform random input. Must fit inside each
            output slice and contain more than one pixel.
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
        if len(latent_shape) != 2 or min(latent_shape) < 1 or math.prod(latent_shape) <= 1:
            raise ValueError('latent_shape must contain two positive sizes and more than one pixel.')
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
            scale = adjoint.abs().square().mean().sqrt().clamp_min(torch.finfo(data.real.dtype).eps)
        if shape[-4] != 1:
            raise ValueError('The acquisition model must map a single complex object to k-space.')
        image_shape = (shape[-2], shape[-1])
        if any(a > b for a, b in zip(self.latent_shape, image_shape, strict=True)):
            raise ValueError('latent_shape must fit inside the reconstructed slice.')
        rng = torch.Generator(device=data.device).manual_seed(self.seed)
        models = nn.ModuleList(
            _ConvDecoder(image_shape, self.latent_shape, self.n_features, self.n_levels, rng, data.real.dtype)
            for _ in range(math.prod(shape[:-2]))
        )
        latents = [
            torch.rand(
                (1, self.n_features, *self.latent_shape), generator=rng, device=data.device, dtype=data.real.dtype
            )
            for _ in models
        ]

        def image() -> torch.Tensor:
            return torch.cat([model(z) for model, z in zip(models, latents, strict=True)]).reshape(shape)

        optimizer = torch.optim.Adam(models.parameters(), lr=self.learning_rate)
        for iteration in range(self.max_iterations):
            optimizer.zero_grad(set_to_none=True)
            (prediction,) = acquisition(image())
            loss = (prediction - data / scale).abs().square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite ConvDecoder fitting loss.')
            loss.backward()
            optimizer.step()
            if self.callback is not None:
                with torch.no_grad():
                    solution = (image() * scale).detach().clone()
                if self.callback({'solution': (solution,), 'iteration_number': iteration}) is False:
                    break
        with torch.no_grad():
            result = (image() * scale).detach()
        return IData.from_tensor_and_kheader(result, kdata.header)
