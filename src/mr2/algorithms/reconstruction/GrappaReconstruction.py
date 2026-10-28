"""Cartesian GRAPPA reconstruction."""

from collections.abc import Callable
from math import isfinite, prod
from typing import Literal

import torch
from einops import rearrange
from typing_extensions import Self

from mr2.algorithms.reconstruction.Reconstruction import Reconstruction
from mr2.data.CsmData import CsmData
from mr2.data.enums import TrajType
from mr2.data.IData import IData
from mr2.data.KData import KData
from mr2.data.KNoise import KNoise
from mr2.operators.CartesianSamplingOp import CartesianSamplingOp
from mr2.operators.FastFourierOp import FastFourierOp
from mr2.operators.SensitivityOp import SensitivityOp
from mr2.utils import pad_or_crop, sliding_window


class GrappaReconstruction(Reconstruction):
    """Calibrate GRAPPA, interpolate Cartesian k-space and reconstruct coil images."""

    acs: torch.Tensor
    """Calibration data in the acquisition coil basis."""

    kernels: torch.Tensor
    """Fitted kernels indexed by source pattern."""

    def __init__(
        self,
        kdata: KData,
        acs: torch.Tensor,
        kernel_size: tuple[int, int, int],
        *,
        regularization: float = 0.01,
        csm: Callable[[IData], CsmData] | CsmData | SensitivityOp | None = CsmData.from_idata_walsh,
    ) -> None:
        """Initialize the reconstruction and calibrate reusable kernels.

        Parameters
        ----------
        kdata
            Cartesian acquisition defining the encoding grid and sampling pattern.
            All batches must share the same pattern. Samples must not repeat.
        acs
            Fully sampled Cartesian calibration tensor ``(..., coils, kz, ky, kx)``.
            Must use the same coil basis and frequency spacing
            as ``kdata``; its batch dimensions broadcast to the acquired data.
        kernel_size
            Positive odd neighborhood widths in z, y, x. Use a z width of one
            for 2D acquisitions. At boundaries, only in-grid acquired sources
            enter the fit. Every missing point needs an acquired neighbor.
        regularization
            Relative ridge strength passed to `calibrate`.
        csm
            Coil combination maps, operator or calculation from completed coil
            images, following `~mr2.algorithms.reconstruction.DirectReconstruction`.
            Set to None to return separate coil images. For differentiable
            reconstruction, construct on the target device.
        """
        super().__init__()
        self.acs = acs
        self.kernel_size = kernel_size
        self.regularization = regularization
        self.noise = None
        self.dcf_op = None
        self.csm_op = None
        self.recalculate_fourierop(kdata)
        if csm is None or isinstance(csm, CsmData | SensitivityOp):
            self.csm_op = csm.as_operator() if isinstance(csm, CsmData) else csm
        else:
            self.recalculate_csm(kdata, csm)

    @staticmethod
    def calibrate(acs: torch.Tensor, source_pattern: torch.Tensor, *, regularization: float = 0.01) -> torch.Tensor:
        """Fit a GRAPPA kernel from fully sampled Cartesian calibration data.

        Parameters
        ----------
        acs
            Fully sampled Cartesian ACS, shape ``(..., coils, kz, ky, kx)``.
            Leading batch dimensions are calibrated independently.
        source_pattern
            Boolean 3D neighborhood with positive odd widths. True selects an
            acquired source offset; the center is the target and must be False.
            Only complete ACS windows are used.
        regularization
            Nonnegative ridge strength, relative to the mean diagonal of ``S.H @ S``
            for source matrix S. Zero uses a minimum-norm pseudoinverse. Batches
            with zero source energy return zero kernels.

        Returns
        -------
            Complex kernel with shape ``(..., coils_out, coils_in, *kernel_size)``.
            Unused offsets are zero.
        """
        kernel_size = tuple(source_pattern.shape)
        if len(kernel_size) != 3 or any(k % 2 != 1 for k in kernel_size):
            raise ValueError('source_pattern must have three positive odd widths (use width one for 2D data).')
        if source_pattern.dtype != torch.bool or source_pattern.flatten()[prod(kernel_size) // 2]:
            raise ValueError('source_pattern must be boolean and exclude the center.')
        if not source_pattern.any():
            raise ValueError('No acquired neighbor within kernel_size; increase the neighborhood widths.')
        if not isfinite(regularization) or regularization < 0:
            raise ValueError('regularization must be finite and nonnegative.')
        n_coils = acs.shape[-4]
        windows = sliding_window(acs, kernel_size, dim=(-3, -2, -1))
        cz, cy, cx = (k // 2 for k in kernel_size)
        targets = rearrange(windows[..., cz, cy, cx], 'z y x ... c -> ... (z y x) c')
        sources = rearrange(windows[..., source_pattern], 'z y x ... c s -> ... (z y x) (c s)')
        if regularization == 0:
            weights = torch.linalg.pinv(sources) @ targets
        else:
            gram = sources.mH @ sources
            energy = gram.diagonal(dim1=-2, dim2=-1).real.mean(-1)
            # Any positive diagonal selects the zero minimum-norm kernel when S is zero.
            ridge = torch.where(energy == 0, 1.0, regularization * energy)
            identity = torch.eye(gram.shape[-1], device=acs.device, dtype=acs.dtype)
            weights = torch.linalg.solve(gram + ridge[..., None, None] * identity, sources.mH @ targets)
        kernel = acs.new_zeros(*acs.shape[:-4], n_coils, n_coils, prod(kernel_size))
        kernel[..., source_pattern.flatten()] = weights.unflatten(-2, (n_coils, -1)).movedim(-1, -3)
        return kernel.reshape(*acs.shape[:-4], n_coils, n_coils, *kernel_size)

    @staticmethod
    def _interpolate_kernel(
        kspace: torch.Tensor, kernel: torch.Tensor, target_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply a calibrated Cartesian GRAPPA kernel.

        Parameters
        ----------
        kspace
            Zero-filled Cartesian grid, shape ``(..., coils, kz, ky, kx)``.
        kernel
            Kernel returned by `calibrate`, shaped ``(..., coils, coils, kz, ky, kx)``,
            with batch dimensions that broadcast to those of ``kspace``. Out-of-grid sources contribute zero.
        target_indices
            Flattened spatial indices to fill, shared across batches and coils.
            Other entries are preserved exactly. If omitted, evaluate the kernel
            on the entire grid. Call separately for each source pattern.

        Returns
        -------
            Interpolated k-space with the input shape, dtype and device.
        """
        kernel_size = kernel.shape[-3:]
        spatial_shape = kspace.shape[-3:]
        padded_shape = tuple(n + k - 1 for n, k in zip(spatial_shape, kernel_size, strict=True))
        windows = sliding_window(pad_or_crop(kspace, padded_shape), kernel_size, dim=(-3, -2, -1))
        result = kspace.flatten(start_dim=-3).clone()
        if target_indices is None:
            target_indices = torch.arange(prod(spatial_shape), device=kspace.device)
        weights = rearrange(kernel, '... o i kz ky kx -> ... (i kz ky kx) o')
        for locations in target_indices.split(1024):
            patches = windows[torch.unravel_index(locations, spatial_shape)]
            sources = rearrange(patches, 'p ... c kz ky kx -> ... p (c kz ky kx)')
            result[..., locations] = (sources @ weights).transpose(-1, -2)
        return result.reshape(kspace.shape)

    def recalculate_fourierop(self, kdata: KData) -> Self:
        """Update the Cartesian sampling operators and recalibrate from stored ACS.

        Parameters
        ----------
        kdata
            Acquisition defining the new encoding grid and sampling pattern.
        """
        if not all(t & TrajType.ONGRID for t in kdata.traj.type_along_kzyx):
            raise ValueError('GRAPPA reconstruction requires Cartesian data.')
        if self.acs.shape[-4] != kdata.data.shape[-4]:
            raise ValueError('ACS and acquisition must have the same number of coils.')
        if torch.broadcast_shapes(self.acs.shape[:-4], kdata.shape[:-4]) != kdata.shape[:-4]:
            raise ValueError('ACS batch dimensions must broadcast to the acquisition batches.')
        sampling_op = CartesianSamplingOp(kdata.header.encoding_matrix, kdata.traj)
        image_fourier_op = FastFourierOp(
            dim=tuple(
                d for d, t in zip((-3, -2, -1), kdata.traj.type_along_kzyx, strict=True) if not t & TrajType.SINGLEVALUE
            ),
            recon_matrix=kdata.header.recon_matrix,
            encoding_matrix=kdata.header.encoding_matrix,
        )
        fourier_op = sampling_op @ image_fourier_op
        ones = kdata.data.new_ones(*kdata.shape[:-4], 1, *kdata.shape[-3:])
        (support,) = sampling_op.H(ones)
        support = support.real.bool().reshape(-1, *support.shape[-3:])
        if (support.flatten(start_dim=-3).sum(-1) != prod(kdata.shape[-3:])).any():
            raise ValueError('GRAPPA reconstruction requires distinct samples inside the encoding grid.')
        if not (support == support[0]).all():
            raise ValueError('GRAPPA reconstruction requires a shared sampling pattern across batches.')
        acquired = support[0]
        missing_indices = (~acquired).flatten().nonzero().flatten()
        padded_shape = tuple(n + k - 1 for n, k in zip(acquired.shape, self.kernel_size, strict=True))
        neighborhoods = sliding_window(pad_or_crop(acquired, padded_shape), self.kernel_size)
        neighborhoods = neighborhoods[torch.unravel_index(missing_indices, acquired.shape)]
        neighborhoods = neighborhoods.reshape(missing_indices.numel(), prod(self.kernel_size))
        patterns, assignments = torch.unique(neighborhoods, dim=0, return_inverse=True)
        kernels = [
            self.calibrate(self.acs, pattern.reshape(self.kernel_size), regularization=self.regularization)
            for pattern in patterns
        ]
        fitted_kernels = torch.stack(kernels) if kernels else self.acs.new_empty(0)
        # Keep the previous reconstruction usable if validation or calibration fails.
        self.sampling_op = sampling_op
        self.fourier_op = fourier_op
        self.image_fourier_op = image_fourier_op
        self.missing_indices = missing_indices
        self.assignments = assignments
        self.kernels = fitted_kernels
        return self

    def interpolate(self, kdata: KData) -> torch.Tensor:
        """Fill the encoded Cartesian grid, preserving acquired samples exactly.

        Parameters
        ----------
        kdata
            Data with the trajectory and coil basis used during initialization.

        Returns
        -------
            Completed k-space tensor, shape ``(..., coils, kz, ky, kx)``.
        """
        (grid,) = self.sampling_op.H(kdata.data)
        for index, kernel in enumerate(self.kernels):
            locations = self.missing_indices[self.assignments == index]
            grid = self._interpolate_kernel(grid, kernel, locations)
        return grid

    def _coil_images(self, kdata: KData) -> IData:
        """Reconstruct completed coil images before sensitivity combination."""
        (images,) = self.image_fourier_op.H(self.interpolate(kdata))
        return IData.from_tensor_and_kheader(images, kdata.header)

    def recalculate_csm(
        self,
        kdata: KData,
        csm_calculation: Callable[[IData], CsmData] = CsmData.from_idata_walsh,
        noise: KNoise | Literal[False] | None = None,
    ) -> Self:
        """Update coil sensitivities from GRAPPA-completed images.

        Parameters
        ----------
        kdata
            Acquisition with the trajectory and coil basis used for calibration.
        csm_calculation
            Sensitivity calculation receiving the completed coil images.
        noise
            Must be None or False. For prewhitening, transform acquisition and
            ACS together before calibrating GRAPPA kernels.
        """
        if noise is not False and (noise is not None or self.noise is not None):
            raise ValueError('Prewhiten acquisition and ACS together before calibrating GRAPPA.')
        self.csm_op = csm_calculation(self._coil_images(kdata)).as_operator()
        return self

    def forward(self, kdata: KData) -> IData:
        """Interpolate and reconstruct data with the calibrated sampling pattern.

        Parameters
        ----------
        kdata
            Data with the trajectory and coil basis used during initialization.

        Returns
        -------
            Reconstructed image data, optionally combined using coil sensitivities.
        """
        image = self._coil_images(kdata)
        if self.csm_op is not None:
            (image.data,) = self.csm_op.H(image.data)
        return image
