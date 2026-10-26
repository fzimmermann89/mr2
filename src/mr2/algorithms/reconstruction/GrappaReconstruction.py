"""Cartesian GRAPPA reconstruction."""

from collections.abc import Callable
from math import prod
from typing import Literal

import torch
from torch.nn.functional import pad
from typing_extensions import Self

from mr2.algorithms.grappa import grappa_calibration, grappa_interpolation
from mr2.algorithms.reconstruction.Reconstruction import Reconstruction
from mr2.data.CsmData import CsmData
from mr2.data.enums import TrajType
from mr2.data.IData import IData
from mr2.data.KData import KData
from mr2.data.KNoise import KNoise
from mr2.operators.CartesianSamplingOp import CartesianSamplingOp
from mr2.operators.FastFourierOp import FastFourierOp
from mr2.operators.FourierOp import FourierOp
from mr2.operators.SensitivityOp import SensitivityOp
from mr2.utils.sliding_window import sliding_window


class GrappaReconstruction(Reconstruction):
    """Calibrate GRAPPA, interpolate Cartesian k-space and reconstruct coil images."""

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
            Fully sampled calibration tensor ``(..., coils, kz, ky, kx)``, e.g.
            ``kdata.acs(size)``. Must use the same coil basis and frequency spacing
            as ``kdata``; its batch dimensions broadcast to the acquired data.
        kernel_size
            Positive odd neighborhood widths in z, y, x. Use a z width of one
            for 2D acquisitions. At boundaries, only in-grid acquired sources
            enter the fit. Every missing point needs an acquired neighbor.
        regularization
            Relative ridge strength passed to `~mr2.algorithms.grappa_calibration`.
        csm
            Coil combination maps, operator or calculation from completed coil
            images, following `~mr2.algorithms.reconstruction.DirectReconstruction`.
            Set to None to return separate coil images.
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

    def recalculate_fourierop(self, kdata: KData) -> Self:
        """Update the Cartesian sampling operators and recalibrate from stored ACS.

        Parameters
        ----------
        kdata
            Acquisition defining the new encoding grid and sampling pattern.
        """
        if not all(t & TrajType.ONGRID for t in kdata.traj.type_along_kzyx):
            raise ValueError('GRAPPA reconstruction requires Cartesian data.')
        if len(self.kernel_size) != 3 or any(k < 1 or k % 2 != 1 for k in self.kernel_size):
            raise ValueError('kernel_size must contain three positive odd widths.')
        if self.acs.shape[-4] != kdata.data.shape[-4]:
            raise ValueError('ACS and acquisition must have the same number of coils.')
        if torch.broadcast_shapes(self.acs.shape[:-4], kdata.shape[:-4]) != kdata.shape[:-4]:
            raise ValueError('ACS batch dimensions must broadcast to the acquisition batches.')
        sampling_op = CartesianSamplingOp(kdata.header.encoding_matrix, kdata.traj)
        fourier_op = FourierOp.from_kdata(kdata)
        image_fourier_op = FastFourierOp(
            recon_matrix=kdata.header.recon_matrix, encoding_matrix=kdata.header.encoding_matrix
        )
        ones = kdata.data.new_ones(*kdata.shape[:-4], 1, *kdata.shape[-3:])
        (support,) = sampling_op.H(ones)
        support = support.real.bool().reshape(-1, *support.shape[-3:])
        if (support.flatten(start_dim=-3).sum(-1) != prod(kdata.shape[-3:])).any():
            raise ValueError('GRAPPA reconstruction requires distinct samples inside the encoding grid.')
        if not (support == support[0]).all():
            raise ValueError('GRAPPA reconstruction requires a shared sampling pattern across batches.')
        acquired = support[0]
        missing_indices = (~acquired).flatten().nonzero().flatten()
        padding = tuple(p for k in reversed(self.kernel_size) for p in (k // 2, k // 2))
        neighborhoods = sliding_window(pad(acquired, padding), self.kernel_size).reshape(-1, prod(self.kernel_size))
        patterns, assignments = torch.unique(neighborhoods[missing_indices], dim=0, return_inverse=True)
        kernels = [
            grappa_calibration(self.acs, pattern.reshape(self.kernel_size), regularization=self.regularization)
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
            grid = grappa_interpolation(grid, kernel, locations)
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
