"""Plug-and-play reconstruction using the tensor-level ADMM optimizer."""

from collections.abc import Callable

import torch

from mr2.algorithms.optimizers.pnp_admm import PNPADMMStatus, pnp_admm
from mr2.algorithms.prewhiten_kspace import prewhiten_kspace
from mr2.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mr2.data.CsmData import CsmData
from mr2.data.DcfData import DcfData
from mr2.data.IData import IData
from mr2.data.KData import KData
from mr2.data.KNoise import KNoise
from mr2.operators.DensityCompensationOp import DensityCompensationOp
from mr2.operators.LinearOperator import LinearOperator
from mr2.operators.SensitivityOp import SensitivityOp


class PlugAndPlayPriorsReconstruction(DirectReconstruction):
    """Reconstruct with a callable image denoiser and quadratic data consistency.

    Uses `~mr2.algorithms.optimizers.pnp_admm.pnp_admm` for iteration and
    `DirectReconstruction` for acquisition setup. The initial image is the adjoint
    reconstruction, including density compensation if available.
    Density compensation only affects initialization. An arbitrary denoiser
    need not define a regularization objective or guarantee convergence.

    Based on the reconstruction proposed in PTB-MR/mrpro PR #1013.
    """

    def __init__(
        self,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable[[IData], CsmData] | CsmData | SensitivityOp | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | DensityCompensationOp | None = None,
        *,
        denoiser: Callable[[torch.Tensor], torch.Tensor],
        tau: float | torch.Tensor,
        max_iterations: int = 128,
        tolerance: float = 0.0,
        cg_max_iterations: int = 128,
        cg_tolerance: float = 1e-6,
        callback: Callable[[PNPADMMStatus], bool | None] | None = None,
    ) -> None:
        """Initialize the acquisition model and solver controls.

        Parameters
        ----------
        kdata
            Data used to estimate missing Fourier operator, DCF and callable CSM.
        fourier_op
            Fourier operator. If None, estimated from kdata.
        csm
            Coil sensitivity maps, operator, or estimation callable. None disables
            coil combination. See `DirectReconstruction` for setup details.
        noise
            Optional noise data for prewhitening, including CSM estimation.
        dcf
            Density compensation used only for the initial adjoint reconstruction.
            If None and kdata is provided, estimated from its trajectory.
        denoiser
            Tensor callable preserving image shape, dtype and device.
        tau
            Positive finite scalar inverse ADMM penalty, as in `pnp_admm`.
            A penalty rho in the upstream reconstruction corresponds to tau=1/rho.
        max_iterations
            Maximum ADMM iterations.
        tolerance
            Relative primal and dual residual tolerance; zero disables stopping.
        cg_max_iterations
            Maximum CG iterations per ADMM update.
        cg_tolerance
            Positive absolute CG residual tolerance.
        callback
            Optional callback receiving `PNPADMMStatus` after each iteration.
        """
        super().__init__(kdata, fourier_op, csm, noise, dcf)
        self.denoiser = denoiser
        self.tau = torch.as_tensor(tau)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cg_max_iterations = cg_max_iterations
        self.cg_tolerance = cg_tolerance
        self.callback = callback

    def forward(self, kdata: KData) -> IData:
        """Reconstruct k-space data, returning the data-consistent iterate as IData."""
        if self.noise is not None:
            kdata = prewhiten_kspace(kdata, self.noise)
        acquisition_model = self.fourier_op
        if self.csm_op is not None:
            acquisition_model = acquisition_model @ self.csm_op
        initial_data = self.dcf_op(kdata.data)[0] if self.dcf_op is not None else kdata.data
        (initial_value,) = acquisition_model.H(initial_data)
        (image,) = pnp_admm(
            self.denoiser,
            acquisition_model,
            kdata.data,
            initial_value,
            tau=self.tau,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            cg_max_iterations=self.cg_max_iterations,
            cg_tolerance=self.cg_tolerance,
            callback=self.callback,
        )
        return IData.from_tensor_and_kheader(image, kdata.header)
