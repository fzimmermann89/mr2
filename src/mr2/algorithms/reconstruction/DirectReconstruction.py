"""Direct Reconstruction by Adjoint Fourier Transform."""

from collections.abc import Callable

import torch

from mr2.algorithms.reconstruction.Reconstruction import Reconstruction
from mr2.data.CsmData import CsmData
from mr2.data.DcfData import DcfData
from mr2.data.IData import IData
from mr2.data.KData import KData
from mr2.data.KNoise import KNoise
from mr2.operators.DensityCompensationOp import DensityCompensationOp
from mr2.operators.FourierOp import FourierOp
from mr2.operators.LinearOperator import LinearOperator
from mr2.operators.SensitivityOp import SensitivityOp
from mr2.utils import unsqueeze_right


class DirectReconstruction(Reconstruction):
    """Direct Reconstruction by Adjoint Fourier Transform."""

    def __init__(
        self,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable[[IData], CsmData] | CsmData | SensitivityOp | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | DensityCompensationOp | None = None,
    ):
        """Initialize DirectReconstruction.

        A direct reconstruction uses the adjoint of the acquisition operator and a
        density compensation to obtain the complex valued images from k-space data.

        If csm is not set to `None`, a single coil combined image will reconstructed.
        The method for estimating sensitivity maps can be adjusted using the `csm` argument.

        Parameters
        ----------
        kdata
            If `kdata` is provided and `fourier_op` or `dcf` are `None`, then `fourier_op` and `dcf` are estimated
            based on `kdata`. Otherwise `fourier_op` and `dcf` are used as provided.
        fourier_op
            Instance of the `~mr2.operators.FourierOp` used for reconstruction.
            If `None`, set up based on `kdata`.
        csm
            Sensitivity maps for coil combination. If `None`, no coil combination is carried out, i.e. images for each
            coil are returned. If a `Callable` is provided, coil images are reconstructed using the adjoint of the
            `~mr2.operators.FourierOp` (including density compensation) and then sensitivity maps are calculated
            using the callable. For this, `kdata` needs also to be provided.
            For examples have a look at the `~mr2.data.CsmData` class e.g. `~mr2.data.CsmData.from_idata_walsh`
            or `~mr2.data.CsmData.from_idata_inati`.
        noise
            Noise used for prewhitening. If `None`, no prewhitening is performed
        dcf
            K-space sampling density compensation. If `None`, set up based on `kdata`.

        Raises
        ------
        `ValueError`
            If the `kdata` and `fourier_op` are `None` or if `csm` is a `Callable` but `kdata` is None.
        """
        super().__init__()
        if fourier_op is None:
            if kdata is None:
                raise ValueError('Either kdata or fourier_op needs to be defined.')
            self.fourier_op = FourierOp.from_kdata(kdata)
        else:
            self.fourier_op = fourier_op

        if kdata is not None and dcf is None:
            self.dcf_op = DcfData.from_traj_voronoi(kdata.traj).as_operator()
        else:
            self.dcf_op = dcf.as_operator() if isinstance(dcf, DcfData) else dcf

        self.noise = noise

        if csm is None or isinstance(csm, CsmData | SensitivityOp):
            self.csm_op = csm.as_operator() if isinstance(csm, CsmData) else csm
        else:
            if kdata is None:
                raise ValueError('kdata needs to be defined to calculate the sensitivity maps.')
            self.recalculate_csm(kdata, csm)

    def forward(self, kdata: KData) -> IData:
        """Apply the reconstruction.

        Parameters
        ----------
        kdata
            k-space data to reconstruct.

        Returns
        -------
            the reconstruced image.
        """
        return self.direct_reconstruction(kdata)

    def _iterative_initial_value(
        self, acquisition_model: LinearOperator, data: torch.Tensor, right_hand_side: torch.Tensor
    ) -> torch.Tensor:
        """Return an initial image estimate for iterative reconstruction.

        If density compensation is available, use the scaled density-compensated adjoint reconstruction. Otherwise,
        return zeros with the image shape expected by the iterative solver.
        """
        if self.dcf_op is None:
            return torch.zeros_like(right_hand_side)

        (u,) = (acquisition_model.H @ self.dcf_op)(data)
        (v,) = (acquisition_model.H @ self.dcf_op @ acquisition_model)(u)
        u_flat = u.flatten(start_dim=-3)
        v_flat = v.flatten(start_dim=-3)
        numerator = torch.linalg.vecdot(u_flat, u_flat).real
        denominator = torch.linalg.vecdot(v_flat, u_flat).real
        valid = denominator > 0
        safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
        scale = torch.where(valid, numerator / safe_denominator, torch.zeros_like(numerator))
        return unsqueeze_right(scale, 3) * u
