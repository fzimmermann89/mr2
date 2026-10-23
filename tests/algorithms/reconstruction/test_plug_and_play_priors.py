"""Integration tests for the thin plug-and-play reconstruction wrapper."""

import pytest
import torch
from mr2.algorithms import prewhiten_kspace
from mr2.algorithms.optimizers import pnp_admm
from mr2.algorithms.optimizers.pnp_admm import PNPADMMStatus
from mr2.algorithms.reconstruction import DirectReconstruction, PlugAndPlayPriorsReconstruction
from mr2.data import CsmData, IData, KData, KNoise
from mr2.operators import DensityCompensationOp, FourierOp


@pytest.mark.parametrize('with_noise', [False, True])
@pytest.mark.parametrize('csm_kind', ['none', 'explicit', 'callable'])
def test_wrapper_matches_tensor_solver(cartesian_kdata: KData, with_noise: bool, csm_kind: str) -> None:
    """Preserve acquisition setup, prewhitening and DCF-only initialization."""
    kdata = cartesian_kdata
    n_coils = kdata.data.shape[-4]
    noise = (
        KNoise(2 * torch.eye(n_coils, dtype=kdata.data.dtype).reshape(n_coils, 1, 1, n_coils)) if with_noise else None
    )
    csm = (
        CsmData.from_idata_walsh
        if csm_kind == 'callable'
        else (CsmData.from_kdata_walsh(kdata) if csm_kind == 'explicit' else None)
    )
    # Spatially varying DCF makes accidental weighting of the data term observable.
    weights = torch.linspace(0.5, 1.5, kdata.data.shape[-1]).reshape(1, 1, 1, -1)
    dcf = DensityCompensationOp(weights)
    fourier = FourierOp.from_kdata(kdata)
    statuses: list[PNPADMMStatus] = []

    def denoiser(x: torch.Tensor) -> torch.Tensor:
        return 0.8 * x

    reconstruction = PlugAndPlayPriorsReconstruction(
        kdata=kdata,
        fourier_op=fourier,
        csm=csm,
        noise=noise,
        dcf=dcf,
        denoiser=denoiser,
        tau=0.7,
        max_iterations=2,
        cg_max_iterations=5,
        callback=statuses.append,
    )
    result = reconstruction(kdata)
    whitened = prewhiten_kspace(kdata, noise) if noise is not None else kdata
    direct = DirectReconstruction(fourier_op=fourier, csm=reconstruction.csm_op, dcf=dcf)
    acquisition = fourier if reconstruction.csm_op is None else fourier @ reconstruction.csm_op
    (expected,) = pnp_admm(
        denoiser,
        acquisition,
        whitened.data,
        direct(whitened).data,
        tau=0.7,
        max_iterations=2,
        cg_max_iterations=5,
    )
    assert isinstance(result, IData)
    assert result.data.shape[-3:] == kdata.header.recon_matrix.zyx
    assert result.data.dtype == kdata.data.dtype
    assert len(statuses) == 2
    torch.testing.assert_close(result.data, expected)
    # Compare CSM estimation to the established noise-aware setup as well.
    reference = DirectReconstruction(kdata=kdata, fourier_op=fourier, csm=csm, noise=noise, dcf=dcf)
    if reference.csm_op is not None:
        assert reconstruction.csm_op is not None
        torch.testing.assert_close(reconstruction.csm_op.csm_tensor, reference.csm_op.csm_tensor)


def test_zero_kspace(cartesian_kdata: KData) -> None:
    """Zero k-space gives finite zero images with DCF initialization."""
    kdata = cartesian_kdata.clone()
    # Assign after construction so Dataclass does not collapse constant spatial axes.
    kdata.data = torch.zeros_like(cartesian_kdata.data)
    reconstruction = PlugAndPlayPriorsReconstruction(
        kdata=kdata, csm=None, denoiser=lambda x: x, tau=1, tolerance=1e-5, max_iterations=5
    )
    result = reconstruction(kdata)
    torch.testing.assert_close(result.data, torch.zeros_like(result.data))
