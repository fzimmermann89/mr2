"""Integration tests for the thin plug-and-play reconstruction wrapper."""

import pytest
import torch
from mr2.algorithms import prewhiten_kspace
from mr2.algorithms.optimizers import admm_l2
from mr2.algorithms.optimizers.admm import ADMML2Status
from mr2.algorithms.reconstruction import DirectReconstruction, PlugAndPlayReconstruction
from mr2.data import CsmData, IData, KData, KNoise
from mr2.operators import DensityCompensationOp, FourierOp, IdentityOp, SensitivityOp


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
    statuses: list[ADMML2Status] = []

    def denoiser(x: torch.Tensor) -> torch.Tensor:
        return 0.8 * x

    reconstruction = PlugAndPlayReconstruction(
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
    (expected,) = admm_l2(
        lambda x, **_: (denoiser(x),),
        acquisition,
        whitened.data,
        IdentityOp(),
        direct(whitened).data,
        stopping_criterion='residual',
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
        if with_noise and csm_kind == 'explicit':
            assert noise is not None
            maps = reference.csm_op.csm_tensor
            covariance = noise.data.flatten(-3) @ noise.data.flatten(-3).mH / noise.data.flatten(-3).shape[-1]
            expected_maps = torch.linalg.solve_triangular(
                torch.linalg.cholesky(covariance), maps.flatten(-3), upper=False
            ).reshape_as(maps)
            torch.testing.assert_close(reconstruction.csm_op.csm_tensor, expected_maps)
        else:
            torch.testing.assert_close(reconstruction.csm_op.csm_tensor, reference.csm_op.csm_tensor)


def test_zero_kspace(cartesian_kdata: KData) -> None:
    """Zero k-space gives finite zero images with DCF initialization."""
    kdata = cartesian_kdata.clone()
    # Assign after construction so Dataclass does not collapse constant spatial axes.
    kdata.data = torch.zeros_like(cartesian_kdata.data)
    reconstruction = PlugAndPlayReconstruction(
        kdata=kdata, csm=None, denoiser=lambda x: x, tau=1, tolerance=1e-5, max_iterations=5
    )
    result = reconstruction(kdata)
    torch.testing.assert_close(result.data, torch.zeros_like(result.data))


@pytest.mark.parametrize('as_operator', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_explicit_csm_whitening(cartesian_kdata: KData, as_operator: bool, device: str) -> None:
    """Non-diagonal noise transforms both measured data and explicit coil maps."""
    kdata = cartesian_kdata
    maps = CsmData.from_kdata_walsh(kdata)
    n_coils = kdata.data.shape[-4]
    mixing = torch.eye(n_coils, dtype=kdata.data.dtype)
    if n_coils > 1:
        mixing[1, 0] = torch.tensor(0.5 + 0.3j, dtype=mixing.dtype)
    mixing[0, 0] = 2
    noise_data = mixing.reshape(n_coils, 1, 1, n_coils)
    noise = KNoise(noise_data)
    covariance = mixing @ mixing.mH / n_coils
    factor = torch.linalg.cholesky(covariance)
    expected_maps = torch.linalg.solve_triangular(factor, maps.data.flatten(-3), upper=False).reshape_as(maps.data)
    fourier = FourierOp.from_kdata(kdata)
    dcf = DensityCompensationOp(torch.tensor(1.0))
    reconstruction = PlugAndPlayReconstruction(
        fourier_op=fourier,
        csm=maps.as_operator() if as_operator else maps,
        noise=noise,
        dcf=dcf,
        denoiser=lambda x: 0.8 * x,
        tau=0.7,
        max_iterations=3,
        cg_max_iterations=8,
    )
    reconstruction = reconstruction.to(device)
    moved_kdata = kdata.to(device)
    acquisition = fourier @ SensitivityOp(expected_maps.to(device))
    whitened = torch.linalg.solve_triangular(factor.to(device), moved_kdata.data.flatten(-3), upper=False).reshape_as(
        moved_kdata.data
    )
    (initial,) = acquisition.H(whitened)
    (expected,) = admm_l2(
        lambda x, **_: (0.8 * x,),
        acquisition,
        whitened,
        IdentityOp(),
        initial,
        stopping_criterion='residual',
        tau=0.7,
        max_iterations=3,
        cg_max_iterations=8,
    )
    torch.testing.assert_close(reconstruction(moved_kdata).data, expected)
    assert noise.data.device.type == 'cpu'
    # The supplied maps remain in their original coil basis.
    torch.testing.assert_close(maps.data, CsmData.from_kdata_walsh(kdata).data)
