"""Tests the CsmData class."""

from collections.abc import Callable

import pytest
import torch
from mr2.algorithms.csm import espirit
from mr2.data import CsmData, KData, KHeader, KTrajectory, SpatialDimension
from mr2.data.traj_calculators.KTrajectoryCartesian import KTrajectoryCartesian
from mr2.data.traj_calculators.KTrajectoryIsmrmrd import KTrajectoryIsmrmrd
from mr2.operators import FourierOp, ZeroPadOp

from tests import relative_image_difference
from tests.algorithms.csm.test_walsh import multi_coil_image
from tests.phantoms import EllipsePhantomTestData


def create_matching_idata_kdata(ismrmrd_cart_single_rep):
    """Create matching idata and kdata for csm tests."""
    kdata = KData.from_file(ismrmrd_cart_single_rep.filename, KTrajectoryCartesian())
    ellipses = EllipsePhantomTestData(n_y=kdata.header.recon_matrix.y, n_x=kdata.header.recon_matrix.x)
    idata, _ = multi_coil_image(n_coils=4, ph_ellipse=ellipses, random_kheader=kdata.header)
    fourier_op = FourierOp.from_kdata(kdata)
    kdata = KData(kdata.header, fourier_op(idata.data)[0], traj=kdata.traj)
    return idata, kdata


@pytest.mark.parametrize('csm_method', [CsmData.from_idata_walsh, CsmData.from_idata_inati])
def test_CsmData_smoothing_width(
    csm_method: Callable, ellipse_phantom: EllipsePhantomTestData, random_kheader: KHeader
) -> None:
    """CsmData SpatialDimension and int for smoothing width."""
    idata, _ = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps using SpatialDimension for smoothing width
    smoothing_width = SpatialDimension(z=1, y=5, x=5)
    csm_using_spatial_dimension = csm_method(idata, smoothing_width)

    # Estimate coil sensitivity maps using int for smoothing width
    csm_using_int = csm_method(idata, smoothing_width=5)

    # assert that both coil sensitivity maps are equal, not just close
    assert torch.equal(csm_using_spatial_dimension.data, csm_using_int.data)


@pytest.mark.cuda
@pytest.mark.parametrize('csm_method', [CsmData.from_idata_walsh, CsmData.from_idata_inati])
def test_CsmData_cuda(csm_method: Callable, ellipse_phantom: EllipsePhantomTestData, random_kheader: KHeader) -> None:
    """CsmData obtained on GPU in CUDA memory."""
    idata, csm_ref = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps
    smoothing_width = SpatialDimension(z=1, y=5, x=5)
    csm = csm_method(idata.cuda(), smoothing_width)

    # Phase is only relative in csm calculation, therefore only the abs values are compared.
    assert relative_image_difference(torch.abs(csm.data), torch.abs(csm_ref.cuda())) <= 0.01


@pytest.mark.parametrize('csm_method', [CsmData.from_idata_walsh, CsmData.from_idata_inati])
def test_CsmData_downsampling(
    csm_method: Callable, ellipse_phantom: EllipsePhantomTestData, random_kheader: KHeader
) -> None:
    """CsmData downsampling does not change smooth coil sensitivity maps."""
    idata, _ = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps on original image size with smoothing width = 3
    smoothing_width = SpatialDimension(z=1, y=3, x=3)
    csm_from_original_image = csm_method(idata, smoothing_width)

    # Estimate coil sensitivity maps on 3x downsampled image with smoothing width = 1
    csm_from_lowres_image = csm_method(idata, downsampled_size=idata.data.shape[-1] // 3, smoothing_width=1)

    # assert that both coil sensitivity maps are equal, not just close
    torch.testing.assert_close(csm_from_original_image.data, csm_from_lowres_image.data, rtol=1e-2, atol=1e-2)


def test_CsmData_kdata_walsh(ismrmrd_cart_single_rep) -> None:
    """CsmData using Walsh method should be the same for idata and kdata."""
    idata, kdata = create_matching_idata_kdata(ismrmrd_cart_single_rep)
    csm_from_kdata = CsmData.from_kdata_walsh(kdata)
    csm_from_idata = CsmData.from_idata_walsh(idata)
    torch.testing.assert_close(csm_from_kdata.data, csm_from_idata.data, rtol=1e-5, atol=1e-5)


def test_CsmData_kdata_inati(ismrmrd_cart_single_rep) -> None:
    """CsmData using Inati method should be the same for idata and kdata."""
    idata, kdata = create_matching_idata_kdata(ismrmrd_cart_single_rep)
    csm_from_kdata = CsmData.from_kdata_inati(kdata)
    csm_from_idata = CsmData.from_idata_inati(idata)
    torch.testing.assert_close(csm_from_kdata.data, csm_from_idata.data, rtol=1e-5, atol=1e-5)


def test_CsmData_walsh_align_phase_matches_explicit_call(
    ellipse_phantom: EllipsePhantomTestData, random_kheader: KHeader
) -> None:
    """CsmData Walsh uses phase alignment by default."""
    idata, _ = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    csm_default = CsmData.from_idata_walsh(idata)
    csm_no_phase = CsmData.from_idata_walsh(idata, align_phase=False)
    csm_with_phase = CsmData.from_idata_walsh(idata, align_phase=True)

    torch.testing.assert_close(csm_default.data, csm_with_phase.data)
    assert not torch.allclose(csm_default.data, csm_no_phase.data)


def test_CsmData_kdata_espirit_requires_covered_center(ismrmrd_cart_random_us) -> None:
    """ESPIRiT rejects Cartesian data without a complete centered region."""
    kdata = KData.from_file(ismrmrd_cart_random_us.filename, KTrajectoryCartesian())

    with pytest.raises(ValueError, match='covered centered ACS'):
        CsmData.from_kdata_espirit(kdata)


def test_CsmData_kdata_espirit_regrids_noncartesian(ismrmrd_rad) -> None:
    """ESPIRiT prepares non-Cartesian ACS and returns finite maps."""
    kdata = KData.from_file(ismrmrd_rad.filename, KTrajectoryIsmrmrd())

    width = kdata.acs().shape[-1]
    if width < 6:
        with pytest.raises(ValueError, match='at least kernel_width'):
            CsmData.from_kdata_espirit(kdata)
    csm = CsmData.from_kdata_espirit(kdata, kernel_width=min(6, width))

    # Dataclass stores repeated map dimensions as singletons.
    expected_shape = (*kdata.shape[:-3], *kdata.header.recon_matrix.zyx)
    assert torch.broadcast_shapes(csm.data.shape, expected_shape) == expected_shape
    assert torch.isfinite(csm.data).all()


def test_CsmData_kdata_espirit_cartesian(ismrmrd_cart_single_rep) -> None:
    """Cartesian ESPIRiT preserves batch and coil axes and uses the reconstruction size."""
    _, kdata = create_matching_idata_kdata(ismrmrd_cart_single_rep)
    csm = CsmData.from_kdata_espirit(kdata)

    assert csm.data.shape == (*kdata.data.shape[:-3], *kdata.header.recon_matrix.zyx)
    assert torch.isfinite(csm.data).all()
    assert csm.data.abs().max() > 0


def test_CsmData_kdata_espirit_indexed_acs(ismrmrd_cart_single_rep) -> None:
    """An indexed central ACS block keeps the original encoding and reconstruction metadata."""
    _, kdata = create_matching_idata_kdata(ismrmrd_cart_single_rep)
    ny, nx = kdata.data.shape[-2:]
    acs = kdata[..., ny // 2 - 8 : ny // 2 + 8, nx // 2 - 8 : nx // 2 + 8]
    assert acs.header.encoding_matrix.zyx == kdata.header.encoding_matrix.zyx
    assert acs.data.shape[-3:] != acs.header.encoding_matrix.zyx

    csm = CsmData.from_kdata_espirit(acs, n_iterations=3)
    expected = espirit(acs.acs()[0], acs.header.encoding_matrix, crop_threshold=0.8, n_iterations=3)
    expected = ZeroPadOp((-3, -2, -1), acs.header.encoding_matrix.zyx, acs.header.recon_matrix.zyx)(expected)[0]

    torch.testing.assert_close(csm.data[0], expected)
    assert csm.data.shape == (*acs.data.shape[:-3], *acs.header.recon_matrix.zyx)


def test_CsmData_kdata_espirit_reversed_readouts(ismrmrd_cart_single_rep) -> None:
    """Trajectory-based reordering handles reversed readouts and permuted phase encodes."""
    _, kdata = create_matching_idata_kdata(ismrmrd_cart_single_rep)
    reversed_lines = (torch.arange(kdata.data.shape[-2]) % 2 == 0).reshape(1, 1, 1, -1, 1)
    kx = kdata.traj.kx.expand(kdata.traj.shape)
    acquired_data = torch.where(reversed_lines, kdata.data.flip(-1), kdata.data).flip(-2)
    acquired_traj = KTrajectory(
        kz=kdata.traj.kz,
        ky=kdata.traj.ky.flip(-2),
        kx=torch.where(reversed_lines, kx.flip(-1), kx).flip(-2),
    )
    acquired = KData(kdata.header, acquired_data, acquired_traj)

    expected = CsmData.from_kdata_espirit(kdata)
    actual = CsmData.from_kdata_espirit(acquired)

    torch.testing.assert_close(actual.data, expected.data)


def test_CsmData_kdata_espirit_duplicate_coordinates(ismrmrd_cart_single_rep) -> None:
    """Shape-matching Cartesian data with a duplicate and missing line is rejected."""
    kdata = KData.from_file(ismrmrd_cart_single_rep.filename, KTrajectoryCartesian())
    ky = kdata.traj.ky.clone()
    center = ky.shape[-2] // 2
    ky[..., center, :] = ky[..., center + 1, :]
    invalid = KData(kdata.header, kdata.data, KTrajectory(kdata.traj.kz, ky, kdata.traj.kx))

    with pytest.raises(ValueError, match='covered centered ACS'):
        CsmData.from_kdata_espirit(invalid)


def test_CsmData_espirit_rejects_inconsistent_fov(ismrmrd_cart_single_rep) -> None:
    """Do not silently apply encoding-grid sensitivities at a different voxel spacing."""
    kdata = KData.from_file(ismrmrd_cart_single_rep.filename, KTrajectoryCartesian())
    kdata.header.recon_fov = kdata.header.recon_fov * 2
    with pytest.raises(ValueError, match='reconstruction geometry'):
        CsmData.from_kdata_espirit(kdata)


def test_CsmData_espirit_rejects_acs_smaller_than_kernel(ismrmrd_cart_single_rep) -> None:
    """Do not silently reduce the calibration kernel on active encoding axes."""
    kdata = KData.from_file(ismrmrd_cart_single_rep.filename, KTrajectoryCartesian())
    with pytest.raises(ValueError, match='at least kernel_width'):
        CsmData.from_kdata_espirit(kdata, calibration_size=3, kernel_width=6)
