"""Tests for KTrajectory class."""

from pathlib import Path

import ismrmrd
import pytest
import torch
from einops import rearrange
from mr2.data import KTrajectory, SpatialDimension
from mr2.data.enums import TrajType
from mr2.data.KData import has_n_coils
from mr2.data.traj_calculators import KTrajectoryIsmrmrd

from tests.conftest import COMMON_MR_TRAJECTORIES, create_traj
from tests.data import IsmrmrdRawTestData


@pytest.mark.parametrize('n_dimensions', [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('use_calculator', [False, True])
def test_ismrmrd_trajectory_dimensions(
    tmp_path: Path, n_dimensions: int, normalize: bool, use_calculator: bool
) -> None:
    """ISMRMRD imports preserve supported coordinates and reject missing or extra ones."""
    raw = IsmrmrdRawTestData(tmp_path / 'source.h5', matrix_size=4, n_coils=1, oversampling=1)
    source = ismrmrd.Dataset(raw.filename)
    header = source.read_xml_header()
    acquisition = source.read_acquisition(raw.n_noise_samples)
    source.close()
    acquisition.resize(4, 1, trajectory_dimensions=n_dimensions)
    coordinates = torch.arange(4 * n_dimensions, dtype=torch.float32).reshape(4, n_dimensions)
    acquisition.traj[:] = coordinates.numpy()
    filename = tmp_path / 'trajectory.h5'
    dataset = ismrmrd.Dataset(filename)
    dataset.write_xml_header(header)
    dataset.append_acquisition(acquisition)
    dataset.close()

    def read() -> KTrajectory:
        if use_calculator:
            return KTrajectoryIsmrmrd(normalize)([acquisition], SpatialDimension(z=1, y=4, x=4))
        return KTrajectory.from_ismrmrd(filename, normalize=normalize)

    if n_dimensions == 0:
        with pytest.raises(ValueError, match='No trajectory information available'):
            read()
    elif n_dimensions > 3:
        with pytest.raises(ValueError, match='Trajectory must have 1, 2 or 3 coordinate dimensions'):
            read()
    else:
        trajectory = read()
        for axis, actual in enumerate((trajectory.kx, trajectory.ky, trajectory.kz)):
            expected = coordinates[:, axis] if axis < n_dimensions else torch.zeros(4)
            if normalize:
                expected = expected * (2 / expected.abs().max()) if axis < 2 and expected.any() else torch.zeros(4)
            torch.testing.assert_close(actual.broadcast_to(1, 1, 1, 1, 4).flatten(), expected)


def test_trajectory_repeat_detection_tol(cartesian_grid) -> None:
    """Test the automatic detection of repeated values."""
    n_k0 = 10
    n_k1 = 20
    n_k2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(n_k2, n_k1, n_k0, jitter=0.1)
    trajectory_sparse = KTrajectory(kz_full, ky_full, kx_full, repeat_detection_tolerance=0.1)

    assert trajectory_sparse.shape == (1, 1, n_k2, n_k1, n_k0)
    assert trajectory_sparse.kx.shape == (1, 1, 1, 1, n_k0)
    assert trajectory_sparse.ky.shape == (1, 1, 1, n_k1, 1)
    assert trajectory_sparse.kz.shape == (1, 1, n_k2, 1, 1)


def test_trajectory_repeat_detection_exact(cartesian_grid) -> None:
    """Test the automatic detection of repeated values."""
    n_k0 = 10
    n_k1 = 20
    n_k2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(n_k2, n_k1, n_k0, jitter=0.1)
    trajectory_full = KTrajectory(kz_full, ky_full, kx_full, repeat_detection_tolerance=None)

    assert trajectory_full.shape == (1, 1, n_k2, n_k1, n_k0)
    assert trajectory_full.kx.shape == (1, 1, n_k2, n_k1, n_k0)
    assert trajectory_full.ky.shape == (1, 1, n_k2, n_k1, n_k0)
    assert trajectory_full.kz.shape == (1, 1, n_k2, n_k1, n_k0)


def test_trajectory_tensor_conversion(cartesian_grid) -> None:
    n_k0 = 10
    n_k1 = 20
    n_k2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(n_k2, n_k1, n_k0, jitter=0.0)
    trajectory = KTrajectory(kz_full, ky_full, kx_full)
    tensor = torch.stack((kz_full, ky_full, kx_full), dim=0).to(torch.float32)

    tensor_from_traj = trajectory.as_tensor()  # stack_dim=0
    tensor_from_traj_dim2 = rearrange(
        trajectory.as_tensor(stack_dim=2), 'other coils dim k2 k1 k0->dim other coils k2 k1 k0'
    )
    tensor_from_traj_from_tensor_dim3 = KTrajectory.from_tensor(
        rearrange(tensor, 'dim other coils k2 k1 k0->other coils k2 dim k1 k0'), stack_dim=3
    ).as_tensor()
    tensor_from_traj_from_tensor = KTrajectory.from_tensor(tensor).as_tensor()  # stack_dim=0

    torch.testing.assert_close(tensor, tensor_from_traj)
    torch.testing.assert_close(tensor, tensor_from_traj_dim2)
    torch.testing.assert_close(tensor, tensor_from_traj_from_tensor)
    torch.testing.assert_close(tensor, tensor_from_traj_from_tensor_dim3)


def test_trajectory_raise_not_broadcastable() -> None:
    """Non broadcastable shapes should raise."""
    kx = ky = torch.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
    kz = torch.arange(1 * 2 * 3 * 100).reshape(1, 2, 3, 100)
    with pytest.raises(ValueError):
        KTrajectory(kz, ky, kx)


def test_trajectory_raise_wrong_dim() -> None:
    """Wrong number of dimensions after broadcasting should raise."""
    kx = ky = kz = torch.arange(1 * 2 * 3).reshape(1, 2, 3)
    with pytest.raises(ValueError):
        KTrajectory(kz, ky, kx)


def test_trajectory_to_float64(cartesian_grid) -> None:
    """Change KTrajectory dtype to float64."""
    n_k0 = 10
    n_k1 = 20
    n_k2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(n_k2, n_k1, n_k0, jitter=0.0)
    trajectory = KTrajectory(kz_full, ky_full, kx_full)
    trajectory_float64 = trajectory.to(dtype=torch.float64)
    assert trajectory_float64.kz.dtype == torch.float64
    assert trajectory_float64.ky.dtype == torch.float64
    assert trajectory_float64.kx.dtype == torch.float64
    assert trajectory.kz.dtype == torch.float32
    assert trajectory.ky.dtype == torch.float32
    assert trajectory.kx.dtype == torch.float32


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.int32, torch.int64])
def test_trajectory_floating_dtype(dtype: torch.dtype) -> None:
    """Test if the trajectory will always be converted to float"""
    ks = torch.ones(3, 1, 1, 1, 1, 1, dtype=dtype)
    traj = KTrajectory(*ks)
    if dtype.is_floating_point:
        # keep as as
        assert traj.kz.dtype == dtype
        assert traj.ky.dtype == dtype
        assert traj.kx.dtype == dtype
    else:
        # convert to float32
        assert traj.kz.dtype == torch.float32
        assert traj.ky.dtype == torch.float32
        assert traj.kx.dtype == torch.float32


@pytest.mark.cuda
def test_trajectory_cuda(cartesian_grid) -> None:
    """Move KTrajectory object to CUDA memory."""
    n_k0 = 10
    n_k1 = 20
    n_k2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(n_k2, n_k1, n_k0, jitter=0.0)
    trajectory = KTrajectory(kz_full, ky_full, kx_full)

    trajectory_cuda = trajectory.cuda()
    assert trajectory_cuda.kz.is_cuda
    assert trajectory_cuda.ky.is_cuda
    assert trajectory_cuda.kx.is_cuda

    assert trajectory.kz.is_cpu
    assert trajectory.ky.is_cpu
    assert trajectory.kx.is_cpu

    assert trajectory_cuda.is_cuda
    assert trajectory.is_cpu

    assert not trajectory_cuda.is_cpu
    assert not trajectory.is_cuda


@pytest.mark.cuda
def test_trajectory_cpu(cartesian_grid) -> None:
    """Move KTrajectory object to CUDA memory and back to CPU memory."""
    n_k0 = 10
    n_k1 = 20
    n_k2 = 30
    kz_full, ky_full, kx_full = cartesian_grid(n_k2, n_k1, n_k0, jitter=0.0)
    trajectory = KTrajectory(kz_full, ky_full, kx_full)

    trajectory_cpu = trajectory.cuda().cpu()
    assert trajectory_cpu.kz.is_cpu
    assert trajectory_cpu.ky.is_cpu
    assert trajectory_cpu.kx.is_cpu


@COMMON_MR_TRAJECTORIES
def test_ktype_along_kzyx(
    img_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
) -> None:
    """Test identification of traj types."""

    # Generate random k-space trajectories
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)

    # Find out the type of the kz, ky and kz dimensions
    single_value_dims = [d for d, s in zip((-3, -2, -1), (type_kz, type_ky, type_kx), strict=True) if s == 'zero']
    on_grid_dims = [d for d, s in zip((-3, -2, -1), (type_kz, type_ky, type_kx), strict=True) if s == 'uniform']
    not_on_grid_dims = [d for d, s in zip((-3, -2, -1), (type_kz, type_ky, type_kx), strict=True) if s == 'non-uniform']

    # Make sure all dimensions are covered
    if len(single_value_dims) + len(on_grid_dims) + len(not_on_grid_dims) != 3:
        raise ValueError(
            f'{single_value_dims=}, {on_grid_dims=} and {not_on_grid_dims=} do not cover all dimensions. ',
            'There must be an error in the test itself.',
        )

    # check dimensions which are of shape 1 and do not need any transform
    assert all(trajectory.type_along_kzyx[dim] & TrajType.SINGLEVALUE for dim in single_value_dims)

    # Check dimensions which are on a grid and require FFT
    assert all(trajectory.type_along_kzyx[dim] & TrajType.ONGRID for dim in on_grid_dims)

    # Check dimensions which are not on a grid and require NUFFT
    assert all(
        ~(trajectory.type_along_kzyx[dim] & (TrajType.SINGLEVALUE | TrajType.ONGRID)) for dim in not_on_grid_dims
    )


@COMMON_MR_TRAJECTORIES
def test_ktype_along_k210(
    img_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
) -> None:
    """Test identification of traj types."""

    # Generate random k-space trajectories
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)

    # Find out the type of the k2, k1 and k0 dimensions
    single_value_dims = [d for d, s in zip((-3, -2, -1), (type_k2, type_k1, type_k0), strict=True) if s == 'zero']
    on_grid_dims = [d for d, s in zip((-3, -2, -1), (type_k2, type_k1, type_k0), strict=True) if s == 'uniform']
    not_on_grid_dims = [d for d, s in zip((-3, -2, -1), (type_k2, type_k1, type_k0), strict=True) if s == 'non-uniform']

    # Make sure all dimensions are covered
    if len(single_value_dims) + len(on_grid_dims) + len(not_on_grid_dims) != 3:
        raise ValueError(
            f'{single_value_dims=}, {on_grid_dims=} and {not_on_grid_dims=} do not cover all dimensions. ',
            'There must be an error in the test itself.',
        )

    # check dimensions which are of shape 1 and do not need any transform
    assert all(trajectory.type_along_k210[dim] & TrajType.SINGLEVALUE for dim in single_value_dims)

    # Check dimensions which are on a grid and require FFT
    assert all(trajectory.type_along_k210[dim] & TrajType.ONGRID for dim in on_grid_dims)

    # Check dimensions which are not on a grid and require NUFFT
    assert all(
        ~(trajectory.type_along_k210[dim] & (TrajType.SINGLEVALUE | TrajType.ONGRID)) for dim in not_on_grid_dims
    )


def test_traj_from_ismrmrd(ismrmrd_cart_random_us) -> None:
    """Test reading trajectory from ISMRMRD file."""
    traj = KTrajectory.from_ismrmrd(ismrmrd_cart_random_us.filename)
    n_other = (
        ismrmrd_cart_random_us.matrix_size // ismrmrd_cart_random_us.acceleration * ismrmrd_cart_random_us.repetitions
        + ismrmrd_cart_random_us.n_noise_samples
    )
    n_k0 = ismrmrd_cart_random_us.matrix_size * ismrmrd_cart_random_us.oversampling
    assert traj.kx.shape == (n_other, 1, 1, 1, n_k0)
    assert traj.ky.shape == (n_other, 1, 1, 1, 1)
    assert traj.kz.shape == (1, 1, 1, 1, 1)


def test_traj_from_ismrmrd_normalize(ismrmrd_rad) -> None:
    """Test reading trajectory from ISMRMRD file with normalization."""
    traj = KTrajectory.from_ismrmrd(ismrmrd_rad.filename, normalize=True)
    n_other = (
        ismrmrd_rad.matrix_size // ismrmrd_rad.acceleration * ismrmrd_rad.repetitions + ismrmrd_rad.n_noise_samples
    )
    n_k0 = ismrmrd_rad.matrix_size * ismrmrd_rad.oversampling
    assert traj.kx.shape == (n_other, 1, 1, 1, n_k0)
    assert traj.ky.shape == (n_other, 1, 1, 1, n_k0)
    assert traj.kz.shape == (1, 1, 1, 1, 1)
    assert traj.kx.abs().amax() == ismrmrd_rad.matrix_size * ismrmrd_rad.oversampling / 2
    assert traj.ky.abs().amax() == ismrmrd_rad.matrix_size * ismrmrd_rad.oversampling / 2


def test_traj_from_ismrmrd_filter(ismrmrd_cart_bodycoil_and_surface_coil) -> None:
    """Test reading trajectory from ISMRMRD file using a filter."""
    with pytest.raises(ValueError, match=r'No matching acquisitions found'):
        _ = KTrajectory.from_ismrmrd(
            ismrmrd_cart_bodycoil_and_surface_coil.filename,
            acquisition_filter_criterion=lambda x: has_n_coils(1, x),  # there are no 1-coil acquisitions
        )
    traj = KTrajectory.from_ismrmrd(
        ismrmrd_cart_bodycoil_and_surface_coil.filename,
        acquisition_filter_criterion=lambda x: has_n_coils(2, x),
    )
    assert traj.shape == (1, 1, 1, 1, 1)  # trajectory is all zeros
