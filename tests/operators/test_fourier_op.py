"""Tests for Fourier operator."""

from typing import TypeAlias, cast

import pytest
import torch
from mr2.data import KData, KTrajectory, SpatialDimension
from mr2.data.enums import TrajType
from mr2.data.traj_calculators import KTrajectoryCartesian
from mr2.operators import FourierOp
from mr2.operators.LinearOperatorMatrix import LinearOperatorMatrix
from mr2.utils import RandomGenerator
from typing_extensions import Unpack

from tests import (
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)
from tests.conftest import COMMON_MR_TRAJECTORIES, create_traj

AtLeast3Ints: TypeAlias = tuple[int, int, int, Unpack[tuple[int, ...]]]


def create_data(
    img_shape: AtLeast3Ints,
    nkx: AtLeast3Ints,
    nky: AtLeast3Ints,
    nkz: AtLeast3Ints,
    type_kx: str,
    type_ky: str,
    type_kz: str,
) -> tuple[torch.Tensor, KTrajectory]:
    """Create k-space trajectory and random image."""
    rng = RandomGenerator(seed=0)
    img = rng.complex64_tensor(size=img_shape)
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)
    return img, trajectory


def create_fourier_op_and_range_domain(
    img_shape: AtLeast3Ints,
    k_shape: AtLeast3Ints,
    nkx: AtLeast3Ints,
    nky: AtLeast3Ints,
    nkz: AtLeast3Ints,
    type_kx: str,
    type_ky: str,
    type_kz: str,
) -> tuple[FourierOp, torch.Tensor, torch.Tensor]:
    """Create a fourier operator and an element from domain and range."""
    # generate random images and k-space trajectories
    _, trajectory = create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    # create operator
    recon_matrix = SpatialDimension(img_shape[-3], img_shape[-2], img_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    # ensure grid_sampling is also tested by simulating partial echo for standard Cartesian readout
    if type_kx == 'uniform' and nky[-1] == 1 and nkz[-1] == 1:
        trajectory.kx = trajectory.kx[..., nkx[-1] // 4 :]
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)

    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=img_shape)
    v = rng.complex64_tensor(size=(*k_shape[:-1], trajectory.kx.shape[-1]))  # type: ignore[misc]
    return fourier_op, u, v


class NufftTrajektory(KTrajectory):
    """Always returns non-grid trajectory type."""

    def _traj_types(
        self,
        tolerance: float,
    ) -> tuple[tuple[TrajType, TrajType, TrajType], tuple[TrajType, TrajType, TrajType]]:
        true_types = super()._traj_types(tolerance)
        modified = tuple(tuple(t & (~TrajType.ONGRID) for t in ts) for ts in true_types)
        return cast(tuple[tuple[TrajType, TrajType, TrajType], tuple[TrajType, TrajType, TrajType]], modified)


@pytest.mark.parametrize('as_matrix', [False, True], ids=['fourier', 'matrix'])
@pytest.mark.parametrize('trajectory_type', ['uniform', 'non-uniform'], ids=['cartesian', 'nufft'])
@pytest.mark.parametrize(
    ('device', 'dtype'),
    [
        ('cpu', torch.float64),
        pytest.param('cuda', torch.float32, marks=pytest.mark.cuda),
        pytest.param('cuda', torch.float64, marks=pytest.mark.cuda),
    ],
)
def test_fourier_op_cached_gram_after_to(
    as_matrix: bool, trajectory_type: str, device: str, dtype: torch.dtype
) -> None:
    """Rebuild Fourier and matrix grams after device or dtype conversion."""
    fourier_op, image, _ = create_fourier_op_and_range_domain(
        (1, 1, 1, 8, 8),
        (1, 1, 1, 8, 8),
        (1, 1, 1, 1, 8),
        (1, 1, 1, 8, 1),
        (1, 1, 1, 1, 1),
        trajectory_type,
        trajectory_type,
        'zero',
    )
    operator = LinearOperatorMatrix([[fourier_op, 2 * fourier_op]]) if as_matrix else fourier_op
    cached_gram = operator.gram
    assert operator.gram is cached_gram
    operator.to(device=device, dtype=dtype)
    image = image.to(device=device, dtype=dtype.to_complex())
    inputs = (image, image) if as_matrix else (image,)

    actual = operator.gram(*inputs)
    expected = (operator.H @ operator)(*inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert operator.gram is not cached_gram
    assert operator.gram is operator.gram
    assert all(buffer.device.type == device for buffer in operator.gram.buffers())
    # Cartesian masks are always constructed in float32, regardless of the operator's dtype.
    if trajectory_type == 'non-uniform':
        assert all(buffer.dtype == dtype for buffer in operator.gram.buffers() if buffer.is_floating_point())


@COMMON_MR_TRAJECTORIES
def test_fourier_op_fwd_adj_property(
    img_shape: AtLeast3Ints,
    k_shape: AtLeast3Ints,
    nkx: AtLeast3Ints,
    nky: AtLeast3Ints,
    nkz: AtLeast3Ints,
    type_kx: str,
    type_ky: str,
    type_kz: str,
    type_k0: str,
    type_k1: str,
    type_k2: str,
) -> None:
    """Test adjoint property of Fourier operator."""
    dotproduct_adjointness_test(
        *create_fourier_op_and_range_domain(img_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)
    )


@COMMON_MR_TRAJECTORIES
def test_fourier_op_grad(
    img_shape: AtLeast3Ints,
    k_shape: AtLeast3Ints,
    nkx: AtLeast3Ints,
    nky: AtLeast3Ints,
    nkz: AtLeast3Ints,
    type_kx: str,
    type_ky: str,
    type_kz: str,
    type_k0: str,
    type_k1: str,
    type_k2: str,
) -> None:
    """Test gradient of Fourier operator."""
    gradient_of_linear_operator_test(
        *create_fourier_op_and_range_domain(img_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)
    )


@COMMON_MR_TRAJECTORIES
def test_fourier_op_forward_mode_autodiff(
    img_shape: AtLeast3Ints,
    k_shape: AtLeast3Ints,
    nkx: AtLeast3Ints,
    nky: AtLeast3Ints,
    nkz: AtLeast3Ints,
    type_kx: str,
    type_ky: str,
    type_kz: str,
    type_k0: str,
    type_k1: str,
    type_k2: str,
) -> None:
    """Test forward-mode autodiff of Fourier operator."""
    forward_mode_autodiff_of_linear_operator_test(
        *create_fourier_op_and_range_domain(img_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)
    )


@COMMON_MR_TRAJECTORIES
def test_fourier_op_gram(
    img_shape: AtLeast3Ints,
    k_shape: AtLeast3Ints,
    nkx: AtLeast3Ints,
    nky: AtLeast3Ints,
    nkz: AtLeast3Ints,
    type_kx: str,
    type_ky: str,
    type_kz: str,
    type_k0: str,
    type_k1: str,
    type_k2: str,
) -> None:
    """Test gram of Fourier operator."""
    fourier_op, img, _ = create_fourier_op_and_range_domain(
        img_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz
    )
    (expected,) = (fourier_op.H @ fourier_op)(img)
    (actual,) = fourier_op.gram(img)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('encoding_offset', [0, 1])
@pytest.mark.parametrize(
    ('recon_shape', 'cartesian_axis'),
    [
        ((1, 1, 8), None),
        ((1, 1, 9), None),
        ((1, 8, 8), None),
        ((1, 9, 9), None),
        ((1, 8, 9), None),
        ((1, 9, 8), None),
        ((4, 6, 8), None),
        ((5, 7, 9), None),
        ((5, 6, 8), None),
        ((4, 7, 8), None),
        ((4, 6, 9), None),
        ((5, 6, 9), 0),
        ((5, 6, 9), 1),
        ((5, 6, 9), 2),
    ],
)
def test_fourier_op_gram_centering(
    recon_shape: tuple[int, int, int], cartesian_axis: int | None, encoding_offset: int, device: str
) -> None:
    """Toeplitz Gram agrees with H @ F for odd, even and mixed axis sizes."""
    rng = RandomGenerator(seed=0)
    encoding_shape = tuple(size + encoding_offset if size > 1 else 1 for size in recon_shape)
    sample_shape = [5 if size > 1 else 1 for size in recon_shape]
    if cartesian_axis is not None:
        sample_shape[cartesian_axis] = encoding_shape[cartesian_axis]
    nonuniform_shape = [1 if axis == cartesian_axis else size for axis, size in enumerate(sample_shape)]
    coordinates = []
    for axis, size in enumerate(encoding_shape):
        if size == 1:
            coordinate = torch.zeros(1, 1, 1, 1, 1)
        elif axis == cartesian_axis:
            shape = [1] * 5
            shape[axis + 2] = size
            coordinate = (torch.arange(size, dtype=torch.float32) - size // 2).reshape(shape)
        else:
            coordinate = rng.float32_tensor((1, 1, *nonuniform_shape), low=-size / 2, high=size / 2)
        coordinates.append(coordinate.to(device))

    kz, ky, kx = coordinates
    trajectory = KTrajectory(kz, ky, kx)
    op = FourierOp(SpatialDimension(*recon_shape), SpatialDimension(*encoding_shape), trajectory).to(device)
    x = rng.complex64_tensor((1, 2, *recon_shape)).to(device)
    (expected,) = (op.H @ op)(x)
    (actual,) = op.gram(x)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_fourier_op_cartesian_sorting(ismrmrd_cart) -> None:
    """Verify correct sorting of Cartesian k-space data before FFT."""
    kdata = KData.from_file(ismrmrd_cart.filename, KTrajectoryCartesian())
    ff_op = FourierOp.from_kdata(kdata)
    (img,) = ff_op.adjoint(kdata.data)

    # shuffle the kspace points along k0
    permutation_index = RandomGenerator(13).randperm(kdata.data.shape[-1])
    kdata_unsorted = KData(
        header=kdata.header,
        data=kdata.data[..., permutation_index],
        traj=KTrajectory.from_tensor(kdata.traj.as_tensor()[..., permutation_index]),
    )
    ff_op_unsorted = FourierOp.from_kdata(kdata_unsorted)
    (img_unsorted,) = ff_op_unsorted.adjoint(kdata_unsorted.data)

    torch.testing.assert_close(img, img_unsorted)


@pytest.mark.parametrize(
    ('img_shape', 'k_shape', 'nkx', 'nky', 'nkz', 'type_kx', 'type_ky', 'type_kz'),  # parameter names
    [
        (  # 3d single shot stack of spiral but cartesian FFT dimension in ky and k2
            (1, 2, 96, 4, 128),  # img_shape
            (1, 2, 4, 1, 192),  # k_shape
            (1, 1, 1, 1, 192),  # nkx
            (1, 1, 4, 1, 1),  # nky
            (1, 1, 1, 1, 192),  # nkz
            'non-uniform',  # type_kx
            'uniform',  # type_ky
            'non-uniform',  # type_kz
        ),
        (  # radial phase encoding, cartesian FFT dimension not aligned with corresponding k2, k1, k0 dimensions
            (2, 3, 48, 16, 32),  # img_shape
            (2, 3, 96, 18, 64),  # k_shape
            (2, 1, 1, 18, 64),  # nkx
            (2, 1, 96, 1, 1),  # nky - Cartesian ky dimension defined along k2 rather than k1
            (2, 1, 1, 18, 64),  # nkz
            'non-uniform',  # type_kx
            'uniform',  # type_ky
            'non-uniform',  # type_kz
        ),
    ],
    ids=['3d_single_shot_stack_of_spirals_ky_in_k2', 'cartesian_fft_dims_not_aligned_with_k2_k1_k0_dims'],
)
def test_fourier_op_not_supported_traj(
    img_shape: AtLeast3Ints,
    k_shape: AtLeast3Ints,
    nkx: AtLeast3Ints,
    nky: AtLeast3Ints,
    nkz: AtLeast3Ints,
    type_kx: str,
    type_ky: str,
    type_kz: str,
) -> None:
    """Test trajectory not supported by Fourier operator."""

    # generate random images and k-space trajectories
    _, trajectory = create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    # create operator
    recon_matrix = SpatialDimension(img_shape[-3], img_shape[-2], img_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    with pytest.raises(UserWarning, match='Cartesian FFT dims need to be aligned'):
        FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)


@COMMON_MR_TRAJECTORIES
def test_fourier_op_fft_nufft_forward(
    img_shape: AtLeast3Ints,
    k_shape: AtLeast3Ints,
    nkx: AtLeast3Ints,
    nky: AtLeast3Ints,
    nkz: AtLeast3Ints,
    type_kx: str,
    type_ky: str,
    type_kz: str,
    type_k0: str,
    type_k1: str,
    type_k2: str,
) -> None:
    """Test Nufft vs FFT for Fourier operator."""
    if not any(t == 'uniform' for t in [type_kx, type_ky, type_kz]):
        return  # only test for uniform trajectories

    img, trajectory = create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    recon_matrix = SpatialDimension(img_shape[-3], img_shape[-2], img_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)

    nufft_fourier_op = FourierOp(
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=NufftTrajektory(trajectory.kz, trajectory.ky, trajectory.kx),
    )

    (result_normal,) = fourier_op(img)
    (result_nufft,) = nufft_fourier_op(img)
    torch.testing.assert_close(result_normal, result_nufft, atol=3e-4, rtol=5e-3)


@COMMON_MR_TRAJECTORIES
def test_fourier_op_fft_nufft_adjoint(
    img_shape: AtLeast3Ints,
    k_shape: AtLeast3Ints,
    nkx: AtLeast3Ints,
    nky: AtLeast3Ints,
    nkz: AtLeast3Ints,
    type_kx: str,
    type_ky: str,
    type_kz: str,
    type_k0: str,
    type_k1: str,
    type_k2: str,
) -> None:
    """Test AdjointNufft vs IFFT for Fourier operator."""
    if not any(t == 'uniform' for t in [type_kx, type_ky, type_kz]):
        return  # only test for uniform trajectories
    img, trajectory = create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)
    recon_matrix = SpatialDimension(img_shape[-3], img_shape[-2], img_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)

    nufft_fourier_op = FourierOp(
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=NufftTrajektory(trajectory.kz, trajectory.ky, trajectory.kx),
    )

    (k,) = fourier_op(img)
    (result_normal,) = fourier_op.H(k)
    (result_nufft,) = nufft_fourier_op.H(k)
    torch.testing.assert_close(result_normal, result_nufft, atol=3e-4, rtol=5e-3)


@COMMON_MR_TRAJECTORIES
def test_fourier_op_fft_nufft_gram(
    img_shape: AtLeast3Ints,
    k_shape: AtLeast3Ints,
    nkx: AtLeast3Ints,
    nky: AtLeast3Ints,
    nkz: AtLeast3Ints,
    type_kx: str,
    type_ky: str,
    type_kz: str,
    type_k0: str,
    type_k1: str,
    type_k2: str,
) -> None:
    """Test Nufft gram vs FFt gram for Fourier operator."""
    if not any(t == 'uniform' for t in [type_kx, type_ky, type_kz]):
        return  # only test for uniform trajectories
    img, trajectory = create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)
    recon_matrix = SpatialDimension(img_shape[-3], img_shape[-2], img_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)

    nufft_fourier_op = FourierOp(
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=NufftTrajektory(trajectory.kz, trajectory.ky, trajectory.kx),
    )

    (result_normal,) = fourier_op.gram(img)
    (result_nufft,) = nufft_fourier_op.gram(img)
    torch.testing.assert_close(result_normal, result_nufft, atol=3e-4, rtol=5e-3)


def test_fourier_op_repr_mixed_fft_nufft():
    """Test the __repr__ method of FourierOp with both FFT and NUFFT components."""

    # Create a trajectory with both Cartesian (on-grid) and non-Cartesian (off-grid) components
    k_shape = (1, 5, 20, 40, 60)
    nkx = (1, 1, 1, 1, 60)
    nky = (1, 1, 1, 40, 1)
    nkz = (1, 1, 20, 1, 1)
    type_kx = 'non-uniform'
    type_ky = 'uniform'
    type_kz = 'uniform'
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)

    recon_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)
    repr_str = repr(fourier_op)

    # Check if __repr__ contains both FFT and NUFFT components
    assert 'FastFourierOp' in repr_str
    assert 'NonUniformFastFourierOp' in repr_str
    assert 'CartesianSamplingOp' in repr_str


@pytest.mark.cuda
def test_fourier_op_cuda() -> None:
    """Test Fourier operator works on CUDA devices."""

    # Create a trajectory with both Cartesian (on-grid) and non-Cartesian (off-grid) components
    img_shape = (1, 2, 20, 20, 20)
    k_shape = (1, 2, 40, 40, 40)
    nkx = (1, 1, 1, 1, 40)
    nky = (1, 1, 1, 40, 1)
    nkz = (1, 1, 40, 1, 1)
    type_kx = 'uniform'
    type_ky = 'uniform'
    type_kz = 'non-uniform'
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)

    recon_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )

    random_generator = RandomGenerator(seed=0)
    x = random_generator.complex64_tensor(size=img_shape)

    # Create on CPU, transfer to GPU, run on GPU
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)
    operator = fourier_op.H @ fourier_op
    operator.cuda()
    (result,) = operator(x.cuda())
    assert result.is_cuda

    # Create on CPU, run on CPU
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory)
    operator = fourier_op.H @ fourier_op
    (result,) = operator(x)
    assert result.is_cpu

    # Create on GPU, run on GPU
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory.cuda())
    operator = fourier_op.H @ fourier_op
    (result,) = operator(x.cuda())
    assert result.is_cuda

    # Create on GPU, transfer to CPU, run on CPU
    fourier_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory.cuda())
    operator = fourier_op.H @ fourier_op
    operator.cpu()
    (result,) = operator(x)
    assert result.is_cpu


@pytest.mark.cuda
def test_fourier_op_gram_cuda() -> None:
    """Test gram Fourier operator works on CUDA devices."""
    # Create a trajectory with both Cartesian (on-grid) and non-Cartesian (off-grid) components
    img_shape = (1, 2, 20, 20, 20)
    k_shape = (1, 2, 40, 40, 40)
    nkx = (1, 1, 1, 1, 40)
    nky = (1, 1, 1, 40, 1)
    nkz = (1, 1, 40, 1, 1)
    type_kx = 'uniform'
    type_ky = 'uniform'
    type_kz = 'non-uniform'
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)

    recon_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )

    random_generator = RandomGenerator(seed=0)
    x = random_generator.complex64_tensor(size=img_shape)

    # Create FourierOp.gram on CPU, transfer to GPU, run on GPU
    gram_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory).gram.cuda()
    operator = gram_op.H @ gram_op
    (result,) = operator(x.cuda())
    assert result.is_cuda

    # Create FourierOp.gram on GPU, run on GPU
    gram_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory).cuda().gram
    operator = gram_op.H @ gram_op
    (result,) = operator(x.cuda())
    assert result.is_cuda

    # Create FourierOp.gram on GPU, transfer to CPU, run on CPU
    gram_op = FourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, traj=trajectory).cuda().gram.cpu()
    operator = gram_op.H @ gram_op
    (result,) = operator(x)
    assert result.is_cpu
