"""Tests for Non-Uniform Fast Fourier operator."""

import copy
import pickle

import einops
import pytest
import torch
from mr2.data import DcfData, KData, KTrajectory, SpatialDimension
from mr2.data.traj_calculators import KTrajectoryIsmrmrd
from mr2.operators import DensityCompensationOp, FastFourierOp, NonUniformFastFourierOp, PCACompressionOp
from mr2.operators.NonUniformFastFourierOp import SubspaceNonUniformFastFourierOpGramOp
from mr2.utils import RandomGenerator
from pytorch_finufft.functional import finufft_type1, finufft_type2
from torch.autograd.gradcheck import gradcheck

from tests.conftest import COMMON_MR_TRAJECTORIES, create_traj
from tests.helper import (
    autodiff_test,
    dotproduct_adjointness_test,
    gradient_of_linear_operator_test,
    relative_image_difference,
)


def create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz) -> tuple[torch.Tensor, KTrajectory]:
    """Create k-space trajectory and random image."""
    rng = RandomGenerator(seed=0)
    img = rng.complex64_tensor(size=img_shape)
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)
    return img, trajectory


def create_time_varying_2d_nufft_op(
    n_timepoints: int = 4,
    image_shape: tuple[int, int] = (12, 10),
) -> NonUniformFastFourierOp:
    """Create a small time-varying 2D NUFFT over y/x."""
    nk = (n_timepoints, 1, 1, *image_shape)
    trajectory = create_traj(
        nkx=nk,
        nky=nk,
        nkz=(n_timepoints, 1, 1, 1, 1),
        type_kx='non-uniform',
        type_ky='non-uniform',
        type_kz='zero',
    )
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    return NonUniformFastFourierOp(
        direction=(-2, -1),
        recon_matrix=SpatialDimension(z=1, y=image_shape[0], x=image_shape[1]),
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )


def test_non_uniform_fast_fourier_op_real_input() -> None:
    """Real inputs give the same result as explicitly promoted complex inputs."""
    rng = RandomGenerator(seed=0)
    operator = create_time_varying_2d_nufft_op(n_timepoints=1, image_shape=(4, 4))
    x = rng.float32_tensor((1, 1, 1, 4, 4))
    for function in (operator, operator.adjoint):
        (actual,) = function(x)
        (expected,) = function(x.to(torch.complex64))
        assert actual.dtype == torch.complex64
        torch.testing.assert_close(actual, expected)


@COMMON_MR_TRAJECTORIES
def test_non_uniform_fast_fourier_op_fwd_adj_property(
    img_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
) -> None:
    """Test adjoint property of non-uniform Fast Fourier operator."""

    # generate random images and k-space trajectories
    _, trajectory = create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    # create operator
    recon_matrix = SpatialDimension(img_shape[-3], img_shape[-2], img_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    direction = [d for d, e in zip(('z', 'y', 'x'), encoding_matrix.zyx, strict=False) if e > 1]
    nufft_op = NonUniformFastFourierOp(
        direction=direction,  # type: ignore[arg-type]
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )

    # test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v
    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=img_shape)
    v = rng.complex64_tensor(size=k_shape)
    dotproduct_adjointness_test(nufft_op, u, v)


@COMMON_MR_TRAJECTORIES
def test_non_uniform_fast_fourier_op_gram(
    img_shape, k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz, type_k0, type_k1, type_k2
) -> None:
    """Test gram of of non-uniform Fast Fourier operator."""
    img, trajectory = create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    recon_matrix = SpatialDimension(img_shape[-3], img_shape[-2], img_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    direction = [d for d, e in zip(('z', 'y', 'x'), encoding_matrix.zyx, strict=False) if e > 1]
    nufft_op = NonUniformFastFourierOp(
        direction=direction,  # type: ignore[arg-type]
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )

    (expected,) = (nufft_op.H @ nufft_op)(img)
    (actual,) = nufft_op.gram(img)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_non_uniform_fast_fourier_op_equal_to_fft(ismrmrd_cart_high_res) -> None:
    """Compare nufft result to fft result for Cartesian data."""
    kdata = KData.from_file(ismrmrd_cart_high_res.filename, KTrajectoryIsmrmrd())
    nufft_op = NonUniformFastFourierOp(
        direction=(-2, -1),
        recon_matrix=[kdata.header.recon_matrix.y, kdata.header.recon_matrix.x],
        encoding_matrix=[kdata.header.encoding_matrix.y, kdata.header.encoding_matrix.x],
        traj=kdata.traj,
    )
    (nufft_img,) = nufft_op.adjoint(kdata.data)
    fft_op = FastFourierOp(
        dim=(-2, -1),
        recon_matrix=[kdata.header.recon_matrix.y, kdata.header.recon_matrix.x],
        encoding_matrix=[kdata.header.encoding_matrix.y, kdata.header.encoding_matrix.x],
    )
    (fft_img,) = fft_op.adjoint(kdata.data)
    torch.testing.assert_close(nufft_img, fft_img, rtol=1e-3, atol=1e-3)


def test_non_uniform_fast_fourier_cartesian_result(ismrmrd_cart_high_res) -> None:
    """Eval result of non-uniform Fast Fourier transform for Cartesian data."""
    kdata = KData.from_file(ismrmrd_cart_high_res.filename, KTrajectoryIsmrmrd())
    nufft_op = NonUniformFastFourierOp(
        direction=(-2, -1),
        recon_matrix=[kdata.header.recon_matrix.y, kdata.header.recon_matrix.x],
        encoding_matrix=[kdata.header.encoding_matrix.y, kdata.header.encoding_matrix.x],
        traj=kdata.traj,
    )
    (reconstructed_img,) = nufft_op.adjoint(kdata.data)
    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert relative_image_difference(reconstructed_img[0, 0, 0, ...], ismrmrd_cart_high_res.img_ref) <= 0.05


def test_non_uniform_fast_fourier_op_empty_dims() -> None:
    """Empty dims do not change the input."""
    nk = (1, 1, 1, 1, 1)
    traj = create_traj(nkx=nk, nky=nk, nkz=nk, type_kx='non-uniform', type_ky='non-uniform', type_kz='non-uniform')
    nufft_op = NonUniformFastFourierOp(direction=(), recon_matrix=(), encoding_matrix=(), traj=traj)

    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=[2, 3, 4, 5])
    torch.testing.assert_close(u, nufft_op(u)[0])


def test_non_uniform_fast_fourier_op_directions() -> None:
    """Test different direction specifiers of non-uniform fast Fourier operator."""

    kdata_shape = (1, 3, 1, 30, 40)
    img_shape = (2, 3, 1, 20, 30)

    # generate random traj and image
    nk = [2, 1, 1, 30, 40]
    img, traj = create_data(
        img_shape,
        nkx=nk,
        nky=nk,
        nkz=nk,
        type_kx='non-uniform',
        type_ky='non-uniform',
        type_kz='non-uniform',
    )

    # create operator
    nufft_op_12 = NonUniformFastFourierOp(
        direction=(-2, -1),
        recon_matrix=SpatialDimension[int](z=img_shape[-3], y=img_shape[-2], x=img_shape[-1]),
        encoding_matrix=SpatialDimension[int](z=kdata_shape[-3], y=kdata_shape[-2], x=kdata_shape[-1]),
        traj=traj,
    )

    nufft_op_yx = NonUniformFastFourierOp(
        direction=('y', 'x'),
        recon_matrix=SpatialDimension[int](z=img_shape[-3], y=img_shape[-2], x=img_shape[-1]),
        encoding_matrix=SpatialDimension[int](z=kdata_shape[-3], y=kdata_shape[-2], x=kdata_shape[-1]),
        traj=traj,
    )

    nufft_op_xy = NonUniformFastFourierOp(
        direction=('x', 'y'),
        recon_matrix=SpatialDimension[int](z=img_shape[-3], y=img_shape[-2], x=img_shape[-1]),
        encoding_matrix=SpatialDimension[int](z=kdata_shape[-3], y=kdata_shape[-2], x=kdata_shape[-1]),
        traj=traj,
    )

    nufft_op_xy_matrix_sequence = NonUniformFastFourierOp(
        direction=('x', 'y'),
        recon_matrix=(img_shape[-1], img_shape[-2]),
        encoding_matrix=(kdata_shape[-1], kdata_shape[-2]),
        traj=traj,
    )
    torch.testing.assert_close(nufft_op_12(img)[0], nufft_op_yx(img)[0])
    torch.testing.assert_close(nufft_op_12(img)[0], nufft_op_xy(img)[0])
    torch.testing.assert_close(nufft_op_12(img)[0], nufft_op_xy_matrix_sequence(img)[0])


def test_non_uniform_fast_fourier_op_error_directions() -> None:
    """Test error for duplicate directions of non-uniform fast Fourier operator."""
    with pytest.raises(ValueError, match='Directions must be unique'):
        NonUniformFastFourierOp(
            direction=(-2, -1, 'x'),
            recon_matrix=[1, 1],
            encoding_matrix=[1, 1],
            traj=KTrajectory.from_tensor(torch.ones((3, 1, 1, 1, 1, 1))),
        )


def test_non_uniform_fast_fourier_op_error_matrix() -> None:
    """Test error for wrong matrix dimensions of non-uniform fast Fourier operator."""
    with pytest.raises(ValueError, match='recon_matrix should have'):
        NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=[1, 1, 1],
            encoding_matrix=[1, 1],
            traj=KTrajectory.from_tensor(torch.ones((3, 1, 1, 1, 1, 1))),
        )

    with pytest.raises(ValueError, match='encoding_matrix should have'):
        NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=[1, 1],
            encoding_matrix=[1],
            traj=KTrajectory.from_tensor(torch.ones((3, 1, 1, 1, 1, 1))),
        )


def test_non_uniform_fast_fourier_op_repr():
    """Test the __repr__ method of NonUniformFastFourierOp."""

    # Create a trajectory with non-Cartesian (off-grid) components
    k_shape = (1, 5, 20, 40, 60)
    nkx = (1, 1, 1, 1, 60)
    nky = (1, 1, 1, 40, 1)
    nkz = (1, 1, 20, 1, 1)
    type_kx = 'non-uniform'
    type_ky = 'non-uniform'
    type_kz = 'non-uniform'
    trajectory = create_traj(nkx, nky, nkz, type_kx, type_ky, type_kz)

    recon_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    nufft_op = NonUniformFastFourierOp(
        direction=['x', 'y', 'z'],
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )
    repr_str = repr(nufft_op)

    # Check if __repr__ contains expected information
    assert 'NonUniformFastFourierOp' in repr_str
    assert 'Dimension(s) along which NUFFT is applied' in repr_str
    assert 'Reconstructed image size' in repr_str
    assert 'device' in repr_str


def test_subspace_non_uniform_fast_fourier_op_gram() -> None:
    """Test subspace Toeplitz Gram against explicit expand-apply-compress reference."""
    rng = RandomGenerator(seed=1)
    n_timepoints, n_coefficients = 4, 2
    image_shape = (12, 10)

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape)
    basis = rng.complex64_tensor((n_timepoints, n_coefficients))
    alpha = rng.complex64_tensor((n_coefficients, 1, 1, *image_shape))

    subspace_gram = nufft_op.toeplitz(subspace=basis)
    assert isinstance(subspace_gram, SubspaceNonUniformFastFourierOpGramOp)

    expanded = einops.einsum(basis, alpha, 'time coeff, coeff joint coil ... -> time joint coil ...')
    (kspace,) = nufft_op(expanded)
    (backprojected,) = nufft_op.H(kspace)
    expected = einops.einsum(basis.conj(), backprojected, 'time coeff, time joint coil ... -> coeff joint coil ...')
    (actual,) = subspace_gram(alpha)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_subspace_non_uniform_fast_fourier_op_gram_non_contiguous_directions() -> None:
    """Test subspace Toeplitz Gram over non-contiguous z/x image axes."""
    rng = RandomGenerator(seed=7)
    n_coefficients = 2
    img_shape = (8, 5, 64, 1, 64)
    nkx = (8, 1, 1, 18, 128)
    nky = (8, 1, 1, 1, 1)
    nkz = (8, 1, 1, 18, 128)
    trajectory = create_traj(nkx, nky, nkz, type_kx='non-uniform', type_ky='zero', type_kz='non-uniform')

    recon_matrix = SpatialDimension(img_shape[-3], img_shape[-2], img_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )
    direction = [d for d, e in zip(('z', 'y', 'x'), encoding_matrix.zyx, strict=False) if e > 1]
    nufft_op = NonUniformFastFourierOp(
        direction=direction,  # type: ignore[arg-type]
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )
    basis = rng.complex64_tensor((img_shape[0], n_coefficients))
    alpha = rng.complex64_tensor((n_coefficients, *img_shape[1:]))

    subspace_gram = nufft_op.toeplitz(subspace=basis)

    expanded = einops.einsum(basis, alpha, 'time coeff, coeff coil ... -> time coil ...')
    (kspace,) = nufft_op(expanded)
    (backprojected,) = nufft_op.H(kspace)
    expected = einops.einsum(basis.conj(), backprojected, 'time coeff, time coil ... -> coeff coil ...')
    (actual,) = subspace_gram(alpha)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize('n_coefficients', [1, 2])
def test_subspace_non_uniform_fast_fourier_op_gram_single_timepoint_basis(n_coefficients: int) -> None:
    """Test single-timepoint bases keep their time and coefficient axes."""
    rng = RandomGenerator(seed=8)
    n_timepoints = 1
    image_shape = (12, 10)

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape)
    basis = rng.complex64_tensor((n_timepoints, n_coefficients))
    alpha = rng.complex64_tensor((n_coefficients, 1, 1, *image_shape))

    expanded = einops.einsum(basis, alpha, 'time coeff, coeff joint coil ... -> time joint coil ...')
    (kspace,) = nufft_op(expanded)
    (backprojected,) = nufft_op.H(kspace)
    expected = einops.einsum(basis.conj(), backprojected, 'time coeff, time joint coil ... -> coeff joint coil ...')
    (actual,) = nufft_op.toeplitz(subspace=basis)(alpha)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_subspace_non_uniform_fast_fourier_op_gram_accepts_pca_operator() -> None:
    """Test subspace Gram accepts a PCACompressionOp as basis input."""
    rng = RandomGenerator(seed=2)
    n_timepoints, n_coefficients = 4, 2
    image_shape = (12, 10)

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape)
    training_signals = rng.complex64_tensor((1, 32, n_timepoints))
    pca_op = PCACompressionOp(training_signals, n_components=n_coefficients, centering=False)
    basis = pca_op.compression_matrix.squeeze(0).mH
    alpha = rng.complex64_tensor((n_coefficients, 1, 1, *image_shape))

    subspace_gram_from_pca = SubspaceNonUniformFastFourierOpGramOp(nufft_op, pca_op)
    subspace_gram_from_basis = SubspaceNonUniformFastFourierOpGramOp(nufft_op, basis)

    (actual_from_pca,) = subspace_gram_from_pca(alpha)
    (actual_from_basis,) = subspace_gram_from_basis(alpha)

    torch.testing.assert_close(actual_from_pca, actual_from_basis)


def test_non_uniform_fast_fourier_op_weighted_toeplitz_matches_explicit_dcf_normal_operator() -> None:
    """Test Toeplitz(weight=dcf) matches the explicit weighted normal operator F^H DCF F."""
    rng = RandomGenerator(seed=6)
    n_timepoints = 4
    image_shape = (12, 10)

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape)
    image = rng.complex64_tensor((n_timepoints, 1, 1, *image_shape))
    dcf = DcfData(data=rng.float32_tensor((n_timepoints, 1, 1, 1, image_shape[-1])))

    explicit_operator = nufft_op.H @ DensityCompensationOp(dcf) @ nufft_op
    weighted_toeplitz = nufft_op.toeplitz(weight=dcf)

    (expected,) = explicit_operator(image)
    (actual,) = weighted_toeplitz(image)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_non_uniform_fast_fourier_op_gram_autograd() -> None:
    """Test autograd of the Toeplitz Gram operator."""
    rng = RandomGenerator(seed=3)
    n_timepoints = 2
    image_shape = (5, 4)
    nufft_op = create_time_varying_2d_nufft_op(
        n_timepoints=n_timepoints,
        image_shape=image_shape,
    )
    operator = nufft_op.toeplitz().double()
    image = rng.complex128_tensor((n_timepoints, 1, 1, *image_shape)).requires_grad_(True)

    gradcheck(operator, (image,), fast_mode=True)


def test_subspace_non_uniform_fast_fourier_op_gram_autograd() -> None:
    """Test autograd of the subspace Toeplitz Gram operator."""
    rng = RandomGenerator(seed=4)
    n_timepoints, n_coefficients = 3, 2
    image_shape = (5, 4)
    nufft_op = create_time_varying_2d_nufft_op(
        n_timepoints=n_timepoints,
        image_shape=image_shape,
    )
    basis = rng.complex128_tensor((n_timepoints, n_coefficients))
    operator = nufft_op.toeplitz(subspace=basis).double()
    coefficients = rng.complex128_tensor((n_coefficients, 1, 1, *image_shape)).requires_grad_(True)

    gradcheck(operator, (coefficients,), fast_mode=True)


@pytest.mark.cuda
def test_non_uniform_fast_fourier_op_gram_cuda() -> None:
    """Test Toeplitz Gram operators work on CUDA devices."""
    rng = RandomGenerator(seed=5)
    n_timepoints = 4
    image_shape = (12, 10)
    image = rng.complex64_tensor((n_timepoints, 1, 1, *image_shape)).cuda()

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape).cuda()
    gram = nufft_op.gram
    (result,) = gram(image)
    assert result.is_cuda

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape)
    gram = nufft_op.gram.cuda()
    (result,) = gram(image)
    assert result.is_cuda


@pytest.mark.cuda
def test_non_uniform_fast_fourier_op_subspace_gram_cuda() -> None:
    """Test subspace Toeplitz Gram operators work on CUDA devices."""
    rng = RandomGenerator(seed=5)
    n_timepoints, n_coefficients = 4, 2
    image_shape = (12, 10)
    coefficients = rng.complex64_tensor((n_coefficients, 1, 1, *image_shape)).cuda()
    basis = rng.complex64_tensor((n_timepoints, n_coefficients))

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape)
    subspace_gram = nufft_op.toeplitz(subspace=basis).cuda()
    (result,) = subspace_gram(coefficients)
    assert result.is_cuda

    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape).cuda()
    subspace_gram = nufft_op.toeplitz(subspace=basis.cuda())
    (result,) = subspace_gram(coefficients)
    assert result.is_cuda


@pytest.mark.cuda
def test_non_uniform_fast_fourier_op_cuda() -> None:
    """Test non-uniform fast Fourier operator works on CUDA devices."""

    # Create a trajectory with non-Cartesian (off-grid) components
    img_shape = (2, 3, 10, 12, 14)
    k_shape = (2, 3, 6, 8, 10)
    nkx = (2, 1, 6, 8, 10)
    nky = (2, 1, 6, 8, 10)
    nkz = (2, 1, 6, 8, 10)
    type_kx = 'non-uniform'
    type_ky = 'non-uniform'
    type_kz = 'non-uniform'

    img, trajectory = create_data(img_shape, nkx, nky, nkz, type_kx, type_ky, type_kz)

    recon_matrix = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    encoding_matrix = SpatialDimension(
        int(trajectory.kz.max() - trajectory.kz.min() + 1),
        int(trajectory.ky.max() - trajectory.ky.min() + 1),
        int(trajectory.kx.max() - trajectory.kx.min() + 1),
    )

    # Create on CPU, transfer to GPU, run on GPU
    nufft_op = NonUniformFastFourierOp(
        direction=['x', 'y', 'z'],
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )
    operator = nufft_op.H @ nufft_op
    operator.cuda()
    (result,) = operator(img.cuda())
    assert result.is_cuda

    # Create on CPU, run on CPU
    nufft_op = NonUniformFastFourierOp(
        direction=['x', 'y', 'z'],
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        traj=trajectory,
    )
    operator = nufft_op.H @ nufft_op
    (result,) = operator(img)
    assert result.is_cpu

    # Create on GPU, run on GPU
    nufft_op = NonUniformFastFourierOp(
        direction=['x', 'y', 'z'],
        recon_matrix=recon_matrix.cuda(),
        encoding_matrix=encoding_matrix.cuda(),
        traj=trajectory.cuda(),
    )
    operator = nufft_op.H @ nufft_op
    (result,) = operator(img.cuda())
    assert result.is_cuda

    # Create on GPU, transfer to CPU, run on CPU
    nufft_op = NonUniformFastFourierOp(
        direction=['x', 'y', 'z'],
        recon_matrix=recon_matrix.cuda(),
        encoding_matrix=encoding_matrix.cuda(),
        traj=trajectory.cuda(),
    )
    operator = nufft_op.H @ nufft_op
    operator.cpu()
    (result,) = operator(img)
    assert result.is_cpu


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_plan_results(device: str, dtype: torch.dtype) -> None:
    """Native plans match the functional backend for both directions."""
    rng = RandomGenerator(seed=0, device=device)
    operator = create_time_varying_2d_nufft_op(image_shape=(5, 6)).to(device=device, dtype=dtype.to_real())
    image = rng.complex128_tensor((4, 2, 1, 5, 6)).to(dtype)
    samples = rng.complex128_tensor((4, 2, 1, 5, 6)).to(dtype)
    expected_forward = torch.vmap(lambda p, x: finufft_type2(p, x, isign=-1, modeord=0, upsampfac=2.0))(
        operator._omega, image[:, :, 0]
    )
    expected_adjoint = torch.vmap(lambda p, x: finufft_type1(p, x, (5, 6), isign=1, modeord=0, upsampfac=2.0))(
        operator._omega, samples.flatten(2)
    )
    torch.testing.assert_close(operator(image)[0], expected_forward.reshape_as(samples) * operator.scale)
    torch.testing.assert_close(operator.adjoint(samples)[0], expected_adjoint.reshape_as(image) * operator.scale)


@pytest.mark.parametrize('device', DEVICES)
def test_plan_cache(device: str) -> None:
    """Reuse plans across frames and calls; invalidate on batching or conversion."""
    rng = RandomGenerator(seed=0, device=device)
    operator = create_time_varying_2d_nufft_op().to(device)
    image = rng.complex64_tensor((4, 2, 1, 12, 10))
    expected = operator(image)[0]
    forward = operator._plan_fwd
    operator.adjoint(expected)
    adjoint = operator._plan_adj
    torch.testing.assert_close(operator(image * 2)[0], expected * 2)
    assert operator._plan_fwd is forward
    assert operator._plan_adj is adjoint
    operator(image[:, :1])
    assert operator._plan_fwd is not forward
    assert '_plan_adj' not in vars(operator)
    operator.double()
    assert '_plan_fwd' not in vars(operator)
    torch.testing.assert_close(
        operator(image.to(torch.complex128))[0], expected.to(torch.complex128), atol=2e-6, rtol=2e-5
    )


@pytest.mark.parametrize('device', DEVICES)
def test_plan_copy(device: str) -> None:
    """Copies and checkpoints exclude native plans."""
    rng = RandomGenerator(seed=0, device=device)
    operator = create_time_varying_2d_nufft_op().to(device)
    image = rng.complex64_tensor((4, 2, 1, 12, 10))
    expected = operator(image)[0]
    operator.adjoint(expected)
    for duplicate in (copy.copy(operator), copy.deepcopy(operator), pickle.loads(pickle.dumps(operator))):  # noqa: S301
        assert '_plan_fwd' not in vars(duplicate)
        assert '_plan_adj' not in vars(duplicate)
        torch.testing.assert_close(duplicate(image)[0], expected)
    assert not any('plan' in key for key in operator.state_dict())


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('adjoint', [False, True])
def test_plan_vmap(device: str, adjoint: bool) -> None:
    """Mapped calls match individual calls in both directions."""
    rng = RandomGenerator(seed=0, device=device)
    operator = create_time_varying_2d_nufft_op().to(device)
    function = operator.adjoint if adjoint else operator
    data = rng.complex64_tensor((3, 4, 2, 1, 12, 10))
    expected = torch.stack([function(x)[0] for x in data])
    torch.testing.assert_close(torch.vmap(function)(data)[0], expected)


@pytest.mark.parametrize('device', DEVICES)
def test_plan_autograd(device: str) -> None:
    """Forward and reverse data derivatives work with native plans."""
    rng = RandomGenerator(seed=0, device=device)
    operator = create_time_varying_2d_nufft_op().to(device)
    image = rng.complex64_tensor((4, 2, 1, 12, 10))
    samples = rng.complex64_tensor((4, 2, 1, 12, 10))
    gradient_of_linear_operator_test(operator, image, samples)
    autodiff_test(operator, image)
    autodiff_test(operator.H, samples)
    for method, data, adjoint in (
        ('_saved_forward', image, operator.adjoint),
        ('_saved_adjoint', samples, operator),
    ):
        function = getattr(operator, method)
        output, tangent, *_ = torch.func.jvp(function, (data,), (data,))
        torch.testing.assert_close(tangent[0], output[0])
        data = data.requires_grad_()
        (output,) = function(data)
        cotangent = rng.randn_like(output)
        (gradient,) = torch.autograd.grad(output, data, cotangent)
        torch.testing.assert_close(gradient, adjoint(cotangent)[0])


@pytest.mark.cuda
def test_plan_cuda_streams() -> None:
    """Changing the CUDA stream rebuilds the plan on the new stream."""
    rng = RandomGenerator(seed=0, device='cuda')
    operator = create_time_varying_2d_nufft_op().cuda()
    image = rng.complex64_tensor((4, 2, 1, 12, 10))
    expected = operator(image)[0]
    plan = operator._plan_fwd
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = operator(image)[0]
        assert operator._plan_fwd is not plan
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, expected)
    with torch.cuda.stream(stream):
        queued = operator(image)[0]
    actual = operator(image)[0]
    torch.testing.assert_close(queued, expected)
    torch.testing.assert_close(actual, expected)
    with torch.cuda.stream(stream):
        queued = operator(image)[0]
    operator.cpu()
    torch.testing.assert_close(queued, expected)
