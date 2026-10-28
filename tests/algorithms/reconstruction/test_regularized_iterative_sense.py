"""Tests for RegularizedIterativeSENSEReconstruction."""

import pytest
import torch
from mr2.algorithms.reconstruction import RegularizedIterativeSENSEReconstruction
from mr2.data import CsmData, DcfData, KData
from mr2.operators import FiniteDifferenceOp, FourierOp, IdentityOp
from mr2.utils import RandomGenerator


def test_regularized_iterative_sense_automatic(cartesian_kdata: KData) -> None:
    """Test automatic setup from kdata only."""
    reconstruction = RegularizedIterativeSENSEReconstruction(
        kdata=cartesian_kdata, n_iterations=2, regularization_weight=0.01
    )
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None
    assert reconstruction.dcf_op is not None


def test_regularized_iterative_sense_with_callable_csm(cartesian_kdata: KData) -> None:
    """Test with callable CSM estimation."""
    reconstruction = RegularizedIterativeSENSEReconstruction(
        kdata=cartesian_kdata,
        csm=CsmData.from_idata_walsh,
        n_iterations=2,
        regularization_weight=0.01,
    )
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None


def test_regularized_iterative_sense_with_explicit_csm(cartesian_kdata: KData) -> None:
    """Test with pre-computed CSM."""
    csm = CsmData.from_kdata_walsh(cartesian_kdata)
    reconstruction = RegularizedIterativeSENSEReconstruction(
        kdata=cartesian_kdata, csm=csm, n_iterations=2, regularization_weight=0.01
    )
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None


def test_regularized_iterative_sense_with_explicit_dcf(cartesian_kdata: KData) -> None:
    """Test with pre-computed DCF."""
    dcf = DcfData.from_traj_voronoi(cartesian_kdata.traj)
    reconstruction = RegularizedIterativeSENSEReconstruction(
        kdata=cartesian_kdata, dcf=dcf, n_iterations=2, regularization_weight=0.01
    )
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.dcf_op is not None


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('prior', ['default', 'tensor_zero', 'differentiable_zero', 'nonzero'])
def test_regularized_iterative_sense_normal_equation(cartesian_kdata: KData, device: str, prior: str) -> None:
    """Check the normal-equation residual for zero and nonzero priors."""
    reconstruction = RegularizedIterativeSENSEReconstruction(
        kdata=cartesian_kdata,
        n_iterations=20,
        regularization_weight=0.5,
        regularization_op=FiniteDifferenceOp(dim=(-1, -2), mode='central'),
    ).to(device)
    kdata = cartesian_kdata.to(device)
    assert reconstruction.csm_op is not None
    acquisition_model = reconstruction.fourier_op @ reconstruction.csm_op
    (right_hand_side,) = acquisition_model.H(kdata.data)
    regularization_op = reconstruction.regularization_op
    (regularization_data,) = regularization_op(torch.zeros_like(right_hand_side))
    if prior == 'default':
        assert reconstruction.regularization_data is None
    elif prior == 'tensor_zero':
        reconstruction.regularization_data = regularization_data
    elif prior == 'differentiable_zero':
        reconstruction.regularization_data = regularization_data.requires_grad_()
    elif prior == 'nonzero':
        regularization_data = RandomGenerator(seed=0).complex64_tensor(regularization_data.shape).to(device)
        reconstruction.regularization_data = regularization_data

    (prior_term,) = regularization_op.H(regularization_data)
    right_hand_side = right_hand_side + reconstruction.regularization_weight * prior_term
    idata = reconstruction(kdata)
    # DCF only initializes CG; the normal equation uses unweighted k-space data (W = I).
    (data_term,) = acquisition_model.H(acquisition_model(idata.data)[0])
    (regularization_term,) = regularization_op.H(regularization_op(idata.data)[0])
    residual = data_term + reconstruction.regularization_weight * regularization_term - right_hand_side
    assert residual.norm() / right_hand_side.norm() < 1e-5
    if prior == 'differentiable_zero':
        assert reconstruction.regularization_data is not None
        (gradient,) = torch.autograd.grad(idata.data.abs().square().sum(), reconstruction.regularization_data)
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('prior', [None, 0.0, 0.5])
def test_regularized_iterative_sense_identity_prior(cartesian_kdata: KData, device: str, prior: float | None) -> None:
    """Check Tikhonov regularization toward a scalar target with identity operators."""
    reconstruction = RegularizedIterativeSENSEReconstruction(
        fourier_op=IdentityOp(), csm=None, n_iterations=1, regularization_weight=0.5, regularization_data=prior
    ).to(device)
    kdata = cartesian_kdata.to(device)
    idata = reconstruction(kdata)
    expected = (kdata.data + 0.5 * (0.0 if prior is None else prior)) / 1.5
    torch.testing.assert_close(idata.data, expected)


def test_regularized_iterative_sense_zero_prior_jvp(cartesian_kdata: KData) -> None:
    """Preserve forward-mode derivatives for a zero prior in the operator range."""
    reconstruction = RegularizedIterativeSENSEReconstruction(
        kdata=cartesian_kdata,
        n_iterations=2,
        regularization_weight=0.5,
        regularization_op=FiniteDifferenceOp(dim=(-1, -2), mode='central'),
    )
    assert reconstruction.csm_op is not None
    acquisition_model = reconstruction.fourier_op @ reconstruction.csm_op
    (image,) = acquisition_model.H(cartesian_kdata.data)
    (prior,) = reconstruction.regularization_op(torch.zeros_like(image))

    with torch.autograd.forward_ad.dual_level():
        reconstruction.regularization_data = torch.autograd.forward_ad.make_dual(prior, torch.ones_like(prior))
        (_, tangent) = torch.autograd.forward_ad.unpack_dual(reconstruction(cartesian_kdata).data)
    assert tangent is not None
    assert torch.isfinite(tangent).all()
    assert tangent.norm() > 0


def test_regularized_iterative_sense_scalar_zero_prior_jvp(cartesian_kdata: KData) -> None:
    """Preserve forward-mode derivatives for a scalar zero prior with identity regularization."""
    reconstruction = RegularizedIterativeSENSEReconstruction(
        fourier_op=IdentityOp(), csm=None, n_iterations=1, regularization_weight=0.5
    )
    with torch.autograd.forward_ad.dual_level():
        reconstruction.regularization_data = torch.autograd.forward_ad.make_dual(torch.tensor(0.0), torch.tensor(1.0))
        (_, tangent) = torch.autograd.forward_ad.unpack_dual(reconstruction(cartesian_kdata).data)
    assert tangent is not None
    torch.testing.assert_close(tangent, torch.full_like(tangent, 1 / 3))


@pytest.mark.cuda
def test_regularized_iterative_sense_cuda_from_kdata(cartesian_kdata: KData) -> None:
    """Test CUDA device transfers for reconstruction created from kdata."""
    reconstruction = RegularizedIterativeSENSEReconstruction(
        kdata=cartesian_kdata, n_iterations=2, regularization_weight=0.01
    ).cuda()
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    reconstruction = RegularizedIterativeSENSEReconstruction(
        kdata=cartesian_kdata.cuda(), n_iterations=2, regularization_weight=0.01
    )
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    reconstruction = RegularizedIterativeSENSEReconstruction(
        kdata=cartesian_kdata.cuda(), n_iterations=2, regularization_weight=0.01
    ).cpu()
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cpu


@pytest.mark.cuda
def test_regularized_iterative_sense_cuda_explicit_components(cartesian_kdata: KData) -> None:
    """Test CUDA device transfers with explicit FourierOp, CSM, and DCF."""

    def explicit_components(kdata: KData) -> tuple[FourierOp, CsmData, DcfData]:
        fourier_op = FourierOp.from_kdata(kdata)
        csm = CsmData.from_kdata_walsh(kdata)
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        return fourier_op, csm, dcf

    fourier_op, csm, dcf = explicit_components(cartesian_kdata)
    reconstruction = RegularizedIterativeSENSEReconstruction(
        fourier_op=fourier_op, csm=csm, dcf=dcf, n_iterations=2, regularization_weight=0.01
    ).cuda()
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    fourier_op, csm, dcf = explicit_components(cartesian_kdata.cuda())
    reconstruction = RegularizedIterativeSENSEReconstruction(
        fourier_op=fourier_op, csm=csm, dcf=dcf, n_iterations=2, regularization_weight=0.01
    )
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    fourier_op, csm, dcf = explicit_components(cartesian_kdata.cuda())
    reconstruction = RegularizedIterativeSENSEReconstruction(
        fourier_op=fourier_op, csm=csm, dcf=dcf, n_iterations=2, regularization_weight=0.01
    ).cpu()
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cpu
