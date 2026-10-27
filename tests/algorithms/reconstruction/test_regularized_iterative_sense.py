"""Tests for RegularizedIterativeSENSEReconstruction."""

import pytest
import torch
from mr2.algorithms.reconstruction import RegularizedIterativeSENSEReconstruction
from mr2.data import CsmData, DcfData, KData
from mr2.operators import FiniteDifferenceOp, FourierOp
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
def test_regularized_iterative_sense_default_data(cartesian_kdata: KData, device: str) -> None:
    """Test finite-difference regularization with the default zero prior."""
    reconstruction = RegularizedIterativeSENSEReconstruction(
        kdata=cartesian_kdata,
        n_iterations=2,
        regularization_weight=0.5,
        regularization_op=FiniteDifferenceOp(dim=(-1, -2), mode='central'),
    ).to(device)
    idata = reconstruction(cartesian_kdata.to(device))
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.data.device.type == device
    assert torch.isfinite(idata.data).all()


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize(
    'prior', ['default', 'scalar_zero', 'tensor_zero', 'range_zero', 'differentiable_zero', 'nonzero']
)
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
    if prior == 'scalar_zero':
        reconstruction.regularization_data = torch.tensor(0.0, device=device)
    elif prior == 'tensor_zero':
        reconstruction.regularization_data = torch.zeros(1, device=device)
    elif prior in ('range_zero', 'differentiable_zero'):
        reconstruction.regularization_data = regularization_data.requires_grad_(prior == 'differentiable_zero')
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
        (gradient,) = torch.autograd.grad(idata.data.abs().square().sum(), reconstruction.regularization_data)
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


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
