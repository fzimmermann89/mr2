"""Tests for ConjugateGradientDC module."""

import pytest
import torch
from mr2.data.CsmData import CsmData
from mr2.data.IHeader import IHeader
from mr2.data.KData import KData
from mr2.data.SpatialDimension import SpatialDimension
from mr2.data.traj_calculators.KTrajectoryCartesian import KTrajectoryCartesian
from mr2.nn.data_consistency.AnalyticCartesianDC import AnalyticCartesianDC
from mr2.nn.data_consistency.ConjugateGradientDC import ConjugateGradientDC
from mr2.nn.data_consistency.GradientDescentDC import GradientDescentDC
from mr2.operators.FourierOp import FourierOp
from mr2.operators.SensitivityOp import SensitivityOp
from mr2.phantoms.EllipsePhantom import EllipsePhantom
from mr2.utils import RandomGenerator


@pytest.fixture
def csm_problem() -> tuple[torch.Tensor, torch.Tensor, KData]:
    matrix = SpatialDimension(z=1, y=16, x=16)
    kdata = EllipsePhantom().kdata(KTrajectoryCartesian.fullysampled(matrix), matrix)
    rng = RandomGenerator(123)
    image = rng.complex64_tensor((1, 1, 1, 16, 16))
    csm = rng.complex64_tensor((1, 3, 1, 16, 16), low=0.3, high=1.3)
    kdata.data = rng.complex64_tensor(csm.shape)
    return image, csm, kdata


def test_conjugate_gradient_dc(
    image_noisy: torch.Tensor, kdata_us: KData, image: torch.Tensor, image_us: torch.Tensor
) -> None:
    image_noisy = image_noisy.clone().requires_grad_(True)
    dc = ConjugateGradientDC(initial_regularization_weight=1.0)
    result = dc(image_noisy, kdata_us)
    analytic_dc = AnalyticCartesianDC(initial_regularization_weight=1.0)
    torch.testing.assert_close(result, analytic_dc(image_noisy, kdata_us), rtol=1e-5, atol=1e-5)
    loss = (result - image).abs().mean()
    assert loss < (image_noisy - image).abs().mean()
    assert loss < (image_us - image).abs().mean()
    loss.backward()
    assert image_noisy.grad is not None
    assert dc.log_weight.grad is not None
    assert not dc.log_weight.grad.isnan()
    assert not image_noisy.grad.isnan().any()


def test_conjugate_gradient_dc_csm(csm_problem: tuple[torch.Tensor, torch.Tensor, KData]) -> None:
    """CSM inputs yield a coil-combined solution of the regularized normal equations."""
    image, csm, kdata = csm_problem
    kdata = kdata[..., ::2, :].clone()
    fourier_op = FourierOp.from_kdata(kdata)
    weight = 0.7
    dc = ConjugateGradientDC(weight)
    result = dc(image, kdata, csm=csm)
    csm_data = CsmData(data=csm, header=IHeader.from_kheader(kdata.header))
    result_csm_data = dc(image, kdata.data, fourier_op, csm_data)

    assert result.shape == image.shape
    torch.testing.assert_close(result, result_csm_data)

    # Apply the normal equations directly, without the CG operator's Gram factory.
    lhs = (csm.conj() * fourier_op.H(fourier_op(csm * result)[0])[0]).sum(-4, keepdim=True) + weight * result
    rhs = (csm.conj() * fourier_op.H(kdata.data)[0]).sum(-4, keepdim=True) + weight * image
    assert (lhs - rhs).norm() / rhs.norm() < 5e-6


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param(
            'cuda',
            id='cuda',
            marks=[pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')],
        ),
    ],
)
def test_conjugate_gradient_dc_csm_gradients(
    csm_problem: tuple[torch.Tensor, torch.Tensor, KData], device: str
) -> None:
    """Implicit gradients agree with the fully sampled closed-form solution."""
    image, csm, kdata = csm_problem
    image, csm, kdata = image.to(device), csm.to(device), kdata.to(device)
    image.requires_grad_(True)
    csm = csm / csm.abs().square().sum(-4, keepdim=True).sqrt()
    csm.requires_grad_(True)
    kdata.data.requires_grad_(True)
    fourier_op = FourierOp.from_kdata(kdata)
    dc = ConjugateGradientDC(0.7).to(image.device)
    result = dc(image, kdata.data, fourier_op, csm)
    weight = dc.log_weight.exp()
    expected = ((csm.conj() * fourier_op.H(kdata.data)[0]).sum(-4, keepdim=True) + weight * image) / (
        csm.abs().square().sum(-4, keepdim=True) + weight
    )
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
    # With normalized CSMs and full sampling, one gradient step also solves the system.
    gd_dc = GradientDescentDC(1 / (1 + 0.7)).to(image.device)
    torch.testing.assert_close(result, gd_dc(image, kdata.data, fourier_op @ SensitivityOp(csm)), rtol=1e-5, atol=1e-6)

    inputs = (dc.log_weight, image, kdata.data, csm)
    grads = torch.autograd.grad(result.abs().square().mean(), inputs)
    expected_grads = torch.autograd.grad(expected.abs().square().mean(), inputs)
    for grad, expected_grad in zip(grads, expected_grads, strict=True):
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
        torch.testing.assert_close(grad, expected_grad, rtol=1e-4, atol=1e-6)
