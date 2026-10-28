"""Tests for ConjugateGradientDC module."""

import pytest
import torch
from mr2.data.CsmData import CsmData
from mr2.data.IHeader import IHeader
from mr2.data.KData import KData
from mr2.nn.data_consistency.AnalyticCartesianDC import AnalyticCartesianDC
from mr2.nn.data_consistency.ConjugateGradientDC import ConjugateGradientDC
from mr2.nn.data_consistency.GradientDescentDC import GradientDescentDC
from mr2.operators.FourierOp import FourierOp
from mr2.operators.SensitivityOp import SensitivityOp


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_conjugate_gradient_dc(
    image_noisy: torch.Tensor, kdata_us: KData, image: torch.Tensor, image_us: torch.Tensor, device: str
) -> None:
    kdata_us = kdata_us.to(device)
    image = image.to(device)
    image_us = image_us.to(device)
    image_noisy = image_noisy.to(device).detach().requires_grad_(True)
    dc = ConjugateGradientDC(initial_regularization_weight=1.0).to(device)
    result = dc(image_noisy, kdata_us)
    assert result.device.type == device
    analytic_dc = AnalyticCartesianDC(initial_regularization_weight=1.0).to(device)
    torch.testing.assert_close(result, analytic_dc(image_noisy, kdata_us), rtol=1e-5, atol=1e-5)
    loss = (result - image).abs().mean()
    assert loss < (image_noisy - image).abs().mean()
    assert loss < (image_us - image).abs().mean()
    loss.backward()
    assert image_noisy.grad is not None
    assert dc.log_weight.grad is not None
    assert dc.log_weight.grad.device.type == device
    assert torch.isfinite(dc.log_weight.grad)
    assert image_noisy.grad.device.type == device
    assert torch.isfinite(image_noisy.grad).all()


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
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
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
