"""Test the SSIM functional."""

from collections.abc import Sequence
from typing import Literal

import pytest
import torch
from mr2.operators.functionals.SSIM import SSIM
from mr2.utils import RandomGenerator


@pytest.mark.parametrize('reduction', ['full', 'volume', 'none'])
@pytest.mark.parametrize('soft_weight', [False, True])
@pytest.mark.parametrize('scale', [0.01, 1e-6, 100.0])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_ssim_weight_scale(
    reduction: Literal['full', 'volume', 'none'], soft_weight: bool, scale: float, device: str
) -> None:
    """Scaling mask or soft weights must not change SSIM values or gradients."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    rng = RandomGenerator(0)
    target = rng.float32_tensor((2, 1, 10, 10)).to(device)
    prediction = (0.8 * target + 0.2 * rng.rand_like(target)).requires_grad_()
    weight = rng.rand_like(target, low=0.1, high=1.0) if soft_weight else torch.ones_like(target)
    weight[..., :1, :] = 0
    expected = SSIM(target, weight, reduction=reduction)(prediction)[0]
    result = SSIM(target, scale * weight, reduction=reduction)(prediction)[0]
    torch.testing.assert_close(result, expected)
    expected_grad = torch.autograd.grad(expected.sum(), prediction)[0]
    result_grad = torch.autograd.grad(result.sum(), prediction)[0]
    torch.testing.assert_close(result_grad, expected_grad)


def test_ssim() -> None:
    """Test the SSIM functional."""
    rng = RandomGenerator(0)
    target = rng.float32_tensor((1, 8, 32, 32), low=0.0, high=1.0)
    ssim = SSIM(target)
    perfect = target.clone()
    assert torch.isclose(ssim(perfect)[0], torch.tensor(1.0))
    bad = rng.float32_tensor((1, 8, 32, 32), low=0.0, high=1.0)
    assert ssim(bad)[0] < 0.1


@pytest.mark.parametrize('shape', [(2, 1, 32, 32), (2, 12, 32, 32)])
def test_ssim_mask(shape: Sequence[int]) -> None:
    """Test masking in the SSIM functional"""
    rng = RandomGenerator(0)
    target = rng.float32_tensor(shape, low=0.0, high=1.0)
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[..., 4:-4, 4:-4] = True
    test = rng.rand_like(target) + 0.5 * target + rng.rand_like(target, high=100) * (~mask).float()
    (masked,) = SSIM(target, mask)(test)
    (cropped,) = SSIM(target[..., 4:-4, 4:-4])(test[..., 4:-4, 4:-4])
    torch.testing.assert_close(masked, cropped)
    assert 0.40 < masked.item() < 0.45


def test_ssim_reduction() -> None:
    """Test the reduction argument of the SSIM functional."""
    rng = RandomGenerator(0)
    target = rng.complex64_tensor((2, 3, 10, 10, 10), low=0.0, high=1.0)
    test = rng.complex64_tensor(target.shape) + rng.float32_tensor((2, 3, 1, 1, 1), low=0.2, high=0.8) * target
    (ssim_volume,) = SSIM(target, reduction='volume')(test)
    (ssim_full,) = SSIM(target, reduction='full')(test)
    (ssim_none,) = SSIM(target, reduction='none')(test)
    torch.testing.assert_close(ssim_volume.mean(), ssim_full)
    torch.testing.assert_close(ssim_none.mean(), ssim_full)
    assert ssim_volume.shape == (2, 3)
    assert ssim_full.shape == ()
    assert ssim_none.shape == (2, 3, 4, 4, 4)


def test_ssim_complex_uses_imaginary_part() -> None:
    """Test that complex SSIM uses real and imaginary components."""
    rng = RandomGenerator(0)
    target = rng.complex64_tensor((1, 8, 32, 32), low=0.0, high=1.0)
    test = target.real + 1j * rng.float32_tensor(target.shape, low=0.0, high=1.0)

    (complex_ssim,) = SSIM(target)(test)
    (real_ssim,) = SSIM(target.real)(test.real)

    assert complex_ssim < real_ssim


def test_ssim_stays_finite_in_degenerate_cases() -> None:
    """SSIM should return finite values and gradients for degenerate but valid inputs."""
    target = torch.ones((1, 7, 7, 7), dtype=torch.float32)
    prediction = torch.ones((1, 7, 7, 7), dtype=torch.float32, requires_grad=True)
    mask = torch.zeros((1, 7, 7, 7), dtype=torch.bool)

    (result,) = SSIM(target, weight=mask, data_range=torch.tensor(1.0))(prediction)

    assert torch.isfinite(result)
    torch.testing.assert_close(result, torch.tensor(0.0))

    result.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()

    target = torch.tensor([[[1.0]]], dtype=torch.float32)
    prediction = torch.tensor([[[1.0]]], dtype=torch.float32, requires_grad=True)

    (result,) = SSIM(target)(prediction)

    assert torch.isfinite(result)
    torch.testing.assert_close(result, torch.tensor(1.0))

    result.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.cuda
def test_ssim_cuda_masked_backward_after_move() -> None:
    """A moved masked SSIM matches CPU and backpropagates finite CUDA gradients."""
    target = torch.linspace(0.0, 1.0, 7**3).reshape(1, 7, 7, 7)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[..., 1:-1, 1:-1, 1:-1] = True
    target = target * (1 + 0.5j)
    prediction = (target * 0.8 + 0.1).requires_grad_()
    cpu_op = SSIM(target, mask, window_size=3, data_range=torch.tensor(1.0))
    expected = cpu_op(prediction)[0]

    moved = cpu_op.cuda()
    cuda_prediction = prediction.detach().cuda().requires_grad_()
    actual = moved(cuda_prediction)[0]
    actual.backward()

    assert actual.is_cuda
    assert actual > 0
    torch.testing.assert_close(actual.cpu(), expected)
    assert cuda_prediction.grad is not None
    assert cuda_prediction.grad.is_cuda
    assert torch.isfinite(cuda_prediction.grad).all()
    assert cuda_prediction.grad.abs().sum() > 0
