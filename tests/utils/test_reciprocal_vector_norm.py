"""Tests for reciprocal Euclidean norms."""

import pytest
import torch
from mr2.utils import reciprocal_vector_norm, vector_norm


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('eps', [0.0, 1e-3])
def test_reciprocal_vector_norm(dtype: torch.dtype, device: str, eps: float) -> None:
    """Match norms and gradients for strided vectors above and below epsilon."""
    x = torch.arange(1, 25, dtype=dtype.to_real(), device=device).to(dtype).reshape(2, 3, 4).transpose(0, 2)
    if dtype.is_complex:
        x = x + 0.5j * x
    x = (x * x.new_tensor([1e-5, 1e2])).requires_grad_()
    actual = reciprocal_vector_norm(x, dim=(0, 1), keepdim=True, eps=eps)
    expected = (torch.linalg.vector_norm(x, dim=(0, 1), keepdim=True) + eps).reciprocal()

    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), x)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize('eps', [0.0, 1e-3])
def test_reciprocal_vector_norm_zero(eps: float) -> None:
    """Zero vectors return infinity or the reciprocal epsilon without NaNs."""
    x = torch.zeros(2, 4, dtype=torch.complex128, requires_grad=True)
    actual = reciprocal_vector_norm(x, dim=-1, eps=eps)
    expected = (x.norm(dim=-1) + eps).reciprocal()
    torch.testing.assert_close(actual, expected)
    if eps:
        gradient = torch.autograd.grad(actual.sum(), x)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(x))


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_vector_norm_zero_gradient(dtype: torch.dtype, device: str) -> None:
    """Match values and gradients for losses with zero and nonzero groups."""
    x = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=dtype, device=device, requires_grad=True)
    actual = vector_norm(x, dim=-1)
    expected = torch.linalg.vector_norm(x, dim=-1)
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), x)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
