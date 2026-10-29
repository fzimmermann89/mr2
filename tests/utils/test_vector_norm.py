"""Tests for Euclidean vector norms."""

import pytest
import torch
from mr2.utils import vector_norm


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_vector_norm(dtype: torch.dtype, device: str) -> None:
    """Match norms and gradients for strided vectors at different scales."""
    x = torch.arange(1, 25, dtype=dtype.to_real(), device=device).to(dtype).reshape(2, 3, 4).transpose(0, 2)
    if dtype.is_complex:
        x = x + 0.5j * x
    x = (x * x.new_tensor([1e-5, 1e2])).requires_grad_()
    actual = vector_norm(x, dim=(0, 1), keepdim=True)
    expected = torch.linalg.vector_norm(x, dim=(0, 1), keepdim=True)

    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), x)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


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


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_vector_norm_reduced_precision(dtype: torch.dtype) -> None:
    """Preserve native accumulation precision when component squares overflow."""
    x = torch.tensor([300, 400], dtype=dtype)
    torch.testing.assert_close(vector_norm(x), torch.linalg.vector_norm(x))


@pytest.mark.parametrize('dtype', [torch.int64, torch.bool])
def test_vector_norm_rejects_nonfloating_input(dtype: torch.dtype) -> None:
    """Reject input types unsupported by the native norm."""
    with pytest.raises(RuntimeError, match='floating point or complex'):
        vector_norm(torch.ones(3, dtype=dtype))


def test_vector_norm_second_derivative() -> None:
    """Preserve curvature at zero components of a nonzero complex vector."""
    x = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)

    def norm(x: torch.Tensor) -> torch.Tensor:
        return vector_norm(torch.view_as_complex(x))

    def reference(x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(torch.view_as_complex(x))

    actual = torch.autograd.functional.hessian(norm, x)
    expected = torch.autograd.functional.hessian(reference, x)
    torch.testing.assert_close(actual, expected)
