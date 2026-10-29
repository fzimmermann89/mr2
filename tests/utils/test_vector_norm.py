"""Tests for Euclidean vector norms."""

import pytest
import torch
from mr2.utils import vector_norm


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_vector_norm(device: str) -> None:
    """Match complex norms and gradients, including a zero vector."""
    x = torch.tensor([[0, 0, 0], [1 + 2j, 0, 3 - 1j]], dtype=torch.complex64, device=device).T
    x.requires_grad_()
    actual = vector_norm(x, dim=0, keepdim=True)
    expected = torch.linalg.vector_norm(x, dim=0, keepdim=True)

    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), x)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


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
