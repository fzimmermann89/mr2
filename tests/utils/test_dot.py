"""Tests for tensor inner products."""

import importlib

import pytest
import torch
from mr2.utils import dot


@pytest.fixture(params=[False, True], ids=['native', 'chunked'])
def dot_path(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the large-tensor path without allocating large tensors."""
    if request.param:
        module = importlib.import_module('mr2.utils.dot')
        monkeypatch.setattr(module, '_MAX_DOT_SIZE', 2)
        monkeypatch.setattr(module, '_DOT_CHUNK_SIZE', 2)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_dot(device: str, dtype: torch.dtype, dot_path: None) -> None:
    """Match values and gradients for noncontiguous real and complex inputs."""
    a = torch.arange(1, 22, dtype=torch.float64, device=device).reshape(3, 7).T.to(dtype).requires_grad_()
    b = (a.detach().flip(0) * (1 + 0.3j if dtype.is_complex else 0.5)).requires_grad_()
    actual = dot(a, b)
    expected = torch.vdot(a.flatten(), b.flatten())
    torch.testing.assert_close(actual, expected)
    grad_output = torch.tensor(0.7 + 0.4j if dtype.is_complex else 0.7, dtype=dtype, device=device)
    actual_gradient = torch.autograd.grad(actual, (a, b), grad_output)
    expected_gradient = torch.autograd.grad(expected, (a, b), grad_output)
    for actual_grad, expected_grad in zip(actual_gradient, expected_gradient, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_dot_derivatives(dot_path: None) -> None:
    """Preserve complex derivatives, curvature and batching."""
    a = torch.tensor([1 + 2j, 3 - 1j, 2 + 0.5j], dtype=torch.complex128, requires_grad=True)
    b = torch.tensor([2 - 1j, 1 + 3j, 4 - 2j], dtype=torch.complex128, requires_grad=True)
    assert torch.autograd.gradcheck(dot, (a, b))
    assert torch.autograd.gradgradcheck(dot, (a, b))
    torch.testing.assert_close(torch.vmap(dot)(a[None], b[None]), torch.linalg.vecdot(a[None], b[None]))
