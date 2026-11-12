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
    actual_jvp = torch.func.jvp(dot, (a, b), (b, a))
    expected_jvp = torch.func.jvp(lambda x, y: torch.vdot(x.flatten(), y.flatten()), (a, b), (b, a))
    for value, reference in zip(actual_jvp, expected_jvp, strict=True):
        torch.testing.assert_close(value, reference)
    with torch.autograd.forward_ad.dual_level():
        dual_a = torch.autograd.forward_ad.make_dual(a, b)
        actual_tangent = torch.autograd.forward_ad.unpack_dual(dot(dual_a, b)).tangent
        torch.testing.assert_close(actual_tangent, torch.vdot(b.flatten(), b.flatten()))


def test_dot_derivatives(dot_path: None) -> None:
    """Preserve complex derivatives, curvature and batching."""
    a = torch.tensor([1 + 2j, 3 - 1j, 2 + 0.5j], dtype=torch.complex128, requires_grad=True)
    b = torch.tensor([2 - 1j, 1 + 3j, 4 - 2j], dtype=torch.complex128, requires_grad=True)
    assert torch.autograd.gradcheck(dot, (a, b), check_forward_ad=True)
    assert torch.autograd.gradgradcheck(dot, (a, b))
    torch.testing.assert_close(torch.vmap(dot)(a[None], b[None]), torch.linalg.vecdot(a[None], b[None]))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_dot_sequence(device: str, dot_path: None) -> None:
    """Match concatenated vectors and their derivatives without concatenating inputs."""
    a = torch.tensor([1 + 2j, 3 - 1j, 2 + 0.5j], dtype=torch.complex128, device=device, requires_grad=True)
    b = torch.tensor([2 - 1j, 1 + 3j, 4 - 2j], dtype=torch.complex128, device=device, requires_grad=True)

    def sequence_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return dot((x[:1], x[1:]), [y[:1], y[1:]])

    actual = sequence_dot(a, b)
    expected = torch.vdot(a, b)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == a.dtype
    assert actual.device == a.device
    assert torch.autograd.gradcheck(sequence_dot, (a, b), check_forward_ad=True)
    assert torch.autograd.gradgradcheck(sequence_dot, (a, b))
    torch.testing.assert_close(dot(a, (b,)), expected)
    torch.testing.assert_close(dot((), ()), torch.tensor(0.0))
    with pytest.raises(ValueError):
        dot((a,), (b, b))
