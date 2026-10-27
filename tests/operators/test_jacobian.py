import pytest
import torch
from mr2.operators import Jacobian
from mr2.operators.functionals import L2NormSquared
from mr2.utils import RandomGenerator

from tests.helper import dotproduct_adjointness_test


def test_jacobian_adjointness() -> None:
    """Test adjointness of Jacobian operator."""
    rng = RandomGenerator(123)
    x = rng.float32_tensor(3)
    y = rng.float32_tensor(())
    x0 = rng.float32_tensor(3)
    op = L2NormSquared()
    jacobian = Jacobian(op, x0)
    dotproduct_adjointness_test(jacobian, x, y)


def test_jacobian_taylor() -> None:
    """Test Taylor expansion"""
    rng = RandomGenerator(123)
    x0 = rng.float32_tensor(3)
    x = x0 + 1e-2 * rng.float32_tensor(3)
    op = L2NormSquared()
    jacobian = Jacobian(op, x0)
    fx = jacobian.taylor(x)
    torch.testing.assert_close(fx, op(x), rtol=1e-3, atol=1e-3)


def test_jacobian_gaussnewton() -> None:
    """Test Gauss Newton approximation of the Hessian"""
    rng = RandomGenerator(123)
    x0 = rng.float32_tensor(3)
    x = x0 + 1e-2 * rng.float32_tensor(3)
    op = L2NormSquared()
    jacobian = Jacobian(op, x0)
    (actual,) = jacobian.gauss_newton(x)
    expected = torch.vdot(x, x0) * 4 * x0  # analytical solution for L2NormSquared
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_jacobian_value_at_x0() -> None:
    """Test value at x0"""
    rng = RandomGenerator(123)
    x0 = rng.float32_tensor(3)
    op = L2NormSquared()
    jacobian = Jacobian(op, x0)
    (actual,) = jacobian.value_at_x0
    (expected,) = op(x0)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.cuda
@pytest.mark.parametrize(('requires_grad', 'grad_enabled'), [(False, True), (True, True), (True, False)])
def test_jacobian_cache_is_cleared_when_moved(requires_grad: bool, grad_enabled: bool) -> None:
    """Cached values and VJPs follow the Jacobian across device moves."""
    x0 = torch.tensor([1.0, 2.0], requires_grad=requires_grad)
    jacobian = Jacobian(L2NormSquared(), x0)
    jacobian.adjoint(torch.tensor(0.5))
    assert jacobian.value_at_x0[0].device.type == 'cpu'

    with torch.set_grad_enabled(grad_enabled):
        jacobian.cuda()
    torch.testing.assert_close(jacobian.value_at_x0[0], x0.cuda().square().sum())
    if requires_grad and grad_enabled:
        (gradient,) = torch.autograd.grad(jacobian.value_at_x0[0], x0)
        torch.testing.assert_close(gradient, 2 * x0)
    (actual,) = jacobian.adjoint(torch.tensor(0.5, device='cuda'))
    torch.testing.assert_close(actual, x0.cuda())

    jacobian.cpu()
    torch.testing.assert_close(jacobian.value_at_x0[0], x0.square().sum())
    (actual,) = jacobian.adjoint(torch.tensor(0.5))
    torch.testing.assert_close(actual, x0)


@pytest.mark.parametrize(('requires_grad', 'grad_enabled'), [(False, True), (True, True), (True, False)])
def test_jacobian_cache_is_cleared_when_cast(requires_grad: bool, grad_enabled: bool) -> None:
    """Cached values and VJPs follow the Jacobian dtype after a cast."""
    x0 = torch.tensor([1.0, 2.0], requires_grad=requires_grad)
    jacobian = Jacobian(L2NormSquared(), x0)
    jacobian.adjoint(torch.tensor(0.5))
    with torch.set_grad_enabled(grad_enabled):
        jacobian.double()
    torch.testing.assert_close(jacobian.value_at_x0[0], x0.double().square().sum())
    if requires_grad and grad_enabled:
        (gradient,) = torch.autograd.grad(jacobian.value_at_x0[0], x0)
        torch.testing.assert_close(gradient, 2 * x0)

    (actual,) = jacobian.adjoint(torch.tensor(0.5, dtype=torch.float64))
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, x0.double())
