"""Test for the adjoint as backward option in the LinearOperator class."""

import pytest
import torch
from mr2.data import KTrajectory, SpatialDimension
from mr2.operators import DownSamplingOp, FourierOp, LinearOperator, NonUniformFastFourierOp
from mr2.utils import RandomGenerator
from torch.autograd.gradcheck import GradcheckError

from tests import dotproduct_adjointness_test


class NonDifferentiableOperator(LinearOperator, adjoint_as_backward=False):
    """Dummy operator for testing purposes."""

    def forward(self, x):
        python_float = x.item()  # breaks autograd as floats do not track gradients
        tensor = torch.tensor([2 * python_float], dtype=x.dtype)
        return (tensor,)

    def adjoint(self, x):
        python_float = x.item()  # breaks autograd as floats do not track gradients
        tensor = torch.tensor([2 * python_float], dtype=x.dtype)
        return (tensor,)


class DifferentiableOperator(LinearOperator, adjoint_as_backward=True):
    """Dummy operator for testing purposes. Should be differentiable due to adjoint_as_backward=True."""

    def forward(self, x):
        python_float = x.item()  # would break autograd as floats do not track gradients
        tensor = torch.tensor([2 * python_float], dtype=x.dtype)
        return (tensor,)

    def adjoint(self, x):
        python_float = x.item()  # would break autograd as floats do not track gradients
        tensor = torch.tensor([2 * python_float], dtype=x.dtype)
        return (tensor,)


def test_linop_autograd_dummy_operator():
    """Ensure correct dummy operator."""
    rng = RandomGenerator(seed=0)
    nondiff = NonDifferentiableOperator()
    diff = DifferentiableOperator()

    u = rng.float32_tensor((1,))
    v = rng.float32_tensor((1,))

    # Differentiable operator and non differentiable operator should be the same
    torch.testing.assert_close(nondiff.adjoint(u), diff.adjoint(u))
    torch.testing.assert_close(nondiff(v), diff(v))

    # and the adjoint should be correct
    dotproduct_adjointness_test(nondiff, u, v)


def test_linop_autograd_nondifferentiable():
    """Test gradient of the forward of the non differentiable operator."""
    nondiff = NonDifferentiableOperator()
    # The non differentiable operator should raise an error,
    # because the forward is non differentiable (python float)
    x = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
    with pytest.raises(GradcheckError):
        torch.autograd.gradcheck(nondiff, x)


def test_linop_autograd_differentiable_forward():
    """Test the gradient of the forward of the differentiable operator."""
    diff = DifferentiableOperator()
    # The differentiable operator should work, as
    # the adjoint can be used as backward.
    x = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
    torch.autograd.gradcheck(diff, x)


def test_linop_autograd_differentiable_adjoint():
    """Test the gradient of the adjoint of the differentiable operator."""
    diff = DifferentiableOperator()
    # Should work, as the forward will be used as the backward
    # of the adjoint.
    x = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
    torch.autograd.gradcheck(diff.H, x)


def test_linop_autograd_differentiable_gradgrad():
    """Test the gradgrad of the differentiable operator."""
    diff = DifferentiableOperator()
    # backward of backward should be working
    # as it should be using the forward
    x = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)
    torch.autograd.gradgradcheck(diff, x)


@pytest.mark.parametrize(
    'operator_name', ['cartesian', 'noncartesian', 'nufft', 'gram', 'subspace_gram', 'downsampling']
)
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('adjoint', [False, True])
@pytest.mark.parametrize(
    'device',
    [
        'cpu',
        pytest.param(
            'cuda',
            marks=[pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')],
        ),
    ],
)
def test_linop_autograd_input_dtype(operator_name, dtype, adjoint, device, monkeypatch):
    """Check real and complex gradients for every adjoint-as-backward operator."""
    rng = RandomGenerator(seed=0)
    zeros = torch.zeros(1, 1, 1, 1, 1, dtype=torch.float64, device=device)
    kx = torch.tensor([-1.8, -0.9, 0.1, 1.2], dtype=torch.float64, device=device)
    if operator_name == 'cartesian':
        kx = torch.arange(-2, 2, dtype=torch.float64, device=device)
    trajectory = KTrajectory(kz=zeros, ky=zeros, kx=kx.reshape(1, 1, 1, 1, 4))
    matrix = SpatialDimension(1, 1, 4)
    nufft = NonUniformFastFourierOp((-1,), matrix, matrix, trajectory).double()
    operator: LinearOperator
    input_shape = (1, 1, 1, 1, 4)
    if operator_name in ('cartesian', 'noncartesian'):
        operator = FourierOp(matrix, matrix, trajectory).double()
    elif operator_name == 'nufft':
        operator = nufft
    elif operator_name == 'gram':
        operator = nufft.gram
    elif operator_name == 'subspace_gram':
        basis = torch.tensor([[1, 1j]], dtype=torch.complex128, device=device)
        operator = nufft.toeplitz(subspace=basis)
        input_shape = (2, 1, 1, 1, 4)
    else:
        operator = DownSamplingOp(-1, (4,), (2,))
        if adjoint:
            input_shape = (1, 1, 1, 1, 2)

    x = rng.randn_tensor(input_shape, dtype=dtype, device=device).requires_grad_()
    function = operator.adjoint if adjoint else operator
    torch.autograd.gradcheck(function, x, fast_mode=True, atol=1e-5, rtol=1e-4)
    torch.autograd.gradgradcheck(function, x, fast_mode=True, atol=1e-5, rtol=1e-4)

    # The Gram implementations use out= and in-place operations, so their
    # derivatives are checked numerically above rather than through autograd.
    if operator_name in ('gram', 'subspace_gram'):
        return
    (output,) = function(x)
    grad_output = rng.randn_like(output)
    (actual,) = torch.autograd.grad(output, x, grad_output)
    with monkeypatch.context() as patch:
        # FourierOp can contain a wrapped NUFFT. Bypass that wrapper as well.
        for method in ('forward', 'adjoint'):
            patch.setattr(NonUniformFastFourierOp, method, getattr(NonUniformFastFourierOp, f'_saved_{method}'))
        (reference,) = getattr(operator, '_saved_adjoint' if adjoint else '_saved_forward')(x)
        (expected,) = torch.autograd.grad(reference, x, grad_output)
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected)
