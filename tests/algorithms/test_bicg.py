"""Regression tests for BiCGStab convergence and breakdown."""

import pytest
import torch
from mr2.algorithms.optimizers import bicg
from mr2.algorithms.optimizers.bicg import BiCGStatus
from mr2.operators import EinsumOp, MultiIdentityOp


@pytest.mark.parametrize('scale', [1.0, 4.0, 2.0 + 1.0j])
@pytest.mark.parametrize('preconditioned', [False, True])
def test_alpha_convergence(scale, preconditioned):
    """An exact alpha step must skip stabilization and report the completed iteration."""
    right_hand_side = (torch.tensor([1.0, 2.0], dtype=torch.complex128),)
    calls = []

    def operator(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Count applications to detect unnecessary stabilization after convergence."""
        calls.append(values)
        return tuple(scale * x for x in values)

    def preconditioner(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Exactly invert the scaled identity."""
        return tuple(x / scale for x in values)

    statuses: list[BiCGStatus] = []
    result = bicg(
        operator,
        right_hand_side,
        preconditioner_inverse=preconditioner if preconditioned else None,
        callback=statuses.append,
    )
    torch.testing.assert_close(result, (right_hand_side[0] / scale,))
    assert len(calls) == 2  # Initial residual and alpha step only.
    assert len(statuses) == 1
    assert statuses[0]['iteration_number'] == 0
    torch.testing.assert_close(statuses[0]['solution'], result)
    torch.testing.assert_close(statuses[0]['residual'], (torch.zeros_like(right_hand_side[0]),))


@pytest.mark.parametrize(('exact_initial', 'tolerance'), [(False, 0.0), (False, 1e-6), (True, 0.0)])
def test_initial_convergence(exact_initial, tolerance):
    """Zero residuals return before dividing, including when tolerance is zero."""
    expected = torch.tensor([1.0, 2.0]) if exact_initial else torch.zeros(2)

    def callback(status):
        """Reject iterations when the initial residual is already zero."""
        pytest.fail('An already converged initial solution must not call the callback')

    def solve():
        """Exercise identical inputs with or without the expected tolerance warning."""
        return bicg(
            MultiIdentityOp(),
            (expected,),
            initial_value=(expected,) if exact_initial else None,
            tolerance=tolerance,
            callback=callback,
        )

    if tolerance == 0:
        with pytest.warns(UserWarning, match='Tolerance is very small'):
            result = solve()
    else:
        result = solve()
    torch.testing.assert_close(result, (expected,))


def test_empty_rhs() -> None:
    """Reject an empty tuple before indexing the right-hand-side dtype."""
    with pytest.raises(ValueError, match='right_hand_side cannot be empty'):
        bicg(MultiIdentityOp(), ())


def test_mixed_tuple_alpha_convergence():
    """Preserve mixed real/complex tuple handling on the converged alpha path."""
    right_hand_side = (torch.tensor([1.0, 2.0]), torch.tensor([1.0 + 2.0j]))
    result = bicg(MultiIdentityOp(), right_hand_side)
    torch.testing.assert_close(result, right_hand_side)


def test_zero_tolerance_alpha_convergence() -> None:
    """An exact alpha step terminates even with a zero stopping tolerance."""
    with pytest.warns(UserWarning, match='Tolerance is very small'):
        result = bicg(MultiIdentityOp(), (torch.tensor([1.0, 2.0]),), tolerance=0)
    torch.testing.assert_close(result, (torch.tensor([1.0, 2.0]),))


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('preconditioned', [False, True])
def test_nonsymmetric_solution(dtype, preconditioned):
    """Compare against a direct solve with and without a diagonal preconditioner."""
    matrix = torch.tensor([[4, 1, -1], [2, 5, 1], [0, -2, 6]], dtype=dtype)
    right_hand_side = torch.tensor([1, -3, 2], dtype=dtype)
    if dtype.is_complex:
        matrix = matrix + 1j * torch.tensor([[1, 2, 0], [-1, 0, 1], [2, 0, -1]], dtype=dtype)
        right_hand_side = right_hand_side + 1j * torch.tensor([2, 1, -1], dtype=dtype)
    statuses: list[BiCGStatus] = []
    (result,) = bicg(
        EinsumOp(matrix),
        (right_hand_side,),
        initial_value=(torch.ones_like(right_hand_side),),
        preconditioner_inverse=EinsumOp(matrix.diagonal().reciprocal().diag()) if preconditioned else None,
        tolerance=1e-12,
        callback=statuses.append,
    )
    torch.testing.assert_close(result, torch.linalg.solve(matrix, right_hand_side))
    torch.testing.assert_close(matrix @ result, right_hand_side)
    assert [status['iteration_number'] for status in statuses] == list(range(len(statuses)))
    for status in statuses:
        torch.testing.assert_close(status['residual'][0], right_hand_side - matrix @ status['solution'][0])


@pytest.mark.parametrize(
    ('matrix', 'message'),
    [
        ([[0, 1], [1, 0]], 'alpha'),
        ([[1, 0], [1, 0]], 'omega'),
        ([[1, 1], [1, 0]], 'zero or non-finite omega'),
        ([[1, 0, 0], [1, 2, 0], [1, 0, 3]], 'alpha'),
        ([[float('nan'), 0], [0, 1]], 'non-finite'),
    ],
)
def test_breakdown(matrix, message):
    """Singular and orthogonal search directions raise instead of returning NaNs."""
    matrix = torch.tensor(matrix, dtype=torch.float64)
    right_hand_side = torch.zeros(matrix.shape[0], dtype=matrix.dtype)
    right_hand_side[0] = 1
    with pytest.raises(RuntimeError, match=f'BiCGStab breakdown: .*{message}'):
        bicg(EinsumOp(matrix), (right_hand_side,))


def test_alpha_convergence_autograd() -> None:
    """The alpha-only solution retains the derivative of the inverse operator."""
    right_hand_side = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    (result,) = bicg(EinsumOp(4 * torch.eye(2, dtype=right_hand_side.dtype)), (right_hand_side,))
    (gradient,) = torch.autograd.grad(result.sum(), right_hand_side)
    torch.testing.assert_close(gradient, torch.full_like(right_hand_side, 0.25))


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('identity', [False, True])
def test_tiny_rhs(dtype, identity):
    """Tiny nonzero data must survive both convergence checks and recurrence products."""
    matrix = torch.eye(2, dtype=dtype) if identity else torch.tensor([[4, 1], [2, 3]], dtype=dtype)
    rhs = torch.tensor([1e-25, 2e-25], dtype=dtype)
    statuses: list[BiCGStatus] = []
    with pytest.warns(UserWarning, match='Tolerance is very small'):
        (result,) = bicg(EinsumOp(matrix), (rhs,), tolerance=0, callback=statuses.append)
    assert statuses
    torch.testing.assert_close(result / 1e-25, torch.linalg.solve(matrix, rhs / 1e-25), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('size', [4, 64])
def test_numerical_convergence_zero_tolerance(size):
    """Recurrence exhaustion after convergence must preserve an accurate iterate."""
    torch.manual_seed(0)
    matrix = torch.randn(size, size, dtype=torch.complex64) + size * torch.eye(size)
    rhs = torch.randn(size, dtype=torch.complex64)
    with pytest.warns(UserWarning, match='Tolerance is very small'):
        (result,) = bicg(EinsumOp(matrix), (rhs,), tolerance=0, max_iterations=60)
    torch.testing.assert_close(matrix @ result, rhs, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('scale', [1.0, 1e-25])
def test_inconsistent_system_zero_tolerance(scale):
    """A small absolute residual cannot excuse a genuine singular-system failure."""
    matrix = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    rhs = scale * torch.tensor([1.0, 0.0])
    with (
        pytest.warns(UserWarning, match='Tolerance is very small'),
        pytest.raises(RuntimeError, match='BiCGStab breakdown'),
    ):
        bicg(EinsumOp(matrix), (rhs,), tolerance=0)


@pytest.mark.parametrize('converged', [False, True])
def test_breakdown_checks_true_residual(monkeypatch, converged):
    """Recurrence exhaustion is acceptable only after the actual system is solved."""
    import importlib

    module = importlib.import_module('mr2.algorithms.optimizers.bicg')
    matrix = torch.tensor([[4.0, 1.0], [2.0, 3.0]])
    rhs = torch.tensor([1.0, 2.0])
    statuses: list[BiCGStatus] = []

    breakdown_calls: list[None] = []

    def exhausted_product(*args):
        """Inject non-finite recurrence coefficients without changing the operator."""
        breakdown_calls.append(None)
        return torch.tensor(float('nan'))

    def callback(status):
        """Trigger breakdown before or after true convergence, according to the case."""
        statuses.append(status)
        # Force the next recurrence to break down, either early or after convergence.
        if not converged or (rhs - matrix @ status['solution'][0]).norm() < 1e-6:
            monkeypatch.setattr(module, 'vdot', exhausted_product)

    if converged:
        with pytest.warns(UserWarning, match='Tolerance is very small'):
            (result,) = bicg(EinsumOp(matrix), (rhs,), tolerance=0, callback=callback)
        torch.testing.assert_close(matrix @ result, rhs)
    else:
        with (
            pytest.warns(UserWarning, match='Tolerance is very small'),
            pytest.raises(RuntimeError, match='BiCGStab breakdown'),
        ):
            bicg(EinsumOp(matrix), (rhs,), tolerance=0, callback=callback)
    assert statuses
    assert breakdown_calls
