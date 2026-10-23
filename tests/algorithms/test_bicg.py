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
        calls.append(values)
        return tuple(scale * x for x in values)

    def preconditioner(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
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


@pytest.mark.parametrize('exact_initial', [False, True])
@pytest.mark.parametrize('tolerance', [0.0, 1e-6])
def test_initial_convergence(exact_initial, tolerance):
    """Zero residuals return before dividing, including when tolerance is zero."""
    expected = torch.tensor([1.0, 2.0]) if exact_initial else torch.zeros(2)

    def callback(status):
        pytest.fail('An already converged initial solution must not call the callback')

    def solve():
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


def test_empty_rhs():
    with pytest.raises(ValueError, match='right_hand_side cannot be empty'):
        bicg(MultiIdentityOp(), ())


def test_mixed_tuple_alpha_convergence():
    """Preserve mixed real/complex tuple handling on the converged alpha path."""
    right_hand_side = (torch.tensor([1.0, 2.0]), torch.tensor([1.0 + 2.0j]))
    result = bicg(MultiIdentityOp(), right_hand_side)
    torch.testing.assert_close(result, right_hand_side)


def test_zero_tolerance_alpha_convergence():
    with pytest.warns(UserWarning, match='Tolerance is very small'):
        result = bicg(MultiIdentityOp(), (torch.tensor([1.0, 2.0]),), tolerance=0)
    torch.testing.assert_close(result, (torch.tensor([1.0, 2.0]),))


@pytest.mark.parametrize(
    ('initial_value', 'message'),
    [((torch.zeros(2), torch.zeros(2)), 'Length mismatch'), ((torch.zeros(3),), 'Shape mismatch')],
)
def test_invalid_initial_value(initial_value, message):
    with pytest.raises(ValueError, match=message):
        bicg(MultiIdentityOp(), (torch.ones(2),), initial_value=initial_value)


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


def test_callback_stop():
    """Returning False stops after the first full iteration with its current solution."""
    matrix = torch.tensor([[4.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
    statuses: list[BiCGStatus] = []

    def callback(status):
        statuses.append(status)
        return False

    result = bicg(EinsumOp(matrix), (torch.tensor([1.0, 2.0], dtype=matrix.dtype),), callback=callback)
    assert len(statuses) == 1
    assert statuses[0]['iteration_number'] == 0
    torch.testing.assert_close(result, statuses[0]['solution'])
    assert statuses[0]['residual'][0].norm() > 1e-6


@pytest.mark.parametrize(
    ('matrix', 'message'),
    [
        ([[0, 1], [1, 0]], 'zero alpha denominator'),
        ([[1, 0], [1, 0]], 'zero omega denominator'),
        ([[1, 1], [1, 0]], 'zero or non-finite omega'),
        ([[1, 0, 0], [1, 2, 0], [1, 0, 3]], 'shadow residual'),
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


def test_alpha_convergence_autograd():
    right_hand_side = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    (result,) = bicg(EinsumOp(4 * torch.eye(2, dtype=right_hand_side.dtype)), (right_hand_side,))
    (gradient,) = torch.autograd.grad(result.sum(), right_hand_side)
    torch.testing.assert_close(gradient, torch.full_like(right_hand_side, 0.25))
