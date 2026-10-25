"""Regression tests for BiCGStab convergence."""

import pytest
import torch
from mr2.algorithms.optimizers import bicg
from mr2.algorithms.optimizers.bicg import BiCGStatus
from mr2.operators import EinsumOp, MultiIdentityOp


def test_initial_convergence() -> None:
    """An exact initial solution returns without iterating at zero tolerance."""
    rhs = torch.tensor([1.0, 2.0])
    statuses: list[BiCGStatus] = []
    with pytest.warns(UserWarning, match='Tolerance is very small'):
        result = bicg(MultiIdentityOp(), (rhs,), initial_value=(rhs,), tolerance=0, callback=statuses.append)
    torch.testing.assert_close(result, (rhs,))
    assert not statuses


@pytest.mark.parametrize('preconditioned', [False, True])
def test_alpha_convergence(preconditioned: bool) -> None:
    """An exact alpha step skips stabilization and reports the completed iteration."""
    rhs = torch.tensor([1.0, 2.0])
    statuses: list[BiCGStatus] = []
    with pytest.warns(UserWarning, match='Tolerance is very small'):
        result = bicg(
            EinsumOp(4 * torch.eye(2)),
            (rhs,),
            preconditioner_inverse=EinsumOp(torch.eye(2) / 4) if preconditioned else None,
            tolerance=0,
            callback=statuses.append,
        )
    torch.testing.assert_close(result, (rhs / 4,))
    assert len(statuses) == 1
    torch.testing.assert_close(statuses[0]['residual'], (torch.zeros_like(rhs),))
