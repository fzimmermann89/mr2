"""Regression tests for BiCGStab convergence."""

import pytest
import torch
from mr2.algorithms.optimizers import bicg
from mr2.algorithms.optimizers.bicg import BiCGStatus
from mr2.operators import EinsumOp, MultiIdentityOp
from tests.conftest import cuda_marks


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=cuda_marks)])
def test_initial_convergence(device: str) -> None:
    """An exact initial solution returns without iterating at zero tolerance."""
    rhs = torch.tensor([1.0, 2.0], device=device)
    statuses: list[BiCGStatus] = []
    with pytest.warns(UserWarning, match='Tolerance is very small'):
        result = bicg(MultiIdentityOp(), (rhs,), initial_value=(rhs,), tolerance=0, callback=statuses.append)
    torch.testing.assert_close(result, (rhs,))
    assert not statuses


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=cuda_marks)])
@pytest.mark.parametrize('preconditioned', [False, True])
def test_alpha_convergence(preconditioned: bool, device: str) -> None:
    """An exact alpha step skips stabilization and reports the completed iteration."""
    rhs = torch.tensor([1.0, 2.0], device=device)
    statuses: list[BiCGStatus] = []
    with pytest.warns(UserWarning, match='Tolerance is very small'):
        result = bicg(
            EinsumOp(4 * torch.eye(2, device=device)),
            (rhs,),
            preconditioner_inverse=EinsumOp(torch.eye(2, device=device) / 4) if preconditioned else None,
            tolerance=0,
            callback=statuses.append,
        )
    torch.testing.assert_close(result, (rhs / 4,))
    assert len(statuses) == 1
    torch.testing.assert_close(statuses[0]['residual'], (torch.zeros_like(rhs),))
