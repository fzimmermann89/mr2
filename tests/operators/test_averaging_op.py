"""Test the averaging operator."""

import pytest
import torch
from mr2.operators import AveragingOp
from mr2.utils import RandomGenerator

from tests.helper import dotproduct_adjointness_test


def test_averageingop_adjointness() -> None:
    """Test the adjointness of the averaging operator."""
    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=(5, 15, 10))
    v = rng.complex64_tensor(size=(5, 3, 10))
    idx = [0, 1, 2], slice(0, 6, 2), torch.tensor([-1, -2, -3])
    op = AveragingOp(dim=-2, idx=idx, domain_size=15)
    dotproduct_adjointness_test(op, u, v)


def test_averageingop_forward() -> None:
    """Test the forward method of the averaging operator."""
    rng = RandomGenerator(seed=1)
    u = rng.complex64_tensor(size=(5, 10))
    idx = [0, 1, 2], slice(0, 6, 2), torch.tensor([-1, -2, -3]), (False, True, True, False, True)
    op = AveragingOp(dim=0, idx=idx, domain_size=5)
    (actual,) = op(u)
    torch.testing.assert_close(actual[0], u[(0, 1, 2),].mean(dim=0))
    torch.testing.assert_close(actual[1], u[(0, 2, 4),].mean(dim=0))
    torch.testing.assert_close(actual[2], u[(-1, -2, -3),].mean(dim=0))
    torch.testing.assert_close(actual[3], u[(1, 2, 4),].mean(dim=0))


def test_averageingop_no_domain_size() -> None:
    """Test the adjoint method of the averaging operator without domain size."""
    rng = RandomGenerator(seed=0)
    v = rng.float32_tensor(size=(5, 2))
    idx = rng.int64_tensor(size=(2, 3), low=0, high=9)
    op = AveragingOp(dim=1, idx=idx)

    with pytest.raises(ValueError, match='Domain'):
        op.adjoint(v)

    u = rng.float32_tensor(size=(5, 10))
    op(u)  # this sets the domain size
    with pytest.warns(match='Domain'):
        (actual,) = op.adjoint(v)
    assert actual.shape == u.shape


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize(
    'group',
    [
        [0, 0, 1],
        (0, 0, 1),
        [0, -2],
        torch.tensor([0, -2]),
        [],
        slice(0, 0),
        torch.tensor([False, False]),
        torch.zeros(2, dtype=torch.uint8),
    ],
)
def test_averaging_adjoint_invalid_groups(
    group: list[int] | tuple[int, ...] | slice | torch.Tensor, device: str
) -> None:
    """Empty groups and repeated indices cannot silently produce an invalid adjoint."""
    if isinstance(group, torch.Tensor):
        group = group.to(device)
    op = AveragingOp(-1, [group], domain_size=2).to(device)
    with pytest.raises(ValueError, match='nonempty groups without repeated indices'):
        op.H(torch.ones(1, device=device))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('group', [[], slice(0, 0), [False, False], torch.tensor([False, False])])
def test_averaging_forward_empty_group(group: list[bool] | slice | torch.Tensor, device: str) -> None:
    """An empty group raises instead of producing a NaN mean."""
    if isinstance(group, torch.Tensor):
        group = group.to(device)
    op = AveragingOp(-1, [group], domain_size=2)
    with pytest.raises(ValueError, match='nonempty groups'):
        op(torch.ones(2, device=device))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize(
    'group',
    [(0, 2), (-3, -1), [True, False, True], (True, False, True), torch.tensor([True, False, True]), slice(0, 3, 2)],
)
def test_averaging_adjoint_sequence_group(
    group: tuple[int, ...] | list[bool] | torch.Tensor | slice, device: str
) -> None:
    """Sequence groups count the selected entries, including boolean masks."""
    if isinstance(group, torch.Tensor):
        group = group.to(device)
    op = AveragingOp(-1, [group], domain_size=3)
    x = torch.tensor([2.0, 3.0, 4.0], device=device)
    y = torch.tensor([3.0], device=device)
    dotproduct_adjointness_test(op, x, y)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('mask', [[1, 1, 1], [2, 0, 3]])
def test_averaging_adjoint_byte_mask(mask: list[int], device: str) -> None:
    """Legacy byte masks select nonzero entries rather than integer indices."""
    op = AveragingOp(-1, [torch.tensor(mask, dtype=torch.uint8, device=device)], domain_size=3)
    x = torch.tensor([2.0, 3.0, 4.0], device=device)
    y = torch.tensor([3.0], device=device)
    with pytest.warns(UserWarning, match='indexing with dtype torch.uint8'):
        dotproduct_adjointness_test(op, x, y)
