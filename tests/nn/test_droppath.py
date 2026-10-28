"""Test DropPath."""

import pytest
import torch
from mr2.nn.DropPath import DropPath
from mr2.utils import RandomGenerator


@pytest.mark.parametrize('scale_by_keep', [False, True])
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
def test_droppath_no_drop(device: str, scale_by_keep: bool) -> None:
    """Test DropPath with zero drop rate (should pass through unchanged)."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5)).to(device).requires_grad_(True)
    droppath = DropPath(0, scale_by_keep=scale_by_keep).to(device)
    y = droppath(x)
    assert y is x
    y.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize('scale_by_keep', [False, True])
@pytest.mark.parametrize('nonfinite', [False, True])
def test_droppath_drop_all(scale_by_keep: bool, nonfinite: bool) -> None:
    """Test DropPath with full drop rate (should output zeros)."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((3, 4))
    if nonfinite:
        x[0, :3] = torch.tensor([float('nan'), float('inf'), -float('inf')])
    x.requires_grad_(True)
    original = x.detach().clone()
    droppath = DropPath(1.0, scale_by_keep=scale_by_keep)
    y = droppath(x)
    assert (y == 0).all()
    y.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    torch.testing.assert_close(x, original, equal_nan=True)


@pytest.mark.parametrize('droprate', [0.0, 1.0])
@pytest.mark.parametrize('scale_by_keep', [False, True])
def test_droppath_eval(droprate: float, scale_by_keep: bool) -> None:
    """Evaluation preserves the input and its gradient at both boundaries."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((3, 4)).requires_grad_(True)
    y = DropPath(droprate, scale_by_keep=scale_by_keep).eval()(x)
    assert y is x
    y.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize('droprate', [-0.1, 1.1, float('nan')])
def test_droppath_invalid_droprate(droprate: float) -> None:
    """Reject drop probabilities outside the allowed interval."""
    with pytest.raises(ValueError, match='droprate must be between 0 and 1'):
        DropPath(droprate)
