"""Test DropPath."""

import pytest
import torch
from mr2.nn.DropPath import DropPath
from mr2.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
def test_droppath_no_drop(device: str) -> None:
    """Test DropPath with zero drop rate (should pass through unchanged)."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5)).to(device)
    droppath = DropPath(0).to(device)
    y = droppath(x)
    assert (y == x).all()


def test_droppath_drop_all() -> None:
    """Test DropPath with full drop rate (should output zeros)."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5))
    droppath = DropPath(1.0)
    y = droppath(x)
    assert (y == 0).all()


@pytest.mark.xfail(strict=True, reason='Known issue: full scaled dropping divides the mask by zero.')
def test_droppath_drop_all_with_keep_scaling_is_finite() -> None:
    """Full drop should not create NaN activations when keep scaling is enabled."""
    output = DropPath(1.0, scale_by_keep=True)(torch.ones(3, 4))
    assert torch.isfinite(output).all()
    assert (output == 0).all()
