"""Tests for upsampling."""

from typing import Literal

import pytest
import torch
from mr2.nn.Upsample import Upsample
from mr2.utils import RandomGenerator


def torch_upsample(
    x: torch.Tensor,
    size: tuple[int, int],
    mode: Literal['nearest', 'linear', 'cubic'],
) -> torch.Tensor:
    """Upsample with torch."""
    torch_mode = {'nearest': 'nearest', 'linear': 'bilinear', 'cubic': 'bicubic'}[mode]
    if x.is_complex():
        return torch.complex(
            torch.nn.functional.interpolate(x.real, size=size, mode=torch_mode),
            torch.nn.functional.interpolate(x.imag, size=size, mode=torch_mode),
        )
    return torch.nn.functional.interpolate(x, size=size, mode=torch_mode)


def test_upsample_nearest_matches_torch() -> None:
    """Test nearest-neighbor upsampling."""
    rng = RandomGenerator(seed=0)
    x = rng.float32_tensor(size=(2, 3, 5, 6))
    y = Upsample(dim=(-2, -1), scale_factor=2, mode='nearest')(x)
    expected = torch_upsample(x, size=(10, 12), mode='nearest')
    torch.testing.assert_close(y, expected)


def test_upsample_linear_matches_torch() -> None:
    """Test linear upsampling."""
    rng = RandomGenerator(seed=0)
    x = rng.complex64_tensor(size=(2, 3, 5, 6))
    y = Upsample(dim=(-2, -1), scale_factor=2, mode='linear')(x)
    expected = torch_upsample(x, size=(10, 12), mode='linear')
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize('mode', ['nearest', 'linear', 'cubic'])
def test_upsample_non_adjacent_dimensions_matches_torch(mode: Literal['nearest', 'linear', 'cubic']) -> None:
    """Test upsampling along non-adjacent dimensions."""
    rng = RandomGenerator(seed=0)
    x = rng.float32_tensor(size=(2, 3, 4, 5))
    y = Upsample(dim=(-3, -1), scale_factor=2, mode=mode)(x)
    expected = torch_upsample(x.swapaxes(-3, -2), size=(6, 10), mode=mode).swapaxes(-3, -2)
    torch.testing.assert_close(y, expected)


def test_upsample_cubic_matches_torch() -> None:
    """Test cubic upsampling."""
    rng = RandomGenerator(seed=0)
    x = rng.float32_tensor(size=(2, 3, 5, 6))
    y = Upsample(dim=(-2, -1), scale_factor=2, mode='cubic')(x)
    expected = torch_upsample(x, size=(10, 12), mode='cubic')
    torch.testing.assert_close(y, expected)


def test_upsample_cubic_requires_two_dimensions() -> None:
    """Test cubic upsampling input validation."""
    with pytest.raises(ValueError, match='Cubic interpolation'):
        Upsample(dim=(-1,), mode='cubic')
