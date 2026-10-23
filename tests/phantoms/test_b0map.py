"""Tests for B0 map generation."""

import pytest
import torch
from mr2.data import SpatialDimension
from mr2.phantoms.b0map import b0map_from_sh_coefficients, random_b0_sh_coefficients, random_b0map


def test_random_b0map_from_coefficients() -> None:
    """Sampling and evaluating coefficients agrees with the convenience function."""
    shape = SpatialDimension(1, 8, 6)
    fov = SpatialDimension(1e-3, 8e-3, 6e-3)
    coefficients = random_b0_sh_coefficients(max_degree=3, sigma=100, seed=4)

    actual = b0map_from_sh_coefficients(shape, fov, coefficients)
    expected = random_b0map(shape, fov, l_max=3, sigma_ppm=100, seed=4)

    assert coefficients.shape == (15,)
    assert actual.shape == shape.zyx
    torch.testing.assert_close(actual, expected)


def test_random_b0_sh_coefficients_degree_scaling() -> None:
    """Degree scaling changes each coefficient using its degree."""
    unscaled = random_b0_sh_coefficients(max_degree=2, sigma=1, scaling_power=0, seed=2)
    scaled = random_b0_sh_coefficients(max_degree=2, sigma=1, scaling_power=-2, seed=2)

    torch.testing.assert_close(scaled[:3], unscaled[:3])
    torch.testing.assert_close(scaled[3:], unscaled[3:] / 4)


@pytest.mark.parametrize('coefficients', [torch.ones(2, 2), torch.ones(4)])
def test_b0map_from_invalid_coefficients(coefficients: torch.Tensor) -> None:
    """Invalid coefficient layouts are rejected."""
    with pytest.raises(ValueError):
        b0map_from_sh_coefficients(SpatialDimension(1, 2, 2), SpatialDimension(1.0, 1.0, 1.0), coefficients)


def test_random_b0_sh_coefficients_invalid_degree() -> None:
    """A negative spherical-harmonic degree is rejected."""
    with pytest.raises(ValueError, match='max_degree'):
        random_b0_sh_coefficients(max_degree=-1)
