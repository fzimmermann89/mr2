"""Random B0 map generation."""

from math import isqrt

import torch
from scipy.special import sph_harm_y

from mr2.data.SpatialDimension import SpatialDimension
from mr2.utils.RandomGenerator import RandomGenerator


def random_b0_sh_coefficients(
    max_degree: int = 3,
    sigma: float = 1000.0,
    scaling_power: float = -1.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample coefficients for a real spherical-harmonic B0 map.

    Parameters
    ----------
    max_degree
        Maximum spherical harmonic degree.
    sigma
        Standard deviation of the degree-one coefficients. This sets the unit of the coefficients and resulting map.
    scaling_power
        Exponent controlling the standard deviation for degree ``n`` as ``sigma * n**scaling_power``.
    seed
        Random seed.

    Returns
    -------
    coefficients
        Coefficients in degree-major order, excluding the constant term.
    """
    if max_degree < 0:
        raise ValueError('max_degree must be non-negative.')
    rng = RandomGenerator(seed)
    coefficients = [
        rng.randn_tensor((1,), torch.float64) * sigma * degree**scaling_power
        for degree in range(1, max_degree + 1)
        for _ in range(2 * degree + 1)
    ]
    return torch.cat(coefficients) if coefficients else torch.empty(0, dtype=torch.float64)


def b0map_from_sh_coefficients(
    shape: SpatialDimension[int],
    fov: SpatialDimension[float],
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a real spherical-harmonic B0 map on a Cartesian grid.

    Parameters
    ----------
    shape
        Grid dimensions.
    fov
        Field of view in meters (fov_z, fov_y, fov_x).
    coefficients
        Coefficients in the order returned by :func:`random_b0_sh_coefficients`. The output has the same unit as these
        coefficients.

    Returns
    -------
    b0_map
        B0 field map with dimensions (z, y, x), in the unit of ``coefficients``.
    """
    if coefficients.ndim != 1:
        raise ValueError('coefficients must be one-dimensional.')
    max_degree = isqrt(coefficients.numel() + 1) - 1
    if (max_degree + 1) ** 2 - 1 != coefficients.numel():
        raise ValueError('The number of coefficients must be (max_degree + 1)**2 - 1.')

    r_ref = max(fov.zyx) / 2
    z = torch.linspace(-fov.z / 2, fov.z / 2, shape.z)
    y = torch.linspace(-fov.y / 2, fov.y / 2, shape.y)
    x = torch.linspace(-fov.x / 2, fov.x / 2, shape.x)
    z, y, x = torch.meshgrid(z, y, x, indexing='ij')
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.arccos(torch.clamp(z / (r + 1e-12), -1, 1))
    phi = torch.atan2(y, x)

    b0_map = torch.zeros(shape.zyx, dtype=torch.float64)
    phi_numpy = phi.numpy()
    theta_numpy = theta.numpy()

    index = 0
    for degree in range(1, max_degree + 1):
        for order in range(-degree, degree + 1):
            if order > 0:
                harmonic = (-1) ** order * 2**0.5 * sph_harm_y(degree, order, phi_numpy, theta_numpy).real
            elif order == 0:
                harmonic = sph_harm_y(degree, 0, phi_numpy, theta_numpy).real
            else:
                harmonic = (-1) ** order * 2**0.5 * sph_harm_y(degree, -order, phi_numpy, theta_numpy).imag

            solid_harmonic = (r / r_ref) ** degree * torch.from_numpy(harmonic)
            b0_map += coefficients[index] * solid_harmonic
            index += 1

    return b0_map


def random_b0map(
    shape: SpatialDimension[int],
    fov: SpatialDimension[float],
    l_max: int = 3,
    sigma_ppm: float = 1000.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Simulate B0 inhomogeneity map via randomized spherical harmonics.

    Parameters
    ----------
    shape
        Grid dimensions
    fov
        Field of view in meters (fov_z, fov_y, fov_x).
    l_max
        Maximum spherical harmonic degree.
    sigma_ppm
        Std of inhomogeneity in ppm.
    seed
        Random seed.

    Returns
    -------
    b0_map
        (z, y, x) B0 field map in ppm.
    """
    coefficients = random_b0_sh_coefficients(max_degree=l_max, sigma=sigma_ppm, seed=seed)
    return b0map_from_sh_coefficients(shape, fov, coefficients)
