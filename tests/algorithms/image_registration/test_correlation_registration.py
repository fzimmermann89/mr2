"""Tests for correlation-based image registration."""

import torch

from mr2.algorithms.image_registration.correlation_registration import correlation_registration
from mr2.data import SpatialDimension


def test_correlation_registration_2d(ellipse_phantom) -> None:
    """Test 2D shift-only correlation registration."""
    image_dimensions = SpatialDimension(z=1, y=ellipse_phantom.n_y, x=ellipse_phantom.n_x)
    fixed = ellipse_phantom.phantom.image_space(image_dimensions).real
    moving = torch.zeros_like(fixed)
    moving[..., 3:, :-5] = fixed[..., :-3, 5:]

    shift = correlation_registration(fixed, moving)

    assert isinstance(shift, SpatialDimension)
    torch.testing.assert_close(shift.z, torch.zeros_like(shift.z))
    torch.testing.assert_close(shift.y, torch.full_like(shift.y, -3.0), atol=1e-2, rtol=0)
    torch.testing.assert_close(shift.x, torch.full_like(shift.x, 5.0), atol=1e-2, rtol=0)


def test_correlation_registration_half_pixel_shift() -> None:
    """Test sub-pixel shift-only correlation registration."""
    fixed = torch.zeros(1, 1, 1, 33, 35)
    fixed[..., 16, 17] = 1.0
    fixed[..., 10, 25] = 0.8
    fixed[..., 24, 8] = 0.5

    y, x = torch.meshgrid(torch.linspace(-1.0, 1.0, 33), torch.linspace(-1.0, 1.0, 35), indexing='ij')
    grid = torch.stack((x + 0.5 * 2.0 / 34.0, y - 0.5 * 2.0 / 32.0), dim=-1)
    moving = torch.nn.functional.grid_sample(
        fixed.flatten(end_dim=-4), grid[None], mode='bilinear', padding_mode='zeros', align_corners=True
    ).unflatten(0, fixed.shape[:-3])

    shift = correlation_registration(fixed, moving)

    torch.testing.assert_close(shift.z, torch.zeros_like(shift.z))
    torch.testing.assert_close(shift.y, torch.full_like(shift.y, -0.5), atol=1e-2, rtol=0)
    torch.testing.assert_close(shift.x, torch.full_like(shift.x, 0.5), atol=1e-2, rtol=0)


def test_correlation_registration_3d() -> None:
    """Test 3D shift-only correlation registration."""
    z = torch.linspace(-1.0, 1.0, 17)
    y = torch.linspace(-1.0, 1.0, 19)
    x = torch.linspace(-1.0, 1.0, 21)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    volume = (
        torch.exp(-((zz + 0.25) ** 2 + 2 * (yy - 0.1) ** 2 + 3 * (xx + 0.15) ** 2) / 0.08)
        + 0.8 * torch.exp(-(2 * (zz - 0.3) ** 2 + (yy + 0.25) ** 2 + 2 * (xx - 0.2) ** 2) / 0.05)
    )[None, None]
    moving = torch.zeros_like(volume)
    moving[..., 1:, :-2, 3:] = volume[..., :-1, 2:, :-3]

    shift = correlation_registration(volume, moving)

    torch.testing.assert_close(shift.z, torch.full_like(shift.z, -1.0), atol=1e-2, rtol=0)
    torch.testing.assert_close(shift.y, torch.full_like(shift.y, 2.0), atol=1e-2, rtol=0)
    torch.testing.assert_close(shift.x, torch.full_like(shift.x, -3.0), atol=1e-2, rtol=0)


def test_correlation_registration_with_mask() -> None:
    """Test masked correlation registration."""
    y = torch.linspace(-1.0, 1.0, 48)
    x = torch.linspace(-1.0, 1.0, 52)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    fixed = (
        torch.exp(-((yy + 0.25) ** 2 + (xx - 0.15) ** 2) / 0.04)
        + 0.7 * torch.exp(-((yy - 0.1) ** 2 + 2 * (xx + 0.3) ** 2) / 0.03)
    )[None, None, None]
    moving = torch.zeros_like(fixed)
    moving[..., :-4, 6:] = fixed[..., 4:, :-6]

    mask = torch.zeros_like(fixed[:, :1])
    mask[..., 8:-8, 8:-8] = 1.0

    fixed_corrupted = fixed.clone()
    moving_corrupted = moving.clone()
    fixed_corrupted[..., :6, :] = 10.0
    moving_corrupted[..., -6:, :] = -10.0

    shift = correlation_registration(fixed_corrupted, moving_corrupted, mask=mask)

    torch.testing.assert_close(shift.z, torch.zeros_like(shift.z))
    torch.testing.assert_close(shift.y, torch.full_like(shift.y, 4.0), atol=1e-2, rtol=0)
    torch.testing.assert_close(shift.x, torch.full_like(shift.x, -6.0), atol=1e-2, rtol=0)


def test_correlation_registration_complex() -> None:
    """Test complex-valued correlation registration."""
    y = torch.linspace(-1.0, 1.0, 31)
    x = torch.linspace(-1.0, 1.0, 35)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    real = torch.exp(-((yy + 0.2) ** 2 + (xx - 0.1) ** 2) / 0.08) + 0.6 * torch.exp(
        -((yy - 0.25) ** 2 + 2 * (xx + 0.25) ** 2) / 0.05
    )
    imag = 0.5 * torch.exp(-((yy - 0.1) ** 2 + 1.5 * (xx - 0.2) ** 2) / 0.06)
    fixed = (real + 1j * imag)[None, None, None]
    moving = torch.zeros_like(fixed)
    moving[..., :-2, 4:] = fixed[..., 2:, :-4]

    shift = correlation_registration(fixed, moving)

    torch.testing.assert_close(shift.z, torch.zeros_like(shift.z))
    torch.testing.assert_close(shift.y, torch.full_like(shift.y, 2.0), atol=1e-2, rtol=0)
    torch.testing.assert_close(shift.x, torch.full_like(shift.x, -4.0), atol=1e-2, rtol=0)
