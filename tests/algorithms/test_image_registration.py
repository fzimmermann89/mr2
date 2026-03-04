"""Tests for image registration."""

import pytest
import torch
from mr2.algorithms.image_registration import affine_registration, register_images, spline_registration
from mr2.data import SpatialDimension
from mr2.operators import GridSamplingOp
from mr2.operators.functionals.NCC import ncc3d
from mr2.utils import RandomGenerator
from tests import relative_image_difference


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_affine_registration(device: str, ellipse_phantom) -> None:
    """Smoke test: affine_registration runs."""
    image_dimensions = SpatialDimension(z=1, y=ellipse_phantom.n_y, x=ellipse_phantom.n_x)
    fixed = ellipse_phantom.phantom.image_space(image_dimensions).real.to(device=device)
    moving = fixed.roll(shifts=(1, -1), dims=(-2, -1))

    operator = affine_registration(
        fixed, moving, downsampling_factor=4, window_size=24, max_iterations=20, regularization_weight=0.0
    )

    assert isinstance(operator, GridSamplingOp)
    (moved,) = operator(moving)
    assert moved.shape == fixed.shape
    assert torch.isfinite(moved).all()
    assert 1 - ncc3d(moved, fixed) < (1 - ncc3d(moving, fixed)) * 0.1


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_spline_registration(device: str, ellipse_phantom) -> None:
    """Smoke test: spline_registration runs."""
    image_dimensions = SpatialDimension(z=1, y=ellipse_phantom.n_y, x=ellipse_phantom.n_x)
    fixed = ellipse_phantom.phantom.image_space(image_dimensions).real.to(device=device)
    rng = RandomGenerator(1)
    control_points_y = rng.float32_tensor(
        (1, (image_dimensions.y - 1) // 32 + 4, (image_dimensions.x - 1) // 32 + 4), low=-10, high=10
    )
    control_points_x = rng.float32_tensor(
        (1, (image_dimensions.y - 1) // 32 + 4, (image_dimensions.x - 1) // 32 + 4), low=-10, high=10
    )
    spline_operator = GridSamplingOp.from_bspline(
        None,
        control_points_y,
        control_points_x,
        input_shape=image_dimensions,
        control_point_spacing=SpatialDimension(32.0, 32.0, 32.0),
        interpolation_mode='bilinear',
        padding_mode='border',
    )
    (moving,) = spline_operator(fixed)

    operator = spline_registration(
        fixed,
        moving,
        downsampling_factor=2,
        window_size=64,
        control_point_spacing=SpatialDimension(32.0, 32.0, 32.0),
        regularization_weight=0.001,
        max_iterations=20,
    )

    assert isinstance(operator, GridSamplingOp)
    (moved,) = operator(moving)
    assert moved.shape == fixed.shape
    assert torch.isfinite(moved).all()
    assert 1 - ncc3d(moved, fixed) < 0.5 * (1 - ncc3d(moving, fixed))


def test_register_images(ellipse_phantom) -> None:
    """Test that optimization improves alignment for a synthetic affine + spline warp."""
    rng = RandomGenerator(7)
    image_dimensions = SpatialDimension(z=1, y=ellipse_phantom.n_y, x=ellipse_phantom.n_x)
    fixed = ellipse_phantom.phantom.image_space(image_dimensions).real

    affine = torch.zeros(1, 2, 3)
    affine[..., 0, 0] = rng.float32(0.9, 1.1)
    affine[..., 1, 1] = rng.float32(0.9, 1.1)
    affine[..., 0, 2] = rng.float32(-0.05, 0.05)
    affine[..., 1, 2] = rng.float32(-0.05, 0.05)
    affine_operator = GridSamplingOp.from_affine(
        affine,
        input_shape=image_dimensions,
        interpolation_mode='bilinear',
        padding_mode='border',
    )

    control_points_y, control_points_x = rng.float32_tensor(
        (2, 1, (image_dimensions.y - 1) // 8 + 4, (image_dimensions.x - 1) // 8 + 4), low=-0, high=0
    )

    spline_operator = GridSamplingOp.from_bspline(
        None,
        control_points_y,
        control_points_x,
        input_shape=image_dimensions,
        control_point_spacing=SpatialDimension(8.0, 8.0, 8.0),
        interpolation_mode='bilinear',
        padding_mode='border',
    )
    warp_operator = spline_operator @ affine_operator
    (moving,) = warp_operator(fixed)

    initial_difference = relative_image_difference(moving, fixed)
    operator = register_images(
        fixed,
        moving,
        spline_downsampling_factors=(4, 2),
        spline_window_sizes=(64, 64),
        spline_control_point_spacings=(SpatialDimension(16.0, 16.0, 16.0), SpatialDimension(8.0, 8.0, 8.0)),
        spline_regularization_weights=(0.1, 0.02),
        affine_downsampling_factor=4,
        affine_window_size=21,
        affine_max_iterations=20,
        spline_max_iterations=20,
    )
    (moved,) = operator(moving)
    recovered_difference = relative_image_difference(moved, fixed)

    assert torch.isfinite(moved).all()
    assert torch.isfinite(initial_difference)
    assert torch.isfinite(recovered_difference)
    assert recovered_difference < initial_difference
