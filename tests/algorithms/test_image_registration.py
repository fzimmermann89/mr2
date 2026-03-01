"""Tests for image registration."""

import pytest
import torch
from mr2.algorithms.image_registration import affine_registration, register_images, spline_registration
from mr2.data import SpatialDimension
from mr2.operators import GridSamplingOp
from mr2.utils import RandomGenerator
from tests import relative_image_difference


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_register_images_runs(device: str) -> None:
    """Smoke test: register_images runs."""
    rng = RandomGenerator(0)
    fixed = rng.float32_tensor((1, 1, 1, 16, 16), low=0.0, high=1.0).to(device=device)
    moving = fixed.clone()

    operator = register_images(
        fixed,
        moving,
        affine_max_iterations=0,
        spline_max_iterations=0,
    )

    assert isinstance(operator, GridSamplingOp)
    assert operator.grid.device.type == device
    (moved,) = operator(moving)
    assert moved.shape == fixed.shape
    assert moved.device.type == device


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_affine_registration_runs(device: str) -> None:
    """Smoke test: affine_registration runs."""
    rng = RandomGenerator(1)
    fixed = rng.float32_tensor((1, 1, 1, 24, 24), low=0.0, high=1.0).to(device=device)
    moving = fixed.roll(shifts=(1, -1), dims=(-2, -1))

    operator = affine_registration(
        fixed,
        moving,
        downsampling_factor=1,
        window_size=9,
        max_iterations=0,
    )

    assert isinstance(operator, GridSamplingOp)
    (moved,) = operator(moving)
    assert moved.shape == fixed.shape
    assert torch.isfinite(moved).all()


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_spline_registration_runs(device: str) -> None:
    """Smoke test: spline_registration runs."""
    rng = RandomGenerator(2)
    fixed = rng.float32_tensor((1, 1, 1, 24, 24), low=0.0, high=1.0).to(device=device)
    moving = fixed.roll(shifts=(1, 1), dims=(-2, -1))

    operator = spline_registration(
        fixed,
        moving,
        downsampling_factor=1,
        window_size=9,
        control_point_spacing=SpatialDimension(8.0, 8.0, 8.0),
        regularization_weight=0.1,
        max_iterations=0,
    )

    assert isinstance(operator, GridSamplingOp)
    (moved,) = operator(moving)
    assert moved.shape == fixed.shape
    assert torch.isfinite(moved).all()


def test_register_images_runs_with_phantom_warp(ellipse_phantom) -> None:
    """Smoke test: register_images runs on a warped ellipse phantom."""
    rng = RandomGenerator(0)
    image_dimensions = SpatialDimension(z=1, y=ellipse_phantom.n_y, x=ellipse_phantom.n_x)
    fixed = ellipse_phantom.phantom.image_space(image_dimensions).real

    affine = torch.zeros(1, 3, 4)
    affine[..., 0, 0] = 1
    affine[..., 1, 1] = 1
    affine[..., 2, 2] = 1
    affine[..., 1, 3] = rng.float32(-1e-3, 1e-3)
    affine[..., 2, 3] = rng.float32(-1e-3, 1e-3)
    affine_operator = GridSamplingOp.from_affine(
        affine,
        input_shape=image_dimensions,
        interpolation_mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )

    control_points_z = torch.zeros((1, 4, 8, 8))
    control_points_y = rng.float32_tensor((1, 4, 8, 8), low=-1e-6, high=1e-6)
    control_points_x = rng.float32_tensor((1, 4, 8, 8), low=-1e-6, high=1e-6)
    spline_operator = GridSamplingOp.from_bspline(
        control_points_z,
        control_points_y,
        control_points_x,
        input_shape=image_dimensions,
        control_point_spacing=SpatialDimension(8.0, 8.0, 8.0),
        interpolation_mode='bilinear',
        padding_mode='border',
    )
    warp_operator = spline_operator @ affine_operator
    (moving,) = warp_operator(fixed)

    operator = register_images(
        fixed,
        moving,
        affine_max_iterations=0,
        spline_max_iterations=0,
    )
    (recovered,) = operator(moving)

    recovered_difference = relative_image_difference(recovered, fixed)
    assert torch.isfinite(recovered).all()
    assert torch.isfinite(recovered_difference)
    assert recovered_difference < 0.2


def test_register_images_recovers_synthetic_affine_bspline_warp(ellipse_phantom) -> None:
    """Test that optimization improves alignment for a synthetic affine + spline warp."""
    rng = RandomGenerator(7)
    image_dimensions = SpatialDimension(z=1, y=ellipse_phantom.n_y, x=ellipse_phantom.n_x)
    fixed = ellipse_phantom.phantom.image_space(image_dimensions).real

    affine = torch.zeros(1, 3, 4)
    affine[..., 0, 0] = 1.0
    affine[..., 1, 1] = 1.0
    affine[..., 2, 2] = 1.0
    affine[..., 1, 3] = rng.float32(-2e-2, 2e-2)
    affine[..., 2, 3] = rng.float32(-2e-2, 2e-2)
    affine_operator = GridSamplingOp.from_affine(
        affine,
        input_shape=image_dimensions,
        interpolation_mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )

    control_points_z = torch.zeros((1, 4, 8, 8))
    control_points_y = rng.float32_tensor((1, 4, 8, 8), low=-3e-3, high=3e-3)
    control_points_x = rng.float32_tensor((1, 4, 8, 8), low=-3e-3, high=3e-3)
    spline_operator = GridSamplingOp.from_bspline(
        control_points_z,
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
        spline_downsampling_factors=(2, 1),
        spline_window_sizes=(11, 9),
        spline_control_point_spacings=(SpatialDimension(12.0, 12.0, 12.0), SpatialDimension(8.0, 8.0, 8.0)),
        spline_regularization_weights=(0.1, 0.02),
        affine_downsampling_factor=1,
        affine_window_size=11,
        affine_max_iterations=30,
        spline_max_iterations=40,
    )
    (recovered,) = operator(moving)
    recovered_difference = relative_image_difference(recovered, fixed)

    assert torch.isfinite(recovered).all()
    assert torch.isfinite(initial_difference)
    assert torch.isfinite(recovered_difference)
    assert recovered_difference < initial_difference
