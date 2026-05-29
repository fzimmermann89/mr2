"""Tests for spline image registration."""

import pytest
import torch
from mr2.algorithms.image_registration.spline_registration import spline_registration
from mr2.data import SpatialDimension
from mr2.operators import GridSamplingOp
from mr2.operators.functionals.NCC import ncc3d
from mr2.utils import RandomGenerator


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
    ).to(device=fixed.device)
    control_points_x = rng.float32_tensor(
        (1, (image_dimensions.y - 1) // 32 + 4, (image_dimensions.x - 1) // 32 + 4), low=-10, high=10
    ).to(device=fixed.device)
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
