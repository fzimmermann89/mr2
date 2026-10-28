"""Tests for multi-level image registration."""

import pytest
import torch
from mr2.algorithms.image_registration.register_images import register_images
from mr2.data import SpatialDimension
from mr2.operators import GridSamplingOp
from mr2.utils import RandomGenerator
from tests import relative_image_difference


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
        (2, 1, (image_dimensions.y - 1) // 8 + 4, (image_dimensions.x - 1) // 8 + 4), low=-0.1, high=0.1
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


@pytest.mark.parametrize(
    'device',
    [
        'cpu',
        pytest.param(
            'cuda',
            marks=(pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')),
        ),
    ],
)
def test_register_images_nonuniform_displacement(device: str, blob_image: torch.Tensor) -> None:
    """The spline pyramid must recover a deformation that affine registration cannot fit."""
    moving = blob_image.to(device)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, 64, device=device), torch.linspace(-1, 1, 64, device=device), indexing='ij'
    )
    displacement_y = 3.0 * torch.sin(torch.pi * x) * torch.cos(torch.pi * y / 2)
    displacement_x = 2.0 * torch.sin(torch.pi * y) * torch.cos(torch.pi * x / 2)
    known_operator = GridSamplingOp.from_displacement(
        None, displacement_y[None], displacement_x[None], padding_mode='border'
    )
    (fixed,) = known_operator(moving)

    operator = register_images(
        fixed,
        moving,
        spline_downsampling_factors=(4, 2),
        spline_window_sizes=(21, 21),
        spline_control_point_spacings=(SpatialDimension(16.0, 16.0, 16.0), SpatialDimension(12.0, 12.0, 12.0)),
        spline_regularization_weights=(0.001, 0.001),
        affine_max_iterations=20,
        spline_max_iterations=80,
    )
    (moved,) = operator(moving)

    assert (moved - fixed).abs().mean() < 0.1 * (moving - fixed).abs().mean()
