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
        pytest.param(
            'cuda',
            marks=(pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')),
            id='cuda',
        ),
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


@pytest.mark.parametrize('downsampling_factor', [1, 2])
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
def test_spline_registration_translation(device: str, downsampling_factor: int, blob_image: torch.Tensor) -> None:
    """A downsampled fit must recover the displacement in full-resolution voxels."""
    moving = blob_image.to(device)
    displacement = torch.tensor([3.15, -1.89], device=device)
    known_operator = GridSamplingOp.from_displacement(
        None,
        displacement[0].expand(1, 64, 64),
        displacement[1].expand(1, 64, 64),
        padding_mode='border',
    )
    (fixed,) = known_operator(moving)

    operator = spline_registration(
        fixed,
        moving,
        downsampling_factor=downsampling_factor,
        window_size=21,
        control_point_spacing=SpatialDimension(16.0, 16.0, 16.0),
        regularization_weight=0.001,
        max_iterations=80,
    )
    (moved,) = operator(moving)

    recovered_displacement = operator.to_displacement()[0, :, 16:48, 16:48].mean(dim=(-2, -1))
    torch.testing.assert_close(recovered_displacement, displacement, atol=0.3, rtol=0)
    assert (moved - fixed).abs().mean() < 0.1 * (moving - fixed).abs().mean()
