"""Tests for affine image registration."""

import pytest
import torch
from mr2.algorithms.image_registration.affine_registration import affine_registration
from mr2.data import SpatialDimension
from mr2.operators import GridSamplingOp
from mr2.operators.functionals.NCC import ncc3d


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
