"""Multi-level image registration."""

from collections.abc import Sequence
from typing import cast

import torch

from mr2.algorithms.image_registration.affine_registration import affine_registration
from mr2.algorithms.image_registration.spline_registration import spline_registration
from mr2.data.SpatialDimension import SpatialDimension
from mr2.operators.GridSamplingOp import GridSamplingOp


def register_images(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    spline_downsampling_factors: Sequence[int] = (8, 4, 2),
    spline_window_sizes: Sequence[int] = (21, 15, 11),
    spline_control_point_spacings: Sequence[SpatialDimension[float]] = (
        SpatialDimension(24.0, 24.0, 24.0),
        SpatialDimension(16.0, 16.0, 16.0),
        SpatialDimension(12.0, 12.0, 12.0),
    ),
    spline_regularization_weights: Sequence[float] = (0.2, 0.1, 0.05),
    weight: torch.Tensor | None = None,
    affine_downsampling_factor: int = 4,
    affine_window_size: int = 11,
    affine_max_iterations: int = 20,
    affine_regularization_weight: float = 1e-4,
    spline_max_iterations: int = 40,
) -> GridSamplingOp:
    """Run affine registration followed by a spline pyramid.

    Parameters
    ----------
    fixed
        Fixed image.
    moving
        Moving image.
    spline_downsampling_factors
        Downsampling factors for the spline pyramid.
    spline_window_sizes
        Window sizes for the NCC similarity measure in the spline pyramid.
    spline_control_point_spacings
        Spacings of the control points in the spline pyramid.
    spline_regularization_weights
        Regularization weights for the L2 regularization of the control points in the spline pyramid.
    weight
        Weight/mask tensor. If None, no weighting is applied.
    affine_downsampling_factor
        Downsampling factor for the affine registration.
    affine_window_size
        Window size for the NCC similarity measure in the affine registration.
    affine_max_iterations
        Maximum number of iterations of the LBFGS optimizer for the affine registration.
    affine_regularization_weight
        Regularization weight for the L2 regularization of the affine matrix.
    spline_max_iterations
        Maximum number of iterations of the LBFGS optimizer for the spline pyramid.
    """
    number_of_levels = len(spline_downsampling_factors)
    if not (
        len(spline_window_sizes)
        == len(spline_control_point_spacings)
        == len(spline_regularization_weights)
        == number_of_levels
    ):
        raise ValueError('Spline level arguments must all have the same length.')

    operator = affine_registration(
        fixed,
        moving,
        weight=weight,
        downsampling_factor=affine_downsampling_factor,
        window_size=affine_window_size,
        max_iterations=affine_max_iterations,
        regularization_weight=affine_regularization_weight,
    )

    (moved,) = operator(moving)
    for (
        downsampling_factor,
        window_size,
        control_point_spacing,
        regularization_weight,
    ) in zip(
        spline_downsampling_factors,
        spline_window_sizes,
        spline_control_point_spacings,
        spline_regularization_weights,
        strict=True,
    ):
        spline_operator = spline_registration(
            fixed,
            moved,
            downsampling_factor=downsampling_factor,
            window_size=window_size,
            control_point_spacing=control_point_spacing,
            regularization_weight=regularization_weight,
            weight=weight,
            max_iterations=spline_max_iterations,
        )
        operator = cast(GridSamplingOp, spline_operator @ operator)
        (moved,) = spline_operator(moved)

    return operator
