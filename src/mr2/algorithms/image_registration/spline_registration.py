"""Spline image registration."""

from typing import cast

import torch

from mr2.algorithms.optimizers.lbfgs import lbfgs
from mr2.data.SpatialDimension import SpatialDimension
from mr2.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mr2.operators.functionals.L2NormSquared import L2NormSquared
from mr2.operators.functionals.NCC import NCC
from mr2.operators.GridSamplingOp import GridSamplingOp
from mr2.utils.interpolate import interpolate


def spline_registration(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    downsampling_factor: int,
    window_size: int,
    control_point_spacing: SpatialDimension[float],
    regularization_weight: float,
    weight: torch.Tensor | None = None,
    max_iterations: int = 120,
) -> GridSamplingOp:
    """Run one B-spline registration level.

    Parameters
    ----------
    fixed
        Fixed image.
    moving
        Moving image.
    downsampling_factor
        Downsampling factor.
    window_size
        Window size for NCC similarity measure.
    control_point_spacing
        Spacing of the control points.
    regularization_weight
        Regularization weight for L2 regularization of the control points.
    weight
        Weight/mask tensor. If None, no weighting is applied.
    max_iterations
        Maximum number of iterations of the LBFGS optimizer.
    """
    if fixed.shape != moving.shape:
        raise ValueError(f'fixed and moving must have same shape, got {fixed.shape=} and {moving.shape=}.')
    if fixed.ndim < 5:
        raise ValueError(f'Expected at least 5 dimensions ``(*batch, channels, z, y, x)``, got {fixed.ndim}.')
    if downsampling_factor <= 0:
        raise ValueError(f'downsampling_factor must be positive, got {downsampling_factor}.')
    if fixed.shape[-3] == 1:
        dim = 2
    else:
        dim = 3

    batch_shape = fixed.shape[:-4]
    fixed_flat = fixed.flatten(end_dim=-5)
    moving_flat = moving.flatten(end_dim=-5)

    downsampled_shape = cast(
        tuple[int, int, int], tuple(max(1, int(size // downsampling_factor)) for size in fixed_flat.shape[-3:])
    )
    window_size_level = min(window_size, *[size for size in downsampled_shape if size > 1])
    fixed_level = interpolate(fixed_flat, downsampled_shape[-dim:], dim=range(-dim, 0), mode='area')
    moving_level = interpolate(moving_flat, downsampled_shape[-dim:], dim=range(-dim, 0), mode='area')
    if weight is not None:
        weight_level = interpolate(
            weight.flatten(end_dim=-5),
            downsampled_shape[-dim:],
            dim=range(-dim, 0),
            mode='area',
        )
    else:
        weight_level = None

    level_spacing = tuple(spacing / downsampling_factor for spacing in control_point_spacing.zyx)
    control_grid_shape = tuple(
        int((size - 1) // spacing) + 4
        for size, spacing in zip(downsampled_shape[-dim:], level_spacing[-dim:], strict=True)
    )

    control_points = torch.zeros(
        (dim, fixed_level.shape[0], *control_grid_shape),
        device=fixed.device,
        dtype=fixed.real.dtype,
    )

    laplace_penalty = L2NormSquared(divide_by_n=True, weight=regularization_weight) @ FiniteDifferenceOp(
        dim=tuple(range(-dim, 0)), mode='second_difference', pad_mode='zeros'
    )
    similarity = NCC(target=fixed_level, weight=weight_level, window_size=window_size_level, reduction='full')

    def objective(control_points: torch.Tensor) -> tuple[torch.Tensor]:
        sampling_operator, displacement = GridSamplingOp.from_bspline(
            control_points[0] if dim == 3 else None,
            control_points[-2],
            control_points[-1],
            input_shape=SpatialDimension(*downsampled_shape),
            control_point_spacing=SpatialDimension(*level_spacing),
            interpolation_mode='bilinear',
            padding_mode='border',
            return_displacement=True,
        )
        (similarity_value,) = (similarity @ sampling_operator)(moving_level)
        (regularization_value,) = laplace_penalty(displacement)
        loss = -similarity_value + regularization_value
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite spline objective encountered. ')
        return (loss,)

    (control_points,) = lbfgs(
        objective,  # type: ignore[arg-type]
        (control_points,),
        max_iterations=max_iterations,
        max_evaluations=max(1, 2 * max_iterations),
        line_search_fn='strong_wolfe',
    )
    control_points = control_points.unflatten(1, batch_shape)
    sampling_operator = GridSamplingOp.from_bspline(
        control_points[0] if dim == 3 else None,
        control_points[-2],
        control_points[-1],
        input_shape=SpatialDimension(*fixed.shape[-3:]),
        control_point_spacing=control_point_spacing,
        interpolation_mode='bilinear',
        padding_mode='border',
    )
    return sampling_operator
