"""Image registration."""

from collections.abc import Sequence
from math import ceil
from typing import cast

import torch

from mr2.algorithms.optimizers.lbfgs import lbfgs
from mr2.data.SpatialDimension import SpatialDimension
from mr2.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mr2.operators.functionals.L2NormSquared import L2NormSquared
from mr2.operators.functionals.NCC import NCC, ncc3d
from mr2.operators.GridSamplingOp import GridSamplingOp
from mr2.utils.interpolate import interpolate


def affine_registration(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    downsampling_factor: int = 1,
    window_size: int = 9,
    learning_rate: float = 1.0,
    max_iterations: int = 80,
    l2_weight: float = 1e-4,
    align_corners: bool = True,
    initial_affine: torch.Tensor | None = None,
) -> GridSamplingOp:
    """Run one affine registration level."""
    if fixed.shape != moving.shape:
        raise ValueError(f'fixed and moving must have same shape, got {fixed.shape=} and {moving.shape=}.')
    if fixed.ndim < 5:
        raise ValueError(f'Expected at least 5 dimensions ``(*batch, channels, z, y, x)``, got {fixed.ndim}.')
    if downsampling_factor <= 0:
        raise ValueError(f'downsampling_factor must be positive, got {downsampling_factor}.')

    batch_shape = fixed.shape[:-4]
    fixed_flat = fixed.flatten(end_dim=-5)
    moving_flat = moving.flatten(end_dim=-5)
    weight_flat = None
    if weight is not None:
        weight_flat = weight.flatten(end_dim=-5)

    downsampled_shape = tuple(max(1, int(size // downsampling_factor)) for size in fixed_flat.shape[-3:])
    non_unit_shape = tuple(size for size in downsampled_shape if size > 1)
    window_size_level = min(window_size, *non_unit_shape) if len(non_unit_shape) > 0 else 1
    fixed_level = interpolate(fixed_flat, downsampled_shape, dim=(-3, -2, -1), mode='area')
    moving_level = interpolate(moving_flat, downsampled_shape, dim=(-3, -2, -1), mode='area')
    weight_level = (
        None
        if weight_flat is None
        else interpolate(
            weight_flat,
            downsampled_shape,
            dim=(-3, -2, -1),
            mode='area',
        )
    )

    if initial_affine is None:
        affine = torch.zeros(
            (fixed_level.shape[0], 3, 4),
            device=fixed.device,
            dtype=fixed.real.dtype,
        )
        affine[:, :3, :3] = torch.eye(3, device=fixed.device, dtype=fixed.real.dtype)
    else:
        affine = initial_affine.flatten(end_dim=-3)

    affine_identity = torch.zeros_like(affine)
    affine_identity[:, :3, :3] = torch.eye(3, device=affine.device, dtype=affine.dtype)

    def objective(*parameters: torch.Tensor) -> tuple[torch.Tensor]:
        (affine_parameters,) = parameters
        sampling_operator = GridSamplingOp.from_affine(
            affine_parameters,
            input_shape=SpatialDimension(*moving_level.shape[-3:]),
            interpolation_mode='bilinear',
            padding_mode='border',
            align_corners=align_corners,
        )
        (moved,) = sampling_operator(moving_level)
        similarity = ncc3d(fixed_level, moved, weight=weight_level, window_size=window_size_level, reduction='full')
        regularization = (affine_parameters - affine_identity).square().mean()
        loss = -similarity + l2_weight * regularization
        if not torch.isfinite(loss):
            raise RuntimeError(
                'Non-finite affine objective encountered. '
                f'got similarity={similarity.detach().item()}, regularization={regularization.detach().item()}, '
                f'loss={loss.detach().item()}.'
            )
        return (loss,)

    (affine,) = lbfgs(
        objective,
        (affine,),
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        line_search_fn='strong_wolfe',
    )
    affine = affine.unflatten(0, batch_shape)
    return GridSamplingOp.from_affine(
        affine,
        input_shape=SpatialDimension(*moving.shape[-3:]),
        interpolation_mode='bilinear',
        padding_mode='border',
        align_corners=align_corners,
    )


def spline_registration(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    downsampling_factor: int,
    window_size: int,
    control_point_spacing: SpatialDimension[float],
    regularization_weight: float,
    weight: torch.Tensor | None = None,
    learning_rate: float = 1.0,
    max_iterations: int = 120,
) -> GridSamplingOp:
    """Run one spline registration level."""
    if fixed.shape != moving.shape:
        raise ValueError(f'fixed and moving must have same shape, got {fixed.shape=} and {moving.shape=}.')
    if fixed.ndim < 5:
        raise ValueError(f'Expected at least 5 dimensions ``(*batch, channels, z, y, x)``, got {fixed.ndim}.')
    if downsampling_factor <= 0:
        raise ValueError(f'downsampling_factor must be positive, got {downsampling_factor}.')

    batch_shape = fixed.shape[:-4]
    fixed_flat = fixed.flatten(end_dim=-5)
    moving_flat = moving.flatten(end_dim=-5)
    weight_flat = None
    if weight is not None:
        weight_flat = weight.flatten(end_dim=-5)

    downsampled_shape = tuple(max(1, int(size // downsampling_factor)) for size in fixed_flat.shape[-3:])
    non_unit_shape = tuple(size for size in downsampled_shape if size > 1)
    window_size_level = min(window_size, *non_unit_shape) if len(non_unit_shape) > 0 else 1
    fixed_level = interpolate(fixed_flat, downsampled_shape, dim=(-3, -2, -1), mode='area')
    moving_level = interpolate(moving_flat, downsampled_shape, dim=(-3, -2, -1), mode='area')
    weight_level = (
        None
        if weight_flat is None
        else interpolate(
            weight_flat,
            downsampled_shape,
            dim=(-3, -2, -1),
            mode='area',
        )
    )

    full_resolution_spacing = control_point_spacing.zyx
    level_spacing = tuple(float(spacing / downsampling_factor) for spacing in full_resolution_spacing)
    control_grid_shape = tuple(
        ceil(size / spacing) + 3 for size, spacing in zip(downsampled_shape, level_spacing, strict=True)
    )

    control_points = torch.zeros(
        (fixed_level.shape[0], 3, *control_grid_shape),
        device=fixed.device,
        dtype=fixed.real.dtype,
    )

    laplace_penalty = L2NormSquared(divide_by_n=True, weight=regularization_weight) @ FiniteDifferenceOp(
        dim=(-3, -2, -1), mode='laplacian', pad_mode='zeros'
    )
    ncc = NCC(target=fixed_level, weight=weight_level, window_size=window_size_level, reduction='full')

    def objective(*parameters: torch.Tensor) -> tuple[torch.Tensor]:
        (control_points_parameters,) = parameters
        control_points_components = control_points_parameters.movedim(-4, 0)
        sampling_operator, displacement = GridSamplingOp.from_bspline(
            control_points_components[0],
            control_points_components[1],
            control_points_components[2],
            input_shape=SpatialDimension(*downsampled_shape),
            control_point_spacing=SpatialDimension(*level_spacing),
            interpolation_mode='bilinear',
            padding_mode='border',
            return_displacement=True,
        )
        (similarity,) = (ncc @ sampling_operator)(moving_level)
        (regularization,) = laplace_penalty(displacement)
        loss = -similarity + regularization
        if not torch.isfinite(loss):
            raise RuntimeError(
                'Non-finite spline objective encountered. '
                f'got similarity={similarity.detach().item()}, regularization={regularization.detach().item()}, '
                f'loss={loss.detach().item()}.'
            )
        return (loss,)

    (control_points,) = lbfgs(
        objective,
        (control_points,),
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        line_search_fn='strong_wolfe',
    )
    control_points = control_points.unflatten(0, batch_shape)
    control_points_components = control_points.movedim(-4, 0)
    sampling_operator = GridSamplingOp.from_bspline(
        control_points_components[0],
        control_points_components[1],
        control_points_components[2],
        input_shape=SpatialDimension(*fixed.shape[-3:]),
        control_point_spacing=SpatialDimension(*full_resolution_spacing),
        interpolation_mode='bilinear',
        padding_mode='border',
    )
    return sampling_operator


def register_images(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    spline_downsampling_factors: Sequence[int] = (8, 4, 2, 1),
    spline_window_sizes: Sequence[int] = (21, 15, 11, 9),
    spline_control_point_spacings: Sequence[SpatialDimension[float]] = (
        SpatialDimension(24.0, 24.0, 24.0),
        SpatialDimension(16.0, 16.0, 16.0),
        SpatialDimension(12.0, 12.0, 12.0),
        SpatialDimension(8.0, 8.0, 8.0),
    ),
    spline_regularization_weights: Sequence[float] = (0.2, 0.1, 0.05, 0.02),
    weight: torch.Tensor | None = None,
    affine_downsampling_factor: int = 2,
    affine_window_size: int = 11,
    affine_learning_rate: float = 1.0,
    affine_max_iterations: int = 80,
    affine_l2_weight: float = 1e-4,
    spline_learning_rate: float = 1.0,
    spline_max_iterations: int = 120,
    align_corners: bool = True,
) -> GridSamplingOp:
    """Run affine registration followed by a spline pyramid."""
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
        learning_rate=affine_learning_rate,
        max_iterations=affine_max_iterations,
        l2_weight=affine_l2_weight,
        align_corners=align_corners,
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
            learning_rate=spline_learning_rate,
            max_iterations=spline_max_iterations,
        )
        operator = cast(GridSamplingOp, spline_operator @ operator)
        (moved,) = spline_operator(moved)

    return operator
