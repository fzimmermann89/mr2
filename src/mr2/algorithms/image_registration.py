"""Image registration."""

from collections.abc import Sequence
from typing import cast

import torch

from mr2.algorithms.optimizers.lbfgs import lbfgs
from mr2.data.SpatialDimension import SpatialDimension
from mr2.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mr2.operators.functionals.L2NormSquared import L2NormSquared
from mr2.operators.functionals.NCC import NCC
from mr2.operators.GridSamplingOp import GridSamplingOp
from mr2.utils.filters import gaussian_filter
from mr2.utils.interpolate import interpolate


def correlation_registration(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    smooth_mask_sigma: float = 1.5,
) -> SpatialDimension[torch.Tensor]:
    """Estimate a 2D or 3D shift between fixed and moving images using FFT-based normalized cross-correlation.

    Parameters
    ----------
    fixed
        Fixed image with shape ``(*other, coils, z, y, x)``.
    moving
        Moving image with shape ``(*other, coils, z, y, x)``.
        Complex-valued inputs are registered phase-sensitively; pass magnitude images to ignore phase.
    mask
        Optional mask/weight tensor. Outside values are suppressed in both fixed and moving.
        The mask is smoothed before it is applied to avoid correlation peaks at hard mask edges.
    smooth_mask_sigma
        Standard deviation of the Gaussian smoothing applied to the mask in voxel units.

    Returns
    -------
        Estimated voxel shift to apply to ``moving`` to align it to ``fixed``.
        For 2D inputs (``z == 1``), the returned z-shift is zero.
    """
    eps = 1e-8
    if fixed.shape != moving.shape:
        raise ValueError(f'fixed and moving must have same shape, got {fixed.shape=} and {moving.shape=}.')
    if fixed.ndim < 5:
        raise ValueError(f'Expected at least 5 dimensions ``(*batch, channels, z, y, x)``, got {fixed.ndim}.')
    if smooth_mask_sigma <= 0:
        raise ValueError(f'smooth_mask_sigma must be positive, got {smooth_mask_sigma}.')

    dim = 2 if fixed.shape[-3] == 1 else 3
    spatial_dims = tuple(range(-dim, 0))
    batch_shape = fixed.shape[:-4]

    fixed_flat = fixed.flatten(end_dim=-5)
    moving_flat = moving.flatten(end_dim=-5)
    spatial_shape = fixed_flat.shape[-dim:]
    padded_shape = tuple(2 * size for size in spatial_shape)
    normalization_dims = tuple(range(1, fixed_flat.ndim))

    if mask is None:
        mask_flat = torch.ones_like(fixed_flat[:, :1], dtype=torch.float32)
    else:
        mask_flat = mask.flatten(end_dim=-5)
        if mask_flat.shape[-3:] != fixed_flat.shape[-3:]:
            raise ValueError(
                'mask must have the same spatial shape as fixed and moving, got '
                f'{mask.shape[-3:]=} and {fixed.shape[-3:]=}.'
            )
        mask_flat = torch.broadcast_to(mask_flat, fixed_flat[:, :1].shape).to(dtype=torch.float32)
        mask_flat = gaussian_filter(mask_flat, smooth_mask_sigma, dim=spatial_dims)
        mask_flat = mask_flat / mask_flat.amax(dim=spatial_dims, keepdim=True).clamp_min(eps)
        mask_flat = mask_flat.clamp_(0.0, 1.0)

    mask_per_channel = mask_flat.expand_as(fixed_flat)
    weight_sum = (2.0 * mask_per_channel.sum(dim=normalization_dims, keepdim=True)).clamp_min(eps)
    joint_mean = (
        (fixed_flat * mask_per_channel).sum(dim=normalization_dims, keepdim=True)
        + (moving_flat * mask_per_channel).sum(dim=normalization_dims, keepdim=True)
    ) / weight_sum
    fixed_weighted = (fixed_flat - joint_mean) * mask_per_channel
    moving_weighted = (moving_flat - joint_mean) * mask_per_channel

    joint_scale = torch.maximum(
        fixed_weighted.abs().amax(dim=normalization_dims, keepdim=True),
        moving_weighted.abs().amax(dim=normalization_dims, keepdim=True),
    ).clamp_min(eps)
    fixed_weighted = fixed_weighted / joint_scale
    moving_weighted = moving_weighted / joint_scale

    fixed_fft = torch.fft.fftn(fixed_weighted, s=padded_shape, dim=spatial_dims)
    moving_fft = torch.fft.fftn(moving_weighted, s=padded_shape, dim=spatial_dims)
    mask_fft = torch.fft.fftn(mask_flat, s=padded_shape, dim=spatial_dims)

    numerator = torch.fft.ifftn((fixed_fft * moving_fft.conj()).sum(dim=1, keepdim=True), dim=spatial_dims).real

    fixed_energy_fft = torch.fft.fftn(
        fixed_weighted.abs().square().sum(dim=1, keepdim=True), s=padded_shape, dim=spatial_dims
    )
    moving_energy_fft = torch.fft.fftn(
        moving_weighted.abs().square().sum(dim=1, keepdim=True), s=padded_shape, dim=spatial_dims
    )
    fixed_energy = torch.fft.ifftn(fixed_energy_fft * mask_fft.conj(), dim=spatial_dims).real
    moving_energy = torch.fft.ifftn(mask_fft * moving_energy_fft.conj(), dim=spatial_dims).real
    correlation = numerator / (fixed_energy.clamp_min(0.0) * moving_energy.clamp_min(0.0)).sqrt().clamp_min(eps)
    overlap = torch.fft.ifftn(mask_fft * mask_fft.conj(), dim=spatial_dims).real
    min_overlap = 0.5 * overlap.amax(dim=spatial_dims, keepdim=True)
    correlation = torch.where(overlap >= min_overlap, correlation, -torch.inf)
    correlation = correlation.squeeze(1)
    max_index = correlation.reshape(correlation.shape[0], -1).argmax(dim=-1)
    peak_index = torch.stack(torch.unravel_index(max_index, padded_shape), dim=-1)
    peak_shift = peak_index.to(dtype=torch.float32)
    peak_shift = torch.where(
        peak_shift >= peak_shift.new_tensor(spatial_shape),
        peak_shift - peak_shift.new_tensor(padded_shape),
        peak_shift,
    )
    peak_shift = peak_shift.unflatten(0, batch_shape)

    if dim == 2:
        zero_shift = torch.zeros_like(peak_shift[..., 0])
        return SpatialDimension(z=zero_shift, y=peak_shift[..., 0], x=peak_shift[..., 1])
    return SpatialDimension(z=peak_shift[..., 0], y=peak_shift[..., 1], x=peak_shift[..., 2])


def affine_registration(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    downsampling_factor: int = 1,
    window_size: int = 9,
    max_iterations: int = 50,
    regularization_weight: float = 1e-4,
    initial_affine: torch.Tensor | None = None,
) -> GridSamplingOp:
    """Run one affine registration level.

    Parameters
    ----------
    fixed
        Fixed image.
    moving
        Moving image.
    weight
        Weight/mask tensor. If None, no weighting is applied.
    downsampling_factor
        Downsampling factor.
    window_size
        Window size for NCC similarity measure.
    max_iterations
        Maximum number of iterations of the LBFGS optimizer.
    regularization_weight
        Regularization weight for L2 regularization of the affine matrix.
    initial_affine
        Initial affine matrix. If None, the identity matrix is used.
    """
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
    if initial_affine is not None:
        initial = initial_affine.flatten(end_dim=-3)
    elif fixed.shape[-3] > 1:  # 3d
        initial = torch.zeros(
            (fixed_level.shape[0], 3, 4),
            device=fixed.device,
            dtype=fixed.real.dtype,
        )
        initial[:, :3, :3] = torch.eye(3, device=fixed.device, dtype=fixed.real.dtype)
    else:  # 2d
        initial = torch.zeros(
            (fixed_level.shape[0], 2, 3),
            device=fixed.device,
            dtype=fixed.real.dtype,
        )
        initial[:, :2, :2] = torch.eye(2, device=fixed.device, dtype=fixed.real.dtype)

    similarity = NCC(target=fixed_level, weight=weight_level, window_size=window_size_level, reduction='full')
    regularization = L2NormSquared(target=initial, divide_by_n=True, weight=regularization_weight)

    def objective(affine: torch.Tensor) -> tuple[torch.Tensor]:
        sampling_operator = GridSamplingOp.from_affine(
            affine,
            input_shape=SpatialDimension(*moving_level.shape[-3:]),
            interpolation_mode='bilinear',
            padding_mode='border',
        )
        (moved,) = sampling_operator(moving_level)
        (similarity_value,) = similarity(moved)
        (regularization_value,) = regularization(affine)
        loss = -similarity_value + regularization_value
        if not torch.isfinite(loss):
            raise RuntimeError(
                'Non-finite affine objective encountered. Got  Similarity '
                f'{similarity_value.detach().item()}, regularization {regularization_value.detach().item()}, and '
                f'loss={loss.detach().item()}.'
            )
        return (loss,)

    (initial,) = lbfgs(
        objective,  # type: ignore[arg-type]
        (initial,),
        max_iterations=max_iterations,
        max_evaluations=max(1, 2 * max_iterations),
        line_search_fn='strong_wolfe',
    )
    initial = initial.unflatten(0, batch_shape)
    return GridSamplingOp.from_affine(
        initial,
        input_shape=SpatialDimension(*moving.shape[-3:]),
        interpolation_mode='bilinear',
        padding_mode='border',
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
