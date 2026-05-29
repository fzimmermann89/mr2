"""Affine image registration."""

import torch

from mr2.algorithms.optimizers.lbfgs import lbfgs
from mr2.data.SpatialDimension import SpatialDimension
from mr2.operators.functionals.L2NormSquared import L2NormSquared
from mr2.operators.functionals.NCC import NCC
from mr2.operators.GridSamplingOp import GridSamplingOp
from mr2.utils.interpolate import interpolate


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
