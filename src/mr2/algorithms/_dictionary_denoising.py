"""Validation shared by analysis and synthesis denoising."""

import math

import torch


def _validate_controls(max_iterations: int, tolerance: float) -> None:
    if max_iterations < 1:
        raise ValueError('max_iterations must be positive')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')


def _image(image: torch.Tensor) -> torch.Tensor:
    """Validate the image without converting caller-owned tensors or operators."""
    if not (image.is_floating_point() or image.is_complex()) or image.numel() == 0:
        raise ValueError('image must be a nonempty floating point or complex tensor')
    return image


def _weight(weight: float | torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    weight = torch.as_tensor(weight, device=coefficients.device)
    if weight.is_complex() or not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('regularization_weight must be real, finite and nonnegative')
    try:
        shape = torch.broadcast_shapes(weight.shape, coefficients.shape)
    except RuntimeError as error:
        raise ValueError('regularization_weight must broadcast to the coefficient shape') from error
    if shape != coefficients.shape:
        raise ValueError('regularization_weight must not expand the coefficient shape')
    weight = weight.to(dtype=coefficients.real.dtype)
    if not torch.isfinite(weight).all():
        raise ValueError('regularization_weight must be finite in the coefficient dtype')
    return weight


def _initial(initial: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if initial is None:
        return reference
    if initial.shape != reference.shape:
        raise ValueError('initial value must have exactly the expected shape')
    if initial.is_complex() and not reference.is_complex():
        raise ValueError('initial value must not introduce a complex dtype')
    return initial.to(reference)
