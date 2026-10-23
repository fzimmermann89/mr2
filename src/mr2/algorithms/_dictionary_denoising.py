"""Validation shared by analysis and synthesis denoising."""

import math
from copy import deepcopy

import torch

from mr2.operators import ConvAnalysisDictionaryOp, ConvSynthesisDictionaryOp, LinearOperator


def _validate_controls(max_iterations: int, tolerance: float) -> None:
    if max_iterations < 1:
        raise ValueError('max_iterations must be positive')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')


def _prepare(image: torch.Tensor, *operators: LinearOperator) -> tuple[torch.Tensor, tuple[LinearOperator, ...]]:
    """Promote floating tensors, copying operators only when conversion is needed."""
    if not (image.is_floating_point() or image.is_complex()) or image.numel() == 0:
        raise ValueError('image must be a nonempty floating point or complex tensor')
    dtype = image.dtype
    for operator in operators:
        for module in operator.modules():
            if isinstance(module, ConvAnalysisDictionaryOp | ConvSynthesisDictionaryOp) and any(
                size % 2 == 0 for size in module.kernel.shape[1:]
            ):
                raise ValueError('Convolutional denoising requires odd spatial kernel sizes')
        for tensor in (*operator.buffers(), *operator.parameters()):
            if tensor.is_floating_point() or tensor.is_complex():
                dtype = torch.promote_types(dtype, tensor.dtype)
    prepared = []
    for operator in operators:
        if any(
            tensor.device != image.device
            or ((tensor.is_floating_point() or tensor.is_complex()) and tensor.dtype != dtype)
            for tensor in (*operator.buffers(), *operator.parameters())
        ):
            operator = deepcopy(operator)
            operator._apply(
                lambda tensor: tensor.to(
                    device=image.device,
                    dtype=dtype if tensor.is_floating_point() or tensor.is_complex() else tensor.dtype,
                )
            )
        prepared.append(operator)
    return image.to(dtype=dtype), tuple(prepared)


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
