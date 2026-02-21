"""Normalized Cross-Correlation (NCC) functional."""

from typing import Literal

import torch

from mr2.operators.Operator import Operator
from mr2.utils.reshape import unsqueeze_at
from mr2.utils.sliding_window import sliding_window


def ncc3d(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    window_size: int | None = None,
    reduction: Literal['full', 'volume', 'none'] = 'full',
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute global or local NCC between two tensors.

    Parameters
    ----------
    target
        Ground truth tensor, shape ``(..., z, y, x)`` or broadcastable with ``prediction``.
    prediction
        Predicted tensor, same shape as target.
    weight
        Optional positive weight (or mask) tensor, broadcastable with target.
        If ``window_size`` is not ``None``, the weight is masked to only include windows that are fully inside the mask.
    window_size
        If ``None``, compute global NCC per volume over the last three dimensions.
        If an integer, compute local NCC using rectangular sliding windows of this size.
        If any of the last 3 dimensions has size 1, the corresponding window size is set to 1.
    reduction
        If ``full``, return scalar mean over volumes.
        If ``volume``, return one value per volume.
        If ``none``, return the local NCC map if ``window_size`` is not ``None``; otherwise equal to ``volume``.
    eps
        Small constant for numerical stability in denominators.
    """
    if target.is_complex() or prediction.is_complex():
        real_ncc = ncc3d(
            target.real, prediction.real, weight=weight, window_size=window_size, reduction=reduction, eps=eps
        )
        imag_ncc = ncc3d(
            target.imag if target.is_complex() else torch.zeros_like(target),
            prediction.imag if prediction.is_complex() else torch.zeros_like(prediction),
            weight=weight,
            window_size=window_size,
            reduction=reduction,
            eps=eps,
        )
        return (real_ncc + imag_ncc) / 2

    if target.ndim < 3:
        raise ValueError('Input must be at least 3D (z, y, x)')
    if window_size is not None and window_size <= 0:
        raise ValueError('window_size must be positive or None')
    if eps <= 0:
        raise ValueError('eps must be positive')

    if reduction not in ('full', 'volume', 'none'):
        raise ValueError("reduction must be one of {'full', 'volume', 'none'}")

    if weight is not None and (weight < 0).any():
        raise ValueError('weight contains negative values')

    if weight is not None:
        target, prediction, weight = torch.broadcast_tensors(target, prediction, weight)
    else:
        target, prediction = torch.broadcast_tensors(target, prediction)

    dims = (-3, -2, -1)

    def window(tensor: torch.Tensor) -> torch.Tensor:
        if window_size is None:
            return unsqueeze_at(tensor, dim=-len(dims), n=len(dims))
        else:
            shape = tuple(window_size if s > 1 else 1 for s in target.shape[-3:])
            w = sliding_window(tensor, window_shape=shape, dim=dims)
            return w.movedim(tuple(range(len(dims))), tuple(range(-2 * len(dims), -len(dims))))

    target_window = window(target)
    prediction_window = window(prediction)

    if weight is None:
        weight_window = target_window.new_ones(()).expand_as(target_window)
    else:
        weight_window = window(weight.to(dtype=torch.float32))
        if window_size is not None:
            weight_window = weight_window * (weight_window > 0).all(dim=dims, keepdim=True)

    w_sum = weight_window.sum(dim=dims, keepdim=True).clamp_min(eps)
    mean_tgt = (weight_window * target_window).sum(dim=dims, keepdim=True) / w_sum
    mean_pred = (weight_window * prediction_window).sum(dim=dims, keepdim=True) / w_sum

    tgt_centered = target_window - mean_tgt
    pred_centered = prediction_window - mean_pred

    w_sum_squeezed = w_sum.squeeze(dims)
    cov = (weight_window * tgt_centered * pred_centered).sum(dim=dims) / w_sum_squeezed
    var_tgt = (weight_window * tgt_centered.square()).sum(dim=dims) / w_sum_squeezed
    var_pred = (weight_window * pred_centered.square()).sum(dim=dims) / w_sum_squeezed

    ncc_map = cov / (torch.sqrt(var_tgt * var_pred) + eps)

    if reduction == 'none':
        if window_size is not None:
            return ncc_map
        return ncc_map.mean(dim=dims)

    window_weight = weight_window.mean(dim=dims)
    window_weight = window_weight / window_weight.sum(dim=dims, keepdim=True).clamp_min(eps)
    ncc_volume = (ncc_map * window_weight).sum(dim=dims)
    if reduction == 'full':
        return ncc_volume.mean()
    return ncc_volume


class NCC(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """(masked) global or local normalized cross-correlation functional."""

    def __init__(
        self,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        *,
        window_size: int | None = None,
        reduction: Literal['full', 'volume', 'none'] = 'full',
        eps: float = 1e-12,
    ) -> None:
        """Initialize NCC.

        Parameters
        ----------
        target
            Target volume. At least 3D in the trailing dimensions.
        weight
            Optional positive weight (or boolean mask), broadcastable with target.
            If ``window_size`` is not ``None``, the weight is masked to only include windows that are fully inside the mask.
        window_size
            If ``None``, compute global NCC over the last three dimensions.
            If integer, compute local NCC with rectangular sliding windows.
        reduction
            If ``full``, return scalar mean over volumes.
            If ``volume``, return one value per volume.
            If ``none``, return local NCC map for local mode or per-volume values for global mode.
        eps
            Small positive constant for numerical stability.
        """
        super().__init__()
        self.target = target
        self.weight = weight
        self.window_size = window_size
        self.reduction = reduction
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Calculate NCC between input and target."""
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward of NCC.

        .. note::
            Prefer calling the instance as ``operator(x)`` over directly calling this method.
        """
        ncc = ncc3d(
            self.target,
            x,
            weight=self.weight,
            window_size=self.window_size,
            reduction=self.reduction,
            eps=self.eps,
        )
        return (ncc,)
