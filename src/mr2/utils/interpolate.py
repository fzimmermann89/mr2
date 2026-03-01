"""Interpolation of data tensor."""

from collections.abc import Callable, Sequence
from typing import Literal

import torch

from mr2.utils.reshape import normalize_indices


def interpolate(
    x: torch.Tensor,
    size: Sequence[int],
    dim: Sequence[int],
    mode: Literal['nearest', 'linear', 'area', 'bicubic'] = 'linear',
    *,
    align_corners: bool | None = None,
) -> torch.Tensor:
    """Interpolate the tensor x along the axes dim to the new size.

    Parameters
    ----------
    x
        Tensor to interpolate
    size
        New size of the tensor
    dim
        Axes to interpolate over. Must have the same length as size.
    mode
        Interpolation mode.
    align_corners
        Forwarded to ``torch.nn.functional.interpolate`` for linear and bicubic modes.

    Returns
    -------
        The interpolated tensor, with the new size.

    """
    if len(dim) != len(size):
        raise ValueError('Must provide matching length size and dim arguments.')

    dim = normalize_indices(x.ndim, dim)

    # return input tensor if old and new size match
    if all(x.shape[d] == s for s, d in zip(size, dim, strict=True)):
        return x

    if mode == 'bicubic' and len(dim) != 2:
        raise ValueError(f"mode='bicubic' requires exactly 2 interpolation dimensions, got {len(dim)}.")
    if mode == 'area' and len(dim) not in (1, 2, 3):
        raise ValueError(f"mode='area' requires 1-3 interpolation dimensions, got {len(dim)}.")
    if align_corners is not None and mode not in ('linear', 'bicubic'):
        raise ValueError("align_corners is only supported for 'linear' and 'bicubic' modes.")
    # torch.nn.functional.interpolate only available for real tensors
    # moveaxis is not implemented for batched tensors, so vmap would fail, thus we use permute.
    x_real = torch.view_as_real(x).permute(-1, *range(x.ndim)) if x.is_complex() else x
    dim = [d + 1 for d in dim] if x.is_complex() else list(dim)

    if mode in ('nearest', 'linear', 'area', 'bicubic') and 1 <= len(dim) <= 3:
        interpolation_mode: str = mode
        if mode == 'linear':
            interpolation_mode = ('linear', 'bilinear', 'trilinear')[len(dim) - 1]
        non_spatial_axes = [axis for axis in range(x_real.ndim) if axis not in dim]
        permutation = [*non_spatial_axes, *dim]
        inverse_permutation = [0] * x_real.ndim
        for new_axis, old_axis in enumerate(permutation):
            inverse_permutation[old_axis] = new_axis
        x_permuted = x_real.permute(permutation)
        x_flat = x_permuted.reshape(-1, 1, *x_permuted.shape[-len(dim) :])
        x_interp = torch.nn.functional.interpolate(
            x_flat, size=size, mode=interpolation_mode, align_corners=align_corners
        )
        x_real = x_interp.reshape(*x_permuted.shape[: -len(dim)], *size).permute(inverse_permutation)
    elif mode in ('nearest', 'linear'):
        for s, d in zip(size, dim, strict=True):
            if s != x_real.shape[d]:
                idx = list(range(x_real.ndim))
                # swapping the last axis and the axis to filter over
                idx[d], idx[-1] = idx[-1], idx[d]
                x_real = x_real.permute(idx)
                x_real = torch.nn.functional.interpolate(
                    x_real.flatten(end_dim=-3), size=s, mode=mode, align_corners=align_corners
                ).reshape(*x_real.shape[:-1], -1)
                # for a single permutation, this undoes the permutation
                x_real = x_real.permute(idx)
    else:
        raise ValueError(f'Interpolation mode {mode} not supported.')

    return torch.view_as_complex(x_real.permute(*range(1, x.ndim + 1), 0).contiguous()) if x.is_complex() else x_real


def apply_lowres(function: Callable[[torch.Tensor], torch.Tensor], size: Sequence[int], dim: Sequence[int]) -> Callable:
    """Apply function f on low-res version of tensor x and then return the upsampled f(x).

    Parameters
    ----------
    function
        Function to be applied on low-resolution version of tensor.
    size
        Low-resolution size of tensor.
    dim
        Low-resolution axes. Must have the same length as size.

    Returns
    -------
        Function which downsamples tensor, applies function and upsamples result.
    """

    def apply_to_lowres_data(x: torch.Tensor) -> torch.Tensor:
        """Downsample tensor, apply function and upsample result.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            Tensor in original size with function applied.
        """
        x_lowres = interpolate(x, size, dim)
        x_lowres = function(x_lowres)
        return interpolate(x_lowres, [x.shape[d] for d in dim], dim)

    return apply_to_lowres_data
