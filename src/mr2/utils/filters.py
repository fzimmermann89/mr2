"""Spatial and temporal filters."""

import math
import warnings
from collections.abc import Sequence
from functools import reduce
from math import ceil
from typing import Literal

import numpy as np
import torch
from einops import repeat

from mr2.utils.reshape import normalize_indices


def filter_separable(
    x: torch.Tensor,
    kernels: Sequence[torch.Tensor],
    dim: Sequence[int],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular', 'none'] = 'constant',
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Apply the separable filter kernels to the tensor x along the axes dim.

    Does padding to keep the output the same size as the input.

    Parameters
    ----------
    x
        Tensor to filter
    kernels
        List of 1D kernels to apply to the tensor x
    dim
        Axes to filter over. Must have the same length as kernels.
    pad_mode
        Padding mode
    pad_value
        Padding value for pad_mode = constant

    Returns
    -------
    The filtered tensor, with the same shape as the input unless pad_mode is 'none' and
    and promoted dtype of the input and the kernels.
    """
    if len(dim) != len(kernels):
        raise ValueError('Must provide matching length kernels and dim arguments.')

    dim = normalize_indices(x.ndim, dim)

    if pad_mode == 'constant' and pad_value == 0:
        # padding is done inside the convolution
        padding_conv = 'same'
    else:
        # padding is done with pad() before the convolution
        padding_conv = 'valid'

    # output will be of the promoted type of the input and the kernels
    target_dtype = reduce(torch.promote_types, [k.dtype for k in kernels], x.dtype)
    x = x.to(target_dtype)

    for kernel, d in zip(kernels, dim, strict=False):
        kernel = kernel.to(device=x.device, dtype=target_dtype)
        # moveaxis is not implemented for batched tensors, so vmap would fail.
        # thus we use permute.
        idx = list(range(x.ndim))
        # swapping the last axis and the axis to filter over
        idx[d], idx[-1] = idx[-1], idx[d]
        x = x.permute(idx)
        # flatten first to allow for circular, replicate and reflection padding for arbitrary tensor size
        x_flat = x.flatten(end_dim=-2)
        if padding_conv == 'valid' and pad_mode != 'none':
            left_pad = (len(kernel) - 1) // 2
            right_pad = (len(kernel) - 1) - left_pad
            x_flat = torch.nn.functional.pad(x_flat, pad=(left_pad, right_pad), mode=pad_mode, value=pad_value)
        x = torch.nn.functional.conv1d(
            repeat(x_flat, 'batch x -> batch channels x', channels=1),
            repeat(kernel, 'x -> batch channels x', batch=1, channels=1),
            padding=padding_conv,
        ).reshape(*x.shape[:-1], -1)
        # for a single permutation, this undoes the permutation
        x = x.permute(idx)
    return x


def gaussian_filter(
    x: torch.Tensor,
    sigmas: float | Sequence[float] | torch.Tensor,
    dim: int | Sequence[int] | None = None,
    truncate: int = 3,
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Apply a and-Gaussian filter.

    Parameters
    ----------
    x
        Tensor to filter
    sigmas
        Standard deviation for Gaussian kernel. If iterable, must have length equal to the number of axes.
    dim
        Axis or axes to filter over. If None, filters over all axes.
    truncate
        Truncate the filter at this many standard deviations.
    pad_mode
        Padding mode
    pad_value
        Padding value for pad_mode = constant
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    sigmas = torch.as_tensor(sigmas) if np.iterable(sigmas) else torch.tensor([sigmas] * len(dim))
    if not torch.all(sigmas > 0):
        raise ValueError('`sigmas` must be positive')

    if len(sigmas) != len(dim):
        raise ValueError('Must provide matching length sigmas and dim arguments. ')

    kernels = tuple(
        [
            torch.exp(-0.5 * (torch.arange(-ceil(truncate * sigma), ceil(truncate * sigma) + 1) / sigma) ** 2)
            for sigma in sigmas
        ]
    )
    kernels = tuple([(k / k.sum()).to(device=x.device) for k in kernels])
    x_filtered = filter_separable(x, kernels, dim, pad_mode, pad_value)
    return x_filtered


def uniform_filter(
    x: torch.Tensor,
    width: int | Sequence[int] | torch.Tensor,
    dim: int | Sequence[int] | None = None,
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Apply a and-uniform filter.

    Parameters
    ----------
    x
        Tensor to filter
    width
        Width of uniform kernel. If iterable, must have length equal to the number of axes.
    dim
        Axis or axes to filter over. If None, filters over all axes.
    pad_mode
        Padding mode
    pad_value
        Padding value for pad_mode = constant
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    width = torch.as_tensor(width) if np.iterable(width) else torch.tensor([width] * len(dim))
    if not torch.all(width > 0):
        raise ValueError('width must be positive.')
    if torch.any(width % 2 != 1):
        warnings.warn('width should be odd.', stacklevel=2)
    if len(width) != len(dim):
        raise ValueError('Must provide matching length width and dim arguments. ')
    width = torch.minimum(width, torch.tensor(x.shape)[(dim), ...])

    kernels = tuple([torch.ones(width, device=x.device) / width for width in width])
    x_filtered = filter_separable(x, kernels, dim, pad_mode, pad_value)
    return x_filtered


def apodization_window(
    coordinates: Sequence[torch.Tensor],
    widths: Sequence[float],
    *,
    window: Literal['hann', 'gaussian'] = 'hann',
    radial: bool = False,
) -> torch.Tensor:
    """Evaluate a centered window at continuous coordinates.

    Parameters
    ----------
    coordinates
        Nonempty sequence of broadcastable real floating-point coordinate tensors
        with the same dtype and device, one per spatial direction.
    widths
        Finite positive full widths in the same units as the coordinates.
        Must have the same length as coordinates. For Hann, this is the support
        width; for Gaussian, the full width at half maximum.
    window
        ``'hann'`` evaluates ``(1 + cos(2*pi*k/width)) / 2`` inside
        ``abs(k) < width/2`` and zero outside. ``'gaussian'`` evaluates
        ``exp(-4*log(2)*(k/width)**2)`` without a hard cutoff.
    radial
        Evaluate one window at the radius of the scaled coordinates instead of
        multiplying per-axis windows. Gaussian weights coincide for both choices.

    Returns
    -------
        Real weights with the broadcast coordinate shape, unit gain at the
        origin and no energy normalization. Radial Hann has zero gradient at DC.
    """
    if not coordinates or len(coordinates) != len(widths):
        raise ValueError('Must provide nonempty coordinates and matching length widths.')
    if window not in ('hann', 'gaussian'):
        raise ValueError("window must be 'hann' or 'gaussian'.")
    if any(not math.isfinite(value) or value <= 0 for value in widths):
        raise ValueError('width must be finite and positive in every direction.')

    weight = coordinates[0].new_ones(())
    radius_squared = torch.zeros_like(weight)
    for coordinate, axis_width in zip(coordinates, widths, strict=True):
        normalized = coordinate / (axis_width / 2)
        if radial or window == 'gaussian':
            radius_squared = radius_squared + normalized.square()
        else:
            weight = weight * torch.where(normalized.abs() < 1, (1 + torch.cos(torch.pi * normalized)) / 2, 0)
    if window == 'gaussian':
        weight = torch.exp(-math.log(2) * radius_squared)
    elif radial:
        at_dc = radius_squared == 0
        radius = torch.where(at_dc, 1, radius_squared).sqrt()
        weight = torch.where(radius < 1, (1 + torch.cos(torch.pi * radius)) / 2, 0)
        weight = torch.where(at_dc, 1, weight)
    return weight
