"""Interpolation of data tensor."""

from collections.abc import Callable, Sequence
from typing import Literal

import torch

from mr2.utils.reshape import normalize_index, normalize_indices, unsqueeze_right


def _interp_along_axis(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    axis: int = 0,
) -> torch.Tensor:
    """One-dimensional linear interpolation of tensor-valued samples along a chosen axis.

    Evaluates the function at the query coordinates ``x`` based on known sample
    points ``xp`` and tensor-valued samples ``fp``. Interpolation is carried out
    along ``axis`` of ``fp``. Out-of-bounds values are clamped to the boundary
    samples, matching the default behavior of ``numpy.interp``.

    Parameters
    ----------
    x
        Query coordinates at which to evaluate the interpolated values.
    xp
        One-dimensional tensor of strictly increasing sample locations.
        This private helper assumes that ``xp`` is already sorted and contains
        no repeated entries.
    fp
        Tensor of sampled function values. The length of ``fp`` along ``axis``
        must match ``len(xp)``. Any remaining dimensions are treated as value
        dimensions and are preserved in the output.
    axis
        Axis of ``fp`` corresponding to the sample locations ``xp``.

    Returns
    -------
        Interpolated values with shape
        ``(*broadcast(x.shape, fp.shape[:axis]), *fp.shape[axis+1:])``.
    """
    if xp.ndim != 1:
        raise ValueError(f'xp must be one-dimensional, got shape {tuple(xp.shape)}.')
    if fp.ndim == 0:
        raise ValueError('fp must have at least one dimension.')

    axis = normalize_index(fp.ndim, axis)
    n_samples = fp.shape[axis]
    if n_samples != len(xp):
        raise ValueError(f'Length mismatch: xp has length {len(xp)}, but fp.shape[{axis}] = {n_samples}.')

    x_clamped = torch.clamp(x, min=xp[0], max=xp[-1])
    idx = torch.searchsorted(xp, x_clamped).clamp(1, len(xp) - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    weight = (x_clamped - x0) / (x1 - x0)
    weight = weight.to(fp.dtype.to_real())

    context_shape = torch.broadcast_shapes(x.shape, fp.shape[:axis])
    trailing_shape = fp.shape[axis + 1 :]
    fp = fp.broadcast_to(*context_shape, n_samples, *trailing_shape)
    sample_dim = len(context_shape)

    idx = idx.broadcast_to(context_shape)
    gather_idx = unsqueeze_right(idx, 1 + len(trailing_shape)).expand(*context_shape, 1, *trailing_shape)
    y0 = torch.take_along_dim(fp, gather_idx - 1, dim=sample_dim).squeeze(sample_dim)
    y1 = torch.take_along_dim(fp, gather_idx, dim=sample_dim).squeeze(sample_dim)
    weight = unsqueeze_right(weight.broadcast_to(context_shape), len(trailing_shape))
    if y0.is_complex():
        # torch.lerp does not support vectorized complex interpolation with real-valued weights.
        return y0 + weight * (y1 - y0)
    return torch.lerp(y0, y1, weight)


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample points.

    Implements numpy.interp

    Evaluates the function at the given coordinates x based on the known points (xp, fp).
    Out-of-bounds values are clamped to fp[0] and fp[-1], matching the default
    behavior of numpy.interp.

    Parameters
    ----------
    x
        The x-coordinates at which to evaluate the interpolated values.
    xp
        1d tensor of x coordinates of data points.
        The tensor will be sorted internally if needed.
    fp
        1d tensor of y coordinates matching the length of xp.

    Returns
    -------
        The interpolated values matching the shape of x.
    """
    if xp.ndim != 1:
        raise ValueError(f'xp must be one-dimensional, got shape {tuple(xp.shape)}.')
    if fp.ndim != 1:
        raise ValueError(f'fp must be one-dimensional for interp, got shape {tuple(fp.shape)}.')
    if len(xp) != len(fp):
        raise ValueError(f'xp and fp must have the same length, got {len(xp)} and {len(fp)}.')

    if not torch.all(xp[:-1] < xp[1:]):
        sorter = torch.argsort(xp)
        xp = xp[sorter]
        fp = fp[sorter]
        if not torch.all(xp[:-1] < xp[1:]):
            raise ValueError('xp must not contain repeated entries.')

    return _interp_along_axis(x, xp, fp, axis=0)


def interpolate(
    x: torch.Tensor, size: Sequence[int], dim: Sequence[int], mode: Literal['nearest', 'linear'] = 'linear'
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

    # torch.nn.functional.interpolate only available for real tensors
    # moveaxis is not implemented for batched tensors, so vmap would fail, thus we use permute.
    x_real = torch.view_as_real(x).permute(-1, *range(x.ndim)) if x.is_complex() else x
    dim = [d + 1 for d in dim] if x.is_complex() else dim

    for s, d in zip(size, dim, strict=True):
        if s != x_real.shape[d]:
            idx = list(range(x_real.ndim))
            # swapping the last axis and the axis to filter over
            idx[d], idx[-1] = idx[-1], idx[d]
            x_real = x_real.permute(idx)
            x_real = torch.nn.functional.interpolate(x_real.flatten(end_dim=-3), size=s, mode=mode).reshape(
                *x_real.shape[:-1], -1
            )
            # for a single permutation, this undoes the permutation
            x_real = x_real.permute(idx)
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
