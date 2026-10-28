"""Cartesian partial Fourier reconstruction with POCS and homodyne."""

from collections.abc import Sequence

import torch

from mr2.operators.FastFourierOp import FastFourierOp
from mr2.utils.reshape import normalize_index, normalize_indices


def _partial_fourier_setup(
    kspace: torch.Tensor, acquired_mask: torch.Tensor, dim: Sequence[int], partial_fourier_dim: int
) -> tuple[torch.Tensor, int, FastFourierOp]:
    """Check tensor inputs and normalize axes for both partial Fourier methods."""
    if not kspace.is_complex():
        raise ValueError('kspace must be complex.')
    if acquired_mask.dtype != torch.bool or acquired_mask.device != kspace.device:
        raise ValueError('acquired_mask must be boolean and on the same device as kspace.')
    dim = normalize_indices(kspace.ndim, dim)
    axis = normalize_index(kspace.ndim, partial_fourier_dim)
    if axis not in dim:
        raise ValueError('Fourier dimensions must be non-empty and include partial_fourier_dim.')
    return torch.broadcast_to(acquired_mask, kspace.shape), axis, FastFourierOp(dim)


def _partial_fourier_center(
    acquired_mask: torch.Tensor, axis: int, dtype: torch.dtype
) -> tuple[torch.Tensor, int, int]:
    """Return the symmetric center taper, calibration radius and acquired side (-1 or +1).

    The mask must vary only along ``axis`` and contain one contiguous interval
    touching either edge. The even-grid Nyquist sample is excluded from calibration.
    """
    lines = acquired_mask.movedim(axis, -1)
    support = lines[(0,) * (lines.ndim - 1)]
    if not torch.all(lines == support):
        raise ValueError('acquired_mask must be identical along all dimensions except the partial Fourier axis.')
    indices = support.nonzero().flatten()
    size = support.numel()
    if not indices.numel():
        raise ValueError('acquired_mask must contain acquired samples.')
    first, last = int(indices[0]), int(indices[-1])
    if last - first + 1 != indices.numel() or (first != 0 and last != size - 1):
        raise ValueError('Acquired support must be contiguous and touch an edge of the partial Fourier axis.')
    radius = min(size // 2 - first, last - size // 2)
    if radius < 1:
        raise ValueError('Partial Fourier phase calibration requires DC and at least one acquired conjugate pair.')
    frequency = torch.arange(size, dtype=dtype, device=acquired_mask.device) - size // 2
    taper = torch.where(
        frequency.abs() <= radius,
        0.5 * (1 + torch.cos(torch.pi * frequency / (radius + 1))),
        0.0,
    )
    shape = [1] * acquired_mask.ndim
    shape[axis] = size
    return taper.reshape(shape), radius, -1 if first == 0 else 1


def _partial_fourier_phase(kspace: torch.Tensor, center: torch.Tensor, fourier_op: FastFourierOp) -> torch.Tensor:
    """Estimate a unit phase from tapered symmetric calibration data; use phase zero at image zeros."""
    (low_resolution,) = fourier_op.H(kspace * center)
    return torch.polar(torch.ones_like(low_resolution.real), low_resolution.angle())


# Define shared helpers before importing the methods that use them.
from mr2.algorithms.partial_fourier.homodyne import homodyne  # noqa: E402
from mr2.algorithms.partial_fourier.pocs import pocs  # noqa: E402

__all__ = ['homodyne', 'pocs']  # noqa: ALL[FastFourierOp,Sequence,normalize_index,normalize_indices,torch]
