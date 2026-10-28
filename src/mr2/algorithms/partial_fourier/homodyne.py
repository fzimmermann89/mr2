"""Ramp-weighted Cartesian partial Fourier reconstruction."""

from collections.abc import Sequence

import torch

from mr2.operators.FastFourierOp import FastFourierOp
from mr2.utils.reshape import normalize_index, normalize_indices


def _partial_fourier_center(
    acquired_mask: torch.Tensor, axis: int, dtype: torch.dtype
) -> tuple[torch.Tensor, int, int]:
    """Return the symmetric center taper, radius and acquired side (-1 or +1)."""
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


def homodyne(
    kspace: torch.Tensor,
    acquired_mask: torch.Tensor,
    dim: Sequence[int],
    partial_fourier_dim: int,
) -> torch.Tensor:
    r"""Complete Cartesian k-space using homodyne ramp weighting.

    Return :math:`F(p\operatorname{Re}(\overline{p}F^H(wk)))`, with phase :math:`p`
    estimated from a tapered symmetric center and linear ramp weights :math:`w`.

    Parameters
    ----------
    kspace
        Complex data on the full centered Cartesian grid.
    acquired_mask
        Boolean acquisition mask; see `~mr2.algorithms.partial_fourier.pocs`.
    dim
        Centered orthonormal FFT axes, including ``partial_fourier_dim``.
    partial_fourier_dim
        Partial Fourier axis. Positive and negative indices are accepted.

    Returns
    -------
        Phase-restored k-space with the input shape, dtype and device. Signed
        image amplitudes are retained. Acquired values may change; fully
        sampled data is unchanged.
    """
    if acquired_mask.dtype != torch.bool or acquired_mask.device != kspace.device:
        raise ValueError('acquired_mask must be boolean and on the same device as kspace.')
    dim = normalize_indices(kspace.ndim, dim)
    axis = normalize_index(kspace.ndim, partial_fourier_dim)
    if axis not in dim:
        raise ValueError('Fourier dimensions must be non-empty and include partial_fourier_dim.')
    mask = torch.broadcast_to(acquired_mask, kspace.shape)
    fourier_op = FastFourierOp(dim)
    if mask.all():
        return kspace.clone()
    center, radius, side = _partial_fourier_center(mask, axis, kspace.real.dtype)
    measured = torch.where(mask, kspace, 0)
    (low_resolution,) = fourier_op.H(measured * center)
    phase = torch.polar(torch.ones_like(low_resolution.real), low_resolution.angle())
    size = kspace.shape[axis]
    frequency = torch.arange(size, dtype=kspace.real.dtype, device=kspace.device) - size // 2
    weight = (1 + side * frequency / radius).clamp(0, 2)
    if size % 2 == 0:
        weight[0] = 1  # Nyquist is self-conjugate; do not double it.
    shape = [1] * kspace.ndim
    shape[axis] = size
    (weighted_image,) = fourier_op.H(measured * weight.reshape(shape))
    corrected = (phase.conj() * weighted_image).real
    (completed,) = fourier_op(phase * corrected)
    return completed
