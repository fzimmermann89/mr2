"""Support and phase estimation for Cartesian partial Fourier reconstruction."""

import torch

from mr2.operators.FastFourierOp import FastFourierOp


def _partial_fourier_center(acquired_mask: torch.Tensor, axis: int) -> torch.Tensor:
    """Return a centered cosine taper within the conjugate-symmetric acquired region.

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
    frequency = torch.arange(size, device=acquired_mask.device) - size // 2
    taper = torch.where(
        frequency.abs() <= radius,
        0.5 * (1 + torch.cos(torch.pi * frequency.to(torch.float64) / (radius + 1))),
        0.0,
    )
    shape = [1] * acquired_mask.ndim
    shape[axis] = size
    return taper.reshape(shape)


def _partial_fourier_phase(kspace: torch.Tensor, center: torch.Tensor, fourier_op: FastFourierOp) -> torch.Tensor:
    """Estimate a unit phase from tapered symmetric calibration data; use phase zero at image zeros."""
    (low_resolution,) = fourier_op.H(kspace * center.to(dtype=kspace.real.dtype))
    return torch.polar(torch.ones_like(low_resolution.real), low_resolution.angle())
