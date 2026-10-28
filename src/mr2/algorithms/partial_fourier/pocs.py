"""Cartesian partial Fourier completion using projection onto convex sets."""

from collections.abc import Sequence

import torch

from mr2.algorithms.partial_fourier.homodyne import _partial_fourier_center
from mr2.operators.FastFourierOp import FastFourierOp
from mr2.utils.reshape import normalize_index, normalize_indices


def pocs(
    kspace: torch.Tensor,
    acquired_mask: torch.Tensor,
    dim: Sequence[int],
    partial_fourier_dim: int,
    n_iterations: int = 10,
) -> torch.Tensor:
    r"""Complete Cartesian k-space using POCS.

    Estimate phase :math:`p` from a tapered symmetric center, then alternate
    :math:`x \mapsto p\operatorname{Re}(\overline{p}x)` with acquired-data replacement.

    Parameters
    ----------
    kspace
        Complex data on the full centered Cartesian grid.
    acquired_mask
        Boolean mask broadcastable to ``kspace``, on the same device. The acquired
        interval must touch either edge along ``partial_fourier_dim``, include
        DC and a conjugate pair, and be identical across batches and coils.
        Other axes must be fully sampled.
    dim
        Centered orthonormal FFT axes, including ``partial_fourier_dim``.
    partial_fourier_dim
        Partial Fourier axis. Positive and negative indices are accepted.
    n_iterations
        Projection count. Zero returns zero-filled acquired data.

    Returns
    -------
        Completed k-space with the input shape, dtype and device. Acquired
        values are preserved exactly; fully sampled data is unchanged.
    """
    if n_iterations < 0:
        raise ValueError('n_iterations must be non-negative.')
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
    center, _, _ = _partial_fourier_center(mask, axis, kspace.real.dtype)
    measured = torch.where(mask, kspace, 0)
    if n_iterations == 0:
        return measured
    (low_resolution,) = fourier_op.H(measured * center)
    phase = torch.polar(torch.ones_like(low_resolution.real), low_resolution.angle())
    completed = measured
    for _ in range(n_iterations):
        (image,) = fourier_op.H(completed)
        projected = phase * (phase.conj() * image).real
        (estimate,) = fourier_op(projected)
        completed = torch.where(mask, measured, estimate)
    return completed
