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
    if not dim or axis not in dim:
        raise ValueError('Fourier dimensions must be non-empty and include partial_fourier_dim.')
    return torch.broadcast_to(acquired_mask, kspace.shape), axis, FastFourierOp(dim)


def _partial_fourier_center(acquired_mask: torch.Tensor, axis: int, dtype: torch.dtype) -> torch.Tensor:
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
    frequency = torch.arange(size, dtype=dtype, device=acquired_mask.device) - size // 2
    taper = torch.where(
        frequency.abs() <= radius,
        0.5 * (1 + torch.cos(torch.pi * frequency / (radius + 1))),
        0.0,
    )
    shape = [1] * acquired_mask.ndim
    shape[axis] = size
    return taper.reshape(shape)


def _partial_fourier_phase(kspace: torch.Tensor, center: torch.Tensor, fourier_op: FastFourierOp) -> torch.Tensor:
    """Estimate a unit phase from tapered symmetric calibration data; use phase zero at image zeros."""
    (low_resolution,) = fourier_op.H(kspace * center)
    return torch.polar(torch.ones_like(low_resolution.real), low_resolution.angle())


def pocs_partial_fourier(
    kspace: torch.Tensor,
    acquired_mask: torch.Tensor,
    dim: Sequence[int],
    partial_fourier_dim: int,
    n_iterations: int = 10,
) -> torch.Tensor:
    r"""Complete a Cartesian grid using a fixed low-resolution phase estimate.

    Alternates the orthogonal image projection
    :math:`x \mapsto p\,\operatorname{Re}(\overline{p}x)` with replacement of
    **all** acquired samples. Here :math:`p` is the unit phase of the inverse FFT
    of a cosine-tapered conjugate-symmetric center. This permits signed real
    amplitudes after demodulation; it does not impose a nonnegative magnitude.

    Parameters
    ----------
    kspace
        Complex tensor on the full encoded Cartesian grid, with zero frequency
        at index ``size // 2``. Values outside ``acquired_mask`` are ignored.
        Batch and coil dimensions are treated independently.
    acquired_mask
        Boolean mask broadcastable to ``kspace``, on the same device. It must
        describe the same contiguous acquired interval for every batch, coil,
        and line, touching either edge along ``partial_fourier_dim``. All other
        directions must be fully sampled. Missing samples are never inferred
        from data values. For trajectory data, use `~mr2.operators.CartesianSamplingOp`
        to place both measurements and an explicit acquisition mask on the grid.
    dim
        Fourier transform dimensions, including ``partial_fourier_dim``.
        Uses the centered, orthonormal conventions of `~mr2.operators.FastFourierOp`.
    partial_fourier_dim
        Single partial Fourier axis, including a readout axis for partial echo.
        Positive and negative indices are accepted. Calibration requires DC and
        at least one conjugate pair. The even-grid Nyquist bin is excluded from
        phase calibration. Arbitrary undersampling and varying masks are unsupported.
    n_iterations
        Number of projections. Zero returns the zero-filled acquired data.

    Returns
    -------
        Completed k-space with the input shape, dtype and device and exactly
        preserved acquired values. Fully sampled input is returned unchanged
        in value. Use ``FastFourierOp(dim).H`` to convert to image space.

    Notes
    -----
    Phase calibration is tapered only along the partial Fourier direction;
    other transformed directions retain their full resolution.
    Recovery assumes sufficiently smooth image phase along the partial Fourier
    direction. Small calibration regions, rapid phase changes and noise can limit
    accuracy. No coil combination, cropping or oversampling removal is performed.
    """
    if n_iterations < 0:
        raise ValueError('n_iterations must be non-negative.')
    mask, axis, fourier_op = _partial_fourier_setup(kspace, acquired_mask, dim, partial_fourier_dim)
    if mask.all():
        return kspace.clone()
    center = _partial_fourier_center(mask, axis, kspace.real.dtype)
    measured = torch.where(mask, kspace, 0)
    if n_iterations == 0:
        return measured
    phase = _partial_fourier_phase(measured, center, fourier_op)
    completed = measured
    for _ in range(n_iterations):
        (image,) = fourier_op.H(completed)
        projected = phase * (phase.conj() * image).real
        (estimate,) = fourier_op(projected)
        completed = torch.where(mask, measured, estimate)
    return completed


def homodyne_partial_fourier(
    kspace: torch.Tensor,
    acquired_mask: torch.Tensor,
    dim: Sequence[int],
    partial_fourier_dim: int,
    *,
    restore_phase: bool = False,
) -> torch.Tensor:
    r"""Reconstruct an image using homodyne ramp weighting and phase correction.

    Returns :math:`\operatorname{Re}(\overline{p} F^H(w k))`, where :math:`p`
    is the unit phase estimated from the same tapered symmetric center as
    `~mr2.algorithms.pocs_partial_fourier`. With ``restore_phase=True``, multiply
    this signed real image by :math:`p`. Neither output takes a magnitude.

    Parameters
    ----------
    kspace
        Complex data on the full encoded Cartesian grid, with DC at ``size // 2``.
        Unacquired values are ignored. Batch and coil images are independent.
        For trajectory data, use `~mr2.operators.CartesianSamplingOp` to place
        measurements and an explicit acquisition mask on this grid first.
    acquired_mask
        Boolean mask broadcastable to ``kspace``, on the same device. Supports
        the same single contiguous interval touching either edge as POCS, shared
        across batches, coils and lines. All other axes must be fully sampled.
        Partial sampling requires DC and at least one acquired conjugate pair.
    dim
        Transform axes, using `~mr2.operators.FastFourierOp` centered orthonormal
        conventions. Must include ``partial_fourier_dim``.
    partial_fourier_dim
        Partial Fourier axis, including readout for partial echo. Positive and
        negative indices, either acquisition side and odd/even grids are supported.
    restore_phase
        If false, return a signed phase-corrected real image with the input's
        real dtype. If true, return a phase-restored complex image with the input
        dtype. This image does not enforce acquired k-space consistency.

    Returns
    -------
        Image on the full encoded grid, with the input shape and device.
        Fully sampled data uses unit weights and its full-resolution image phase:
        the real output is its amplitude and phase restoration returns its inverse FFT.

    Notes
    -----
    The ramp is one at DC, rises to two toward the acquired outer region and
    falls to zero at the opposite edge of the symmetric calibration interval.
    Conjugate weights sum to two. An acquired even-grid Nyquist bin has weight
    one; a missing Nyquist bin cannot be recovered by homodyne. No extra FFT
    scaling is applied. Phase calibration is tapered only along the partial
    Fourier axis. Accuracy depends on smooth phase and sufficient calibration.
    No coil combination, cropping or oversampling removal is performed.
    """
    mask, axis, fourier_op = _partial_fourier_setup(kspace, acquired_mask, dim, partial_fourier_dim)
    if mask.all():
        (image,) = fourier_op.H(kspace)
        return image if restore_phase else image.abs()
    center = _partial_fourier_center(mask, axis, kspace.real.dtype)
    measured = torch.where(mask, kspace, 0)
    phase = _partial_fourier_phase(measured, center, fourier_op)
    size = kspace.shape[axis]
    radius = int((center > 0).sum()) // 2
    support = mask.movedim(axis, -1)[(0,) * (mask.ndim - 1)]
    side = -1 if bool(support[0]) else 1
    frequency = torch.arange(size, dtype=kspace.real.dtype, device=kspace.device) - size // 2
    weight = (1 + side * frequency / radius).clamp(0, 2)
    if size % 2 == 0:
        weight[0] = 1  # Nyquist is self-conjugate; do not double it.
    shape = [1] * kspace.ndim
    shape[axis] = size
    (weighted_image,) = fourier_op.H(measured * weight.reshape(shape))
    corrected = (phase.conj() * weighted_image).real
    return phase * corrected if restore_phase else corrected
