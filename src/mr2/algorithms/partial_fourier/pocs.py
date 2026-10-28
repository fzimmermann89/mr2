"""Cartesian partial Fourier completion using projection onto convex sets."""

from collections.abc import Sequence

import torch

from mr2.algorithms.partial_fourier import _partial_fourier_center, _partial_fourier_phase, _partial_fourier_setup


def pocs(
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
    center, _, _ = _partial_fourier_center(mask, axis, kspace.real.dtype)
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
