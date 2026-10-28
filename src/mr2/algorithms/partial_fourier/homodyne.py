"""Ramp-weighted Cartesian partial Fourier reconstruction."""

from collections.abc import Sequence

import torch

from mr2.algorithms.partial_fourier import _partial_fourier_center, _partial_fourier_phase, _partial_fourier_setup


def homodyne(
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
    `~mr2.algorithms.partial_fourier.pocs`. With ``restore_phase=True``, multiply
    this signed real image by :math:`p`. For partial sampling, neither output takes a magnitude.

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
    center, radius, side = _partial_fourier_center(mask, axis, kspace.real.dtype)
    measured = torch.where(mask, kspace, 0)
    phase = _partial_fourier_phase(measured, center, fourier_op)
    size = kspace.shape[axis]
    frequency = torch.arange(size, dtype=kspace.real.dtype, device=kspace.device) - size // 2
    weight = (1 + side * frequency / radius).clamp(0, 2)
    if size % 2 == 0:
        weight[0] = 1  # Nyquist is self-conjugate; do not double it.
    shape = [1] * kspace.ndim
    shape[axis] = size
    (weighted_image,) = fourier_op.H(measured * weight.reshape(shape))
    corrected = (phase.conj() * weighted_image).real
    return phase * corrected if restore_phase else corrected
