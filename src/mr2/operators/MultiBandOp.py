"""Phase encoding and summation of simultaneously acquired slices."""

import math
from collections.abc import Sequence

import torch
from typing_extensions import Self

from mr2.operators.EinsumOp import EinsumOp


class MultiBandOp(EinsumOp):
    r"""Apply slice-dependent phases and sum slices into collapsed measurements.

    Maps ``(... slices coils k2 k1 k0)`` to ``(... coils k2 k1 k0)`` using
    :math:`y = \sum_s \exp(-2\pi i k\,\mathrm{shift}_s) x_s`.
    The slice axis is a leading axis, separate from the spatial/Fourier axes.
    The adjoint distributes measurements to all slices with conjugate phases.

    For joint SMS encoding, compose ``collapse @ fourier @ sensitivities``.
    Images and sensitivity maps must contain the same ordered group of slices.
    Other leading axes, such as contrasts or slice groups, remain batched.
    """

    def __init__(self, k: torch.Tensor, shifts: Sequence[float] | torch.Tensor) -> None:
        """Initialize slice encoding from coordinates and explicit shifts.

        Parameters
        ----------
        k
            Real coordinates of shape ``(... 1 k2 k1 k0)``, with a singleton coil
            axis, as in a `~mr2.data.KTrajectory` component. Singleton sampling
            axes broadcast. Use the actual acquired coordinates, including after
            selecting or reordering measurements.
        shifts
            Nonempty one-dimensional sequence of shifts, one per slice. The
            product of `k` and each shift must be dimensionless. For an in-plane
            coordinate in full-grid units, shifts are fractions of the full FOV.
            Alternatively, slice-encoding coordinates in cycles/metre can be
            paired with slice positions in metres.
        """
        if k.ndim < 4 or k.shape[-4] != 1 or k.is_complex() or not torch.isfinite(k).all():
            raise ValueError('k must contain finite real coordinates with shape (... 1 k2 k1 k0).')
        if not k.is_floating_point():
            k = k.to(torch.get_default_dtype())
        shift_tensor = torch.as_tensor(shifts, device=k.device)
        if (
            shift_tensor.ndim != 1
            or not shift_tensor.numel()
            or shift_tensor.is_complex()
            or not torch.isfinite(shift_tensor).all()
        ):
            raise ValueError('shifts must be a nonempty one-dimensional sequence of finite real values.')
        shift_tensor = torch.as_tensor(shifts, device=k.device, dtype=k.dtype)
        phase = torch.exp(-2j * torch.pi * k * shift_tensor.reshape(-1, 1, 1, 1))
        super().__init__(phase, '... s k2 k1 k0, ... s c k2 k1 k0 -> ... c k2 k1 k0')

    @classmethod
    def from_caipi(cls, k: torch.Tensor, multiband_factor: int, fov_shift_factor: float) -> Self:
        """Initialize regular CAIPI shifts relative to the full FOV.

        Parameters
        ----------
        k
            Actual acquired coordinates in full-grid units, with shape
            ``(... 1 k2 k1 k0)``. For accelerated Cartesian data these coordinates
            retain their original spacing; do not replace them with consecutive
            indices or divide them by the acceleration factor.
        multiband_factor
            Number of simultaneously acquired slices.
        fov_shift_factor
            Denominator of the shift between successive slices relative to the
            full FOV. For example, 3 gives shifts ``(0, 1/3, 2/3, ...)``.
            Must be finite and at least 1. A value of 1 means no CAIPI shift.

        Notes
        -----
        Scanner settings may express the shift relative to the acceleration-reduced
        FOV. In that case, multiply the scanner's FOV-shift factor by the nominal
        in-plane acceleration before passing it. For example, scanner factor 3
        with acceleration 2 gives ``fov_shift_factor=6``. This convention occurs
        in Siemens SMS reconstruction; confirm the sequence's convention, slice
        order and phase origin when applying it to raw data. A scanner setting
        meaning no CAIPI shift should remain 1.
        """
        if not isinstance(multiband_factor, int) or isinstance(multiband_factor, bool) or multiband_factor < 1:
            raise ValueError('multiband_factor must be a positive integer.')
        if not math.isfinite(fov_shift_factor) or fov_shift_factor < 1:
            raise ValueError('fov_shift_factor must be finite and at least 1.')
        shifts = torch.arange(multiband_factor, device=k.device, dtype=k.dtype if k.is_floating_point() else None)
        shifts = torch.zeros_like(shifts) if fov_shift_factor == 1 else shifts / fov_shift_factor
        return cls(k, shifts)
