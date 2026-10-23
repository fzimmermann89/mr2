"""Fourier noise correlation and noise-scaled image pairs."""

from collections.abc import Sequence

import torch

from mr2.operators.FastFourierOp import FastFourierOp
from mr2.utils.reshape import normalize_indices


def gaussian_kspace_window(
    size: int,
    sigma: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create a Gaussian window on centered, normalized k-space coordinates.

    Parameters
    ----------
    size
        Length of the Fourier axis.
    sigma
        Nonnegative strength in ``exp(-0.5 * (sigma * coordinate)**2)``.
        Coordinates run from -1 to 1, inclusive. Larger values suppress more
        high frequencies. Zero gives an all-ones window. This is not an
        image-domain Gaussian standard deviation.
    device
        Device for the window.
    dtype
        Real floating-point dtype; defaults to the PyTorch default dtype.

    Returns
    -------
        One-dimensional, unnormalized Fourier window. Unlike the upstream
        SNRAware even-length convention, the first element is not forced to zero.
    """
    if size < 1 or sigma < 0:
        raise ValueError('size must be positive and sigma must be nonnegative.')
    coordinate = torch.linspace(-1, 1, size, device=device, dtype=dtype)
    if size == 1:
        coordinate = torch.zeros_like(coordinate)
    return torch.exp(-0.5 * (sigma * coordinate).square())


def tapered_kspace_window(
    size: int,
    start: int,
    stop: int,
    taper: int = 0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create a supported Fourier window with optional raised-cosine transitions.

    Parameters
    ----------
    size
        Length of the centered Fourier axis.
    start, stop
        Retained interval ``[start, stop)``. A centered interval reduces resolution;
        an asymmetric interval models partial-Fourier noise correlation.
    taper
        Number of retained samples to taper at each cut boundary. Array boundaries
        are not tapered. Zero gives a binary mask. Transitions must not overlap.
    device
        Device for the window.
    dtype
        Real floating-point dtype; defaults to the PyTorch default dtype.

    Returns
    -------
        Unnormalized one-dimensional window, zero outside the retained interval.
        The taper samples are ``sin(pi*j/(2*(taper+1)))**2``, j=1,...,taper.
        These explicit boundaries differ from the upstream tapered-Hanning recipe.
    """
    boundaries = int(start > 0) + int(stop < size)
    if not 0 <= start < stop <= size or taper < 0 or boundaries * taper > stop - start:
        raise ValueError('Invalid support or overlapping taper transitions.')
    window = torch.zeros(size, device=device, dtype=dtype)
    window[start:stop] = 1
    if taper:
        ramp = torch.arange(1, taper + 1, device=device, dtype=dtype)
        ramp = (torch.pi * ramp / (2 * (taper + 1))).sin().square()
        if start > 0:
            window[start : start + taper] = ramp
        if stop < size:
            window[stop - taper : stop] = ramp.flip(0)
    return window


def filter_noise(noise: torch.Tensor, windows: Sequence[torch.Tensor], dim: Sequence[int]) -> torch.Tensor:
    """Correlate complex noise with variance-preserving Fourier windows.

    Parameters
    ----------
    noise
        Complex noise tensor. For unit per-real-component Gaussian noise, use
        ``sqrt(2) * rng.randn_like(image)`` with ``mr2.utils.RandomGenerator``.
    windows
        One-dimensional centered Fourier windows. Multiply windows referring to
        the same axis before energy normalization. Each length must match its
        input dimension. Create windows for the full matrix, filter, then crop;
        filtering a crop generally produces a different correlation length.
    dim
        Axis for each window. Repeated axes are allowed and their windows combine.

    Returns
    -------
        Complex filtered noise with the same shape. Each combined axis window is
        divided by its root mean squared magnitude. For stationary white noise
        this preserves expected per-component variance, not realized sample
        variance. Nonzero-lag real/imaginary covariance can arise from asymmetric
        windows, although same-pixel covariance stays zero for circular noise.
    """
    if not noise.is_complex():
        raise ValueError('noise must be complex.')
    if len(windows) != len(dim):
        raise ValueError('windows and dim must have the same length.')
    combined: dict[int, torch.Tensor] = {}
    for window, axis in zip(windows, normalize_indices(noise.ndim, dim, unique=False), strict=True):
        if window.ndim != 1 or window.numel() != noise.shape[axis]:
            raise ValueError('Each window must be one-dimensional and match its Fourier axis.')
        window = window.to(device=noise.device, dtype=noise.dtype)
        magnitude = window.abs().max()
        if not torch.isfinite(magnitude) or magnitude <= 0:
            raise ValueError('Fourier windows must have finite positive energy.')
        # Overall window amplitude cancels during energy normalization. Remove it
        # before multiplication and squaring to avoid underflow or overflow.
        window = window / magnitude
        combined[axis] = combined[axis] * window if axis in combined else window
    if not combined:
        return noise
    transform = FastFourierOp(dim=tuple(combined))
    (spectrum,) = transform(noise)
    for axis, window in combined.items():
        energy = window.abs().square().mean()
        if not torch.isfinite(energy) or energy <= 0:
            raise ValueError('Fourier windows must have finite positive energy.')
        shape = [1] * noise.ndim
        shape[axis] = window.numel()
        spectrum = spectrum * (window / energy.sqrt()).reshape(shape)
    (result,) = transform.H(spectrum)
    return result


def noise_pair(
    image: torch.Tensor,
    unit_noise: torch.Tensor,
    sigma: float | torch.Tensor,
    gmap: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a corrupted image and target using the released SNRAware scaling.

    Parameters
    ----------
    image
        Complex high-SNR image. Baseline noise calibration is the caller's
        responsibility; this function does not estimate noise from the image.
    unit_noise
        Independent complex noise with unit variance in each real component,
        optionally correlated by ``filter_noise``. Supply noise before g-map
        multiplication and without sigma scaling.
    sigma
        Nonnegative added per-component noise standard deviation. Applied once.
        Tensor values use normal PyTorch broadcasting; reshape per-batch values
        with singleton spatial axes explicitly.
    gmap
        Nonnegative real noise-amplification map, broadcastable to the image.
        One gives spatially uniform amplification. Filtering precedes modulation.

    Returns
    -------
    noisy, target
        ``(image + sigma*gmap*unit_noise)/sqrt(1+sigma**2)`` and
        ``image/sqrt(1+sigma**2)``. Keep the map unchanged when supplying it to a
        network. For baseline component SD s0(p), output component variance is
        ``(s0(p)**2 + sigma**2*gmap(p)**2)/(1+sigma**2)``. This need not be unity,
        nor preserve baseline spatial covariance. Targets retain acquisition noise.
    """
    if not image.is_complex() or not unit_noise.is_complex():
        raise ValueError('image and unit_noise must be complex.')
    if any(isinstance(value, torch.Tensor) and value.is_complex() for value in (sigma, gmap)):
        raise ValueError('sigma and gmap must be real.')
    sigma = torch.as_tensor(sigma, device=image.device, dtype=image.real.dtype)
    gmap = torch.as_tensor(gmap, device=image.device, dtype=image.real.dtype)
    for value in (sigma, gmap):
        if not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError('sigma and gmap must be finite and nonnegative.')
    scale = torch.hypot(sigma, torch.ones_like(sigma))
    return (image + sigma * gmap * unit_noise) / scale, image / scale
