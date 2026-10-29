MRI noise augmentation
======================

``mr2.utils.noise`` provides Fourier noise filtering and the noise-based scaling
used by the released `SNRAware method <https://arxiv.org/abs/2503.18162>`_. These
functions are independent of the network and operate on complex tensors.

Generating correlated noise
---------------------------

``RandomGenerator.randn_like`` already generates reproducible complex Gaussian
noise. PyTorch's standard complex normal has variance one in complex magnitude,
so each real component has variance one half. Multiply by ``sqrt(2)`` to get unit
variance in each real component::

    import torch
    from mr2.utils import RandomGenerator
    from mr2.utils.noise import (
        filter_noise, gaussian_kspace_window, noise_pair, tapered_kspace_window,
    )

    # image: complex (batch, frames, height, width)
    rng = RandomGenerator(seed=42, device=image.device)
    unit_noise = 2**0.5 * rng.randn_like(image)
    height, width = image.shape[-2:]
    gaussian = gaussian_kspace_window(
        width, sigma=2.0, device=image.device, dtype=image.real.dtype,
    )
    partial_fourier = tapered_kspace_window(
        width, start=width//4, stop=width, taper=width//16,
        device=image.device, dtype=image.real.dtype,
    )
    unit_noise = filter_noise(unit_noise, (gaussian, partial_fourier), dim=(-1, -1))

Windows use centered Fourier coordinates. Multiple windows on the same axis
multiply before normalization. The combined transfer function is divided by its
root mean squared magnitude. Centered orthonormal transforms use the existing
``FastFourierOp``. This preserves expected component variance for stationary white
noise while changing its spatial covariance. It does not force each sampled
realization to have exactly the requested sample variance.

A centered retained interval in ``tapered_kspace_window`` models resolution
reduction; an asymmetric interval models partial-Fourier noise correlation.
Temporal filtering works the same way by supplying a window along the frame axis.
Generate and filter noise on the intended full matrix before cropping training
patches: using the same normalized window on a smaller matrix changes the
correlation length. G-factor modulation is applied after filtering, matching the
released code; the paper's description places modulation before filtering.

The window builders have explicit conventions rather than reproduce all upstream
endpoints. Gaussian coordinates peak at FFT DC; ``sigma`` controls frequency
suppression, not image-domain standard deviation. Tapered windows use half-open
support and taper only cut boundaries. See the docstrings for exact coordinates
and transition values; supplied windows can model other reconstruction filters.

Noise-scaled pairs
------------------

Choose a nonnegative added component noise standard deviation and a real g-map.
For each batch member, reshape scalar noise levels to broadcast over frames and
spatial axes explicitly::

    # gmap: real tensor broadcastable to image; 1 means uniform amplification
    sigma = torch.full(
        (image.shape[0], 1, 1, 1), 4.0, device=image.device, dtype=image.real.dtype,
    )
    noisy, target = noise_pair(image, unit_noise, sigma=sigma, gmap=gmap)

The pair is ``(image + sigma*gmap*unit_noise)/sqrt(1+sigma**2)`` and
``image/sqrt(1+sigma**2)``. Supply unit-component-variance noise; sigma is applied
exactly once. Supply the unchanged g-map as the network's additional channel if
needed. Images and targets remain complex; channel packing is separate.

This scaling does not calibrate arbitrary source data. If the original component
noise standard deviation at location p is s0(p), independent added noise gives
component variance ``(s0(p)**2 + sigma**2*gmap(p)**2)/(1+sigma**2)``. The result
preserves local standard deviation gmap(p) only if s0(p) already equals gmap(p).
Spatial covariance can still differ. High-SNR targets retain their acquisition
noise. Real/imaginary covariance stays zero at a single pixel for circular noise;
asymmetric Fourier windows can introduce cross-component correlations between
different pixels.

Choosing noise levels, selecting measured g-maps, and matched image/map geometry
transforms are ordinary caller operations. The functions do not estimate g-maps,
perform SNR calibration, or provide a dataset or training framework.
