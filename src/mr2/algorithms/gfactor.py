"""Trajectory-independent g-factor estimation."""

import math
from collections.abc import Callable

import torch


def gfactor(
    data: torch.Tensor,
    reconstruction: Callable[[torch.Tensor], torch.Tensor | tuple[torch.Tensor]],
    reference_data: torch.Tensor,
    reference_reconstruction: Callable[[torch.Tensor], torch.Tensor | tuple[torch.Tensor]],
    *,
    acceleration: float,
    noise_covariance: torch.Tensor,
    n_replicas: int = 100,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Estimate g-factor using pseudo multiple replicas [ROB2008]_.

    Independent measurement-domain noise is added before each reconstruction,
    including any whitening, coil mixing, Fourier sampling and coil combination.
    Trajectories are defined entirely by the supplied reconstructions; no
    Cartesian alias grouping or sampling-density assumption is used.

    Parameters
    ----------
    data
        Accelerated complex measurements with shape ``(..., coils, k2, k1, k0)``.
        Supply acquired samples only, or have the callable mask noise at
        unacquired positions of a padded grid. For a fixed linear reconstruction
        these may be zeros. For a nonlinear reconstruction use the measured data
        to probe fluctuations about it.
    reconstruction
        Callable mapping measurements to an image tensor or a one-tensor tuple
        (as returned by mr2 operators). Capture fixed calibration, sensitivity
        maps, weights and solver settings in the callable. Calibration is not
        estimated by this function; refitting it inside the callable changes the
        experiment to include calibration uncertainty. Calls must be deterministic
        apart from the supplied noise, and return a consistent shape and dtype.
    reference_data
        Matching fully sampled measurements, with the same batch/coil dimensions,
        dtype and device as ``data``. Sample dimensions may differ. Zeros suffice
        for a fixed linear reference reconstruction.
    reference_reconstruction
        Fully sampled reconstruction producing the same image shape and signal
        units as ``reconstruction``. Both must have matching unbiased signal
        amplitude, spatial resolution and contrast for a classical g-factor.
        Include any needed signal normalization in these callables; for example,
        orthonormal transforms with different matrix sizes may need rescaling.
    acceleration
        Explicit acquisition-time reduction ``T_full / T_acc >= 1``, including
        calibration overhead where applicable. With equal dwell time and no other
        overhead this is the full/acquired sample-count ratio. It is never inferred
        from tensor sizes. Both experiments must have equal per-sample noise
        covariance (including bandwidth, receiver gain and dwell-time corrections).
    noise_covariance
        Positive definite Hermitian coil covariance ``E[n n^H]`` with shape
        ``(..., coils, coils)``, broadcastable over the batch dimensions. Noise is
        circular complex Gaussian, independent between samples and batches, with
        this same covariance for both acquisitions. This is the total complex
        variance convention used by ``prewhiten_kspace``, twice the per-real-part
        covariance in [ROB2008]_. Use identity for already whitened data and put
        whitening inside the callable when using unwhitened data/covariance.
    n_replicas
        Number of replicas for each acquisition, at least two. More replicas
        reduce Monte Carlo uncertainty; no fixed count guarantees accuracy.
    generator
        Optional PyTorch generator on the measurement device, advanced in place.
        A freshly seeded generator reproduces the estimate.

    Returns
    -------
        Real tensor ``sqrt(var_acc / (acceleration * var_full))`` on the image
        device. Variance is the unbiased sample estimate of ``E[abs(x-E[x])**2]``;
        only online means and squared deviations are retained. For circular
        complex linear outputs this is twice the real-part variance and the
        factor cancels in the ratio. Zero reference variance yields NaN.

    Notes
    -----
    This is a classical geometry factor only for consistently normalized,
    unbiased linear reconstructions. For nonlinear, magnitude, or regularized
    outputs the result is a reference-normalized noise ratio, potentially signal
    dependent, and does not measure bias or resolution loss. Values below one
    must not be interpreted as improved encoding efficiency in that setting.
    Added replicas keep the original acquired noise fixed; nonlinear results
    describe this finite-noise perturbation experiment, not a universal noise law.
    Reconstruction graphs are detached before variance accumulation.

    References
    ----------
    .. [ROB2008] Robson et al. (2008). Comprehensive Quantification of Signal-to-Noise
       Ratio and g-Factor for Image-Based and k-Space-Based Parallel Imaging
       Reconstructions. https://doi.org/10.1002/mrm.21728
    """
    if not math.isfinite(acceleration) or acceleration < 1:
        raise ValueError('acceleration must be finite and at least one.')
    if not isinstance(n_replicas, int) or n_replicas < 2:
        raise ValueError('n_replicas must be an integer of at least two.')
    if data.ndim < 4 or not data.is_complex():
        raise ValueError('data must be complex with at least four dimensions.')
    if reference_data.shape[:-3] != data.shape[:-3]:
        raise ValueError('reference_data must have the same batch and coil dimensions as data.')
    if reference_data.dtype != data.dtype or reference_data.device != data.device:
        raise ValueError('reference_data must have the same dtype and device as data.')
    n_coils = data.shape[-4]
    covariance = noise_covariance.detach().to(device=data.device, dtype=data.dtype)
    if covariance.shape[-2:] != (n_coils, n_coils):
        raise ValueError('noise_covariance must have shape (..., coils, coils).')
    covariance = torch.broadcast_to(covariance, (*data.shape[:-4], n_coils, n_coils))
    if not torch.isfinite(covariance).all() or not torch.allclose(covariance, covariance.mH):
        raise ValueError('noise_covariance must be finite and Hermitian.')
    # Match prewhiten_kspace: E[n n^H] = L L^H, without spectral flooring.
    cholesky = torch.linalg.cholesky(covariance)

    def variance(
        measurements: torch.Tensor, reconstruct: Callable[[torch.Tensor], torch.Tensor | tuple[torch.Tensor]]
    ) -> torch.Tensor:
        mean = squared_deviations = None
        for count in range(1, n_replicas + 1):
            # Complex torch.randn already has unit TOTAL variance, not two.
            white = torch.randn(measurements.shape, dtype=data.dtype, device=data.device, generator=generator)
            noise = (cholesky @ white.flatten(start_dim=-3)).reshape(measurements.shape)
            result = reconstruct(measurements.detach() + noise)
            image = (result[0] if isinstance(result, tuple) else result).detach()
            if mean is None:
                mean = image.clone()
                squared_deviations = torch.zeros_like(image.real)
            else:
                if image.shape != mean.shape or image.dtype != mean.dtype or image.device != mean.device:
                    raise ValueError(
                        'Reconstruction output shape, dtype and device must be consistent across replicas.'
                    )
                delta = image - mean
                mean = mean + delta / count
                assert squared_deviations is not None
                squared_deviations += (delta.conj() * (image - mean)).real
        assert squared_deviations is not None
        return squared_deviations.clamp_min(0) / (n_replicas - 1)

    accelerated_variance = variance(data, reconstruction)
    reference_variance = variance(reference_data, reference_reconstruction)
    if accelerated_variance.shape != reference_variance.shape:
        raise ValueError('Accelerated and reference images must have the same shape.')
    return torch.where(
        reference_variance > 0,
        (accelerated_variance / (acceleration * reference_variance)).sqrt(),
        torch.nan,
    )
