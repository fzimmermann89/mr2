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
    r"""Estimate g-factor from noisy replicas of two reconstructions [ROB2008]_.

    Parameters
    ----------
    data
        Accelerated complex measurements, shape ``(..., coils, k2, k1, k0)``.
        Supply acquired samples only, or mask unacquired positions in the callable.
        Zeros suffice for linear reconstructions; otherwise use measured data.
    reconstruction
        Deterministic tensor-to-image callable returning a tensor or one-tensor
        tuple. Keep calibration, sensitivity maps and solver settings fixed.
        Include whitening and coil combination as needed.
    reference_data
        Fully sampled measurements with matching batch/coil dimensions, dtype and
        device. Sample dimensions may differ; zeros suffice for a linear reference.
    reference_reconstruction
        Reference callable returning the same image shape, signal scale, resolution
        and contrast. Apply any signal normalization inside the callables.
    acceleration
        Acquisition-time ratio ``T_full / T_acc >= 1``, including calibration
        overhead. Both acquisitions must have the same per-sample noise covariance.
    noise_covariance
        Positive definite Hermitian ``E[n n^H]``, shape ``(..., coils, coils)``,
        broadcastable over batches. Noise is circular complex and independent across
        samples/batches. This uses total complex variance, as in ``prewhiten_kspace``;
        use identity for whitened data. Robson uses half this covariance.
    n_replicas
        Replicas per acquisition, at least two. More reduce sampling uncertainty.
    generator
        Optional PyTorch generator on the measurement device, advanced in place.

    Returns
    -------
        Real g-factor map with the reconstructed image shape:
        ``sqrt(var_acc / (acceleration * var_full))``. Variances are unbiased total
        complex variances accumulated online. Zero reference variance gives NaN.

    Notes
    -----
    A classical geometry factor requires unbiased linear reconstructions with
    matching signal normalization. Nonlinear, magnitude or regularized outputs
    instead yield a possibly signal-dependent noise ratio that does not measure
    bias or resolution loss. Replicas perturb the supplied data at finite noise
    amplitude; they do not remove its original noise or guarantee convergence.

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
    if not torch.isfinite(covariance).all() or not torch.allclose(covariance, covariance.mH, atol=0):
        raise ValueError('noise_covariance must be finite and Hermitian.')
    # Match prewhiten_kspace: E[n n^H] = L L^H, without spectral flooring.
    cholesky = torch.linalg.cholesky(covariance)

    def variance(
        measurements: torch.Tensor, reconstruct: Callable[[torch.Tensor], torch.Tensor | tuple[torch.Tensor]]
    ) -> torch.Tensor:
        for count in range(1, n_replicas + 1):
            # Complex torch.randn already has unit TOTAL variance, not two.
            white = torch.randn(measurements.shape, dtype=data.dtype, device=data.device, generator=generator)
            noise = (cholesky @ white.flatten(start_dim=-3)).reshape(measurements.shape)
            result = reconstruct(measurements.detach() + noise)
            image = (result[0] if isinstance(result, tuple) else result).detach()
            del result  # Release saved autograd activations before the next reconstruction.
            if count == 1:
                mean = image.clone()
                squared_deviations = torch.zeros_like(image.real)
            else:
                delta = image - mean
                mean = mean + delta / count
                squared_deviations += delta.abs().square() * ((count - 1) / count)
        return squared_deviations / (n_replicas - 1)

    accelerated_variance = variance(data, reconstruction)
    reference_variance = variance(reference_data, reference_reconstruction)
    if accelerated_variance.shape != reference_variance.shape:
        raise ValueError('Accelerated and reference images must have the same shape.')
    return torch.where(
        reference_variance > 0,
        (accelerated_variance / (acceleration * reference_variance)).sqrt(),
        torch.nan,
    )
