"""ROVir coil compression."""

import torch
from einops import rearrange

from mr2.operators import EinsumOp


def rovir(img: torch.Tensor, roi_mask: torch.Tensor, *, n_compressed_coils: int) -> EinsumOp:
    r"""Orthonormal ROVir coil compression operator.

    ROVir emphasizes signal variation inside an ROI while suppressing a background
    region. This implementation estimates ROI and background covariances, solves
    the generalized eigenproblem via background whitening, and keeps the dominant
    eigenvectors.

    An additional QR step produces an orthonormal compression matrix. This differs
    from the original formulation [KIM2021]_, which uses the generalized
    eigenvectors directly (they are background-orthogonal, not Euclidean-
    orthonormal). The QR step preserves the compressed subspace but loses the
    individual signal-to-interference interpretation and the strict eigenvalue
    ordering. The orthonormal basis is convenient for prewhitened input because it
    leaves noise statistics unchanged.


    Parameters
    ----------
    img
        Prewhitened coil images, shape ``(coil, z, y, x)``.
    roi_mask
        Boolean mask of shape ``(z, y, x)`` selecting the ROI.
        The background is defined as the complement of ``roi_mask``.
    n_compressed_coils
        Number of virtual coils to retain.

    Returns
    -------
        Operator ``(..., coil, z, y, x) -> (..., compressed_coil, z, y, x)``.

    References
    ----------
    .. [KIM2021] Kim D, Cauley SF, Nayak KS, Leahy RM, Haldar JP.
    Region-optimized virtual (ROVir) coils: Localization and/or suppression
    of spatial regions using sensor-domain beamforming. Magn Reson Med.
    2021;86(1):197--212. https://doi.org/10.1002/mrm.28706
    """
    if img.ndim != 4:
        raise ValueError(f'img must have shape (coil, z, y, x), got {tuple(img.shape)}')

    n_coils = img.shape[0]

    if roi_mask.shape != img.shape[1:]:
        raise ValueError(f'roi must have shape {tuple(img.shape[1:])}, got {tuple(roi_mask.shape)}')

    if not 1 <= n_compressed_coils <= n_coils:
        raise ValueError(f'n_compressed_coils must be in [1, {n_coils}], got {n_compressed_coils}')

    voxel_data = rearrange(img, 'coil ... -> (...) coil')
    roi_mask = roi_mask.to(device=img.device, dtype=torch.bool).flatten()

    if not roi_mask.any():
        raise ValueError('roi does not contain any voxels')

    if roi_mask.all():
        raise ValueError('roi covers the full image; ROVir needs background voxels')

    roi_voxels = voxel_data[roi_mask]
    background_voxels = voxel_data[~roi_mask]

    roi_covariance = roi_voxels.conj().mT @ roi_voxels / roi_voxels.shape[0]
    background_covariance = background_voxels.conj().mT @ background_voxels / background_voxels.shape[0]

    identity = torch.eye(n_coils, dtype=img.dtype, device=img.device)

    # Stabilize the generalized eigenvalue problem if the background covariance is close to singular.
    background_scale = background_covariance.diagonal().real.mean().clamp_min(1e-10)
    background_covariance = background_covariance + 1e-6 * background_scale * identity

    background_cholesky = torch.linalg.cholesky(background_covariance)
    background_whitener = torch.linalg.solve_triangular(background_cholesky, identity, upper=False)

    whitened_roi_covariance = background_whitener @ roi_covariance @ background_whitener.conj().mT
    whitened_roi_covariance = 0.5 * (whitened_roi_covariance + whitened_roi_covariance.conj().mT)

    eigenvalues, eigenvectors = torch.linalg.eigh(whitened_roi_covariance)
    component_order = torch.argsort(eigenvalues.real, descending=True)

    compression_vectors = background_whitener.conj().mT @ eigenvectors[:, component_order[:n_compressed_coils]]
    orthonormal_vectors, _ = torch.linalg.qr(compression_vectors, mode='reduced')
    compression_matrix = orthonormal_vectors.conj().mT.to(img.dtype)

    return EinsumOp(
        compression_matrix,
        'compressed_coil coil, ... coil z y x -> ... compressed_coil z y x',
    )
