"""ESPIRiT method for coil sensitivity map calculation."""

from math import prod

import torch
from einops import rearrange

from mr2.data.SpatialDimension import SpatialDimension


def _espirit_correlations(kernels: torch.Tensor) -> torch.Tensor:
    """Compute kernel cross-correlations on their small signed-lag grid."""
    support = tuple(2 * n - 1 for n in kernels.shape[-3:])
    if kernels.shape[0] == 0:
        return kernels.new_zeros(kernels.shape[1], kernels.shape[1], *support)
    transformed = torch.fft.fftn(kernels, s=support, dim=(-3, -2, -1))
    return torch.fft.ifftn(torch.einsum('nczyx,ndzyx->cdzyx', transformed, transformed.conj()), dim=(-3, -2, -1))


def espirit(
    coil_k_space: torch.Tensor,
    img_shape: SpatialDimension[int],
    singular_value_threshold: float = 0.02,
    kernel_width: int = 6,
    crop_threshold: float = 0.8,
    n_iterations: int = 10,
) -> torch.Tensor:
    """Calculate a coil sensitivity map (csm) using the ESPIRIT method.

    This is for a single set of Cartesian coil k-space data. The input should be a
    tensor with dimensions `(coils, kz, ky, kx)`. The output has shape
    `(coils, *img_shape.zyx)`. Either apply this function individually to each set of coil
    k-space data, or see `~mr2.data.CsmData.from_kdata_espirit` which performs
    this operation on a whole dataset [UEC2013]_.

    This function is inspired by
    `sigpy.mri.app.EspiritCalib <https://sigpy.readthedocs.io/en/latest/_modules/sigpy/mri/app.html#EspiritCalib>`_.

    Covariance and power iterations are evaluated in spatial slabs to limit working memory.

    Parameters
    ----------
    coil_k_space
        Cartesian k-space for each coil element.
    img_shape
        Shape of the reconstructed image.
    singular_value_threshold
        Relative singular-value cutoff for calibration kernels, matching SigPy:
        retain ``s_i / s_max > singular_value_threshold``. For BART ``ecalib -t t``,
        use ``sqrt(t)``; BART thresholds relative calibration-Gram eigenvalues.
        For example, BART ``-t 0.02`` corresponds to approximately ``0.141421``.
        The Gram eigendecomposition loses precision for very small cutoffs.
    kernel_width
        edge size of the scanning window
    crop_threshold
        Eigenvalue cutoff for map support, independent of the kernel cutoff.
        Maps with eigenvalues at or below this value are set to zero.
    n_iterations
        number of iterations in the power method

    Returns
    -------
        csm of the shape (coils, z, y, x)

    References
    ----------
    .. [UEC2013] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M (2013) ESPIRiT
    - an eigenvalue approach to autocalibrating parallel MRI: Where SENSE meets GRAPPA. MRM
    """
    if coil_k_space.ndim != 4 or any(n < 1 for n in coil_k_space.shape):
        raise ValueError('ESPIRiT expects nonempty (coils, kz, ky, kx) calibration data.')
    if kernel_width < 1 or n_iterations < 1:
        raise ValueError('kernel_width and n_iterations must be positive.')
    if not 0 <= singular_value_threshold <= 1 or not 0 <= crop_threshold <= 1:
        raise ValueError('ESPIRiT thresholds must lie between zero and one.')
    if any(out < min(n, kernel_width) for n, out in zip(coil_k_space.shape[-3:], img_shape.zyx, strict=True)):
        raise ValueError('The output grid must be at least as large as the calibration kernel.')

    # expecting coil_k_space to be of shape (coils, z, y, x)
    # inspired by https://sigpy.readthedocs.io/en/latest/_modules/sigpy/mri/app.html#EspiritCalib

    # Get calibration matrix.
    # Shape will be [n_coils] + n_blks + [kernel_width] * img_ndim
    mat = coil_k_space
    for ax in (1, 2, 3):
        mat = mat.unfold(dimension=ax, size=min(coil_k_space.shape[ax], kernel_width), step=1)

    n_coils, _, _, _, c, b, a = mat.shape
    mat = rearrange(mat, 'coils z y x c b a -> (z y x) (coils c b a)')

    # Gram eigenvalues are squared singular values; keep the SigPy cutoff.
    eigenvalues, eigenvectors = torch.linalg.eigh(mat.H @ mat)
    keep = eigenvalues > singular_value_threshold**2 * eigenvalues[-1]
    vh = eigenvectors[:, keep].H
    kernels = rearrange(vh, 'n (coils c b a) -> n coils c b a', coils=n_coils, c=c, b=b, a=a)
    correlations = _espirit_correlations(kernels)
    # Split along the largest image axis, keeping the other two transforms as FFTs.
    axis = max(range(3), key=lambda i: img_shape.zyx[i])
    correlations = correlations.movedim(axis + 2, -3)
    shape = list(img_shape.zyx)
    shape.insert(0, shape.pop(axis))
    support = correlations.shape[-3:]
    lags = [torch.fft.ifftshift(torch.arange(n, device=kernels.device) - n // 2) for n in support]
    index = (lags[1][:, None].remainder(shape[1]) * shape[2] + lags[2].remainder(shape[2])).flatten()
    # Center and normalize the small correlations instead of copying every image slab.
    centering = sum(
        -(2j * torch.pi / size) * (size // 2) * lag.to(kernels.real.dtype)
        for size, lag in zip(shape[1:], (lags[1][:, None], lags[2]), strict=True)
    )
    correlations = correlations * centering.exp().to(kernels.dtype) / prod(kernels.shape[-3:])
    maps = kernels.new_empty(n_coils, *shape)
    BATCHSIZE = 2**18  # noqa: N806
    # Target 512 MiB per covariance slab; a single plane is the minimum.
    batchsize = min(BATCHSIZE, (512 * 1024**2) // (n_coils**2 * kernels.element_size()))
    planes = max(1, batchsize // prod(shape[1:]))
    offsets = torch.arange(shape[0], device=kernels.device, dtype=kernels.real.dtype) - shape[0] // 2
    for out, offset in zip(maps.split(planes, dim=1), offsets.split(planes), strict=True):
        # Evaluate only these centered slices along the split axis. The remaining
        # signed lags are accumulated modulo the output size before the 2D FFT.
        fourier_phase = (2j * torch.pi / shape[0]) * offset[:, None] * lags[0]
        values = torch.einsum('pz,cdzyx->cdpyx', fourier_phase.exp().to(kernels.dtype), correlations)
        aha = kernels.new_zeros(n_coils, n_coils, len(offset), prod(shape[1:]))
        aha.scatter_add_(-1, index.expand(n_coils, n_coils, len(offset), -1), values.flatten(start_dim=-2))
        aha = torch.fft.ifft2(aha.unflatten(-1, shape[1:]), norm='forward')
        # A covariance column with maximal diagonal avoids cancellation of summed coils.
        seed = aha.diagonal(dim1=0, dim2=1).real.argmax(dim=-1)
        fallback = aha.gather(1, seed[None, None].expand(n_coils, 1, *aha.shape[-3:])).squeeze(1)
        # A small column component breaks symmetry in the summed seed, including
        # opposite coils. Fall back to the column if the combined seed nearly cancels.
        v = aha.sum(dim=1) + 0.01 * fallback
        stable = (
            v.abs().square().sum(dim=0, keepdim=True).sqrt()
            > 1e-3 * fallback.abs().square().sum(dim=0, keepdim=True).sqrt()
        )
        v = torch.where(stable, v, fallback)
        for _ in range(n_iterations):
            v /= v.abs().square().sum(dim=0).sqrt().clamp_min(1e-12)
            v = torch.einsum('abzyx,bzyx->azyx', aha, v)
        max_eig = v.abs().square().sum(dim=0).sqrt()
        csm = v / max_eig.clamp_min(1e-12)

        # Normalize phase with respect to first channel
        reference = csm[0]
        strongest = csm.gather(0, csm.abs().argmax(dim=0, keepdim=True)).squeeze(0)
        reference = torch.where(reference.abs() > 1e-12, reference, strongest)
        phase = torch.where(reference.abs() > 1e-12, reference.conj() / reference.abs().clamp_min(1e-12), 1)
        csm = csm * phase

        csm *= max_eig > crop_threshold
        out.copy_(csm)

    return maps.movedim(1, axis + 1)
