"""ESPIRiT method for coil sensitivity map calculation."""

from math import prod

import torch
from einops import rearrange

from mr2.data.SpatialDimension import SpatialDimension


def _espirit_covariance(kernels: torch.Tensor, img_shape: tuple[int, int, int]) -> torch.Tensor:
    """Evaluate kernel cross-correlations on the image grid without full-size kernel FFTs."""
    n_coils = kernels.shape[1]
    if kernels.shape[0] == 0:
        return kernels.new_zeros(n_coils, n_coils, *img_shape)
    support = tuple(2 * n - 1 for n in kernels.shape[-3:])
    transformed = torch.fft.fftn(kernels, s=support, dim=(-3, -2, -1))
    correlations = torch.fft.ifftn(
        torch.einsum('nczyx,ndzyx->cdzyx', transformed, transformed.conj()), dim=(-3, -2, -1)
    )
    # Correlations contain signed lags modulo support. Accumulate them modulo the
    # image size as well: multiple lags can coincide on small output grids.
    index = torch.zeros(support, dtype=torch.int64, device=kernels.device)
    coordinates = torch.meshgrid(*(torch.arange(n, device=kernels.device) for n in support), indexing='ij')
    for coordinate, n, size in zip(coordinates, support, img_shape, strict=True):
        lag = torch.where(coordinate < (n + 1) // 2, coordinate, coordinate - n)
        index = index * size + lag.remainder(size)
    grid = kernels.new_zeros(n_coils, n_coils, prod(img_shape))
    grid.scatter_add_(-1, index.flatten().expand(n_coils, n_coils, -1), correlations.flatten(start_dim=-3))
    grid = torch.fft.ifftn(grid.reshape(n_coils, n_coils, *img_shape), dim=(-3, -2, -1))
    grid = torch.fft.fftshift(grid, dim=(-3, -2, -1))
    grid *= prod(img_shape) / prod(kernels.shape[-3:])
    return grid


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

    Covariance storage scales with coils squared times output voxels; large 3D inputs can require substantial memory.

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
    aha = _espirit_covariance(kernels, img_shape.zyx)

    # A covariance column with maximal diagonal avoids cancellation of summed coils.
    diagonal = aha.diagonal(dim1=0, dim2=1).real
    seed = diagonal.argmax(dim=-1)
    fallback = aha.gather(1, seed[None, None].expand(n_coils, 1, *img_shape.zyx)).squeeze(1)
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

    return csm
