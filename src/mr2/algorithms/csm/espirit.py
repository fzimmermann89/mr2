"""ESPIRiT method for coil sensitivity map calculation."""

import torch
from einops import rearrange

from mr2.data.SpatialDimension import SpatialDimension


def espirit(
    coil_k_space: torch.Tensor,
    img_shape: SpatialDimension[int],
    singular_value_threshold: float = 0.02,
    kernel_width: int = 6,
    crop_threshold: float = 0.1,
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

    Parameters
    ----------
    coil_k_space
        Cartesian k-space for each coil element.
    img_shape
        Shape of the reconstructed image.
    singular_value_threshold
        threshold for singular value decomposition
    kernel_width
        edge size of the scanning window
    crop_threshold
        threshold for cropping the csm, values below this threshold are set to 0
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

    # Perform SVD on calibration matrix
    _, s, vh = torch.linalg.svd(mat, full_matrices=False)

    # Get kernels
    vh = vh[s > singular_value_threshold * s.max()]
    kernels = rearrange(vh, 'n (coils c b a) -> n coils c b a', coils=n_coils, c=c, b=b, a=a)

    # Get covariance matrix in image domain
    aha = torch.zeros((n_coils, n_coils, *img_shape.zyx), dtype=coil_k_space.dtype, device=coil_k_space.device)

    for kernel in kernels:
        img_kernel = torch.fft.ifftn(kernel, s=img_shape.zyx, dim=(-3, -2, -1), norm='ortho')
        img_kernel = torch.fft.fftshift(img_kernel, dim=(-1, -2, -3))
        aha += torch.einsum('c z y x, d z y x->c d z y x ', img_kernel, img_kernel.conj())

    kernel_size = kernels.shape[-3] * kernels.shape[-2] * kernels.shape[-1]
    aha *= aha[0, 0].numel() / kernel_size

    # A covariance column with maximal diagonal avoids cancellation of summed coils.
    diagonal = aha.diagonal(dim1=0, dim2=1).real
    seed = diagonal.argmax(dim=-1)
    fallback = aha.gather(1, seed[None, None].expand(n_coils, 1, *img_shape.zyx)).squeeze(1)
    # A small column component breaks symmetry in the summed seed, including
    # opposite coils. Fall back to the column if the combined seed nearly cancels.
    v = aha.sum(dim=1) + 0.01 * fallback
    stable = v.norm(dim=0, keepdim=True) > 1e-3 * fallback.norm(dim=0, keepdim=True)
    v = torch.where(stable, v, fallback)
    for _ in range(n_iterations):
        v /= v.norm(dim=0).clamp_min(1e-12)
        v = torch.einsum('abzyx,bzyx->azyx', aha, v)
    max_eig = v.norm(dim=0)
    csm = v / max_eig.clamp_min(1e-12)

    # Normalize phase with respect to first channel
    reference = csm[0]
    strongest = csm.gather(0, csm.abs().argmax(dim=0, keepdim=True)).squeeze(0)
    reference = torch.where(reference.abs() > 1e-12, reference, strongest)
    phase = torch.where(reference.abs() > 1e-12, reference.conj() / reference.abs().clamp_min(1e-12), 1)
    csm = csm * phase

    csm *= max_eig > crop_threshold

    return csm
