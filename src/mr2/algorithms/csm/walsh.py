"""Walsh method for coil sensitivity map calculation."""

import torch
from einops import einsum, reduce

from mr2.algorithms.csm.extrapolate import extrapolate_csm
from mr2.data.SpatialDimension import SpatialDimension
from mr2.utils.filters import uniform_filter
from mr2.utils.reshape import normalize_index


def walsh(
    coil_images: torch.Tensor,
    smoothing_width: SpatialDimension[int] | int,
    align_phase: bool = True,
    extrapolate: bool = False,
    *,
    pool_dim: int | None = None,
) -> torch.Tensor:
    """Calculate a coil sensitivity map (csm) using Walsh's method [WAL2000]_.

    This function computes CSMs from a set of complex coil images assuming spatially
    slowly changing sensitivity maps using Walsh's method [WAL2000]_.

    The algorithm follows these steps:

    1. **Compute Pointwise Covariance**:
       Calculate the covariance matrix of the coil images at each voxel to capture inter-coil signal relationships.

    2. **Apply Smoothing Filter**:
       Smooth the covariance matrices across spatial dimensions using a uniform filter of specified width
       to reduce noise and enforce spatial consistency.

    3. **Dominant Eigenvector Estimation via Power Iteration**:
       Perform power iterations to approximate the dominant eigenvector of the covariance matrix at each voxel,
       representing the principal component of the signal.

    4. **Normalize Sensitivity Maps**:
       Normalize the resulting eigenvectors to produce the final CSMs.

    5. **Phase Alignment (Optional)**:
       If `align_phase` is True, aligns the eigenvectors' global phase to a reference derived from the
       coil data [INA2013]_. This prevents phase singularities that otherwise cause destructive
       interference when spatially interpolating or downsampling the maps.

    This function supports one or more sets of coil images with leading dimensions. The input
    should be a tensor with dimensions `(..., coils, z, y, x)`. Prefer using
    `~mr2.data.CsmData` when sensitivity estimation should stay synchronized with MR2 data
    containers and metadata.

    Parameters
    ----------
    coil_images
        Images for each coil element, shape `(..., coils, z, y, x)`.
    smoothing_width
        Width of the smoothing filter.
    align_phase
        If True, resolve the phase ambiguity of eigenvectors relative to the data [INA2013]_.
    extrapolate
        If `True`, extrapolate the estimated CSMs into low-signal regions.
    pool_dim
        Optional leading dimension over which to sum coil covariance matrices,
        for example an echo axis. Retained as a singleton in the output. Phase
        alignment uses the first image along this axis, without complex averaging.
        Negative indices are supported. If `None`, estimate each image independently.

    Returns
    -------
    csm
        Coil sensitivity map, shape `(..., coils, z, y, x)`, with
        `pool_dim` reduced to size one when specified.

    References
    ----------
    .. [WAL2000] Walsh DO, Gmitro AF, Marcellin MW (2000) Adaptive reconstruction of phased array MR imagery. MRM 43
    .. [INA2013] Inati S, Hansen M, Kellman P (2013) A solution to the phase problem in adaptive coil combination.
       in Proceedings of the 21st Annual Meeting of ISMRM, Salt Lake City, USA, 2672.
    """
    n_power_iterations = 10
    eps = 1e-12

    if pool_dim is not None:
        pool_dim = normalize_index(coil_images.ndim, pool_dim)
        if pool_dim >= coil_images.ndim - 4:
            raise ValueError('pool_dim must index a leading dimension, not coils or space.')

    if isinstance(smoothing_width, int):
        smoothing_width = SpatialDimension(
            z=smoothing_width if coil_images.shape[-3] > 1 else 1, y=smoothing_width, x=smoothing_width
        )

    # Pointwise covariance
    images = coil_images.unsqueeze(-1) if pool_dim is None else coil_images.movedim(pool_dim, -1).unsqueeze(pool_dim)
    coil_covariance = einsum(
        images,
        images.conj(),
        '... coil_out z y x pool, ... coil_in z y x pool -> ... coil_out coil_in z y x',
    )

    # Smooth covariance
    coil_covariance = uniform_filter(coil_covariance, width=smoothing_width.zyx, dim=(-3, -2, -1))

    # Power iterations for dominant eigenvector
    v = coil_covariance.sum(dim=-4)
    for _ in range(n_power_iterations):
        v = v / (v.norm(dim=-4, keepdim=True) + eps)
        v = einsum(coil_covariance, v, '... coil_out coil_in z y x, ... coil_in z y x -> ... coil_out z y x')

    csm = v / (v.norm(dim=-4, keepdim=True) + eps)

    if align_phase:
        phase_reference = images[..., 0]
        d_sum = reduce(phase_reference, '... coils z y x -> ... coils', 'sum')
        d_sum = d_sum / (d_sum.norm(dim=-1, keepdim=True) + eps)
        phase_map = einsum(d_sum.conj(), csm, '... coils, ... coils z y x -> ... z y x').angle()
        csm = csm * torch.exp(-1j * phase_map).unsqueeze(-4)

    if extrapolate:
        energy = reduce(images.abs().square(), '... coils z y x pool -> ... z y x', 'sum')
        csm = extrapolate_csm(csm, energy.sqrt(), smoothing_width)

    norm = csm.norm(dim=-4, keepdim=True)
    fallback = torch.zeros_like(csm)
    fallback[..., 0, :, :, :] = 1
    valid = torch.isfinite(norm) & (norm > eps)
    csm = csm / norm.clamp_min(eps)
    return torch.where(valid.expand_as(csm), csm, fallback)
