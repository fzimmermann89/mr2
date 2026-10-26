"""Gridding-based Pipe density compensation."""

from collections.abc import Sequence
from itertools import product
from math import ceil, isfinite, prod
from typing import cast

import torch
from pytorch_finufft.functional import finufft_type1, finufft_type2


@torch.no_grad()
def dcf_pipe(
    traj: torch.Tensor,
    encoding_matrix: Sequence[int],
    n_iterations: int = 10,
    oversampling: float = 2.0,
) -> torch.Tensor:
    r"""Calculate positive real Pipe density compensation.

    Iterate :math:`w \leftarrow w / |G^H G w|`, starting from ones, where
    :math:`G` spreads samples onto a periodic Cartesian grid. Only spreading
    and interpolation are used, without FFTs or deconvolution.

    Parameters
    ----------
    traj
        Coordinates in encoded-grid units, shaped `(..., directions, k2, k1, k0)`.
        Leading dimensions describe independent trajectories. One to three
        coordinate directions are supported. Unbatched inputs can omit leading
        singleton sample axes, for example `(directions, k1, k0)`.
    encoding_matrix
        Encoded matrix sizes in the same order as the coordinate directions.
        Coordinates have period equal to these sizes. Every supplied direction
        contributes to the normalization, including constant directions.
    n_iterations
        Number of reciprocal iterations, at least one.
    oversampling
        Grid size relative to the encoding matrix, at least one. Each grid axis
        has `max(32, ceil(oversampling * size))` points. The minimum accommodates
        the kernel support. FINUFFT does not oversample a spread-only grid.

    Returns
    -------
        Real weights shaped `(..., k2, k1, k0)`, on the input device and with its
        dtype. Omitted singleton sample axes stay omitted. Each trajectory's
        weights sum to the product of `encoding_matrix`. This normalization does
        not estimate the volume of a partially sampled region.

    Notes
    -----
    Trajectory gradients are not supported by the FINUFFT spread-only backend.

    References
    ----------
    .. [1] Pipe and Menon, Sampling density compensation in MRI: Rationale and an
       iterative numerical solution. Magn. Reson. Med. 41 (1999), 179-186.
    """
    sizes = tuple(encoding_matrix)
    direction_dim = -4 if traj.ndim >= 4 else 0
    if not 1 <= len(sizes) <= 3 or traj.ndim < 2 or traj.shape[direction_dim] != len(sizes):
        raise ValueError('Expected one to three coordinate directions matching encoding_matrix.')
    if any(size < 1 for size in sizes):
        raise ValueError('encoding_matrix sizes must be positive.')
    if n_iterations < 1 or not isfinite(oversampling) or oversampling < 1:
        raise ValueError('n_iterations must be positive and oversampling must be finite and at least one.')
    if traj.dtype not in (torch.float32, torch.float64) or not traj.isfinite().all() or traj.numel() == 0:
        raise ValueError('traj must contain finite float32 or float64 coordinates.')

    grid_shape = cast(
        tuple[int] | tuple[int, int] | tuple[int, int, int],
        tuple(max(32, ceil(oversampling * size)) for size in sizes),
    )
    output_shape = (*traj.shape[:-4], *traj.shape[-3:]) if traj.ndim >= 4 else traj.shape[1:]
    while traj.ndim < 4:
        traj = traj.unsqueeze(1)
    batch_shape = traj.shape[:-4]
    sample_shape = traj.shape[-3:]
    options = {'eps': 1e-6, 'upsampfac': 2.0, 'modeord': 0}
    options['gpu_spreadinterponly' if traj.is_cuda else 'spreadinterponly'] = 1
    scale = traj.new_tensor([2 * torch.pi / size for size in sizes])[:, None]
    complex_dtype = torch.complex128 if traj.dtype == torch.float64 else torch.complex64
    results = []
    for index in product(*(range(size) for size in batch_shape)):
        # FINUFFT takes one point list per geometry, with all sample axes combined.
        points = traj[index].flatten(1) * scale
        points = ((points + torch.pi).remainder(2 * torch.pi) - torch.pi).contiguous()
        values = torch.ones(points.shape[-1], dtype=complex_dtype, device=traj.device)
        weight = values.real
        for _ in range(n_iterations):
            grid = finufft_type1(points, values, grid_shape, isign=1, **options)
            denominator = finufft_type2(points, grid, isign=-1, **options).abs()
            weight.div_(denominator.clamp_min(torch.finfo(weight.dtype).tiny))
        weight.mul_(prod(sizes) / weight.sum())
        if not (weight.isfinite() & (weight > 0)).all():
            raise RuntimeError('Pipe density compensation did not produce finite positive weights.')
        results.append(weight.reshape(sample_shape))
    return torch.stack(results).reshape(output_shape)
