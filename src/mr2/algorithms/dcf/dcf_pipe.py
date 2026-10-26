"""Gridding-based Pipe density compensation."""

from collections.abc import Sequence
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
    r"""Calculate positive real Pipe density compensation for one trajectory.

    Iterate :math:`w \leftarrow w / |G^H G w|`, starting from ones, where
    :math:`G` spreads samples onto a periodic Cartesian grid. Only spreading
    and interpolation are used, without FFTs or deconvolution.

    Parameters
    ----------
    traj
        Coordinates in encoded-grid units, shaped `(directions, *samples)`.
        One to three directions are supported. There are no batch or coil axes;
        use `~mr2.data.DcfData.from_traj_pipe` for batched trajectories.
    encoding_matrix
        Encoded matrix sizes in the same order as the coordinate directions.
        Coordinates have period equal to these sizes.
    n_iterations
        Number of reciprocal iterations, at least one.
    oversampling
        Grid size relative to the encoding matrix, at least one. Each grid axis
        has `max(32, ceil(oversampling * size))` points. The minimum accommodates
        the kernel support. FINUFFT does not oversample a spread-only grid.

    Returns
    -------
    torch.Tensor
        Real weights shaped `(*samples)`, on the input device and with its dtype.
        Their sum is the product of `encoding_matrix`, giving a unit central PSF
        value with the corresponding orthonormal Fourier operator. This global
        convention does not estimate the volume of a partially sampled region.

    Notes
    -----
    FINUFFT's kernel is fixed by `eps=1e-6` and `upsampfac=2`, independently of
    the explicit grid oversampling. Both calls use `modeord=0` and opposite
    Fourier signs; signs have no effect on the real spreading kernel. Denominators
    are floored at the smallest positive normal value of the real dtype.
    Failed or nonpositive iterations raise an error instead of replacing weights.
    CPU and CUDA kernels can give slightly different normalized weights.
    Kernel sampling can also cause a small ripple on Cartesian trajectories
    when the grid size is not a multiple of the encoded size.

    Trajectory gradients are not supported: the backend wrapper's Fourier
    coordinate derivative is not the derivative of this gridding kernel.
    DCF weights compensate sampling density; they are not inverse noise variances.

    References
    ----------
    .. [1] Pipe and Menon, Sampling density compensation in MRI: Rationale and an
       iterative numerical solution. Magn. Reson. Med. 41 (1999), 179-186.
    """
    sizes = tuple(encoding_matrix)
    if not 1 <= len(sizes) <= 3 or traj.ndim < 2 or traj.shape[0] != len(sizes):
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
    points = traj.flatten(1) * traj.new_tensor([2 * torch.pi / size for size in sizes])[:, None]
    points = ((points + torch.pi).remainder(2 * torch.pi) - torch.pi).contiguous()
    options = {'eps': 1e-6, 'upsampfac': 2.0, 'modeord': 0}
    options['gpu_spreadinterponly' if traj.is_cuda else 'spreadinterponly'] = 1
    weight = torch.ones(points.shape[-1], dtype=traj.dtype, device=traj.device)
    for _ in range(n_iterations):
        grid = finufft_type1(
            points,
            weight.to(torch.complex128 if traj.dtype == torch.float64 else torch.complex64),
            grid_shape,
            isign=1,
            **options,
        )
        denominator = finufft_type2(points, grid, isign=-1, **options).abs()
        weight = weight / denominator.clamp_min(torch.finfo(weight.dtype).tiny)
    weight = weight * (prod(sizes) / weight.sum())
    if not (weight.isfinite() & (weight > 0)).all():
        raise RuntimeError('Pipe density compensation did not produce finite positive weights.')
    return weight.reshape(traj.shape[1:])
