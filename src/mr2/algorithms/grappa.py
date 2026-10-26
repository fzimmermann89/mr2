"""Pattern-specific Cartesian GRAPPA interpolation."""

from math import isfinite, prod

import torch
from torch.nn.functional import pad

from mr2.utils.sliding_window import sliding_window


def _patches(data: torch.Tensor, kernel_size: tuple[int, ...]) -> torch.Tensor:
    """Lift to (..., positions, coils, neighborhood), keeping batches independent."""
    n_spatial = len(kernel_size)
    n_batch = data.ndim - n_spatial - 1
    windows = sliding_window(data, kernel_size, dim=tuple(range(-n_spatial, 0)))
    windows = windows.movedim(tuple(range(n_spatial)), tuple(range(n_batch, n_batch + n_spatial)))
    return windows.reshape(*data.shape[:n_batch], -1, data.shape[-n_spatial - 1], prod(kernel_size))


def grappa(
    kspace: torch.Tensor,
    acs: torch.Tensor,
    acquired: torch.Tensor,
    kernel_size: tuple[int, ...],
    *,
    regularization: float = 0.01,
) -> torch.Tensor:
    """Calibrate GRAPPA kernels and fill missing Cartesian k-space samples.

    Parameters
    ----------
    kspace
        Cartesian grid with shape ``(..., coils, ky, kx)`` for a 2D kernel or
        ``(..., coils, kz, ky, kx)`` for a 3D kernel. Missing entries are ignored.
        Must use complex64 or complex128. No normalization is applied.
    acs
        Fully sampled rectangular calibration region with the same coil basis,
        frequency spacing, dtype and device as ``kspace``. Its batch dimensions
        must broadcast to those of ``kspace``. Each batch is calibrated separately.
        For 2D interpolation of KData tensors, move kz before the coil axis.
        A 3D kernel uses the standard KData axis order directly.
    acquired
        Boolean mask with exactly the spatial shape of ``kspace``, shared by
        all coils and batches. True denotes an acquired sample, including real
        zero measurements. Different masks across batches require separate calls.
    kernel_size
        Two or three positive odd widths, including the target at the center.
        Only acquired neighbors inside this window are sources; the center is
        never a source. Choose widths large enough to bridge missing lines.
    regularization
        Nonnegative dimensionless ridge strength. For source matrix S, solve
        ``(S.H @ S + lambda * I) W = S.H @ T`` with
        ``lambda = regularization * ||S||_F**2 / S.shape[-1]``.
        Zero selects the minimum-norm least-squares solution via a pseudoinverse.

    Returns
    -------
    torch.Tensor
        Filled grid with the input shape, dtype and device. Acquired samples are
        preserved exactly. Inputs are not modified.

    Notes
    -----
    Each distinct local mask around a missing sample gets its own calibrated
    kernel. Uniform Cartesian line undersampling, with optional acquired ACS
    lines, is supported in 2D and 3D. Other Cartesian masks are accepted only
    when every hole has acquired neighbors and enough calibration equations;
    this does not guarantee accurate interpolation for arbitrary masks.
    Non-Cartesian samples must not be passed to this function.

    Boundaries do not wrap: out-of-grid neighbors are excluded and the reduced
    pattern is calibrated separately. Calibration uses only complete ACS
    windows, without padding. Synthesized values are never reused as sources.
    Calibration neighborhoods are lifted in memory. Interpolation gathers only
    acquired sources in bounded chunks to avoid lifting the full input grid.

    Raises
    ------
    ValueError
        If shapes, dtype, device or parameters are incompatible, a hole has no
        acquired neighbors, a calibration matrix has fewer rows than columns,
        or a batch has no calibration source energy.

    References
    ----------
    .. [1] Griswold et al. Generalized autocalibrating partially parallel
       acquisitions (GRAPPA). Magnetic Resonance in Medicine 47, 1202-1210
       (2002). https://doi.org/10.1002/mrm.10171
    """
    if len(kernel_size) not in (2, 3) or any(
        isinstance(k, bool) or not isinstance(k, int) or k < 1 or k % 2 == 0 for k in kernel_size
    ):
        raise ValueError('kernel_size must contain two or three positive odd integers.')
    if not isfinite(regularization) or regularization < 0:
        raise ValueError('regularization must be finite and nonnegative.')
    n_spatial = len(kernel_size)
    if kspace.ndim < n_spatial + 1 or acs.ndim < n_spatial + 1 or not kspace.numel() or not acs.numel():
        raise ValueError('kspace and acs must have nonempty coil and spatial dimensions.')
    if kspace.dtype not in (torch.complex64, torch.complex128) or acs.dtype != kspace.dtype:
        raise ValueError('kspace and acs must have the same complex64 or complex128 dtype.')
    if acs.device != kspace.device or acquired.device != kspace.device:
        raise ValueError('kspace, acs and acquired must be on the same device.')
    if acquired.dtype != torch.bool or acquired.shape != kspace.shape[-n_spatial:]:
        raise ValueError('acquired must be a boolean mask with exactly the spatial shape of kspace.')
    n_coils = kspace.shape[-n_spatial - 1]
    if acs.shape[-n_spatial - 1] != n_coils:
        raise ValueError('kspace and acs must have the same number of coils.')
    batch_shape = kspace.shape[: -n_spatial - 1]
    try:
        if torch.broadcast_shapes(acs.shape[: -n_spatial - 1], batch_shape) != batch_shape:
            raise ValueError('acs batch dimensions must broadcast to the kspace batch dimensions.')
    except RuntimeError as error:
        raise ValueError('acs batch dimensions must broadcast to the kspace batch dimensions.') from error
    if not torch.isfinite(acs).all() or not torch.isfinite(kspace[..., acquired]).all():
        raise ValueError('acs and acquired samples must be finite.')
    if acquired.all():
        return kspace.clone()
    if any(a < k for a, k in zip(acs.shape[-n_spatial:], kernel_size, strict=True)):
        raise ValueError('acs spatial dimensions must be at least kernel_size.')

    padding = tuple(p for k in reversed(kernel_size) for p in (k // 2, k // 2))
    mask_patches = sliding_window(pad(acquired, padding), kernel_size).reshape(-1, prod(kernel_size))
    missing = ~acquired.flatten()
    patterns, assignments = torch.unique(mask_patches[missing], dim=0, return_inverse=True)
    calibration = _patches(acs, kernel_size)
    targets = calibration[..., prod(kernel_size) // 2]
    data = kspace.flatten(start_dim=-n_spatial).transpose(-1, -2)
    result = data.clone()
    # Offsets use the original grid strides; the mask excludes out-of-grid sources.
    coordinates = torch.unravel_index(torch.arange(prod(kernel_size), device=kspace.device), kernel_size)
    offsets = torch.stack(
        [
            (coordinate - width // 2) * prod(acquired.shape[axis + 1 :])
            for axis, (coordinate, width) in enumerate(zip(coordinates, kernel_size, strict=True))
        ]
    ).sum(0)
    missing_indices = missing.nonzero().flatten()
    for index, pattern in enumerate(patterns):
        if not pattern.any():
            raise ValueError('Every missing sample must have acquired neighbors inside kernel_size.')
        sources = calibration[..., pattern].flatten(start_dim=-2)
        if sources.shape[-2] < sources.shape[-1]:
            raise ValueError('Insufficient ACS: calibration must have at least as many rows as source features.')
        energy = sources.abs().square().sum(dim=(-2, -1)) / sources.shape[-1]
        if (energy == 0).any():
            raise ValueError('Every batch must have nonzero calibration source energy.')
        if regularization == 0:
            weights = torch.linalg.pinv(sources) @ targets
        else:
            gram = sources.mH @ sources
            identity = torch.eye(gram.shape[-1], dtype=kspace.dtype, device=kspace.device)
            weights = torch.linalg.solve(
                gram + (regularization * energy)[..., None, None] * identity, sources.mH @ targets
            )
        for locations in missing_indices[assignments == index].split(1024):
            source_indices = locations[:, None] + offsets[pattern]
            sources = data[..., source_indices, :].transpose(-1, -2).flatten(start_dim=-2)
            result[..., locations, :] = sources @ weights
    return result.transpose(-1, -2).reshape(kspace.shape)
