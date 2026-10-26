"""Cartesian GRAPPA kernel calibration and interpolation."""

from math import isfinite, prod

import torch

from mr2.utils.sliding_window import sliding_window


def grappa_calibration(
    acs: torch.Tensor, source_pattern: torch.Tensor, *, regularization: float = 0.01
) -> torch.Tensor:
    """Fit a GRAPPA kernel from fully sampled Cartesian calibration data.

    Parameters
    ----------
    acs
        Fully sampled ACS, shape ``(..., coils, kz, ky, kx)``, as returned by ``KData.acs()``.
        Leading batch dimensions are calibrated independently.
    source_pattern
        Boolean 3D neighborhood with positive odd widths. True selects an
        acquired source offset; the center is the target and must be False.
        Only complete ACS windows are used.
    regularization
        Nonnegative ridge strength, relative to the mean diagonal of ``S.H @ S``
        for source matrix S. Zero uses a minimum-norm pseudoinverse.

    Returns
    -------
        Complex kernel with shape ``(..., coils_out, coils_in, *kernel_size)``.
        Unused offsets are zero. Apply using `grappa_interpolation`. Unbatched
        kernels also work with `~mr2.operators.ConvolutionOp` using constant
        padding and correlation.
    """
    kernel_size = tuple(source_pattern.shape)
    n_spatial = len(kernel_size)
    if n_spatial != 3 or any(k % 2 != 1 for k in kernel_size):
        raise ValueError('source_pattern must have three positive odd widths (use width one for 2D data).')
    if (
        source_pattern.dtype != torch.bool
        or not source_pattern.any()
        or source_pattern.flatten()[prod(kernel_size) // 2]
    ):
        raise ValueError('source_pattern must select boolean source offsets excluding the center.')
    if not isfinite(regularization) or regularization < 0:
        raise ValueError('regularization must be finite and nonnegative.')
    n_batch = acs.ndim - n_spatial - 1
    n_coils = acs.shape[-n_spatial - 1]
    windows = sliding_window(acs, kernel_size, dim=tuple(range(-n_spatial, 0)))
    windows = windows.movedim(tuple(range(n_spatial)), tuple(range(n_batch, n_batch + n_spatial)))
    calibration = windows.reshape(*acs.shape[:n_batch], -1, n_coils, prod(kernel_size))
    targets = calibration[..., prod(kernel_size) // 2]
    sources = calibration[..., source_pattern.flatten()].flatten(start_dim=-2)
    if sources.shape[-2] < sources.shape[-1]:
        raise ValueError('Insufficient ACS: calibration must have at least as many rows as source features.')
    if regularization == 0:
        weights = torch.linalg.pinv(sources) @ targets
    else:
        gram = sources.mH @ sources
        ridge = regularization * gram.diagonal(dim1=-2, dim2=-1).real.mean(-1)
        identity = torch.eye(gram.shape[-1], device=acs.device, dtype=acs.dtype)
        weights = torch.linalg.solve(gram + ridge[..., None, None] * identity, sources.mH @ targets)
    kernel = acs.new_zeros(*acs.shape[:n_batch], n_coils, n_coils, prod(kernel_size))
    kernel[..., source_pattern.flatten()] = weights.unflatten(-2, (n_coils, -1)).movedim(-1, -3)
    return kernel.reshape(*acs.shape[:n_batch], n_coils, n_coils, *kernel_size)


def grappa_interpolation(
    kspace: torch.Tensor, kernel: torch.Tensor, target_indices: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply a calibrated Cartesian GRAPPA kernel.

    Parameters
    ----------
    kspace
        Zero-filled Cartesian grid, shape ``(..., coils, kz, ky, kx)``.
    kernel
        Kernel returned by `grappa_calibration`, shaped ``(..., coils, coils, kz, ky, kx)``,
        with batch dimensions that broadcast to those of ``kspace``. Out-of-grid sources contribute zero.
    target_indices
        Flattened spatial indices to fill, shared across batches and coils.
        Other entries are preserved exactly. If omitted, evaluate the kernel
        on the entire grid. Call separately for each source pattern.

    Returns
    -------
        Interpolated k-space with the input shape, dtype and device.
    """
    kernel_size = tuple(kernel.shape[-3:])
    data = kspace.flatten(start_dim=-3).transpose(-1, -2)
    result = data.clone()
    if target_indices is None:
        target_indices = torch.arange(data.shape[-2], device=kspace.device)
    weights = kernel.flatten(start_dim=-3).flatten(start_dim=-2).transpose(-1, -2)
    coordinates = torch.stack(
        torch.unravel_index(torch.arange(prod(kernel_size), device=kspace.device), kernel_size), dim=-1
    )
    offsets = coordinates - coordinates.new_tensor(kernel_size) // 2
    strides = offsets.new_tensor([prod(kspace.shape[-2:]), kspace.shape[-1], 1])
    for locations in target_indices.split(1024):
        positions = torch.stack(torch.unravel_index(locations, kspace.shape[-3:]), dim=-1)
        neighbors = positions[:, None, :] + offsets
        inside = ((neighbors >= 0) & (neighbors < neighbors.new_tensor(kspace.shape[-3:]))).all(-1)
        indices = (neighbors * strides).sum(-1).clamp(0, data.shape[-2] - 1)
        sources = torch.where(inside[..., None], data[..., indices, :], 0)
        sources = sources.transpose(-1, -2).flatten(start_dim=-2)
        result[..., locations, :] = sources @ weights
    return result.transpose(-1, -2).reshape(kspace.shape)
