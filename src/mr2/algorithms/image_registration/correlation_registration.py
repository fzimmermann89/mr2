"""Correlation-based image registration."""

import torch

from mr2.data.SpatialDimension import SpatialDimension
from mr2.utils.filters import gaussian_filter
from mr2.utils.pad_or_crop import pad_or_crop


def correlation_registration(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    smooth_mask_sigma: float = 1.5,
    search_mask: torch.Tensor | None = None,
) -> SpatialDimension[torch.Tensor]:
    """Estimate a 2D or 3D shift using FFT-based normalized cross-correlation.

    Parameters
    ----------
    fixed
        Fixed image with shape ``(*batch, channels, z, y, x)``.
    moving
        Moving image with shape ``(*batch, channels, z, y, x)``.
        Complex-valued inputs are registered phase-sensitively; pass magnitude
        images to ignore phase.
    mask
        Optional mask/weight tensor. Outside values are suppressed in both fixed
        and moving. The mask is smoothed before it is applied.
    smooth_mask_sigma
        Standard deviation of Gaussian mask smoothing in voxel units.
    search_mask
        Optional boolean mask restricting the allowed shifts in shift space
        with shape ``(*batch, 1, z-shift, y-shift, x-shift)``.
        This mask is defined in shift space with zero shift at the center.
        Values outside the dimensions of the mask are treated as false / forbidden.
        Example: torch.ones(1,1,10,10,10) would allow shifts in the range [-5, 4] in each dimension.

    Returns
    -------
        Estimated voxel shift to apply to ``moving`` to align it to ``fixed``.
        For 2D inputs (``z == 1``), the returned z-shift is zero.
    """
    eps = 1e-8

    if fixed.shape != moving.shape:
        raise ValueError(f'fixed and moving must have same shape, got {fixed.shape=} and {moving.shape=}.')
    if fixed.ndim < 5:
        raise ValueError(f'Expected at least 5 dimensions ``(*batch, channels, z, y, x)``, got {fixed.ndim}.')
    if smooth_mask_sigma <= 0:
        raise ValueError(f'smooth_mask_sigma must be positive, got {smooth_mask_sigma}.')

    dim = 2 if fixed.shape[-3] == 1 else 3
    spatial_dims = tuple(range(-dim, 0))
    batch_shape = fixed.shape[:-4]

    fixed_flat = fixed.flatten(end_dim=-5)
    moving_flat = moving.flatten(end_dim=-5)

    spatial_shape = fixed_flat.shape[-dim:]
    crop_shape = fixed_flat.shape[-3:]
    padded_shape = tuple(2 * size for size in spatial_shape)
    normalization_dims = tuple(range(1, fixed_flat.ndim))

    if mask is None:
        mask_flat = torch.ones_like(fixed_flat[:, :1], dtype=torch.float32)
    else:
        mask_flat = mask.flatten(end_dim=-5)
        if mask_flat.shape[-3:] != fixed_flat.shape[-3:]:
            raise ValueError(
                'mask must have the same spatial shape as fixed and moving, got '
                f'{mask.shape[-3:]=} and {fixed.shape[-3:]=}.'
            )
        mask_flat = torch.broadcast_to(mask_flat, fixed_flat[:, :1].shape).to(dtype=torch.float32)
        mask_flat = gaussian_filter(mask_flat, smooth_mask_sigma, dim=spatial_dims)
        mask_flat = mask_flat / mask_flat.amax(dim=spatial_dims, keepdim=True).clamp_min(eps)
        mask_flat = mask_flat.clamp_(0.0, 1.0)

    mask_per_channel = mask_flat.expand_as(fixed_flat)

    # Joint centering removes the common DC component before correlation.
    weight_sum = (2.0 * mask_per_channel.sum(dim=normalization_dims, keepdim=True)).clamp_min(eps)
    joint_mean = (
        (fixed_flat * mask_per_channel).sum(dim=normalization_dims, keepdim=True)
        + (moving_flat * mask_per_channel).sum(dim=normalization_dims, keepdim=True)
    ) / weight_sum

    fixed_weighted = (fixed_flat - joint_mean) * mask_per_channel
    moving_weighted = (moving_flat - joint_mean) * mask_per_channel

    # Shared scaling keeps both images on the same numerical scale.
    joint_scale = torch.maximum(
        fixed_weighted.abs().amax(dim=normalization_dims, keepdim=True),
        moving_weighted.abs().amax(dim=normalization_dims, keepdim=True),
    ).clamp_min(eps)

    fixed_weighted = fixed_weighted / joint_scale
    moving_weighted = moving_weighted / joint_scale

    mask_fft = torch.fft.fftn(mask_flat, s=padded_shape, dim=spatial_dims)

    correlation = torch.fft.ifftn(
        (
            torch.fft.fftn(fixed_weighted, s=padded_shape, dim=spatial_dims)
            * torch.fft.fftn(moving_weighted, s=padded_shape, dim=spatial_dims).conj()
        ).sum(dim=1, keepdim=True),
        dim=spatial_dims,
    ).real
    correlation = torch.fft.fftshift(correlation, dim=spatial_dims)
    correlation = pad_or_crop(correlation, crop_shape)

    if search_mask is not None:
        if search_mask.dtype != torch.bool:
            raise ValueError(f'search_mask must be boolean, got {search_mask.dtype}.')
        if search_mask.ndim < 5:
            raise ValueError(
                f'Expected search_mask with at least 5 dimensions ``(*batch, 1, z, y, x)``, got {search_mask.ndim}.'
            )
        if search_mask.shape[-4] != 1:
            raise ValueError(f'search_mask must have a singleton channel dimension, got {search_mask.shape[-4]=}.')

        search_mask_flat = pad_or_crop(search_mask.flatten(end_dim=-5), crop_shape)
        correlation = torch.where(search_mask_flat, correlation, -torch.inf)

    # Local energy normalization gives NCC instead of raw cross-correlation.
    fixed_energy = torch.fft.ifftn(
        torch.fft.fftn(fixed_weighted.abs().square().sum(dim=1, keepdim=True), s=padded_shape, dim=spatial_dims)
        * mask_fft.conj(),
        dim=spatial_dims,
    ).real
    fixed_energy = torch.fft.fftshift(fixed_energy, dim=spatial_dims)
    fixed_energy = pad_or_crop(fixed_energy, crop_shape)

    moving_energy = torch.fft.ifftn(
        mask_fft
        * torch.fft.fftn(
            moving_weighted.abs().square().sum(dim=1, keepdim=True), s=padded_shape, dim=spatial_dims
        ).conj(),
        dim=spatial_dims,
    ).real
    moving_energy = torch.fft.fftshift(moving_energy, dim=spatial_dims)
    moving_energy = pad_or_crop(moving_energy, crop_shape)

    # Suppress shifts with too little support.
    overlap = torch.fft.ifftn(mask_fft * mask_fft.conj(), dim=spatial_dims).real
    overlap = torch.fft.fftshift(overlap, dim=spatial_dims)
    overlap = pad_or_crop(overlap, crop_shape)

    denominator = (fixed_energy.clamp_min(0.0) * moving_energy.clamp_min(0.0)).sqrt()
    correlation = correlation / denominator.clamp_min(eps)

    min_overlap = 0.5 * overlap.amax(dim=spatial_dims, keepdim=True)
    correlation = torch.where(overlap >= min_overlap, correlation, -torch.inf)
    correlation = correlation.squeeze(1)

    if dim == 2:
        correlation = correlation.squeeze(-3)

    # After fftshift + center crop, zero shift is at the correlation center.
    max_index = correlation.reshape(correlation.shape[0], -1).argmax(dim=-1)
    peak_index = torch.stack(torch.unravel_index(max_index, spatial_shape), dim=-1)

    # Fit q(t) through q(-1)=corr_minus, q(0)=corr, q(1)=corr_plus.
    # Vertex offset is 0.5 * (corr_minus - corr_plus) / (corr_minus - 2*c0 + corr_plus).
    # Only finite, non-boundary triplets are refined.
    batch_index = torch.arange(correlation.shape[0], device=correlation.device)
    corr = correlation[(batch_index, *peak_index.unbind(-1))]

    if not bool(torch.isfinite(corr).all().item()):
        raise ValueError('No finite correlation peak found.')

    center_index = peak_index.new_tensor([size // 2 for size in spatial_shape])
    peak_shift = (peak_index - center_index).float()

    for axis, size in enumerate(spatial_shape):
        index_minus = peak_index.clone()
        index_plus = peak_index.clone()
        index_minus[:, axis] = (index_minus[:, axis] - 1).clamp_min(0)
        index_plus[:, axis] = (index_plus[:, axis] + 1).clamp_max(size - 1)
        corr_minus = correlation[(batch_index, *index_minus.unbind(-1))]
        corr_plus = correlation[(batch_index, *index_plus.unbind(-1))]
        inside = (peak_index[:, axis] > 0) & (peak_index[:, axis] < size - 1)
        finite = inside & torch.isfinite(corr_minus) & torch.isfinite(corr_plus)
        curvature = corr_minus - 2.0 * corr + corr_plus
        usable = finite & torch.isfinite(curvature) & (curvature.abs() > eps)
        safe_curvature = torch.where(usable, curvature, torch.ones_like(curvature))
        delta = 0.5 * (corr_minus - corr_plus) / safe_curvature
        delta = torch.where(usable, delta.clamp(-0.5, 0.5), torch.zeros_like(delta))
        peak_shift[:, axis] = peak_shift[:, axis] + delta

    peak_shift = peak_shift.unflatten(0, batch_shape)
    if dim == 2:
        z = torch.zeros_like(peak_shift[..., :1])
        peak_shift = torch.cat((z, peak_shift), dim=-1)
    return SpatialDimension.from_array_zyx(peak_shift)
