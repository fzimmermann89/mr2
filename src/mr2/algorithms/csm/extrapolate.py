"""CSM extrapolation utilities."""

from math import ceil

import torch

from mr2.data.SpatialDimension import SpatialDimension
from mr2.utils.filters import uniform_filter


def extrapolate_csm(
    csm: torch.Tensor,
    confidence: torch.Tensor,
    smoothing_width: SpatialDimension[int],
) -> torch.Tensor:
    """Extrapolate CSMs from high-signal regions by normalized convolution.

    Parameters
    ----------
    csm
        Coil sensitivity maps with shape `(..., coils, z, y, x)`.
    confidence
        Confidence map with shape `(..., z, y, x)`.
    smoothing_width
        Spatial width used for the normalized convolution kernel.

    Returns
    -------
        Extrapolated coil sensitivity maps with the same shape as `csm`.
    """
    eps = 1e-12
    scale = confidence.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(eps)
    confidence = (confidence / scale).clamp(min=0, max=1)
    valid = confidence > 1e-3

    width = smoothing_width.zyx
    n_iterations = min(10, max(1, ceil(max(csm.shape[-3:]) / max(width))))

    valid = valid.unsqueeze(-4)
    filled = torch.where(valid, csm, torch.zeros_like(csm))

    for _ in range(n_iterations):
        confidence_filtered = uniform_filter(confidence, width=width, dim=(-3, -2, -1)).clamp(max=1)
        filled_filtered = uniform_filter(
            filled * confidence.unsqueeze(-4),
            width=width,
            dim=(-3, -2, -1),
        )
        filled_filtered = filled_filtered / confidence_filtered.unsqueeze(-4).clamp_min(eps)
        filled = torch.where(valid, csm, filled_filtered)
        confidence = confidence_filtered

    return torch.where(torch.isfinite(filled), filled, 0.0)
