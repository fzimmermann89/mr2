"""Upsampling by interpolation."""

from collections.abc import Sequence
from typing import Literal

import torch
from torch.nn import Module

from mr2.utils.interpolate import interpolate
from mr2.utils.reshape import normalize_indices


class Upsample(Module):
    """Upsampling by interpolation."""

    def __init__(
        self, dim: Sequence[int], scale_factor: int = 2, mode: Literal['nearest', 'linear', 'cubic'] = 'linear'
    ):
        """Initialize the upsampling layer.

        Parameters
        ----------
        dim
            Dimensions which to upsample
        scale_factor
            Factor by which to upsample
        mode
            Interpolation mode.
        """
        super().__init__()
        self.dim = tuple(dim)
        self.scale_factor = scale_factor
        self.mode = mode
        if mode not in ('nearest', 'linear', 'cubic'):
            raise ValueError("mode must be one of 'nearest', 'linear', or 'cubic'.")
        if scale_factor <= 0:
            raise ValueError('scale_factor should be positive.')
        if mode == 'cubic' and len(self.dim) != 2:
            raise ValueError('Cubic interpolation is only supported for 2D images.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample the input tensor."""
        dim = normalize_indices(x.ndim, self.dim)
        size = tuple(x.shape[d] * self.scale_factor for d in dim)
        return interpolate(x, size=size, dim=dim, mode=self.mode)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample the input tensor.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
            Upsampled tensor
        """
        return super().__call__(x)
