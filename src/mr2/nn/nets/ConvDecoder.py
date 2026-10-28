"""Convolutional decoder for untrained image priors."""

import math
from collections.abc import Sequence
from functools import partial

import torch
from torch import nn
from torchnd import ConvNd

from mr2.nn.ndmodules import instanceNormND
from mr2.nn.Sequential import Sequential
from mr2.utils.interpolate import interpolate


class ConvDecoder(nn.Module):
    """Generate spatial data by progressively upsampling a latent tensor.

    Stages apply nearest interpolation, dense ``torchnd.ConvNd`` convolution,
    ReLU and affine instance normalization. A full-resolution block and linear
    output follow. This ND variant uses InstanceNorm instead of the BatchNorm
    in [DARESTANI2021]_.

    Input and output use real ``(batch, channels, *spatial)`` layout.
    Use `~mr2.nn.ComplexAsChannel` for complex output, and save network,
    optimizer and latent state to resume fitting.

    .. [DARESTANI2021] Darestani MZ, Heckel R (2021). Accelerated MRI with
       Un-trained Neural Networks. https://doi.org/10.1109/TCI.2021.3097596
    """

    def __init__(
        self,
        n_dim: int,
        n_channels_in: int,
        n_channels_out: int,
        output_shape: Sequence[int],
        n_features: int = 256,
        n_levels: int = 6,
    ) -> None:
        """Initialize the decoder.

        Parameters
        ----------
        n_dim
            Positive number of spatial dimensions to convolve over jointly.
        n_channels_in
            Number of latent channels.
        n_channels_out
            Number of real output channels.
        output_shape
            Spatial output size, with `n_dim` entries. Intermediate sizes are
            geometrically spaced between the input and this size.
        n_features
            Hidden channel count.
        n_levels
            Number of upsampling stages, followed by one full-resolution block.
        """
        super().__init__()
        if n_dim < 1:
            raise ValueError('n_dim must be positive.')

        convolution = partial(ConvNd, dim=tuple(range(-n_dim, 0)))

        if len(output_shape) != n_dim or any(size < 1 for size in output_shape):
            raise ValueError('output_shape must contain n_dim positive sizes.')
        if min(n_channels_in, n_channels_out, n_features, n_levels) < 1:
            raise ValueError('Channel counts and n_levels must be positive.')
        self.n_dim = n_dim
        self.output_shape = tuple(output_shape)
        self.n_levels = n_levels
        self.blocks = nn.ModuleList(
            Sequential(
                convolution(n_channels_in if level == 0 else n_features, n_features, 3, padding=1),
                nn.ReLU(),
                instanceNormND(n_dim)(n_features, affine=True, track_running_stats=False)
                if n_dim <= 3
                else nn.GroupNorm(n_features, n_features),
            )
            for level in range(n_levels + 1)
        )
        self.output = convolution(n_features, n_channels_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode real latent channels into real output channels."""
        if x.ndim != self.n_dim + 2:
            raise ValueError('Expected (batch, channels, *spatial) input with n_dim spatial dimensions.')
        latent_shape = x.shape[2:]
        if any(a > b for a, b in zip(latent_shape, self.output_shape, strict=True)):
            raise ValueError('The latent must fit inside output_shape.')
        dim = tuple(range(-self.n_dim, 0))
        for level, block in enumerate(self.blocks):
            if level < self.n_levels:
                size = tuple(
                    math.ceil(a * (b / a) ** ((level + 1) / self.n_levels))
                    for a, b in zip(latent_shape, self.output_shape, strict=True)
                )
                if level == self.n_levels - 1:
                    size = self.output_shape
                if level == 0 and math.prod(size) <= 1:
                    raise ValueError('The first decoder stage must contain more than one pixel for InstanceNorm.')
                x = interpolate(x, size, dim=dim, mode='nearest')
            x = block(x)
        return self.output(x)
