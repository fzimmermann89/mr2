"""Convolutional decoder for untrained image priors."""

import math
from collections.abc import Sequence

import torch
from torch import nn

from mr2.nn.ndmodules import convND, instanceNormND
from mr2.nn.Sequential import Sequential
from mr2.utils.interpolate import interpolate


class ConvDecoder(nn.Module):
    """Generate spatial data by progressively upsampling a latent tensor.

    Each stage applies nearest-neighbor interpolation, a 3-tap convolution,
    ReLU and stateless affine instance normalization, in that order. A further
    convolution/ReLU/normalization block at full resolution precedes a linear
    1-tap output convolution. This variant uses instance normalization instead
    of the BatchNorm in [DARESTANI2021]_.

    Input and output use real channels in ``(batch, channels, *spatial)`` layout.
    Use `~mr2.nn.ComplexAsChannel` for complex output. The caller owns the latent
    and optimizer; save these together with ``state_dict()`` to resume fitting.
    Loading weights never reinitializes them or changes the latent.

    References
    ----------
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
            Number of spatial dimensions, 1, 2 or 3.
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
        convolution = convND(n_dim)
        normalization = instanceNormND(n_dim)
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
                normalization(n_features, affine=True, track_running_stats=False),
            )
            for level in range(n_levels + 1)
        )
        self.output = convolution(n_features, n_channels_out, 1)

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        """Reset weights with PyTorch defaults, optionally using a local generator.

        Parameters
        ----------
        generator
            Random generator on the parameters' device. If provided, global RNG
            state is unchanged. Useful after construction on meta and `to_empty`.
        """
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Conv1d | nn.Conv2d | nn.Conv3d):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
                    if module.bias is not None:
                        bound = 1 / math.sqrt(module.weight[0].numel())
                        nn.init.uniform_(module.bias, -bound, bound, generator=generator)
                elif isinstance(module, nn.InstanceNorm1d | nn.InstanceNorm2d | nn.InstanceNorm3d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

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
