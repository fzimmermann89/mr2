"""Convolutional attention for imaging transformers."""

from typing import Literal

import torch
from einops import rearrange
from torch.nn import Module, Parameter
from torch.nn.functional import normalize

from mr2.nn.ndmodules import convND
from mr2.nn.PermutedBlock import PermutedBlock


class ConvolutionalAttention(Module):
    """Temporal or patch attention with spatial convolutional projections.

    Implements the attention used by the released SNRAware imaging transformers [XUE25]_.
    Spatial relative-position bias is added after softmax, following the official
    implementation [SNRAWARE]_ rather than the usual logit bias.

    References
    ----------
    .. [XUE25] Xue et al. SNRAware: Improved Deep Learning MRI Denoising with SNR
       Unit Training and G-factor Map Augmentation. https://arxiv.org/abs/2503.18162
    .. [SNRAWARE] Microsoft SNRAware. https://github.com/microsoft/SNRAware
    """

    relative_position_index: torch.Tensor

    def __init__(
        self,
        n_channels: int,
        n_heads: int,
        image_size: int,
        mode: Literal['temporal', 'local', 'global'],
    ) -> None:
        """Initialize convolutional attention.

        Parameters
        ----------
        n_channels
            Number of input and output channels.
        n_heads
            Number of heads. Must divide the channel count.
        image_size
            Height and width of the input. Must be divisible by eight.
        mode
            Attend across frames, four patches within each local window, or all
            spatial patches within each frame.
        """
        super().__init__()
        if n_heads <= 0 or n_channels % n_heads:
            raise ValueError('n_heads must be positive and divide n_channels.')
        if image_size < 8 or image_size % 8:
            raise ValueError('image_size must be a positive multiple of eight.')
        if mode not in ('temporal', 'local', 'global'):
            raise ValueError(f'Unknown attention mode: {mode}')
        self.n_heads = n_heads
        self.image_size = image_size
        self.mode = mode

        def projection() -> Module:
            return PermutedBlock((-2, -1), convND(2)(n_channels, n_channels, 3, padding=1, padding_mode='reflect'))

        self.to_q = projection()
        self.to_k = projection()
        self.to_v = projection()
        self.to_out = projection()
        if mode != 'temporal':
            grid = 2 if mode == 'local' else image_size // 4
            coords = torch.stack(torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')).flatten(1)
            relative = coords[:, :, None] - coords[:, None, :] + grid - 1
            self.register_buffer(
                'relative_position_index', relative[0] * (2 * grid - 1) + relative[1], persistent=False
            )
            self.relative_position_bias_table = Parameter(torch.empty((2 * grid - 1) ** 2, n_heads))
            torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        else:
            self.register_parameter('relative_position_bias_table', None)

    def _pack(self, x: torch.Tensor) -> torch.Tensor:
        match self.mode:
            case 'temporal':
                return rearrange(
                    x,
                    'batch (heads channels) frames height width -> batch heads frames (channels height width)',
                    heads=self.n_heads,
                )
            case 'local':
                x = rearrange(
                    x,
                    'batch channels frames (window_rows patch_rows patch_h) (window_cols patch_cols patch_w) '
                    '-> batch frames (window_rows window_cols) (patch_rows patch_cols) (patch_h patch_w channels)',
                    patch_rows=2,
                    patch_cols=2,
                    patch_h=4,
                    patch_w=4,
                )
                return rearrange(
                    x,
                    'batch frames windows sequence (heads features) -> batch frames windows heads sequence features',
                    heads=self.n_heads,
                )
            case 'global':
                x = rearrange(
                    x,
                    'batch channels frames (window_rows patch_h) (window_cols patch_w) '
                    '-> batch frames 1 (window_rows window_cols) (patch_h patch_w channels)',
                    patch_h=4,
                    patch_w=4,
                )
                return rearrange(
                    x,
                    'batch frames windows sequence (heads features) -> batch frames windows heads sequence features',
                    heads=self.n_heads,
                )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolutional attention.

        Parameters
        ----------
        x
            Real input of shape ``(batch, channels, frames, image_size, image_size)``.

        Returns
        -------
            Features with the same shape as the input.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention to a real tensor of shape ``(batch, channels, frames, height, width)``."""
        if x.ndim != 5 or x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f'Expected (batch, channels, frames, {self.image_size}, {self.image_size}).')
        q, k, v = (self._pack(proj(x)) for proj in (self.to_q, self.to_k, self.to_v))
        weights = (normalize(q, dim=-1) @ normalize(k, dim=-1).transpose(-2, -1)).softmax(-1)
        if self.relative_position_bias_table is not None:
            bias = self.relative_position_bias_table[self.relative_position_index].permute(2, 0, 1)
            weights = weights + bias
        y = weights @ v
        match self.mode:
            case 'temporal':
                y = rearrange(
                    y,
                    'batch heads frames (channels height width) -> batch (heads channels) frames height width',
                    height=self.image_size,
                    width=self.image_size,
                )
            case 'local' | 'global':
                patches_per_window = 2 if self.mode == 'local' else self.image_size // 4
                windows_per_axis = self.image_size // (4 * patches_per_window)
                y = rearrange(
                    y, 'batch frames windows heads sequence features -> batch frames windows sequence (heads features)'
                )
                y = rearrange(
                    y,
                    'batch frames (window_rows window_cols) (patch_rows patch_cols) (patch_h patch_w channels) '
                    '-> batch channels frames (window_rows patch_rows patch_h) (window_cols patch_cols patch_w)',
                    window_rows=windows_per_axis,
                    window_cols=windows_per_axis,
                    patch_rows=patches_per_window,
                    patch_cols=patches_per_window,
                    patch_h=4,
                    patch_w=4,
                )
        return self.to_out(y)
