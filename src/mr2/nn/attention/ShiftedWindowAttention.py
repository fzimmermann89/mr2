"""Shifted Window Attention."""

import warnings
from types import EllipsisType

import torch
from einops import rearrange
from torch.nn import Linear, Module

from mr2.utils.reshape import ravel_multi_index
from mr2.utils.sliding_window import sliding_window

_INDUCTOR_SOFTMAX_WARNING = (
    r'\s*Online softmax is disabled on the fly since Inductor decides to\s+'
    r'split the reduction\. Cut an issue to PyTorch if this is an\s+'
    r'important use case and you want to speed it up with online\s+softmax\.\s*'
)
# Keep warning-state changes outside forward so they are not traced by torch.compile.
warnings.filterwarnings(
    'ignore', message=_INDUCTOR_SOFTMAX_WARNING, category=UserWarning, module=r'torch\._inductor\.lowering$'
)


class ShiftedWindowAttention(Module):
    """Shifted Window Attention.

    (Shifted) Window Attention calculates attention over windows of the input.
    It was introduced in Swin Transformer [SWIN]_ and is used in Uformer.

    References
    ----------
    .. [SWIN] Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." ICCV 2021.
    """

    rel_position_index: torch.Tensor

    def __init__(
        self,
        n_dim: int,
        n_channels_in: int,
        n_channels_out: int,
        n_heads: int,
        window_size: int = 7,
        shifted: bool = True,
        features_last: bool = False,
    ):
        """Initialize the ShiftedWindowAttention module.

        Parameters
        ----------
        n_dim
            The dimension of the input.
        n_channels_in
            The number of channels in the input tensor.
        n_channels_out
            The number of channels in the output tensor.
        n_heads
            The number of attention heads. The number if channels per head is ``channels // n_heads``.
        window_size
            The size of the window.
        shifted
            Whether to shift the window.
        features_last
            Whether the features are last in the input tensor or in the second dimension.
        """
        super().__init__()
        if n_heads <= 0:
            raise ValueError('n_heads must be positive')
        if n_channels_in % n_heads:
            raise ValueError('n_channels_in must be divisible by n_heads')
        self.n_heads = n_heads
        self.window_size = window_size
        self.shifted = shifted
        self.features_last = features_last
        channels_per_head = n_channels_in // n_heads
        self.to_qkv = Linear(channels_per_head * n_heads, 3 * channels_per_head * n_heads)
        self.to_out = Linear(channels_per_head * n_heads, n_channels_out)
        self.n_dim = n_dim
        coords_1d = torch.arange(window_size)
        coords_nd = torch.stack(torch.meshgrid(*([coords_1d] * n_dim), indexing='ij'), 0).flatten(1)
        rel_coords = coords_nd[:, :, None] - coords_nd[:, None, :]  # (dim, window_size**dim, window_size**dim)
        rel_coords += window_size - 1  # shift to >=0
        rel_position_index = ravel_multi_index(tuple(rel_coords), (2 * window_size - 1,) * n_dim)
        self.register_buffer('rel_position_index', rel_position_index)

        self.relative_position_bias_table = torch.nn.Parameter(torch.empty((2 * window_size - 1) ** n_dim, n_heads))
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02, a=-0.04, b=0.04)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the ShiftedWindowAttention.

        Parameters
        ----------
        x
            The input tensor.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the ShiftedWindowAttention."""
        if not self.features_last:
            x = x.moveaxis(1, -1)  # now it is features last

        padding = []
        for s in x.shape[-self.n_dim - 1 : -1]:
            target = ((s + self.window_size - 1) // self.window_size) * self.window_size
            padding.extend([target - s, 0])
        x_padded = torch.nn.functional.pad(x, (0, 0, *padding[::-1]), mode='circular') if any(padding) else x
        spatial_dims = tuple(range(-self.n_dim - 1, -1))
        apply_shift = self.shifted and all(size > self.window_size for size in x_padded.shape[-self.n_dim - 1 : -1])
        if apply_shift:
            x_padded = torch.roll(x_padded, (-(self.window_size // 2),) * self.n_dim, dims=spatial_dims)

        qkv = self.to_qkv(x_padded)
        windowed = sliding_window(
            qkv, window_shape=self.window_size, stride=self.window_size, dim=range(-self.n_dim - 1, -1)
        )
        q, k, v = rearrange(
            windowed.flatten(-self.n_dim - 1, -2),
            '... sequence (qkv heads channels)->qkv ... heads sequence channels',
            heads=self.n_heads,
            qkv=3,
        )
        bias = rearrange(self.relative_position_bias_table[self.rel_position_index], 'wd1 wd2 heads -> 1 heads wd1 wd2')
        if apply_shift:
            shift = self.window_size // 2
            region_mask = torch.zeros(x_padded.shape[-self.n_dim - 1 : -1], device=x.device, dtype=torch.int64)
            for dim, size in enumerate(region_mask.shape):
                coordinate = torch.arange(size, device=x.device)
                region = (coordinate >= size - self.window_size).to(torch.int64)
                region += (coordinate >= size - shift).to(torch.int64)
                shape = [1] * self.n_dim
                shape[dim] = size
                region_mask += region.reshape(shape) * 3**dim
            windowed_mask = sliding_window(
                region_mask, window_shape=self.window_size, stride=self.window_size, dim=range(self.n_dim)
            ).flatten(-self.n_dim)
            windowed_mask = windowed_mask[..., :, None] != windowed_mask[..., None, :]
            bias = bias + windowed_mask[..., None, None, :, :] * torch.finfo(q.dtype).min
        attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        attention = rearrange(attention, '... head sequence channels->... sequence (head channels)')
        attention = attention.unflatten(-2, windowed.shape[-self.n_dim - 1 : -1])
        # The window-grid axes precede batch; interleave only grid and within-window axes.
        attention = attention.moveaxis(list(range(self.n_dim)), list(range(1, 2 * self.n_dim, 2)))
        attention = attention.reshape(x_padded.shape)
        if apply_shift:
            attention = torch.roll(attention, (self.window_size // 2,) * self.n_dim, dims=spatial_dims)
        if any(padding):
            crop_idx: tuple[EllipsisType | slice, ...] = (
                Ellipsis,
                *[slice(0, s) for s in x.shape[-self.n_dim - 1 : -1]],
                slice(None),
            )
            attention = attention[crop_idx]
        out = self.to_out(attention)
        if not self.features_last:
            out = out.moveaxis(-1, 1)
        return out
