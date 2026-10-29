"""Tests for ConvolutionalAttention."""

from typing import Literal

import pytest
import torch
from einops import rearrange
from mr2.nn.attention import ConvolutionalAttention
from mr2.utils import RandomGenerator
from torch.nn.functional import normalize


@torch.no_grad()
@pytest.mark.parametrize('mode', ['local', 'global'])
def test_convolutional_attention_spatial_reference(mode: Literal['local', 'global']) -> None:
    """Compare window attention with dense attention masked by patch coordinates.

    Heads split flattened (patch row, patch column, channel) features. Relative
    position bias is added after softmax.
    """
    size = 16
    x = RandomGenerator(seed=4).float64_tensor((2, 4, 3, size, size))
    layer = ConvolutionalAttention(n_channels=4, n_heads=2, image_size=size, mode=mode).double()
    assert layer.relative_position_bias_table is not None
    torch.nn.init.normal_(layer.relative_position_bias_table)
    q, k, v = (
        rearrange(projection(x), 'b c t (gy py) (gx px) -> b t (gy gx) (py px c)', py=4, px=4)
        .unflatten(-1, (layer.n_heads, -1))
        .transpose(-3, -2)
        for projection in (layer.to_q, layer.to_k, layer.to_v)
    )
    patch = torch.cartesian_prod(torch.arange(size // 4), torch.arange(size // 4))
    patches_per_window = 2 if mode == 'local' else size // 4
    window, position = patch // patches_per_window, patch % patches_per_window
    same_window = (window[:, None] == window[None]).all(-1)
    relative = position[:, None] - position[None] + patches_per_window - 1
    index = relative[..., 0] * (2 * patches_per_window - 1) + relative[..., 1]
    bias = layer.relative_position_bias_table[index].permute(2, 0, 1)
    logits = (normalize(q, dim=-1) @ normalize(k, dim=-1).mT).masked_fill(~same_window, -torch.inf)
    weights = logits.softmax(-1) + bias * same_window
    expected = rearrange(weights @ v, 'b t heads s d -> b t s (heads d)')
    expected = rearrange(expected, 'b t (gy gx) (py px c) -> b c t (gy py) (gx px)', gy=size // 4, py=4, px=4)
    torch.testing.assert_close(layer(x), layer.to_out(expected))


@torch.no_grad()
def test_convolutional_attention_temporal_reference() -> None:
    """Each head attends across frames using its channel group and all pixels as features."""
    x = RandomGenerator(seed=4).float64_tensor((2, 4, 3, 16, 16))
    layer = ConvolutionalAttention(n_channels=4, n_heads=2, image_size=16, mode='temporal').double()
    # Keep channel and pixel axes explicit instead of packing them into token features.
    q, k, v = (projection(x).unflatten(1, (layer.n_heads, -1)) for projection in (layer.to_q, layer.to_k, layer.to_v))
    q, k = normalize(q.flatten(-2), dim=(2, 4)), normalize(k.flatten(-2), dim=(2, 4))
    weights = torch.einsum('bncti,bncsi->bnts', q, k).softmax(-1)
    expected = torch.einsum('bnts,bncsyx->bnctyx', weights, v).flatten(1, 2)
    torch.testing.assert_close(layer(x), layer.to_out(expected))


def test_convolutional_attention_wrong_shape() -> None:
    """Inputs not matching image_size are rejected."""
    layer = ConvolutionalAttention(n_channels=4, n_heads=2, image_size=16, mode='local')
    with pytest.raises(ValueError, match='Expected'):
        layer(torch.zeros(1, 4, 2, 16, 24))
