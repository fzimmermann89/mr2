"""Numerical checks for convolutional attention packing."""

from typing import Literal

import pytest
import torch
from mr2.nn.attention import ConvolutionalAttention
from torch.nn.functional import normalize


@pytest.mark.parametrize('mode', ['temporal', 'local', 'global'])
def test_attention_reference(mode: Literal['temporal', 'local', 'global']) -> None:
    """Match explicit token gathering, head splitting and post-softmax position bias."""
    torch.manual_seed(4)
    layer = ConvolutionalAttention(4, 2, 16, mode).double()
    x = torch.randn(1, 4, 2, 16, 16, dtype=torch.float64)
    q, k, v = [projection(x) for projection in (layer.to_q, layer.to_k, layer.to_v)]
    result = torch.zeros_like(x)
    if mode == 'temporal':
        for head in range(2):
            indices = slice(head * 2, (head + 1) * 2)
            tokens = [z[0, indices].transpose(0, 1).reshape(2, -1) for z in (q, k, v)]
            query, key, value = tokens
            weights = (normalize(query, dim=-1) @ normalize(key, dim=-1).T).softmax(-1)
            result[0, indices] = (weights @ value).reshape(2, 2, 16, 16).transpose(0, 1)
    else:
        assert layer.relative_position_bias_table is not None
        # Make the bias nonzero and structured so before/after-softmax placement matters.
        with torch.no_grad():
            layer.relative_position_bias_table.copy_(
                torch.linspace(-0.2, 0.3, layer.relative_position_bias_table.numel()).reshape_as(
                    layer.relative_position_bias_table
                )
            )
        groups = [(h, w) for h in (0, 8) for w in (0, 8)] if mode == 'local' else [(0, 0)]
        offsets = [
            (h, w) for h in range(0, 8 if mode == 'local' else 16, 4) for w in range(0, 8 if mode == 'local' else 16, 4)
        ]
        grid = 2 if mode == 'local' else 4
        for frame in range(2):
            for top, left in groups:
                tokens = [
                    torch.stack(
                        [
                            z[0, :, frame, top + h : top + h + 4, left + w : left + w + 4].permute(1, 2, 0).flatten()
                            for h, w in offsets
                        ]
                    )
                    .reshape(len(offsets), 2, 32)
                    .transpose(0, 1)
                    for z in (q, k, v)
                ]
                query, key, value = tokens
                weights = (normalize(query, dim=-1) @ normalize(key, dim=-1).transpose(-1, -2)).softmax(-1)
                bias = torch.stack(
                    [
                        torch.stack(
                            [
                                layer.relative_position_bias_table[
                                    (h1 // 4 - h2 // 4 + grid - 1) * (2 * grid - 1) + w1 // 4 - w2 // 4 + grid - 1
                                ]
                                for h2, w2 in offsets
                            ]
                        )
                        for h1, w1 in offsets
                    ]
                ).permute(2, 0, 1)
                patches = ((weights + bias) @ value).transpose(0, 1).reshape(len(offsets), 4, 4, 4)
                for patch, (h, w) in zip(patches, offsets, strict=True):
                    result[0, :, frame, top + h : top + h + 4, left + w : left + w + 4] = patch.permute(2, 0, 1)
    torch.testing.assert_close(layer(x), layer.to_out(result))
