"""Tests for TransposedAttention module."""

from collections.abc import Sequence

import pytest
import torch
from einops import rearrange
from mr2.nn.attention import TransposedAttention
from mr2.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('dim', 'channels', 'num_heads', 'input_shape'),
    [
        (2, 32, 4, (1, 32, 32, 32)),
        (3, 64, 8, (2, 64, 16, 16, 16)),
    ],
)
def test_transposed_attention(
    dim: int,
    channels: int,
    num_heads: int,
    input_shape: Sequence[int],
    device: str,
) -> None:
    """Test TransposedAttention output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)
    attn = TransposedAttention(n_dim=dim, n_channels_in=channels, n_channels_out=channels, n_heads=num_heads).to(device)
    output = attn(x)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not output.isnan().any(), 'NaN values in output'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert attn.to_qkv.weight.grad is not None, 'No gradient computed for qkv'
    assert attn.qkv_dwconv.weight.grad is not None, 'No gradient computed for qkv_dwconv'
    assert attn.to_out.weight.grad is not None, 'No gradient computed for project_out'
    assert attn.temperature.grad is not None, 'No gradient computed for temperature'


def test_transposed_attention_matches_explicit_channel_attention() -> None:
    """Verify that attention is calculated across channels, not spatial points."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 8, 4, 5))
    module = TransposedAttention(n_dim=2, n_channels_in=8, n_channels_out=8, n_heads=2)

    qkv = module.qkv_dwconv(module.to_qkv(x))
    q, k, v = rearrange(qkv, 'b (qkv heads channels) ... -> qkv b heads channels (...)', heads=2, qkv=3)
    q = torch.nn.functional.normalize(q, dim=-1) * module.temperature
    k = torch.nn.functional.normalize(k, dim=-1)
    weights = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
    expected = rearrange(weights @ v, 'b heads channels points -> b (heads channels) points').unflatten(-1, x.shape[2:])
    expected = module.to_out(expected)

    torch.testing.assert_close(module(x), expected)


@pytest.mark.parametrize(
    ('n_channels_in', 'n_heads', 'message'),
    [(7, 2, 'divisible'), (8, 0, 'positive')],
)
def test_transposed_attention_invalid_heads(n_channels_in: int, n_heads: int, message: str) -> None:
    """Invalid head configurations raise a clear error."""
    with pytest.raises(ValueError, match=message):
        TransposedAttention(n_dim=2, n_channels_in=n_channels_in, n_channels_out=8, n_heads=n_heads)
