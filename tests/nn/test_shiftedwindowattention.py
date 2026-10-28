"""Tests for ShiftedWindowAttention module."""

from collections.abc import Sequence

import pytest
import torch
from einops import rearrange
from mr2.nn.attention import ShiftedWindowAttention
from mr2.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('n_dim', 'window_size', 'shifted'),
    [
        (2, 8, False),
        (4, 4, True),
    ],
)
def test_shifted_window_attention(n_dim: int, window_size: int, shifted: bool, device: str) -> None:
    """Test ShiftedWindowAttention output shape and backpropagation."""
    n_batch, n_channels, n_heads = 2, 8, 2
    spatial_shape = (window_size * 4,) * n_dim
    rng = RandomGenerator(13)
    x = rng.float32_tensor((n_batch, n_channels, *spatial_shape)).to(device).requires_grad_(True)
    swin = ShiftedWindowAttention(
        n_dim=n_dim,
        n_channels_in=n_channels,
        n_channels_out=n_channels,
        n_heads=n_heads,
        window_size=window_size,
        shifted=shifted,
    ).to(device)
    out = swin(x)
    assert out.shape == x.shape, f'Output shape {out.shape} != input shape {x.shape}'
    assert not out.isnan().any(), 'NaN values in output'
    out.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert swin.to_qkv.weight.grad is not None, 'No gradient computed for to_qkv.weight'
    assert swin.relative_position_bias_table.grad is not None, 'No gradient computed for relative_position_bias_table'


@pytest.mark.parametrize('shifted', [True, False], ids=['shifted', 'non-shifted'])
def test_shifted_window_attention_size_mismatch(shifted: bool):
    n_batch, n_channels, n_heads, n_dim, window_size = 3, 4, 2, 2, 7
    spatial_shape = (window_size * 4 + 1,) * n_dim
    rng = RandomGenerator(13)
    x = rng.float32_tensor((n_batch, n_channels, *spatial_shape))
    swin = ShiftedWindowAttention(
        n_dim=n_dim,
        n_channels_in=n_channels,
        n_channels_out=n_channels,
        n_heads=n_heads,
        window_size=window_size,
        shifted=shifted,
    )
    out = swin(x)
    assert out.shape == x.shape, f'Output shape {out.shape} != input shape {x.shape}'


def test_shifted_window_attention_single_window_dimension() -> None:
    """Shifting is disabled if a spatial dimension contains only one window."""
    rng = RandomGenerator(13)
    x = rng.float32_tensor((2, 4, 7, 14))
    shifted = ShiftedWindowAttention(2, 4, 4, 2, window_size=7, shifted=True)
    non_shifted = ShiftedWindowAttention(2, 4, 4, 2, window_size=7, shifted=False)
    non_shifted.load_state_dict(shifted.state_dict())

    torch.testing.assert_close(shifted(x), non_shifted(x))


def shifted_window_attention_reference(swin: ShiftedWindowAttention, x: torch.Tensor) -> torch.Tensor:
    """Brute-force (shifted) window attention over all tokens with an explicit window mask.

    Windows tile the unpadded input, starting at the origin. If shifted, the window boundaries are offset by
    `window_size // 2` without wrapping around, as in the Swin Transformer.
    """
    window_size, n_dim = swin.window_size, swin.n_dim
    spatial_shape = x.shape[2:]
    query, key, value = rearrange(
        swin.to_qkv(x.moveaxis(1, -1).flatten(1, -2)),
        'batch sequence (qkv heads channels) -> qkv batch heads sequence channels',
        qkv=3,
        heads=swin.n_heads,
    )
    coords = torch.stack(
        torch.meshgrid(*[torch.arange(s, device=x.device) for s in spatial_shape], indexing='ij'), -1
    ).flatten(0, -2)
    padded_shape = [-(-s // window_size) * window_size for s in spatial_shape]
    shifted = swin.shifted and all(s > window_size for s in padded_shape)
    offset = window_size - window_size // 2 if shifted else 0
    window = (coords + offset) // window_size
    same_window = (window[:, None] == window[None, :]).all(-1)
    relative = (coords[:, None] - coords[None, :] + window_size - 1).clamp(0, 2 * window_size - 2)
    index = sum(relative[..., d] * (2 * window_size - 1) ** (n_dim - 1 - d) for d in range(n_dim))
    bias = swin.relative_position_bias_table[index].moveaxis(-1, 0)
    bias = bias.masked_fill(~same_window, float('-inf'))
    out = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=bias)
    out = swin.to_out(rearrange(out, 'batch heads sequence channels -> batch sequence (heads channels)'))
    return out.unflatten(1, spatial_shape).moveaxis(-1, 1)


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize('shifted', [True, False], ids=['shifted', 'non-shifted'])
@pytest.mark.parametrize(
    ('spatial_shape', 'window_size'),
    [
        ((3,), 7),
        ((10,), 4),
        ((16,), 4),
        ((3, 3), 7),
        ((8, 8), 4),
        ((9, 6), 4),
        ((8, 8, 8), 4),
        ((5, 5, 5), 4),
        ((5, 9, 3), 4),
        ((2, 3, 2), 5),
    ],
)
@torch.no_grad()
def test_shifted_window_attention_reference(
    spatial_shape: Sequence[int], window_size: int, shifted: bool, device: str
) -> None:
    """Compare to a brute-force reference, including inputs smaller than the window and non-multiples."""
    n_dim = len(spatial_shape)
    rng = RandomGenerator(13)
    x = rng.float64_tensor((2, 4, *spatial_shape)).to(device)
    swin = ShiftedWindowAttention(n_dim, 4, 6, 2, window_size=window_size, shifted=shifted).double().to(device)
    torch.nn.init.normal_(swin.relative_position_bias_table)
    torch.testing.assert_close(swin(x), shifted_window_attention_reference(swin, x))


@pytest.mark.parametrize('shifted', [True, False], ids=['shifted', 'non-shifted'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_shifted_window_attention_padded_backward(shifted: bool, device: str, dtype: torch.dtype) -> None:
    """Padded queries must not introduce non-finite gradients into valid tokens or parameters."""
    rng = RandomGenerator(13)
    x = rng.float32_tensor((2, 4, 5, 9)).to(device=device, dtype=dtype).requires_grad_(True)
    swin = ShiftedWindowAttention(2, 4, 4, 2, window_size=4, shifted=shifted).to(device=device, dtype=dtype)
    output = swin(x)
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in swin.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    'device',
    ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)],
)
def test_shifted_window_attention_large_scores(device: str) -> None:
    """Large float16 scores cannot override the shifted-window mask."""
    swin = ShiftedWindowAttention(1, 1, 1, 1, window_size=4, shifted=True).half().to(device)
    with torch.no_grad():
        # Scores of +/-40,000 stay finite in float16, but their gap exceeds its finite mask value.
        swin.to_qkv.weight.copy_(torch.tensor([[200.0], [-200.0], [1.0]], device=device))
        swin.to_qkv.bias.zero_()
        swin.to_out.weight.fill_(1)
        swin.to_out.bias.zero_()
        swin.relative_position_bias_table.zero_()
    x = torch.tensor([[[1, 1, 1, 1, -1, -1, -1, -1]]], dtype=torch.float16, device=device, requires_grad=True)
    output = swin(x)
    torch.testing.assert_close(output, shifted_window_attention_reference(swin, x))
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in swin.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
