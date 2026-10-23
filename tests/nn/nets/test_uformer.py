"""Tests for Uformer network."""

from typing import cast

import pytest
import torch
from mr2.nn.LayerNorm import LayerNorm
from mr2.nn.nets.Uformer import LeWinTransformerBlock, Uformer


def test_lewin_transformer_block_structure() -> None:
    """LeWin blocks use channel normalization and two residual connections."""
    block = LeWinTransformerBlock(n_dim=2, n_channels_per_head=2, n_heads=1, window_size=2)
    assert isinstance(block.norm1, LayerNorm)
    assert isinstance(block.norm2, LayerNorm)
    assert not block.norm1.features_last
    assert not block.norm2.features_last

    with torch.no_grad():
        x = torch.rand(1, 2, 2, 2)
        attended = x + block.drop_path(block.attn(block.norm1(x) + block.modulator))
        expected = attended + block.drop_path(block.ff(block.norm2(attended)))
        torch.testing.assert_close(block(x), expected)


@pytest.mark.parametrize(
    ('shape', 'window_size'),
    [((8, 8), 2), ((5,), 2), ((5,), 4), ((15, 16), 2), ((5, 7), 4), ((1, 3), 2), ((4, 4, 4), 2)],
)
@pytest.mark.parametrize('shifted', [False, True])
def test_lewin_periodic_modulator(shape: tuple[int, ...], window_size: int, shifted: bool) -> None:
    """Partial windows preserve periodic modulation and its input/parameter gradients."""
    block = LeWinTransformerBlock(len(shape), 2, 2, window_size=window_size, shifted=shifted)
    x = torch.randn(2, 4, *shape, requires_grad=True)
    modulator: torch.Tensor = block.modulator
    normalized = block.norm1(x)
    for dim, size in enumerate(shape, start=1):
        positions = torch.arange(((size + window_size - 1) // window_size) * window_size)
        modulator = modulator.index_select(dim, positions % window_size)
        normalized = normalized.index_select(dim + 1, positions % size)
    attended = x + block.attn(normalized + modulator)[tuple(slice(0, size) for size in x.shape)]
    expected = attended + block.ff(block.norm2(attended))
    actual = block(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    cotangent = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (x, block.modulator), cotangent)
    expected_gradients = torch.autograd.grad(expected, (x, block.modulator), cotangent)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize('shape', [(16, 16), (14, 18)])
@pytest.mark.parametrize('torch_compile', [True, False], ids=['compiled', 'uncompiled'])
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_uformer_forward(shape: tuple[int, int], torch_compile: bool, device: str) -> None:
    """Test the forward pass of the uformer."""
    uformer = Uformer(
        n_dim=2, n_channels_in=1, n_channels_out=1, n_heads=(1, 1), cond_dim=32, n_channels_per_head=4, window_size=2
    )

    x = torch.zeros(1, 1, *shape, device=device)
    cond = torch.zeros(1, 32, device=device)
    uformer = uformer.to(device)
    x = x.to(device)
    cond = cond.to(device)
    if torch_compile:
        uformer = cast(Uformer, torch.compile(uformer))
    y = uformer(x, cond=cond)
    assert y.shape == (1, 1, *shape)


@pytest.mark.parametrize('size', [16, 12])
def test_uformer_backward(size: int) -> None:
    uformer = Uformer(
        n_dim=1,
        n_channels_in=1,
        n_channels_out=1,
        n_heads=(1, 2, 4),
        cond_dim=32,
        n_channels_per_head=8,
        window_size=2,
    )

    x = torch.zeros(1, 1, size, requires_grad=True)
    cond = torch.zeros(1, 32, requires_grad=True)
    y = uformer(x, cond=cond)
    y.sum().backward()
    assert x.grad is not None, 'x.grad is None'
    assert not x.grad.isnan().any(), 'x.grad is NaN'
    assert cond.grad is not None, 'cond.grad is None'
    assert not cond.grad.isnan().any(), 'cond.grad is NaN'
    for name, parameter in uformer.named_parameters():
        assert parameter.grad is not None, f'{name}.grad is None'
        assert not parameter.grad.isnan().any(), f'{name}.grad is NaN'
