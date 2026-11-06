"""Tests for conditioning dispatch through module wrappers."""

import pytest
import torch
from mr2.nn import ComplexAsChannel, CondMixin, FiLM, PermutedBlock, Residual, SeparableResBlock, Sequential
from mr2.nn.CondMixin import call_with_cond
from mr2.nn.join import Add
from mr2.nn.nets import DiT, SwinIR, UNet
from mr2.nn.nets.UNet import UNetDecoder, UNetEncoder
from torch.nn import Module


@pytest.mark.parametrize('module_type', [UNet, SeparableResBlock, DiT, SwinIR])
@pytest.mark.parametrize('wrapper', ['complex', 'sequential', 'permuted', 'residual', 'nested'])
def test_conditioning_dispatch(module_type: type[Module], wrapper: str) -> None:
    """Wrappers preserve conditioning outputs and gradients."""
    torch.manual_seed(0)
    module: Module
    if module_type is UNet:
        module = UNet(
            n_dim=2,
            n_channels_in=2,
            n_channels_out=2,
            n_features=(8, 16),
            attention_depths=(),
            cond_dim=16,
        )
    elif module_type is SeparableResBlock:
        module = SeparableResBlock(dim_groups=((-2, -1),), n_channels_in=2, n_channels_out=2, cond_dim=16)
    elif module_type is DiT:
        module = DiT(
            n_dim=2,
            n_channels_in=2,
            n_channels_out=2,
            input_size=16,
            patch_size=2,
            hidden_dim=16,
            depth=1,
            n_heads=2,
            cond_dim=16,
        )
    else:
        module = SwinIR(
            n_dim=2,
            n_channels_in=2,
            n_channels_out=2,
            n_channels_per_head=4,
            n_heads=2,
            n_blocks=1,
            n_attn_per_block=2,
            window_size=4,
            emb_dim=16,
        )
    module.eval()
    x = torch.randn(1, 2, 16, 16)
    cond = torch.randn(1, 16, requires_grad=True)
    other_cond = torch.randn(1, 16)

    wrapped: Module
    if wrapper in ('complex', 'nested'):
        wrapped = ComplexAsChannel(module)
        if wrapper == 'nested':
            wrapped = Sequential(wrapped)
        x_wrapped = torch.complex(x[:, :1], x[:, 1:])
    else:
        x_wrapped = x
        if wrapper == 'sequential':
            wrapped = Sequential(module)
        elif wrapper == 'permuted':
            wrapped = PermutedBlock((-2, -1), module)
        else:
            wrapped = Residual(module)

    output = wrapped(x_wrapped, cond=cond)
    expected = module(x, cond=cond)
    if wrapper in ('complex', 'nested'):
        expected = torch.complex(expected[:, :1], expected[:, 1:])
    elif wrapper == 'residual':
        expected = expected + x
    torch.testing.assert_close(output, expected)
    assert not torch.allclose(output, wrapped(x_wrapped, cond=other_cond))

    output.abs().square().mean().backward()
    assert cond.grad is not None, 'No gradient computed for conditioning'
    assert torch.isfinite(cond.grad).all(), 'Non-finite conditioning gradients'
    assert cond.grad.abs().max() > 0, 'Conditioning gradients are zero'


@pytest.mark.parametrize('wrapper', ['complex', 'residual', 'nested'])
def test_multiple_inputs(wrapper: str) -> None:
    """Conditioning wrappers retain the multiple-input module call contract."""
    x, y = torch.randn(2, 3), torch.randn(2, 3)
    module = Sequential(Add(), FiLM(3, 4))
    cond = torch.randn(2, 4, requires_grad=True)
    wrapped: Module
    if wrapper == 'residual':
        wrapped = Residual(module, skip=Add())
    else:
        wrapped = ComplexAsChannel(module, convert_back=False)
        if wrapper == 'nested':
            wrapped = Sequential(Residual(wrapped, skip=Add()))
    output = wrapped(x, y, cond=cond)
    expected = module(x, y, cond=cond)
    if wrapper != 'complex':
        expected = expected + x + y
    torch.testing.assert_close(output, expected)
    output.square().mean().backward()
    assert cond.grad is not None
    assert cond.grad.abs().max() > 0


def test_unet_encoder_decoder_dispatch() -> None:
    """UNet components preserve conditioning when used in containers."""
    encoder = UNetEncoder(torch.nn.Identity(), [], [], FiLM(2, 4))
    decoder = UNetDecoder([], [], [], FiLM(2, 4))
    assert isinstance(encoder, CondMixin)
    assert isinstance(decoder, CondMixin)
    x = torch.randn(2, 2, 8, 8)
    cond = torch.randn(2, 4, requires_grad=True)
    encoded = Sequential(encoder)(x, cond=cond)
    torch.testing.assert_close(encoded, encoder(x, cond=cond)[0])
    output = call_with_cond(decoder, (encoded,), cond=cond)  # type: ignore[arg-type]
    torch.testing.assert_close(output, decoder((encoded,), cond=cond))
    assert not torch.allclose(output, decoder((encoded,), cond=cond + 1))
    output.square().mean().backward()
    assert cond.grad is not None
    assert cond.grad.abs().max() > 0
