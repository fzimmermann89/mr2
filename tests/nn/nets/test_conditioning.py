"""Tests that all conditioned networks pass the conditioning to every conditioned layer."""

import sys
from collections.abc import Callable

import pytest
import torch
from mr2.nn.FiLM import FiLM
from mr2.nn.LayerNorm import LayerNorm
from mr2.nn.nets import DiT, HourglassTransformer, Restormer, SwinIR, Uformer
from mr2.nn.nets.UNet import AttentionGatedUNet, UNet
from mr2.nn.Sequential import Sequential
from mr2.utils import RandomGenerator
from packaging.version import parse as parse_version
from tests.conftest import maximum_python_313, minimal_torch_26
from torch.nn import Module

COND_DIM = 8

NETS: list = [
    pytest.param(
        lambda: SwinIR(2, 1, 1, n_channels_per_head=4, n_heads=2, window_size=4, n_blocks=2, emb_dim=COND_DIM),
        id='SwinIR',
    ),
    pytest.param(
        lambda: Uformer(2, 1, 1, n_channels_per_head=4, n_heads=(1, 2), n_blocks=1, window_size=4, cond_dim=COND_DIM),
        id='Uformer',
    ),
    pytest.param(
        lambda: Restormer(
            2, 1, 1, n_blocks=(1, 1), n_refinement_blocks=1, n_heads=(1, 2), n_channels_per_head=4, cond_dim=COND_DIM
        ),
        id='Restormer',
    ),
    pytest.param(lambda: UNet(2, 1, 1, n_features=(8, 16), n_heads=2, cond_dim=COND_DIM), id='UNet'),
    pytest.param(lambda: AttentionGatedUNet(2, 1, 1, n_features=(8, 16), cond_dim=COND_DIM), id='AttentionGatedUNet'),
    pytest.param(
        lambda: DiT(2, 1, cond_dim=COND_DIM, input_size=16, patch_size=4, hidden_dim=16, depth=2, n_heads=2), id='DiT'
    ),
    pytest.param(
        lambda: HourglassTransformer(
            2, 1, 1, n_features=(16, 32), depths=1, attention_neighborhood=(3, None), n_heads=1, cond_dim=COND_DIM
        ),
        id='HourglassTransformer',
        marks=(
            maximum_python_313,
            minimal_torch_26,
            pytest.mark.skipif(
                sys.platform == 'darwin' and parse_version(torch.__version__).release[:2] == (2, 6),
                reason='PyTorch 2.6 does not support compiled CPU FlexAttention on macOS',
            ),
        ),
    ),
]


def is_conditioned(module: Module) -> bool:
    """Check if a module is a layer that uses the conditioning."""
    return (isinstance(module, FiLM) and module.project is not None) or (
        isinstance(module, LayerNorm) and module.cond_proj is not None
    )


@pytest.mark.parametrize('net_factory', NETS)
@torch.no_grad()
def test_conditioning_reaches_all_layers(net_factory: Callable[[], Module]) -> None:
    """Every conditioned layer receives the conditioning tensor."""
    net = net_factory()
    conditioned = {name: module for name, module in net.named_modules() if is_conditioned(module)}
    assert conditioned, 'network has no conditioned layers'
    received: dict[str, torch.Tensor | None] = {}

    def make_hook(name: str) -> Callable:
        def hook(_module: Module, _args: tuple, kwargs: dict) -> None:
            received[name] = kwargs.get('cond')

        return hook

    for name, module in conditioned.items():
        module.register_forward_pre_hook(make_hook(name), with_kwargs=True)
    rng = RandomGenerator(seed=0)
    cond = rng.float32_tensor((2, COND_DIM))
    net(rng.float32_tensor((2, 1, 16, 16)), cond=cond)

    missing = [name for name in conditioned if received.get(name) is not cond]
    assert not missing, f'layers did not receive the conditioning: {missing}'


@pytest.mark.parametrize('net_factory', NETS)
@torch.no_grad()
def test_conditioning_changes_output(net_factory: Callable[[], Module]) -> None:
    """The output depends on the conditioning."""
    net = net_factory().eval()
    for parameter in net.parameters():  # avoid zero-initialized conditioning projections
        torch.nn.init.normal_(parameter, std=0.2)
    rng = RandomGenerator(seed=0)
    x = rng.float32_tensor((2, 1, 16, 16))
    output1 = net(x, cond=rng.float32_tensor((2, COND_DIM)))
    output2 = net(x, cond=rng.float32_tensor((2, COND_DIM)))
    assert torch.isfinite(output1).all()
    assert torch.isfinite(output2).all()
    assert not torch.allclose(output1, output2)


@torch.no_grad()
def test_swinir_conditioning_through_sequential() -> None:
    """Wrapping SwinIR in Sequential preserves its conditioning."""
    net = SwinIR(
        2,
        1,
        1,
        n_channels_per_head=4,
        n_heads=2,
        window_size=4,
        n_blocks=1,
        n_attn_per_block=1,
        emb_dim=COND_DIM,
    ).eval()
    rng = RandomGenerator(seed=0)
    x = rng.float32_tensor((2, 1, 8, 8))
    cond = rng.float32_tensor((2, COND_DIM))
    torch.testing.assert_close(Sequential(net)(x, cond=cond), net(x, cond=cond))
