"""Tests for NeighborhoodSelfAttention module."""

from collections.abc import Sequence

import pytest
import torch
from einops import rearrange
from mr2.nn.attention.NeighborhoodSelfAttention import NeighborhoodSelfAttention
from mr2.utils import RandomGenerator
from tests.conftest import maximum_python_313, minimal_torch_26


@maximum_python_313
@minimal_torch_26
@pytest.mark.parametrize(
    'device',
    [
        pytest.param(
            'cpu',
            id='cpu',
            marks=pytest.mark.skip(
                reason='Flex Attention backward not supported on CPU. https://github.com/pytorch/pytorch/issues/148752'
            ),
        ),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('n_channels_in', 'n_channels_out', 'n_heads', 'kernel_size', 'input_shape', 'features_last'),
    [
        (32, 32, 2, 2, (1, 32, 16, 16), False),
        (32, 32, 2, 4, (1, 32, 8, 8, 8, 8), True),
    ],
    ids=['2d_kernel2', '4d_features-last_kernel4'],
)
def test_neighborhood_self_attention_backward(
    n_channels_in: int,
    n_channels_out: int,
    n_heads: int,
    kernel_size: int,
    input_shape: tuple[int, ...],
    features_last: bool,
    device: str,
) -> None:
    """Test NeighborhoodSelfAttention output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)

    attention = NeighborhoodSelfAttention(
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        n_heads=n_heads,
        kernel_size=kernel_size,
        features_last=features_last,
    ).to(device)

    if features_last:
        output = attention(x.moveaxis(1, -1)).moveaxis(-1, 1)
    else:
        output = attention(x)

    expected_shape = (input_shape[0], n_channels_out, *input_shape[2:])
    assert output.shape == expected_shape
    assert not output.isnan().any(), 'NaN values in output'

    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'

    assert attention.to_qkv.weight.grad is not None, 'No gradient computed for to_qkv.weight'
    assert attention.to_qkv.bias.grad is not None, 'No gradient computed for to_qkv.bias'
    assert attention.to_out.weight.grad is not None, 'No gradient computed for to_out.weight'
    assert attention.to_out.bias.grad is not None, 'No gradient computed for to_out.bias'


@maximum_python_313
@minimal_torch_26
@pytest.mark.cuda
@pytest.mark.parametrize(
    ('kernel_size', 'dilation', 'circular', 'rope'),
    [
        (3, 1, False, True),
        (5, 2, True, False),
        (7, 1, False, True),
    ],
)
def test_neighborhood_attention_variants(kernel_size: int, dilation: int, circular: bool, rope: bool) -> None:
    """Test NeighborhoodSelfAttention with different neighborhood configurations."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 32, 16, 16)).cuda()

    attention = NeighborhoodSelfAttention(
        n_channels_in=32,
        n_channels_out=32,
        n_heads=2,
        kernel_size=kernel_size,
        dilation=dilation,
        circular=circular,
        rope_embed_fraction=1.0 if rope else 0.0,
    ).cuda()
    output = attention(x)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'


@maximum_python_313
@minimal_torch_26
@pytest.mark.parametrize(
    ('kernel_size', 'circular', 'input_shape'),
    [
        (11, False, (1, 8, 32, 32)),
        (3, True, (1, 8, 64, 64)),
    ],
    ids=['regular', 'circular'],
)
@torch.no_grad()
def test_neighborhood_constraint(kernel_size: int, circular: bool, input_shape: tuple[int, int, int, int]) -> None:
    """Test that neighborhood attention only affects pixels within the kernel window."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape)
    attention = NeighborhoodSelfAttention(
        n_channels_in=8,
        n_channels_out=8,
        n_heads=2,
        kernel_size=kernel_size,
        dilation=1,
        circular=circular,
    )
    output_original = attention(x)
    x_modified = x.clone()
    test_point = (input_shape[-2] - 2, input_shape[-1] - 2)
    x_modified[..., test_point[0], test_point[1]] += 1.0
    output_modified = attention(x_modified)

    diff = output_modified - output_original
    changed_pixels = torch.abs(diff).sum(dim=(0, 1)) > 1e-6

    half_kernel = kernel_size // 2
    h, w = input_shape[2], input_shape[3]

    i_coords, j_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')

    if circular:
        h_dist = torch.minimum((i_coords - test_point[0]) % h, (test_point[0] - i_coords) % h)
        w_dist = torch.minimum((j_coords - test_point[1]) % w, (test_point[1] - j_coords) % w)
        in_neighborhood = (h_dist <= half_kernel) & (w_dist <= half_kernel)
    else:
        h_min, h_max = max(0, test_point[0] - half_kernel), min(h, test_point[0] + half_kernel + 1)
        w_min, w_max = max(0, test_point[1] - half_kernel), min(w, test_point[1] + half_kernel + 1)
        in_neighborhood = (i_coords >= h_min) & (i_coords < h_max) & (j_coords >= w_min) & (j_coords < w_max)

    neighborhood_changed = changed_pixels[in_neighborhood].all()
    outside_changed = changed_pixels[~in_neighborhood].any()

    assert neighborhood_changed, 'Not all pixels in the neighborhood changed, which indicates a problem'
    assert not outside_changed, 'Pixels outside the neighborhood changed, which violates the constraint'


def neighborhood_attention_reference(
    attention: NeighborhoodSelfAttention, x: torch.Tensor, kernel_size: int, circular: bool
) -> torch.Tensor:
    """Brute-force neighborhood attention with an explicit dense mask and axial RoPE on the spatial grid."""
    spatial_shape = x.shape[2:]
    query, key, value = rearrange(
        attention.to_qkv(x.moveaxis(1, -1)),
        'batch ... (qkv heads channels) -> qkv batch heads ... channels',
        qkv=3,
        heads=attention.n_head,
    )
    query, key = attention.rope(query, key)
    query, key, value = (t.flatten(2, -2) for t in (query, key, value))
    coords = torch.stack(
        torch.meshgrid(*[torch.arange(s, device=x.device) for s in spatial_shape], indexing='ij'), -1
    ).flatten(0, -2)
    half = kernel_size // 2
    q_coord, kv_coord = coords[:, None], coords[None, :]
    size = torch.tensor(spatial_shape, device=x.device)
    if circular:
        distance = torch.minimum((q_coord - kv_coord) % size, (kv_coord - q_coord) % size)
    else:
        center = torch.minimum(torch.maximum(q_coord, torch.tensor(half)), size - 1 - half)
        distance = (kv_coord - center).abs()
    mask = (distance <= half).all(-1)
    out = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    out = attention.to_out(rearrange(out, 'batch heads sequence channels -> batch sequence (heads channels)'))
    return out.unflatten(1, spatial_shape).moveaxis(-1, 1)


@maximum_python_313
@minimal_torch_26
@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('spatial_shape', 'kernel_size', 'circular'),
    [
        ((9, 7), 3, False),
        ((8, 8), 5, True),
        ((5, 6, 7), 3, False),
    ],
    ids=['2d', '2d_circular', '3d'],
)
@torch.no_grad()
def test_neighborhood_attention_reference(
    spatial_shape: Sequence[int], kernel_size: int, circular: bool, device: str
) -> None:
    """Compare to a brute-force reference, with rotary embeddings on the true N-D positions."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((2, 32, *spatial_shape)).to(device)
    attention = NeighborhoodSelfAttention(
        n_channels_in=32,
        n_channels_out=32,
        n_heads=2,
        kernel_size=kernel_size,
        circular=circular,
        rope_embed_fraction=1.0,
    ).to(device)
    torch.testing.assert_close(attention(x), neighborhood_attention_reference(attention, x, kernel_size, circular))
