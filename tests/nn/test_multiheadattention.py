"""Tests for MultiHeadAttention."""

import pytest
import torch
from mr2.nn.attention.MultiHeadAttention import MultiHeadAttention


@pytest.mark.xfail(strict=True, reason='Known issue: cross-attention projects context to an incompatible head width.')
def test_multiheadattention_cross_attention_with_distinct_context_width() -> None:
    """Cross-attention should project distinct context channels to the query attention width."""
    attention = MultiHeadAttention(n_channels_in=4, n_channels_out=4, n_heads=2, n_channels_cross=6)
    output = attention(torch.randn(1, 4, 3, 3), torch.randn(1, 6, 3, 3))
    assert output.shape == (1, 4, 3, 3)


@pytest.mark.xfail(strict=True, reason='Known issue: incompatible channel/head settings fail only during forward.')
def test_multiheadattention_rejects_channels_not_divisible_by_heads() -> None:
    """Invalid channel/head combinations should fail at construction with an actionable error."""
    with pytest.raises(ValueError, match=r'n_channels_in.*n_heads'):
        _ = MultiHeadAttention(n_channels_in=5, n_channels_out=5, n_heads=2)
