"""Tests for MultiHeadAttention."""

import pytest
import torch
from mr2.nn.attention.MultiHeadAttention import MultiHeadAttention


def test_multiheadattention_cross_attention_with_distinct_context_width() -> None:
    """Cross-attention should project distinct context channels to the query attention width."""
    attention = MultiHeadAttention(n_channels_in=4, n_channels_out=4, n_heads=2, n_channels_cross=6)
    output = attention(torch.randn(1, 4, 3, 3), torch.randn(1, 6, 3, 3))
    assert output.shape == (1, 4, 3, 3)


def test_multiheadattention_rejects_channels_not_divisible_by_heads() -> None:
    """Invalid channel/head combinations should fail at construction with an actionable error."""
    with pytest.raises(ValueError, match=r'n_channels_in.*n_heads'):
        _ = MultiHeadAttention(n_channels_in=5, n_channels_out=5, n_heads=2)
