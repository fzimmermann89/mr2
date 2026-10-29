"""Euclidean vector norms with explicit magnitude reductions."""

import torch


def vector_norm(x: torch.Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> torch.Tensor:
    """Compute Euclidean norms, retaining zero gradients at zero vectors.

    Explicit magnitude reductions improve complex norm performance on CPU.
    Squared magnitudes can underflow or overflow at extreme scales.

    Parameters
    ----------
    x
        Real or complex input tensor.
    dim
        Dimensions to reduce, or all dimensions if `None`.
    keepdim
        Retain reduced dimensions as singletons.

    Returns
    -------
    torch.Tensor
        Real-valued Euclidean norms.
    """
    squared_norm = x.abs().square().sum(dim=dim, keepdim=keepdim)
    zero = squared_norm == 0
    return squared_norm.masked_fill(zero, 1).sqrt().masked_fill(zero, 0)
