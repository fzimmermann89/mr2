"""Euclidean vector norms."""

import torch


def vector_norm(x: torch.Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> torch.Tensor:
    """Compute Euclidean norms with zero gradients at zero vectors.

    Use explicit magnitude reductions on CPU for performance and native norms
    on other devices.

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
    if x.device.type != 'cpu':
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)
    squared_norm = x.abs().square().sum(dim=dim, keepdim=keepdim)
    zero = squared_norm == 0
    return squared_norm.masked_fill(zero, 1).sqrt().masked_fill(zero, 0)
