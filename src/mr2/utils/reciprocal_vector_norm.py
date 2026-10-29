"""Reciprocal Euclidean vector norms."""

import torch


def reciprocal_vector_norm(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """Compute reciprocal Euclidean norms.

    Use a magnitude reduction and rsqrt on CPU for performance, and reciprocal
    native norms on other devices. Zero vectors return infinity.

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
        Real-valued reciprocal norms.
    """
    if x.device.type != 'cpu':
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim).reciprocal()
    return x.abs().square().sum(dim=dim, keepdim=keepdim).rsqrt()
