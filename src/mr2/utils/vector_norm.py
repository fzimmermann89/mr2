"""Euclidean vector norms."""

import torch


def vector_norm(x: torch.Tensor, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> torch.Tensor:
    """Compute Euclidean norms with zero gradients at zero vectors.

    Use real and imaginary squares for complex64/complex128 inputs on CPU.
    Other inputs use the native norm.

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
    if x.device.type != 'cpu' or x.dtype not in (torch.complex64, torch.complex128):
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)
    squared_norm = (x.real.square() + x.imag.square()).sum(dim=dim, keepdim=keepdim)
    zero = squared_norm == 0
    return squared_norm.masked_fill(zero, 1).sqrt().masked_fill(zero, 0)
