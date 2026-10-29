"""Reciprocal Euclidean vector norms."""

import torch


def reciprocal_vector_norm(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    eps: float = 0.0,
) -> torch.Tensor:
    """Compute ``1 / (norm(x) + eps)`` using a magnitude reduction and rsqrt.

    Explicit magnitude reductions improve complex norm performance on CPU.
    With no epsilon, zero vectors return infinity and have undefined gradients.
    Squared magnitudes can underflow or overflow at extreme scales.

    Parameters
    ----------
    x
        Real or complex input tensor.
    dim
        Dimensions to reduce, or all dimensions if `None`.
    keepdim
        Retain reduced dimensions as singletons.
    eps
        Nonnegative offset added to the norm, rather than its square.

    Returns
    -------
    torch.Tensor
        Real-valued reciprocal norms.
    """
    squared_norm = x.abs().square().sum(dim=dim, keepdim=keepdim)
    if eps == 0:
        return squared_norm.rsqrt()
    zero = squared_norm == 0
    inverse_norm = squared_norm.masked_fill(zero, 1).rsqrt()
    return (inverse_norm / (1 + eps * inverse_norm)).masked_fill(zero, 1 / eps)
