"""Reciprocal Euclidean vector norms."""

import torch


def reciprocal_vector_norm(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """Compute ``1 / norm(x)`` using a magnitude reduction and rsqrt.

    Explicit magnitude reductions improve complex norm performance on CPU.
    Zero vectors return infinity and have undefined gradients.
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
        Real-valued reciprocal norms.
    """
    return x.abs().square().sum(dim=dim, keepdim=keepdim).rsqrt()
