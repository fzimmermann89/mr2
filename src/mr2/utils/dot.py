"""Inner products with bounded temporary memory for large tensors."""

from collections.abc import Sequence
from typing import Any

import torch

_MAX_DOT_SIZE = 2**31 - 1
_DOT_CHUNK_SIZE = 2**22


@torch.compiler.allow_in_graph
class _ChunkedDot(torch.autograd.Function):
    """Reduce chunks with analytic derivatives and bounded temporary memory."""

    generate_vmap_rule = True

    @staticmethod
    def forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        result = a.new_zeros(())
        for x, y in zip(a.split(_DOT_CHUNK_SIZE), b.split(_DOT_CHUNK_SIZE), strict=True):
            result = result + torch.vdot(x, y)
        return result

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor], _output: torch.Tensor) -> None:  # noqa: ANN401
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:  # noqa: ANN401
        if grad_output is None:
            return None, None
        a, b = ctx.saved_tensors
        return (
            grad_output.conj() * b if ctx.needs_input_grad[0] else None,
            grad_output * a if ctx.needs_input_grad[1] else None,
        )

    @staticmethod
    def jvp(ctx: Any, grad_a: torch.Tensor | None, grad_b: torch.Tensor | None) -> torch.Tensor:  # noqa: ANN401
        a, b = ctx.saved_tensors
        result = a.new_zeros(())
        if grad_a is not None:
            result = result + dot(grad_a, b)
        if grad_b is not None:
            result = result + dot(a, grad_b)
        return result


def dot(a: torch.Tensor | Sequence[torch.Tensor], b: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    r"""Compute the inner product of tensors or sequences of tensors.

    Computes :math:`\sum_i \overline{a_i} b_i`, conjugating the first input.
    Corresponding tensors must have the same number of elements, dtype and device.
    Sequences are treated as concatenated vectors without allocating the concatenation.
    They must have the same length; empty sequences return a scalar zero on CPU.
    Uses `torch.vdot` for at most 2**31 - 1 elements. Larger inputs are reduced
    in chunks to avoid inaccurate CPU reductions and CUDA size limits.
    Computation and output use the input dtype. Derivatives are computed
    analytically to bound memory.

    Parameters
    ----------
    a
        First input tensor or sequence of tensors.
    b
        Second input tensor or sequence of tensors.

    Returns
    -------
    torch.Tensor
        Scalar inner product.
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        a = (a,) if isinstance(a, torch.Tensor) else a
        b = (b,) if isinstance(b, torch.Tensor) else b
        return sum((dot(x, y) for x, y in zip(a, b, strict=True)), start=torch.tensor(0.0))
    a, b = a.flatten(), b.flatten()
    if a.numel() <= _MAX_DOT_SIZE:
        return torch.vdot(a, b)
    return _ChunkedDot.apply(a, b)
