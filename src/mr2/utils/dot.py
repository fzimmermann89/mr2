"""Inner products with bounded temporary memory for large tensors."""

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
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:  # noqa: ANN401
        a, b = ctx.saved_tensors
        return (
            grad_output.conj() * b if ctx.needs_input_grad[0] else None,
            grad_output * a if ctx.needs_input_grad[1] else None,
        )

    @staticmethod
    def jvp(ctx: Any, grad_a: torch.Tensor, grad_b: torch.Tensor) -> torch.Tensor:  # noqa: ANN401
        a, b = ctx.saved_tensors
        return dot(grad_a, b) + dot(a, grad_b)


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Compute the inner product of two flattened tensors.

    Computes :math:`\sum_i \overline{a_i} b_i`, conjugating the first input.
    Inputs must have the same number of elements, dtype and device.
    Uses `torch.vdot` for at most 2**31 - 1 elements. Larger inputs are reduced
    in chunks to avoid inaccurate CPU reductions and CUDA size limits.
    Computation and output use the input dtype. Analytic reverse- and
    forward-mode derivatives save only the original inputs.

    Parameters
    ----------
    a
        First input tensor.
    b
        Second input tensor.

    Returns
    -------
    torch.Tensor
        Scalar inner product.
    """
    a, b = a.flatten(), b.flatten()
    if a.numel() <= _MAX_DOT_SIZE:
        return torch.vdot(a, b)
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise RuntimeError('Inputs must have the same number of elements, dtype and device.')
    return _ChunkedDot.apply(a, b)
