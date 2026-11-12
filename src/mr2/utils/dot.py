"""Inner products with bounded temporary memory for large tensors."""

from typing import Any

import torch

_MAX_DOT_SIZE = 2**31 - 1
_DOT_CHUNK_SIZE = 2**22


class _ChunkedDot(torch.autograd.Function):
    """Accumulate wide-precision chunks without saving converted inputs."""

    generate_vmap_rule = True

    @staticmethod
    def forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dtype = torch.complex128 if a.is_complex() else torch.float64
        result = torch.zeros((), dtype=dtype, device=a.device)
        for x, y in zip(a.split(_DOT_CHUNK_SIZE), b.split(_DOT_CHUNK_SIZE), strict=True):
            result = result + torch.vdot(x.to(dtype), y.to(dtype))
        return result.to(a.dtype)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor], _output: torch.Tensor) -> None:  # noqa: ANN401
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:  # noqa: ANN401
        a, b = ctx.saved_tensors
        return (
            grad_output.conj() * b if ctx.needs_input_grad[0] else None,
            grad_output * a if ctx.needs_input_grad[1] else None,
        )


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Compute the inner product of two flattened tensors.

    Computes :math:`\sum_i \overline{a_i} b_i`, conjugating the first input.
    Inputs must have the same number of elements, dtype and device.
    Uses `torch.vdot` for at most 2**31 - 1 elements. Larger inputs are reduced
    in float64 or complex128 chunks to avoid inaccurate CPU reductions and
    CUDA size limits. The result retains the input dtype. The backward saves
    only the original inputs, keeping temporary conversion memory bounded.
    Forward-mode differentiation is only supported on the native path.

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
