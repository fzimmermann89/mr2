"""Tests for N-dimensional convolution and its Hermitian adjoint."""

from typing import Literal

import pytest
import torch
from mr2.operators import ConvAnalysisDictionaryOp, ConvolutionOp, ConvSynthesisDictionaryOp, RearrangeOp
from mr2.utils import RandomGenerator
from torch.nn import functional

from tests import (
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


@pytest.mark.parametrize('n_dim', [1, 2, 3])
@pytest.mark.parametrize('pad_mode', ['constant', 'circular', 'reflect', 'replicate'])
@pytest.mark.parametrize(
    ('input_dtype', 'kernel_dtype'),
    [
        (torch.float64, torch.float64),
        (torch.complex128, torch.complex128),
        (torch.complex128, torch.float64),
        (torch.float64, torch.complex128),
    ],
)
def test_forward_and_adjoint(
    n_dim: int,
    pad_mode: Literal['constant', 'circular', 'reflect', 'replicate'],
    input_dtype: torch.dtype,
    kernel_dtype: torch.dtype,
) -> None:
    """Multi-channel filtering matches PyTorch and its adjoint satisfies the complex dot product."""
    rng = RandomGenerator(seed=0)
    x = rng.rand_tensor((2, 2, *[4, 5, 6][:n_dim]), dtype=input_dtype)
    kernel_size = (3, 1, 3)[:n_dim]
    kernel = rng.rand_tensor((3, 2, *kernel_size), dtype=kernel_dtype)
    op = ConvolutionOp(kernel, pad_mode=pad_mode)
    padding = tuple(p for size in reversed(kernel_size) for p in (size // 2, size // 2))
    # PyTorch has batch then channels, whereas the operator defaults to channels then batch.
    padded = functional.pad(x.movedim(0, 1), padding, mode=pad_mode)
    conv = (functional.conv1d, functional.conv2d, functional.conv3d)[n_dim - 1]
    dtype = torch.promote_types(input_dtype, kernel_dtype)
    expected = conv(padded.to(dtype), kernel.to(dtype)).movedim(1, 0)
    torch.testing.assert_close(op(x)[0], expected)
    # A complex filter bank acts on a complex domain, including real-valued images.
    v = rng.rand_tensor(expected.shape, dtype=expected.dtype)
    dotproduct_adjointness_test(op, x.to(dtype), v, relative_tolerance=1e-12, absolute_tolerance=1e-12)


@pytest.mark.parametrize('pad_mode', ['constant', 'reflect'])
def test_autodiff(pad_mode: Literal['constant', 'reflect']) -> None:
    """Forward and reverse differentiation respect the explicit Hermitian adjoint."""
    rng = RandomGenerator(seed=0)
    op = ConvolutionOp(rng.complex128_tensor((3, 2, 3, 3)), pad_mode=pad_mode)
    x = rng.complex128_tensor((2, 1, 4, 5))
    y = rng.complex128_tensor((3, 1, 4, 5))
    gradient_of_linear_operator_test(op, x, y)
    forward_mode_autodiff_of_linear_operator_test(op, x, y)


@pytest.mark.parametrize('view', [False, True])
@pytest.mark.parametrize('pad_mode', ['constant', 'reflect'])
def test_kernel_gradient(view: bool, pad_mode: Literal['constant', 'reflect']) -> None:
    """A Parameter or non-leaf view receives correct gradients through forward and adjoint."""
    rng = RandomGenerator(seed=0)
    parameter = torch.nn.Parameter(rng.complex128_tensor((2, 3, 3) if view else (2, 1, 3, 3)))
    kernel = parameter[:, None] if view else parameter
    op = ConvolutionOp(kernel, pad_mode=pad_mode)
    x = rng.complex128_tensor((1, 1, 4, 5))
    y = rng.complex128_tensor((2, 1, 4, 5))
    reference_input = x.movedim(0, 1).detach().requires_grad_()
    reference_forward = functional.conv2d(functional.pad(reference_input, (1, 1, 1, 1), mode=pad_mode), kernel)
    (reference_adjoint,) = torch.autograd.grad(
        reference_forward, reference_input, grad_outputs=y.movedim(0, 1), create_graph=True
    )
    reference_loss = reference_forward.abs().square().sum() + reference_adjoint.abs().square().sum()
    loss = op(x)[0].abs().square().sum() + op.H(y)[0].abs().square().sum()
    actual_gradient = torch.autograd.grad(loss, parameter)[0]
    expected_gradient = torch.autograd.grad(reference_loss, parameter)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert op.kernel is kernel


@pytest.mark.parametrize('n_dim', [2, 3])
def test_axis_layout(n_dim: int) -> None:
    """Spatial axes can be reordered and separated from a nonleading channel axis."""
    rng = RandomGenerator(seed=0)
    kernel = rng.complex128_tensor((3, 2, *([3] * n_dim)))
    x = rng.complex128_tensor((2, 1, *([4] * n_dim)))
    permutation = (2, 0, 3, 1) if n_dim == 2 else (4, 1, 3, 0, 2)
    dim = tuple(permutation.index(axis) for axis in range(2, x.ndim))
    channel_dim = permutation.index(0)
    permuted = x.permute(permutation)
    op = ConvolutionOp(kernel, dim=dim, channel_dim=channel_dim)
    negative_axes = ConvolutionOp(kernel, dim=tuple(d - x.ndim for d in dim), channel_dim=channel_dim - x.ndim)
    expected = ConvolutionOp(kernel)(x)[0].permute(permutation)
    torch.testing.assert_close(op(permuted)[0], expected)
    torch.testing.assert_close(negative_axes(permuted)[0], expected)
    y = rng.complex128_tensor(expected.shape)
    dotproduct_adjointness_test(op, permuted, y)


def test_four_spatial_dimensions() -> None:
    """The backend also supports more than three spatial dimensions."""
    rng = RandomGenerator(seed=0)
    x = rng.complex128_tensor((1, 2, 4, 3, 2, 2))
    kernel = torch.tensor([1, 2, -3j], dtype=x.dtype).reshape(1, 1, 3, 1, 1, 1)
    op = ConvolutionOp(kernel)
    expected = x.roll(1, dims=2) + 2 * x - 3j * x.roll(-1, dims=2)
    torch.testing.assert_close(op(x)[0], expected)
    dotproduct_adjointness_test(op, x, rng.complex128_tensor(x.shape))


@pytest.mark.parametrize('pad_mode', ['constant', 'circular', 'reflect', 'replicate'])
def test_dictionary_compositions(pad_mode: Literal['constant', 'circular', 'reflect', 'replicate']) -> None:
    """Both old dictionary conventions are reproduced without changing boundary behavior."""
    rng = RandomGenerator(seed=0)
    kernel = rng.complex128_tensor((3, 3, 3))
    x = rng.complex128_tensor((2, 4, 5))
    codes = rng.complex128_tensor((3, *x.shape))
    insert_channel = RearrangeOp('... -> 1 ...')
    analysis = ConvolutionOp(kernel[:, None], pad_mode=pad_mode) @ insert_channel
    synthesis = insert_channel.H @ ConvolutionOp(kernel[None], pad_mode=pad_mode)
    old_analysis = ConvAnalysisDictionaryOp(kernel, pad_mode)
    old_synthesis = ConvSynthesisDictionaryOp(kernel, pad_mode)
    torch.testing.assert_close(analysis(x)[0], old_analysis(x)[0])
    torch.testing.assert_close(analysis.H(codes)[0], old_analysis.H(codes)[0])
    torch.testing.assert_close(synthesis(codes)[0], old_synthesis(codes)[0])
    torch.testing.assert_close(synthesis.H(x)[0], old_synthesis.H(x)[0])


@pytest.mark.parametrize('parameter', [False, True])
def test_tensor_registration(parameter: bool) -> None:
    """Stored tensors retain identity and use mr2's buffer/parameter movement conventions."""
    kernel = torch.ones(2, 1, 3)
    if parameter:
        kernel = torch.nn.Parameter(kernel)
    op = ConvolutionOp(kernel)
    assert op.kernel is kernel
    registered = dict(op.named_parameters()) if parameter else dict(op.named_buffers())
    assert registered['kernel'] is kernel
    torch.testing.assert_close(op.state_dict()['kernel'], kernel)
    op = op.double()
    assert op.kernel.dtype == torch.float64
    assert op(torch.ones(1, 5, dtype=torch.float64))[0].dtype == torch.float64


@pytest.mark.parametrize('shape', [(2, 3), (0, 1, 3), (1, 0, 3), (1, 1, 0), (1, 1, 2), (1, 1, 3, 4)])
def test_invalid_kernel(shape: tuple[int, ...]) -> None:
    """Kernels require nonempty channel axes and positive odd spatial sizes."""
    with pytest.raises(ValueError, match='kernel'):
        ConvolutionOp(torch.ones(shape))


def test_invalid_arguments() -> None:
    """Invalid boundary modes and mismatched spatial axis counts are rejected."""
    with pytest.raises(ValueError, match='pad_mode'):
        ConvolutionOp(torch.ones(1, 1, 3), pad_mode='invalid')  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='dim'):
        ConvolutionOp(torch.ones(1, 1, 3), dim=(-2, -1))


@pytest.mark.parametrize('adjoint', [False, True])
def test_invalid_axes_and_channels(adjoint: bool) -> None:
    """Axis aliases are normalized before validation, in both operator directions."""
    x = torch.ones(1, 4, 5)
    for dim, channel_dim, error, message in [
        ((1, -2), 0, IndexError, 'unique'),
        ((0, 2), -3, ValueError, 'distinct'),
        ((1, 3), 0, IndexError, 'Invalid index'),
        ((1, 2), 3, IndexError, 'Invalid index'),
        ((1, 2), 0, ValueError, 'channel size'),
    ]:
        op = ConvolutionOp(torch.ones(2, 2, 3, 3), dim=dim, channel_dim=channel_dim)
        with pytest.raises(error, match=message):
            (op.H if adjoint else op)(x)


@pytest.mark.cuda
def test_cuda() -> None:
    """Moving the operator moves its kernel and supports forward and adjoint on CUDA."""
    rng = RandomGenerator(seed=0)
    kernel = rng.complex64_tensor((3, 2, 3, 3))
    x = rng.complex64_tensor((2, 1, 4, 5))
    op = ConvolutionOp(kernel)
    expected = op.H(op(x)[0])[0]
    op.cuda()
    actual = op.H(op(x.cuda())[0])[0]
    assert op.kernel.is_cuda
    torch.testing.assert_close(actual.cpu(), expected)
