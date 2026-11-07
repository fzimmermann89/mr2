"""Tests for reusable native NUFFT plans."""

import copy
import pickle

import pytest
import torch
from mr2.utils import RandomGenerator
from pytorch_finufft.functional import finufft_type1, finufft_type2

from tests.helper import autodiff_test, gradient_of_linear_operator_test
from tests.operators.test_non_uniform_fast_fourier_op import create_time_varying_2d_nufft_op

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_plan_results(device: str, dtype: torch.dtype) -> None:
    """Native plans match the functional backend for both directions."""
    rng = RandomGenerator(seed=0, device=device)
    operator = create_time_varying_2d_nufft_op(image_shape=(5, 6)).to(device=device, dtype=dtype.to_real())
    image = rng.complex128_tensor((4, 2, 1, 5, 6)).to(dtype)
    samples = rng.complex128_tensor((4, 2, 1, 5, 6)).to(dtype)
    expected_forward = torch.vmap(lambda p, x: finufft_type2(p, x, isign=-1, modeord=0, upsampfac=2.0))(
        operator._omega, image[:, :, 0]
    )
    expected_adjoint = torch.vmap(lambda p, x: finufft_type1(p, x, (5, 6), isign=1, modeord=0, upsampfac=2.0))(
        operator._omega, samples.flatten(2)
    )
    torch.testing.assert_close(operator(image)[0], expected_forward.reshape_as(samples) * operator.scale)
    torch.testing.assert_close(operator.adjoint(samples)[0], expected_adjoint.reshape_as(image) * operator.scale)


@pytest.mark.parametrize('device', DEVICES)
def test_plan_cache(device: str) -> None:
    """Reuse plans across frames and calls; invalidate on batching or conversion."""
    rng = RandomGenerator(seed=0, device=device)
    operator = create_time_varying_2d_nufft_op().to(device)
    image = rng.complex64_tensor((4, 2, 1, 12, 10))
    expected = operator(image)[0]
    forward = operator._plan_fwd
    operator.adjoint(expected)
    adjoint = operator._plan_adj
    torch.testing.assert_close(operator(image * 2)[0], expected * 2)
    assert operator._plan_fwd is forward
    assert operator._plan_adj is adjoint
    operator(image[:, :1])
    assert operator._plan_fwd is not forward
    assert '_plan_adj' not in vars(operator)
    operator.double()
    assert '_plan_fwd' not in vars(operator)
    torch.testing.assert_close(
        operator(image.to(torch.complex128))[0], expected.to(torch.complex128), atol=2e-6, rtol=2e-5
    )


@pytest.mark.parametrize('device', DEVICES)
def test_plan_copy(device: str) -> None:
    """Copies and checkpoints exclude native plans."""
    rng = RandomGenerator(seed=0, device=device)
    operator = create_time_varying_2d_nufft_op().to(device)
    image = rng.complex64_tensor((4, 2, 1, 12, 10))
    expected = operator(image)[0]
    operator.adjoint(expected)
    for duplicate in (copy.copy(operator), copy.deepcopy(operator), pickle.loads(pickle.dumps(operator))):  # noqa: S301
        assert '_plan_fwd' not in vars(duplicate)
        assert '_plan_adj' not in vars(duplicate)
        torch.testing.assert_close(duplicate(image)[0], expected)
    assert not any('plan' in key for key in operator.state_dict())


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('adjoint', [False, True])
def test_plan_vmap(device: str, adjoint: bool) -> None:
    """Mapped calls match individual calls in both directions."""
    rng = RandomGenerator(seed=0, device=device)
    operator = create_time_varying_2d_nufft_op().to(device)
    function = operator.adjoint if adjoint else operator
    data = rng.complex64_tensor((3, 4, 2, 1, 12, 10))
    expected = torch.stack([function(x)[0] for x in data])
    torch.testing.assert_close(torch.vmap(function)(data)[0], expected)


@pytest.mark.parametrize('device', DEVICES)
def test_plan_autograd(device: str) -> None:
    """Forward and reverse data derivatives work with native plans."""
    rng = RandomGenerator(seed=0, device=device)
    operator = create_time_varying_2d_nufft_op().to(device)
    image = rng.complex64_tensor((4, 2, 1, 12, 10))
    samples = rng.complex64_tensor((4, 2, 1, 12, 10))
    gradient_of_linear_operator_test(operator, image, samples)
    autodiff_test(operator, image)
    autodiff_test(operator.H, samples)


@pytest.mark.cuda
def test_plan_cuda_streams() -> None:
    """Changing the CUDA stream rebuilds the plan on the new stream."""
    rng = RandomGenerator(seed=0, device='cuda')
    operator = create_time_varying_2d_nufft_op().cuda()
    image = rng.complex64_tensor((4, 2, 1, 12, 10))
    expected = operator(image)[0]
    plan = operator._plan_fwd
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = operator(image)[0]
        assert operator._plan_fwd is not plan
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, expected)
