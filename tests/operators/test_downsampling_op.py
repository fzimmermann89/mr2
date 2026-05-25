"""Tests for downsampling operator."""

from collections.abc import Sequence

import pytest
import torch
from mr2.operators import DownSamplingOp
from mr2.utils import RandomGenerator

from tests import autodiff_test, dotproduct_adjointness_test, forward_mode_autodiff_of_linear_operator_test


@pytest.mark.parametrize(
    ('dim', 'domain_shape', 'range_shape', 'input_shape', 'output_shape'),
    [
        (-1, (16,), (7,), (2, 3, 16), (2, 3, 7)),
        ((1, 3), (12, 14), (6, 7), (2, 12, 3, 14), (2, 6, 3, 7)),
        ((1, 2, 3, 4), (3, 4, 5, 6), (4, 3, 4, 5), (2, 3, 4, 5, 6), (2, 4, 3, 4, 5)),
    ],
)
def test_downsampling_op_adjointness(
    dim: Sequence[int] | int,
    domain_shape: Sequence[int],
    range_shape: Sequence[int],
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> None:
    """Test adjointness and shape of DownSamplingOp."""
    rng = RandomGenerator(seed=0)
    u = rng.complex64_tensor(size=input_shape)
    v = rng.complex64_tensor(size=output_shape)
    operator = DownSamplingOp(dim=dim, domain_shape=domain_shape, range_shape=range_shape)
    dotproduct_adjointness_test(operator, u, v)


@pytest.mark.parametrize(
    ('dim', 'domain_shape', 'range_shape', 'input_shape', 'output_shape'),
    [
        (-1, (16,), (7,), (2, 3, 16), (2, 3, 7)),
        ((1, 3), (12, 14), (6, 7), (2, 12, 3, 14), (2, 6, 3, 7)),
        ((1, 2, 3, 4), (3, 4, 5, 6), (4, 3, 4, 5), (2, 3, 4, 5, 6), (2, 4, 3, 4, 5)),
    ],
)
def test_downsampling_op_autodiff(
    dim: Sequence[int] | int,
    domain_shape: Sequence[int],
    range_shape: Sequence[int],
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> None:
    """Test autodiff works for DownSamplingOp."""
    rng = RandomGenerator(seed=1)
    u = rng.complex64_tensor(size=input_shape)
    v = rng.complex64_tensor(size=output_shape)
    operator = DownSamplingOp(dim=dim, domain_shape=domain_shape, range_shape=range_shape)
    autodiff_test(operator, u)
    forward_mode_autodiff_of_linear_operator_test(operator, u, v)


def test_downsampling_op_matches_torch_interpolate() -> None:
    """Test DownSamplingOp matches torch interpolation for standard tensor layouts."""
    rng = RandomGenerator(seed=2)
    u = rng.float32_tensor(size=(2, 3, 12, 14))
    operator = DownSamplingOp(dim=(-2, -1), domain_shape=(12, 14), range_shape=(6, 7))

    (actual,) = operator(u)
    expected = torch.nn.functional.interpolate(u, size=(6, 7), mode='bilinear', align_corners=False)

    torch.testing.assert_close(actual, expected)


def test_downsampling_op_invalid() -> None:
    """Test invalid parameters for DownSamplingOp."""
    with pytest.raises(ValueError, match='same length'):
        DownSamplingOp(dim=(0, 1), domain_shape=(10,), range_shape=(5, 5))

    with pytest.raises(ValueError, match='positive'):
        DownSamplingOp(dim=0, domain_shape=(10,), range_shape=(0,))

    operator = DownSamplingOp(dim=(0, 0), domain_shape=(10, 10), range_shape=(5, 5))
    with pytest.raises(IndexError, match='unique'):
        operator(torch.ones(10, 10))

    operator = DownSamplingOp(dim=-1, domain_shape=(10,), range_shape=(5,))
    with pytest.raises(ValueError, match='Expected domain size'):
        operator(torch.ones(9))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_downsampling_op_cuda() -> None:
    """Test DownSamplingOp works on cuda devices."""
    rng = RandomGenerator(seed=3)
    u = rng.complex64_tensor((2, 3, 12, 14))

    operator = DownSamplingOp(dim=(-2, -1), domain_shape=(12, 14), range_shape=(6, 7)).cuda()
    (actual,) = (operator.H @ operator)(u.cuda())

    assert actual.is_cuda
    assert actual.isfinite().all()
