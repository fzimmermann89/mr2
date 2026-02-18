"""Tests for the Differentiable Dictionary Matching Operator."""

import pytest
import torch
from mr2.operators import DifferentiableDictionaryMatchOp
from mr2.operators.models import InversionRecovery
from mr2.utils import RandomGenerator

SHAPE = (5, 4, 3)


def _setup(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    index_of_scaling_parameter: int | None,
    batch_size: int = 1024 * 1024,
    seed: int = 2,
):
    rng = RandomGenerator(seed)
    model = InversionRecovery(rng.float32_tensor(5))
    m0 = rng.rand_tensor(shape, dtype=dtype, low=0.2, high=1.0)
    t1 = rng.rand_tensor(shape, dtype=dtype.to_real(), low=0.1, high=1.0)
    (y,) = model(m0, t1)
    operator = DifferentiableDictionaryMatchOp(
        model,
        index_of_scaling_parameter=index_of_scaling_parameter,
        batch_size=batch_size,
    )
    operator.append(m0, t1)
    return operator, y, m0, t1


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('index_of_scaling_parameter', [None, -2, 0], ids=['no_scale', 'scale_neg', 'scale_pos'])
@pytest.mark.parametrize('batch_size', [1, 1024 * 1024], ids=['batched', 'single_chunk'])
def test_matching(
    dtype: torch.dtype,
    index_of_scaling_parameter: int | None,
    batch_size: int,
) -> None:
    """Match against own dictionary entries."""
    operator, y, m0, t1 = _setup(SHAPE, dtype, index_of_scaling_parameter, batch_size)
    m0_matched, t1_matched = operator(y)
    torch.testing.assert_close(t1_matched, t1, atol=1e-4, rtol=0.0)
    if index_of_scaling_parameter is not None:
        torch.testing.assert_close(m0_matched, m0, atol=1e-3, rtol=0.0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
def test_append_consistency(
    dtype: torch.dtype,
    index_of_scaling_parameter: int | None,
) -> None:
    """Appending individually vs concatenated gives same result."""
    rng = RandomGenerator(2)
    m0_1 = rng.rand_tensor(SHAPE, dtype=dtype, low=0.2, high=1.0)
    t1_1 = rng.rand_tensor(SHAPE, dtype=dtype.to_real(), low=0.1, high=1.0)
    m0_2 = rng.rand_tensor(SHAPE, dtype=dtype, low=0.2, high=1.0)
    t1_2 = rng.rand_tensor(SHAPE, dtype=dtype.to_real(), low=0.1, high=1.0)
    model = InversionRecovery(rng.float32_tensor(5))

    m0_cat = torch.cat((m0_1, m0_2))
    t1_cat = torch.cat((t1_1, t1_2))
    (y,) = model(m0_cat, t1_cat)

    op_seq = DifferentiableDictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter)
    op_seq.append(m0_1, t1_1)
    op_seq.append(m0_2, t1_2)

    op_cat = DifferentiableDictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter)
    op_cat.append(m0_cat, t1_cat)

    result_seq = op_seq(y)
    result_cat = op_cat(y)

    for r_seq, r_cat in zip(result_seq, result_cat, strict=True):
        torch.testing.assert_close(r_seq, r_cat, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
def test_empty_dictionary(index_of_scaling_parameter: int | None) -> None:
    rng = RandomGenerator(2)
    model = InversionRecovery(rng.float32_tensor(5))
    operator = DifferentiableDictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter)
    with pytest.raises(KeyError, match='No keys'):
        operator(torch.zeros(5, 5, 4, 3))


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
def test_prior_zero_weight_is_noop(
    dtype: torch.dtype,
    index_of_scaling_parameter: int | None,
) -> None:
    operator, y, m0, t1 = _setup(SHAPE, dtype, index_of_scaling_parameter)
    prior = (torch.zeros_like(m0), torch.zeros_like(t1))

    result_no_prior = operator(y)
    result_with_prior = operator(y, prior=prior, prior_weight=0.0)

    for r1, r2 in zip(result_no_prior, result_with_prior, strict=True):
        torch.testing.assert_close(r1, r2, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
def test_differentiable_in_input_signal(
    dtype: torch.dtype,
    index_of_scaling_parameter: int | None,
) -> None:
    operator, y, m0, t1 = _setup(SHAPE, dtype, index_of_scaling_parameter)
    y = y.detach().requires_grad_(True)

    out = operator(y)
    loss = sum(o.real.float().abs().mean() for o in out)
    loss.backward()

    assert y.grad is not None
    assert torch.isfinite(y.grad).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
def test_prior_pulls_result(
    dtype: torch.dtype,
    index_of_scaling_parameter: int | None,
) -> None:
    """Strong prior should pull the result towards the prior values."""
    operator, y, m0, t1 = _setup(SHAPE, dtype, index_of_scaling_parameter)
    prior = (torch.ones_like(m0) * 0.5, torch.ones_like(t1) * 0.5)

    result_no_prior = operator(y)
    result_with_prior = operator(y, prior=prior, prior_weight=1e6)

    for r, p in zip(result_with_prior, prior, strict=True):
        dist_to_prior = (r - p).abs().mean()
        assert dist_to_prior < 0.01, f'Strong prior should pull result close, got distance {dist_to_prior}'

    for r_np, r_wp in zip(result_no_prior, result_with_prior, strict=True):
        assert not torch.allclose(r_np, r_wp), 'Prior with large weight should change the result'
