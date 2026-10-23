"""Tests for the Dictionary Matching Operator."""

import pytest
import torch
from mr2.operators import DictionaryMatchOp
from mr2.operators.models import InversionRecovery
from mr2.utils import RandomGenerator


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('shape', [(5, 4, 3)])
@pytest.mark.parametrize(
    'index_of_scaling_parameter',
    [None, -2, 0],
    ids=['dont_predict_scale', 'predict_scale_negative_index', 'predict_scale'],
)
@pytest.mark.parametrize('batch_size', [1, 1024 * 1024], ids=['batched', 'single_chunk'])
def test_dictionary_matching_op(
    shape: tuple[int], dtype: torch.dtype, index_of_scaling_parameter: int | None, batch_size: int
) -> None:
    """Test dictionary matching."""
    rng = RandomGenerator(2)
    model = InversionRecovery(rng.float32_tensor(5))
    m0 = rng.rand_tensor(shape, dtype=dtype, low=0.2, high=1.0)
    t1 = rng.rand_tensor(shape, dtype=dtype.to_real(), low=0.1, high=1.0)
    (y,) = model(m0, t1)

    operator = DictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter, batch_size=batch_size)
    operator.append(m0, t1)

    m0_matched, t1_matched = operator(y)

    if index_of_scaling_parameter is not None:
        torch.testing.assert_close(m0_matched, m0, atol=1e-3, rtol=0.0)
    torch.testing.assert_close(t1_matched, t1, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('shape', [(5, 4, 3)])
@pytest.mark.parametrize(
    'index_of_scaling_parameter',
    [None, -2, 0],
    ids=['dont_predict_scale', 'predict_scale_negative_index', 'predict_scale'],
)
def test_dictionary_matching_op_append(
    shape: tuple[int], dtype: torch.dtype, index_of_scaling_parameter: int | None
) -> None:
    """Test dictionary matching with concatenated entries."""
    rng = RandomGenerator(2)
    m0_1 = rng.rand_tensor(shape, dtype=dtype, low=0.2, high=1.0)
    t1_1 = rng.rand_tensor(shape, dtype=dtype.to_real(), low=0.1, high=1.0)

    m0_2 = rng.rand_tensor(shape, dtype=dtype, low=0.2, high=1.0)
    t1_2 = rng.rand_tensor(shape, dtype=dtype.to_real(), low=0.1, high=1.0)

    model = InversionRecovery(rng.float32_tensor(5))

    # dictionary matching when appending the individual tensors
    m0_cat = torch.cat((m0_1, m0_2))
    t1_cat = torch.cat((t1_1, t1_2))
    operator = DictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter)
    operator.append(m0_1, t1_1)
    operator.append(m0_2, t1_2)
    (y,) = model(m0_cat, t1_cat)
    m0_matched, t1_matched = operator(y)

    # dictionary matching when appending the concatenated tensors
    operator_cat = DictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter)
    operator_cat.append(m0_cat, t1_cat)
    (y_cat,) = model(m0_cat, t1_cat)
    m0_matched_cat, t1_matched_cat = operator(y_cat)

    if index_of_scaling_parameter is not None:
        torch.testing.assert_close(m0_matched, m0_matched_cat, atol=1e-3, rtol=0.0)
    torch.testing.assert_close(t1_matched, t1_matched_cat, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize(
    'index_of_scaling_parameter',
    [None, -2, 0],
    ids=['dont_predict_scale', 'predict_scale_negative_index', 'predict_scale'],
)
def test_dictionary_matching_op_empty_dictionary(index_of_scaling_parameter: int | None) -> None:
    """Test dictionary matching when no entries have been appended before."""
    rng = RandomGenerator(2)
    model = InversionRecovery(rng.float32_tensor(5))

    operator = DictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter)
    y = torch.zeros(5, 5, 4, 3)
    with pytest.raises(KeyError, match=r'No keys in the dictionary\. Please first add some x values using `append`\.'):
        _ = operator(y)


def test_dictionary_matching_op_not_differentiable() -> None:
    """Test dictionary matching when the input signal requires gradients."""
    rng = RandomGenerator(2)
    model = InversionRecovery(rng.float32_tensor(5))
    operator = DictionaryMatchOp(model)
    operator.append(torch.ones(5, 4, 3), torch.ones(5, 4, 3))
    y = torch.zeros(5, 5, 4, 3)
    with pytest.raises(ValueError, match='differentiable'):
        _ = operator(y.requires_grad_(True))


@pytest.mark.cuda
def test_dictionary_matching_op_cuda() -> None:
    """Test dictionary matching operator works on CUDA devices."""
    rng = RandomGenerator(2)
    shape = (5, 4, 3)
    ti = rng.float32_tensor(5)
    m0 = rng.rand_tensor(shape, dtype=torch.float32, low=0.2, high=1.0)
    t1 = rng.rand_tensor(shape, dtype=torch.float32, low=0.1, high=1.0)

    # Create on CPU, transfer to GPU, run on GPU.
    model = InversionRecovery(ti)
    (y,) = model(m0, t1)
    operator = DictionaryMatchOp(model)
    operator.append(m0, t1)
    operator.cuda()
    result = operator(y.cuda())
    assert all(tensor.is_cuda for tensor in result)

    # Create directly on GPU, run on GPU.
    model = InversionRecovery(ti.cuda())
    operator = DictionaryMatchOp(model)
    operator.append(m0.cuda(), t1.cuda())
    result = operator(y.cuda())
    assert all(tensor.is_cuda for tensor in result)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('invalid_value', [0.0, float('nan'), float('inf')])
@pytest.mark.parametrize('predict_scale', [False, True])
@pytest.mark.parametrize('populated', [False, True])
def test_dictionary_matching_op_invalid_atoms(
    dtype: torch.dtype, invalid_value: float, predict_scale: bool, populated: bool
) -> None:
    """Reject an invalid append without changing any dictionary entries."""

    def model(scale: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (scale * x.unsqueeze(0).expand(2, -1),)

    operator = DictionaryMatchOp(model, index_of_scaling_parameter=0 if predict_scale else None)
    scale = torch.ones(1, dtype=dtype)
    if populated:
        operator.append(scale, torch.ones(1, dtype=dtype))
    old_x = tuple(operator.x)
    old_y = operator.y
    old_inverse_norm_y = operator.inverse_norm_y

    with pytest.raises(ValueError, match='finite, nonzero norms'):
        operator.append(scale, torch.tensor([2.0, invalid_value], dtype=dtype))

    assert all(old is new for old, new in zip(old_x, operator.x, strict=True))
    assert operator.y is old_y
    assert operator.inverse_norm_y is old_inverse_norm_y
    operator.append(scale, torch.ones(1, dtype=dtype))
    matched_scale, matched_x = operator(torch.ones(2, 1, dtype=dtype))
    torch.testing.assert_close(matched_x, torch.ones(1, dtype=dtype))
    if predict_scale:
        torch.testing.assert_close(matched_scale, matched_x.reciprocal())
    assert torch.isfinite(operator.y).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('amplitude', [1e-30, 1e30])
@pytest.mark.parametrize('predict_scale', [False, True])
def test_dictionary_matching_op_signal_magnitude(dtype: torch.dtype, amplitude: float, predict_scale: bool) -> None:
    """Normalize finite nonzero signals without squaring their original magnitudes."""

    def model(scale: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor]:
        signal = torch.stack((x, 1 - x)).to(dtype) * amplitude
        if signal.is_complex():
            signal = signal * 1j
        return (scale * signal,)

    operator = DictionaryMatchOp(model, index_of_scaling_parameter=0 if predict_scale else None)
    scale = torch.ones(1)
    operator.append(scale, torch.tensor([0.0]))
    operator.append(scale, torch.tensor([1.0]))
    (signal,) = model(scale, torch.tensor([1.0]))
    matched_scale, matched_x = operator(signal / amplitude)
    torch.testing.assert_close(matched_x, torch.ones(1))
    if predict_scale:
        torch.testing.assert_close(matched_scale, torch.full((1,), 1 / amplitude, dtype=dtype))
    assert operator.y.dtype == dtype
    torch.testing.assert_close(torch.linalg.norm(operator.y, dim=0), torch.ones(2))


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('predict_scale', [False, True])
def test_dictionary_matching_op_subnormal_signal(dtype: torch.dtype, predict_scale: bool) -> None:
    """Only require a representable inverse norm when estimating scale."""

    def model(scale: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (scale * torch.stack((x, torch.zeros_like(x))),)

    operator = DictionaryMatchOp(model, index_of_scaling_parameter=0 if predict_scale else None)
    scale = torch.ones(1, dtype=dtype)
    x = torch.tensor([1e-40], dtype=dtype)
    if predict_scale:
        old_y = operator.y
        old_inverse_norm_y = operator.inverse_norm_y
        with pytest.raises(ValueError, match='finite inverse norms'):
            operator.append(scale, x)
        assert not operator.x
        assert operator.y is old_y
        assert operator.inverse_norm_y is old_inverse_norm_y
    else:
        operator.append(scale, x)
        torch.testing.assert_close(operator.y, torch.tensor([[1.0], [0.0]], dtype=dtype))
        _, matched_x = operator(torch.tensor([[1.0], [0.0]], dtype=dtype))
        torch.testing.assert_close(matched_x, x)
