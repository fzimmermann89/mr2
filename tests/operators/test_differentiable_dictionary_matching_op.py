"""Tests for the Differentiable Dictionary Matching Operator."""

from typing import cast

import pytest
import torch
from mr2.operators import DifferentiableDictionaryMatchOp
from mr2.operators.models import InversionRecovery
from mr2.utils import RandomGenerator


def _make_operator_and_signal(
    dtype: torch.dtype,
    index_of_scaling_parameter: int | None,
    batch_size: int = 1024 * 1024,
    shape: tuple[int, ...] = (5, 4, 3),
    seed: int = 2,
):
    rng = RandomGenerator(seed)
    model = InversionRecovery(rng.float32_tensor(5))
    m0 = rng.rand_tensor(shape, dtype=dtype, low=0.2, high=1.0)
    t1 = rng.rand_tensor(shape, dtype=dtype.to_real(), low=0.1, high=1.0)
    (y,) = model(m0, t1)
    op = DifferentiableDictionaryMatchOp(
        model,
        index_of_scaling_parameter=index_of_scaling_parameter,
        batch_size=batch_size,
    )
    op.append(m0, t1)
    return op, y, m0, t1


class TestDifferentiableDictionaryMatchOp:
    """Tests for DifferentiableDictionaryMatchOp."""

    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
    @pytest.mark.parametrize('index_of_scaling_parameter', [None, -2, 0], ids=['no_scale', 'scale_neg', 'scale_pos'])
    @pytest.mark.parametrize('batch_size', [1, 1024 * 1024], ids=['batched', 'single_chunk'])
    def test_matching(
        self,
        dtype: torch.dtype,
        index_of_scaling_parameter: int | None,
        batch_size: int,
    ):
        """Matching against own dictionary entries recovers parameters."""
        op, y, m0, t1 = _make_operator_and_signal(dtype, index_of_scaling_parameter, batch_size)
        m0_hat, t1_hat = op(y)
        torch.testing.assert_close(t1_hat, t1, atol=1e-4, rtol=0)
        if index_of_scaling_parameter is not None:
            torch.testing.assert_close(m0_hat, m0, atol=1e-3, rtol=0)

    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
    @pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
    def test_append_consistency(
        self,
        dtype: torch.dtype,
        index_of_scaling_parameter: int | None,
    ):
        """Sequential appends match a single concatenated append."""
        rng = RandomGenerator(2)
        m0_1 = rng.rand_tensor((5, 4, 3), dtype=dtype, low=0.2, high=1.0)
        t1_1 = rng.rand_tensor((5, 4, 3), dtype=dtype.to_real(), low=0.1, high=1.0)
        m0_2 = rng.rand_tensor((5, 4, 3), dtype=dtype, low=0.2, high=1.0)
        t1_2 = rng.rand_tensor((5, 4, 3), dtype=dtype.to_real(), low=0.1, high=1.0)
        model = InversionRecovery(rng.float32_tensor(5))

        m0_cat, t1_cat = torch.cat((m0_1, m0_2)), torch.cat((t1_1, t1_2))
        (y,) = model(m0_cat, t1_cat)

        op_seq = DifferentiableDictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter)
        op_seq.append(m0_1, t1_1).append(m0_2, t1_2)

        op_cat = DifferentiableDictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter)
        op_cat.append(m0_cat, t1_cat)

        for r_seq, r_cat in zip(op_seq(y), op_cat(y), strict=True):
            torch.testing.assert_close(r_seq, r_cat, atol=1e-4, rtol=0)

    @pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
    def test_empty_dictionary_raises(self, index_of_scaling_parameter: int | None):
        """Calling operator without append raises KeyError."""
        rng = RandomGenerator(2)
        model = InversionRecovery(rng.float32_tensor(5))
        op = DifferentiableDictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter)
        with pytest.raises(KeyError, match='No keys'):
            op(torch.zeros(5, *(5, 4, 3)))

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex64], ids=['float64', 'complex64'])
    @pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
    def test_prior_zero_precision_is_noop(
        self,
        dtype: torch.dtype,
        index_of_scaling_parameter: int | None,
    ):
        """Prior with zero precision does not change the result."""
        op, y, m0, t1 = _make_operator_and_signal(dtype, index_of_scaling_parameter)
        prior = (torch.zeros_like(m0), torch.zeros_like(t1))
        prior_precision = (
            torch.tensor(0.0, dtype=dtype.to_real()),
            torch.tensor(0.0, dtype=dtype.to_real()),
        )
        input_signal = 1.3 * y
        for r1, r2 in zip(
            op(input_signal),
            op(input_signal, prior=prior, prior_precision=prior_precision),
            strict=True,
        ):
            torch.testing.assert_close(r1, r2, atol=1e-6, rtol=0)

    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
    @pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
    def test_prior_pulls_result(
        self,
        dtype: torch.dtype,
        index_of_scaling_parameter: int | None,
    ):
        """Strong prior pulls result towards prior values."""
        op, y, m0, t1 = _make_operator_and_signal(dtype, index_of_scaling_parameter)
        prior = (torch.ones_like(m0) * 0.5, torch.ones_like(t1) * 0.5)
        prior_precision = (
            torch.tensor(1e6, dtype=dtype.to_real()),
            torch.tensor(1e6, dtype=dtype.to_real()),
        )

        result_no_prior = op(y)
        result_with_prior = op(y, prior=prior, prior_precision=prior_precision)

        for r, p in zip(result_with_prior, prior, strict=True):
            assert (r - p).abs().mean() < 0.01
        for r_np, r_wp in zip(result_no_prior, result_with_prior, strict=True):
            assert not torch.allclose(r_np, r_wp)

    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
    @pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
    def test_single_prior_precision_applies_to_all_priors(
        self,
        dtype: torch.dtype,
        index_of_scaling_parameter: int | None,
    ):
        """Single prior precision tensor is reused for every prior parameter."""
        op, y, m0, t1 = _make_operator_and_signal(dtype, index_of_scaling_parameter)
        prior = (torch.ones_like(m0) * 0.5, torch.ones_like(t1) * 0.5)
        prior_precision = torch.tensor(10.0, dtype=dtype.to_real())

        result_single = op(y, prior=prior, prior_precision=prior_precision)
        result_tuple = op(y, prior=prior, prior_precision=(prior_precision, prior_precision))

        for single, tuple_ in zip(result_single, result_tuple, strict=True):
            torch.testing.assert_close(single, tuple_)

    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
    @pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
    def test_return_signal_matches_input_signal_for_dictionary_entries(
        self,
        dtype: torch.dtype,
        index_of_scaling_parameter: int | None,
    ):
        """Returned signal approximation matches exact dictionary entries."""
        op, y, m0, t1 = _make_operator_and_signal(dtype, index_of_scaling_parameter)
        params, y_hat = op(y, return_signal=True)
        m0_hat, t1_hat = params

        torch.testing.assert_close(t1_hat, t1, atol=1e-4, rtol=0)
        if index_of_scaling_parameter is not None:
            torch.testing.assert_close(m0_hat, m0, atol=1e-3, rtol=0)
        assert y_hat.shape == y.shape
        assert y_hat.dtype == y.dtype
        torch.testing.assert_close(y_hat, y, atol=1e-4, rtol=0)

    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
    @pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
    def test_return_signal_backward_in_input_signal(
        self,
        dtype: torch.dtype,
        index_of_scaling_parameter: int | None,
    ):
        """Backward pass through returned signal approximation produces finite gradients."""
        op, y, _, _ = _make_operator_and_signal(dtype, index_of_scaling_parameter)
        y = y.detach().requires_grad_(True)
        _, y_hat = op(y, return_signal=True)
        loss = y_hat.real.float().square().mean()
        loss.backward()
        assert y.grad is not None
        assert torch.isfinite(y.grad).all()

    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
    @pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
    def test_differentiable_in_input_signal(
        self,
        dtype: torch.dtype,
        index_of_scaling_parameter: int | None,
    ):
        """Backward pass through input signal produces finite gradients."""
        op, y, _, _ = _make_operator_and_signal(dtype, index_of_scaling_parameter)
        y = y.detach().requires_grad_(True)
        loss = sum(o.real.float().abs().mean() for o in op(y))
        loss.backward()
        assert y.grad is not None
        assert torch.isfinite(y.grad).all()

    @pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
    def test_autograd_gradcheck(self, index_of_scaling_parameter: int | None):
        """Verify gradient correctness via finite differences."""
        op, y, _, _ = _make_operator_and_signal(torch.float64, index_of_scaling_parameter, shape=(3, 2))
        y = y.detach().requires_grad_(True)
        torch.autograd.gradcheck(op, (y,), raise_exception=True)

    @pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
    def test_autograd_gradcheck_complex(self, index_of_scaling_parameter: int | None):
        """Verify gradient correctness for complex inputs via finite differences."""
        op, y, _, _ = _make_operator_and_signal(torch.complex128, index_of_scaling_parameter, shape=(3, 2))
        y = y.detach().requires_grad_(True)
        torch.autograd.gradcheck(op, (y,), raise_exception=True)

    @pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128], ids=['float64', 'complex128'])
    def test_autograd_gradcheck_with_prior(self, index_of_scaling_parameter: int | None, dtype: torch.dtype):
        """Verify gradient correctness with prior via finite differences."""
        op, y, m0, t1 = _make_operator_and_signal(dtype, index_of_scaling_parameter, shape=(3, 2))
        prior = (m0.detach(), t1.detach())
        prior_precision = (
            torch.tensor(1.0, dtype=dtype.to_real()),
            torch.ones(1, 2, dtype=dtype.to_real()),
        )
        y = y.detach().requires_grad_(True)
        torch.autograd.gradcheck(
            lambda sig: op(sig, prior=prior, prior_precision=prior_precision),
            (y,),
            raise_exception=True,
        )

    @pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
    def test_spatially_varying_prior_precision_pulls_high_precision_region_more(
        self, index_of_scaling_parameter: int | None
    ):
        """Higher broadcasted spatial precision pulls estimates closer to the prior."""
        op, y, _, _ = _make_operator_and_signal(torch.float32, index_of_scaling_parameter, shape=(4, 3))
        batch_shape = y.shape[1:]
        prior = (torch.full(batch_shape, 0.5), torch.full(batch_shape, 0.5))
        high_precision = torch.zeros(batch_shape[0], 1)
        high_precision[:2] = 200.0
        result = op(y, prior=prior, prior_precision=(high_precision, high_precision))

        high_region = high_precision.broadcast_to(batch_shape) > 0
        low_region = ~high_region
        for result_k, prior_k in zip(result, prior, strict=True):
            high_error = (result_k[high_region] - prior_k[high_region]).abs().mean()
            low_error = (result_k[low_region] - prior_k[low_region]).abs().mean()
            assert high_error < low_error

    def test_refinement_with_offgrid_target(self) -> None:
        """Test refinement between two points with scaling!= 1"""
        rng = RandomGenerator(2)
        model = InversionRecovery(rng.float32_tensor(5))
        t1_dict = torch.linspace(0.1, 1.0, 20)
        m0_dict = torch.ones_like(t1_dict)
        op = DifferentiableDictionaryMatchOp(model, index_of_scaling_parameter=0)
        op.append(m0_dict, t1_dict)
        t1_true = torch.tensor([0.37])
        m0_true = torch.tensor([0.6])
        (y,) = model(m0_true, t1_true)
        m0_hat, t1_hat = op(y)
        torch.testing.assert_close(t1_hat, t1_true, atol=5e-2, rtol=0)
        torch.testing.assert_close(m0_hat, m0_true, atol=5e-2, rtol=0)

    @pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['no_scale', 'scale'])
    def test_prior_validation_raises(self, index_of_scaling_parameter: int | None):
        """Invalid prior inputs raise clear errors."""
        op, y, m0, t1 = _make_operator_and_signal(torch.float32, index_of_scaling_parameter)
        prior = (m0, t1)
        prior_precision = (torch.tensor(1.0), torch.tensor(1.0))

        with pytest.raises(ValueError, match='prior and prior_precision'):
            op(y, prior=prior)
        with pytest.raises(ValueError, match='prior and prior_precision'):
            op(y, prior_precision=prior_precision)
        with pytest.raises(ValueError, match='Prior must be a tuple of tensors matching the number of parameters'):
            op(y, prior=(m0,), prior_precision=(torch.tensor(1.0),))
        with pytest.raises(ValueError, match='Prior must be a tuple of tensors matching the number of parameters'):
            op(y, prior=(m0, cast(torch.Tensor, 1.0)), prior_precision=(torch.tensor(1.0), torch.tensor(1.0)))
        with pytest.raises(ValueError, match='prior_precision must be a tuple of finite non-negative real tensors'):
            op(y, prior=prior, prior_precision=(torch.tensor(1.0),))
        with pytest.raises(ValueError, match='prior_precision must be a tuple of finite non-negative real tensors'):
            op(y, prior=prior, prior_precision=(torch.tensor(1.0), cast(torch.Tensor, 1.0)))
        with pytest.raises(ValueError, match='prior_precision must be a tuple of finite non-negative real tensors'):
            op(y, prior=(m0, t1), prior_precision=(torch.tensor(-1.0), torch.tensor(1.0)))
        with pytest.raises(ValueError, match='prior_precision must be a tuple of finite non-negative real tensors'):
            op(y, prior=(m0, t1), prior_precision=(torch.tensor(1.0 + 1.0j), torch.tensor(1.0)))
        for bad_value in (torch.tensor(float('nan')), torch.tensor(float('inf'))):
            with pytest.raises(ValueError, match='prior_precision must be a tuple of finite non-negative real tensors'):
                op(y, prior=(m0, t1), prior_precision=(bad_value, torch.tensor(1.0)))
        with pytest.raises(RuntimeError):
            op(y, prior=(torch.ones(2, 2), t1), prior_precision=prior_precision)
        with pytest.raises(RuntimeError):
            op(y, prior=prior, prior_precision=(torch.ones(2, 2), torch.tensor(1.0)))
