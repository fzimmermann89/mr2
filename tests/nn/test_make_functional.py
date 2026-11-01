"""Tests for networks with explicit parameter inputs."""

from functools import partial

import pytest
import torch
from mr2.algorithms.optimizers import adam
from mr2.nn import make_functional
from mr2.operators.functionals import L2NormSquared
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_forward_and_gradients(dtype: torch.dtype) -> None:
    """Match forward values and derivatives without detaching weights or inputs."""
    module = nn.Linear(2, 1, dtype=dtype)
    x = torch.ones(3, 2, dtype=dtype, requires_grad=True)
    operator, params = make_functional(module)
    assert all(p is original for p, original in zip(params, module.parameters(), strict=True))
    expected = module(x)
    (actual,) = operator(*params, x)
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected.abs().square().sum(), (*params, x))
    actual_grad = torch.autograd.grad(actual.abs().square().sum(), (*params, x))
    torch.testing.assert_close(actual_grad, expected_grad)


def test_replacement_weights() -> None:
    """Use the supplied weights while leaving the original module untouched."""
    module = nn.Linear(2, 1, bias=False)
    original = module.weight.detach().clone()
    operator, _ = make_functional(module)
    (output,) = operator(torch.full_like(module.weight, 3), torch.ones(1, 2))
    torch.testing.assert_close(output, torch.tensor([[6.0]]))
    torch.testing.assert_close(module.weight, original)


def test_frozen_parameters() -> None:
    """Expose trainable weights only and retain the frozen bias."""
    module = nn.Linear(2, 1)
    module.bias.requires_grad_(False)
    operator, params = make_functional(module)
    assert len(params) == 1
    assert params[0] is module.weight
    (actual,) = operator(torch.zeros_like(params[0]), torch.ones(1, 2))
    torch.testing.assert_close(actual, module.bias[None])


def test_explicit_arguments_and_tuple_output() -> None:
    """Pass multiple positional inputs and preserve multiple outputs."""

    class Pair(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x + y, x - y

    x, y = torch.tensor(2.0), torch.tensor(3.0)
    operator, params = make_functional(Pair())
    assert params == ()
    actual = operator(x, y)
    torch.testing.assert_close(actual, (torch.tensor(5.0), torch.tensor(-1.0)))


def test_parameter_count() -> None:
    """Reject missing parameter arguments instead of falling back silently."""
    operator, params = make_functional(nn.Linear(2, 1))
    with pytest.raises(ValueError, match='Expected at least 2 parameter tensors'):
        operator(*params[:-1])


def test_unsupported_output() -> None:
    """Reject outputs that cannot be composed as MR2 tensor operators."""

    class Mapping(nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {'output': x}

    operator, _ = make_functional(Mapping())
    with pytest.raises(TypeError, match='tensor or a tuple of tensors'):
        operator(torch.ones(1))


def test_tied_parameters() -> None:
    """Expose a shared weight once and use replacements for both occurrences."""
    linear = nn.Linear(2, 2, bias=False)
    module = nn.Sequential(linear, linear)
    operator, params = make_functional(module)
    assert len(params) == 1
    (actual,) = operator(torch.ones_like(params[0]), torch.ones(1, 2))
    torch.testing.assert_close(actual, torch.full((1, 2), 4.0))


def test_parametrization() -> None:
    """Keep registered parametrizations active with explicit original weights."""

    class Square(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.square()

    module = nn.Linear(2, 1, bias=False)
    parametrize.register_parametrization(module, 'weight', Square())
    operator, params = make_functional(module)
    (actual,) = operator(torch.full_like(params[0], -2.0), torch.ones(1, 2))
    torch.testing.assert_close(actual, torch.tensor([[8.0]]))


def test_batchnorm_buffers() -> None:
    """Isolate BatchNorm updates and retain the chosen training mode."""
    module = nn.BatchNorm1d(2).double()
    x = torch.tensor([[1.0, 2.0], [3.0, 6.0]], dtype=torch.float64)
    initial = tuple(buffer.clone() for buffer in module.buffers())
    operator, params = make_functional(module)
    (first,) = operator(*params, x)
    (second,) = operator(*params, x)
    torch.testing.assert_close(first, second)
    assert module.training
    torch.testing.assert_close(tuple(module.buffers()), initial)
    gradients = torch.autograd.grad(first.square().sum(), params)
    assert all(torch.isfinite(g).all() for g in gradients)


def test_buffer_snapshot_in_eval_mode() -> None:
    """Evaluate against the captured buffer state without updating the original."""
    module = nn.BatchNorm1d(2).eval()
    x = torch.ones(2, 2)
    expected = module(x)
    operator, params = make_functional(module)
    assert module.running_mean is not None
    module.running_mean.fill_(10.0)
    (actual,) = operator(*params, x)
    torch.testing.assert_close(actual, expected)
    assert not module.training
    torch.testing.assert_close(module.running_mean, torch.full((2,), 10.0))


@pytest.mark.parametrize('checkpointed', [False, True])
def test_derivatives(checkpointed: bool) -> None:
    """Preserve first and second derivatives, also with operator checkpointing."""
    operator, _ = make_functional(nn.Linear(2, 1, bias=False).double())
    weight = torch.tensor([[2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    x = torch.ones(1, 2, dtype=torch.float64, requires_grad=True)
    evaluate = partial(checkpoint, operator, use_reentrant=False) if checkpointed else operator
    assert torch.autograd.gradcheck(evaluate, (weight, x))
    assert torch.autograd.gradgradcheck(evaluate, (weight, x))


@pytest.mark.parametrize('fit_input', [False, True])
def test_optimization(fit_input: bool) -> None:
    """Fit weights, optionally with the input, without changing initial tensors."""
    module = nn.Linear(1, 1, bias=False)
    nn.init.constant_(module.weight, 0.25)
    z = torch.ones(1, 1)
    operator, params = make_functional(module)
    objective = 0.5 * L2NormSquared(torch.ones(1, 1)) @ operator
    if fit_input:
        fitted = adam(objective, (*params, z), learning_rate=0.05, max_iterations=100)
        assert not torch.allclose(fitted[-1], z)
    else:
        fitted = (*adam(lambda *p: objective(*p, z), params, learning_rate=0.05, max_iterations=100), z)
    (loss,) = objective(*fitted)
    assert loss < 1e-3
    assert not torch.allclose(fitted[0], params[0])
    torch.testing.assert_close(module.weight, torch.full_like(module.weight, 0.25))
    torch.testing.assert_close(z, torch.ones_like(z))
