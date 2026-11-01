"""Tests for networks with explicit parameter inputs."""

import pytest
import torch
from mr2.algorithms.optimizers import adam
from mr2.nn import make_functional
from mr2.operators import Operator
from mr2.operators.functionals import L2NormSquared
from torch import nn
from torch.nn.utils import parametrize


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_forward_and_gradients(dtype: torch.dtype) -> None:
    """Match forward values and derivatives without detaching weights or inputs."""
    module = nn.Linear(2, 1, dtype=dtype)
    x = torch.ones(3, 2, dtype=dtype, requires_grad=True)
    operator, params = make_functional(module)
    assert isinstance(operator, Operator)
    assert all(p is original for p, original in zip(params, module.parameters(), strict=True))
    expected = module(x)
    (actual,) = operator(*params, x)
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected.abs().square().sum(), (*params, x))
    actual_grad = torch.autograd.grad(actual.abs().square().sum(), (*params, x))
    for actual_value, expected_value in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual_value, expected_value)


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

    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(2, 2, bias=False)
            self.second = nn.Linear(2, 2, bias=False)
            self.second.weight = self.first.weight

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.first(x) + self.second(x)

    module = Tied()
    operator, params = make_functional(module)
    assert len(params) == 1
    (actual,) = operator(torch.ones_like(params[0]), torch.ones(1, 2))
    torch.testing.assert_close(actual, torch.full((1, 2), 4.0))
    assert module.first.weight is module.second.weight


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
    initial = {name: value.clone() for name, value in module.named_buffers()}
    operator, params = make_functional(module)
    (first,) = operator(*params, x)
    (second,) = operator(*params, x)
    torch.testing.assert_close(first, second)
    assert module.training
    for name, buffer in module.named_buffers():
        torch.testing.assert_close(buffer, initial[name])
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


def test_adam_composed_objective() -> None:
    """Optimize the exposed weights through a composed functional, not the input."""
    module = nn.Linear(2, 1, bias=False)
    nn.init.zeros_(module.weight)
    z = torch.tensor([[1.0, 2.0]])
    operator, params = make_functional(module)
    objective = 0.5 * L2NormSquared(torch.ones(1, 1)) @ operator

    class FixedInput(Operator):
        def forward(self, *p: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return objective(*p, z)

    fitted = adam(FixedInput(), params, learning_rate=0.05, max_iterations=50)
    (loss,) = objective(*fitted, z)
    assert loss < 1e-3
    torch.testing.assert_close(module.weight, torch.zeros_like(module.weight))
    torch.testing.assert_close(z, torch.tensor([[1.0, 2.0]]))


def test_higher_order_derivatives() -> None:
    """Keep gradients through parameter substitution for second derivatives."""
    operator, _ = make_functional(nn.Linear(2, 1, bias=False).double())
    weight = torch.tensor([[2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    x = torch.ones(1, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(operator, (weight, x))
    assert torch.autograd.gradgradcheck(operator, (weight, x))


def test_joint_weight_and_input_optimization() -> None:
    """Fit both weights and the latent input through the composed operator."""
    module = nn.Linear(1, 1, bias=False)
    nn.init.constant_(module.weight, 0.25)
    z = torch.ones(1, 1)
    operator, params = make_functional(module)
    objective = 0.5 * L2NormSquared(torch.ones(1, 1)) @ operator
    fitted = adam(objective, (*params, z), learning_rate=0.05, max_iterations=100)
    (loss,) = objective(*fitted)
    assert loss < 1e-3
    assert not torch.allclose(fitted[0], params[0])
    assert not torch.allclose(fitted[-1], z)
    torch.testing.assert_close(module.weight, torch.full_like(module.weight, 0.25))
    torch.testing.assert_close(z, torch.ones_like(z))


def test_inputs_are_not_bound() -> None:
    """Reuse the same adapter and weights with different forward inputs."""
    module = nn.Linear(1, 1, bias=False)
    nn.init.ones_(module.weight)
    operator, params = make_functional(module)
    (first,) = operator(*params, torch.tensor([[2.0]]))
    (second,) = operator(*params, torch.tensor([[3.0]]))
    torch.testing.assert_close(first, torch.tensor([[2.0]]))
    torch.testing.assert_close(second, torch.tensor([[3.0]]))
