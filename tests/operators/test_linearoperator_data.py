"""Linear operators accept objects exposing tensor data."""

from dataclasses import dataclass

import pytest
import torch
from mr2.data import IData, IHeader, KData
from mr2.operators import FastFourierOp, IdentityOp, Jacobian, LinearOperator, Operator
from mr2.utils.typing import HasTensorData
from typing_extensions import assert_type

from tests.operators.test_autograd_linop import DifferentiableOperator


@dataclass
class TensorData:
    """A minimal data container independent of mr2's data classes."""

    data: torch.Tensor


@dataclass(frozen=True)
class FrozenTensorData:
    """A container whose data cannot be reassigned."""

    data: torch.Tensor


@pytest.mark.parametrize('container', [TensorData, FrozenTensorData])
@pytest.mark.parametrize(
    'operator',
    [
        IdentityOp(),
        FastFourierOp(dim=(-1,)),
        FastFourierOp(dim=(-1,)).H,
        FastFourierOp(dim=(-1,)).gram,
        FastFourierOp(dim=(-1,)) @ FastFourierOp(dim=(-1,)),
        FastFourierOp(dim=(-1,)) + IdentityOp(),
        FastFourierOp(dim=(-1,)) * 2j,
        2j * FastFourierOp(dim=(-1,)),
    ],
)
def test_linearoperator_data(container: type[TensorData] | type[FrozenTensorData], operator: LinearOperator) -> None:
    """Wrapped inputs give the same tensors and gradients as bare tensor inputs."""
    tensor = torch.randn(2, 3, dtype=torch.complex64, requires_grad=True)
    wrapped: HasTensorData = container(tensor)
    (expected,) = operator(tensor)
    (actual,) = assert_type(operator(wrapped), tuple[torch.Tensor])
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected)
    (actual_gradient,) = torch.autograd.grad(actual.abs().sum(), tensor)
    (expected_gradient,) = torch.autograd.grad(expected.abs().sum(), tensor)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_linearoperator_mr_data(consistently_shaped_kdata: KData) -> None:
    """KData and IData satisfy the input protocol for concrete operators."""
    kdata = consistently_shaped_kdata
    idata = IData(data=kdata.data, header=IHeader.from_kheader(kdata.header))
    identity = IdentityOp()
    assert assert_type(identity(kdata), tuple[torch.Tensor])[0] is kdata.data
    assert assert_type(identity(idata), tuple[torch.Tensor])[0] is idata.data
    fft = FastFourierOp(dim=(-1,))
    torch.testing.assert_close(assert_type(fft(idata), tuple[torch.Tensor]), fft(idata.data))
    torch.testing.assert_close(assert_type(fft.H(kdata), tuple[torch.Tensor]), fft.H(kdata.data))


@pytest.mark.parametrize('adjoint', [False, True])
def test_linearoperator_data_adjoint_as_backward(adjoint: bool) -> None:
    """Unwrap before entering the custom autograd function."""
    tensor = torch.tensor([3.0], requires_grad=True)
    operator = DifferentiableOperator().H if adjoint else DifferentiableOperator()
    (actual,) = operator(TensorData(tensor))
    torch.testing.assert_close(actual, 2 * tensor)
    (gradient,) = torch.autograd.grad(actual.sum(), tensor)
    torch.testing.assert_close(gradient, torch.full_like(tensor, 2))


def test_linearoperator_data_multiple_inputs() -> None:
    """Keep Jacobian's multiple-input calls working with tensors and containers."""

    class Product(Operator[torch.Tensor, torch.Tensor, tuple[torch.Tensor]]):
        """A two-input operator for checking its Jacobian."""

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor]:
            """Multiply two inputs."""
            return (x * y,)

    x0, y0 = torch.tensor([2.0]), torch.tensor([3.0])
    x, y = torch.tensor([5.0]), torch.tensor([7.0])
    jacobian = Jacobian(Product(), x0, y0)  # type: ignore[arg-type]
    torch.testing.assert_close(jacobian(x, y), (y0 * x + x0 * y,))
    torch.testing.assert_close(jacobian(TensorData(x), FrozenTensorData(y)), jacobian(x, y))
