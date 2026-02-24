from typing import assert_type

import pytest
import torch
from typing_extensions import Unpack

from mr2.operators import LinearOperator, LinearOperatorMatrix, Operator, OperatorStack
from mr2.utils import RandomGenerator


class DummyOperator(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Dummy single-input operator for testing."""

    def __init__(self, exponent: float):
        super().__init__()
        self.exponent = exponent

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return ((x**self.exponent).sum().unsqueeze(0),)


class IdentityLinearOp(LinearOperator):
    """Simple linear identity operator used for type-inference tests."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x,)


class SingleInputOp(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Simple single-input, single-output operator for type-inference tests."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x,)


def test_operatorstack_shape():
    a = DummyOperator(2.0)
    b = DummyOperator(3.0)
    matrix = OperatorStack([[a, b], [b, a]])
    assert matrix.shape == (2, 2)


def test_operatorstack_shorthand_vertical():
    a = DummyOperator(2.0)
    b = DummyOperator(3.0)
    rng = RandomGenerator(0)
    x = rng.float32_tensor(8)

    matrix1 = a % b
    assert isinstance(matrix1, OperatorStack)
    assert matrix1.shape == (2, 1)
    torch.testing.assert_close(matrix1(x), (*a(x), *b(x)))

    matrix2 = b % (matrix1 % a)
    assert isinstance(matrix2, OperatorStack)
    assert matrix2.shape == (4, 1)
    torch.testing.assert_close(matrix2(x), (*b(x), *matrix1(x), *a(x)))


def test_operatorstack_shorthand_horizontal():
    a = DummyOperator(2.0)
    b = DummyOperator(3.0)
    rng = RandomGenerator(0)
    x1 = rng.float32_tensor(8)
    x2 = rng.float32_tensor(8)
    x3 = rng.float32_tensor(8)

    matrix1 = a | b
    assert isinstance(matrix1, OperatorStack)
    assert matrix1.shape == (1, 2)
    torch.testing.assert_close(matrix1(x1, x2), (a(x1)[0] + b(x2)[0],))

    matrix2 = b | (matrix1 | a)
    assert isinstance(matrix2, OperatorStack)
    assert matrix2.shape == (1, 4)
    torch.testing.assert_close(matrix2(x1, x2, x3, x1), (b(x1)[0] + matrix1(x2, x3)[0] + a(x1)[0],))


def test_operatorstack_stacking_error():
    a = DummyOperator(2.0)
    b = DummyOperator(3.0)
    matrix1 = OperatorStack([[a, b], [b, a]])
    matrix2 = OperatorStack([[a], [b]])
    matrix3 = OperatorStack([[a, b]])

    with pytest.raises(ValueError, match='Shape mismatch'):
        matrix1 % matrix2
    with pytest.raises(ValueError, match='Shape mismatch'):
        matrix1 | matrix3
    with pytest.raises(ValueError, match='Shape mismatch'):
        matrix1 | a
    with pytest.raises(ValueError, match='Shape mismatch'):
        matrix1 % a


def test_operatorstack_linear_stacking_assert_type() -> None:
    lin1: LinearOperator = IdentityLinearOp()
    lin2: LinearOperator = IdentityLinearOp()
    linear_row: LinearOperatorMatrix = lin1 | lin2
    linear_col: LinearOperatorMatrix = lin1 % lin2

    assert_type(lin1 | lin2, LinearOperatorMatrix)
    assert_type(lin1 % lin2, LinearOperatorMatrix)
    assert_type(lin1 | linear_row, LinearOperatorMatrix)
    assert_type(lin1 % linear_col, LinearOperatorMatrix)
    assert_type(linear_row | lin1, LinearOperatorMatrix)
    assert_type(linear_col % lin1, LinearOperatorMatrix)


def test_operatorstack_mixed_stacking_assert_type() -> None:
    lin: LinearOperator = IdentityLinearOp()
    linear_row: LinearOperatorMatrix = lin | lin
    linear_col: LinearOperatorMatrix = lin % lin

    nonlinear = SingleInputOp()
    nonlinear_stack = OperatorStack([[nonlinear]])

    assert_type(lin | nonlinear, OperatorStack)
    assert_type(lin % nonlinear, OperatorStack)
    assert_type(linear_row | nonlinear, OperatorStack)
    assert_type(linear_col % nonlinear, OperatorStack)
    assert_type(lin | nonlinear_stack, OperatorStack)
    assert_type(lin % nonlinear_stack, OperatorStack)
    assert_type(linear_row | nonlinear_stack, OperatorStack)
    assert_type(linear_col % nonlinear_stack, OperatorStack)
    assert_type(nonlinear | nonlinear, OperatorStack)
    assert_type(nonlinear % nonlinear, OperatorStack)
