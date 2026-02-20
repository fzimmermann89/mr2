import pytest
import torch
from mr2.operators import Operator, OperatorMatrix
from mr2.utils import RandomGenerator


class DummyOperator(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Dummy single-input operator for testing."""

    def __init__(self, exponent: float):
        super().__init__()
        self.exponent = exponent

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return ((x**self.exponent).sum().unsqueeze(0),)


def test_operatormatrix_shape():
    a = DummyOperator(2.0)
    b = DummyOperator(3.0)
    matrix = OperatorMatrix([[a, b], [b, a]])
    assert matrix.shape == (2, 2)


def test_operatormatrix_shorthand_vertical():
    a = DummyOperator(2.0)
    b = DummyOperator(3.0)
    rng = RandomGenerator(0)
    x = rng.float32_tensor(8)

    matrix1 = a % b
    assert isinstance(matrix1, OperatorMatrix)
    assert matrix1.shape == (2, 1)
    torch.testing.assert_close(matrix1(x), (*a(x), *b(x)))

    matrix2 = b % (matrix1 % a)
    assert isinstance(matrix2, OperatorMatrix)
    assert matrix2.shape == (4, 1)
    torch.testing.assert_close(matrix2(x), (*b(x), *matrix1(x), *a(x)))


def test_operatormatrix_shorthand_horizontal():
    a = DummyOperator(2.0)
    b = DummyOperator(3.0)
    rng = RandomGenerator(0)
    x1 = rng.float32_tensor(8)
    x2 = rng.float32_tensor(8)
    x3 = rng.float32_tensor(8)

    matrix1 = a | b
    assert isinstance(matrix1, OperatorMatrix)
    assert matrix1.shape == (1, 2)
    torch.testing.assert_close(matrix1(x1, x2), (a(x1)[0] + b(x2)[0],))

    matrix2 = b | (matrix1 | a)
    assert isinstance(matrix2, OperatorMatrix)
    assert matrix2.shape == (1, 4)
    torch.testing.assert_close(matrix2(x1, x2, x3, x1), (b(x1)[0] + matrix1(x2, x3)[0] + a(x1)[0],))


def test_operatormatrix_stacking_error():
    a = DummyOperator(2.0)
    b = DummyOperator(3.0)
    matrix1 = OperatorMatrix([[a, b], [b, a]])
    matrix2 = OperatorMatrix([[a], [b]])
    matrix3 = OperatorMatrix([[a, b]])

    with pytest.raises(ValueError, match='Shape mismatch'):
        matrix1 % matrix2
    with pytest.raises(ValueError, match='Shape mismatch'):
        matrix1 | matrix3
    with pytest.raises(ValueError, match='Shape mismatch'):
        matrix1 | a
    with pytest.raises(ValueError, match='Shape mismatch'):
        matrix1 % a
