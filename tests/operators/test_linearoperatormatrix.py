from typing import Any

import pytest
import torch
from mr2.operators import EinsumOp, IdentityOp, LinearOperator, MagnitudeOp, ZeroOp
from mr2.operators.LinearOperatorMatrix import LinearOperatorMatrix
from mr2.utils import RandomGenerator
from typing_extensions import assert_type

from tests import dotproduct_adjointness_test


def random_linearop(size, rng):
    """Create a random LinearOperator."""
    return EinsumOp(rng.complex64_tensor(size), '... i j, ... j -> ... i')


def random_linearoperatormatrix(size, inner_size, rng):
    """Create a random LinearOperatorMatrix."""
    operators = [[random_linearop(inner_size, rng) for i in range(size[1])] for j in range(size[0])]
    return LinearOperatorMatrix(operators)


def test_linearoperatormatrix_shape():
    """Test creation and shape of LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((5, 3), (3, 10), rng)
    assert matrix.shape == (5, 3)


def test_linearoperatormatrix_add_matrix():
    """Test addition of two LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    matrix2 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    vector = rng.complex64_tensor((3, 10))
    result = (matrix1 + matrix2)(*vector)
    expected = tuple(a + b for a, b in zip(matrix1(*vector), matrix2(*vector), strict=False))
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_sub_matrix():
    """Subtract matrices with matching shapes and reject mismatched shapes."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((2, 3), (3, 10), rng)
    matrix2 = random_linearoperatormatrix((2, 3), (3, 10), rng)
    vector = rng.complex64_tensor((3, 10))
    expected = tuple(a - b for a, b in zip(matrix1(*vector), matrix2(*vector), strict=True))
    torch.testing.assert_close((matrix1 - matrix2)(*vector), expected)
    with pytest.raises(ValueError, match='shapes do not match'):
        matrix1 - random_linearoperatormatrix((3, 2), (3, 10), rng)


def test_linearoperatormatrix_add_tensor_nonsquare():
    """Test failure of addition of tensor to non-square matrix."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    other = rng.complex64_tensor(3)
    with pytest.raises(ValueError, match='square'):
        (matrix1 + other)
    with pytest.raises(ValueError, match='square'):
        (other + matrix1)
    with pytest.raises(ValueError, match='square'):
        (matrix1 + 1.0)


def test_linearoperatormatrix_add_tensor_square():
    """Add tensor to square matrix."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((3, 3), (2, 2), rng)
    other = rng.complex64_tensor(2)
    vector = rng.complex64_tensor((3, 2))
    result = (matrix1 + other)(*vector)
    expected = tuple((mv + other * v for mv, v in zip(matrix1(*vector), vector, strict=True)))
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('scalar', [2, 2.0, 2j], ids=['int', 'float', 'complex'])
def test_linearoperatormatrix_add_scalar(scalar: complex):
    """Adding a scalar adds scalar*Identity to the diagonal blocks, from either side."""
    rng = RandomGenerator(0)
    matrix: LinearOperatorMatrix = random_linearoperatormatrix((3, 3), (2, 2), rng)
    vector = rng.complex64_tensor((3, 2))
    expected = tuple(mv + scalar * v for mv, v in zip(matrix(*vector), vector, strict=True))
    assert_type(matrix + scalar, LinearOperatorMatrix)
    assert_type(scalar + matrix, LinearOperatorMatrix)
    for result in (matrix + scalar, scalar + matrix):
        assert isinstance(result, LinearOperatorMatrix)
        torch.testing.assert_close(result(*vector), expected)


def test_linearoperatormatrix_tensor_left():
    """Tensor+LinearOperatorMatrix and Tensor-LinearOperatorMatrix with the tensor as left operand."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((2, 2), (2, 2), rng)
    other = rng.complex64_tensor(2)
    vector = rng.complex64_tensor((2, 2))
    result = other + matrix
    assert isinstance(result, LinearOperatorMatrix)
    torch.testing.assert_close(
        result(*vector), tuple(mv + other * v for mv, v in zip(matrix(*vector), vector, strict=True))
    )
    result = other - matrix
    assert isinstance(result, LinearOperatorMatrix)
    torch.testing.assert_close(
        result(*vector), tuple(other * v - mv for mv, v in zip(matrix(*vector), vector, strict=True))
    )


def test_linearoperatormatrix_sub_scalar():
    """Subtracting a scalar from either side."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((2, 2), (2, 2), rng)
    vector = rng.complex64_tensor((2, 2))
    torch.testing.assert_close(
        (matrix - 1.0)(*vector), tuple(mv - v for mv, v in zip(matrix(*vector), vector, strict=True))
    )
    torch.testing.assert_close(
        (1.0 - matrix)(*vector), tuple(v - mv for mv, v in zip(matrix(*vector), vector, strict=True))
    )


def test_linearoperatormatrix_add_linearoperator():
    """LinearOperator+LinearOperatorMatrix adds the operator to the diagonal blocks, from either side."""
    rng = RandomGenerator(0)
    matrix: LinearOperatorMatrix = random_linearoperatormatrix((2, 2), (3, 3), rng)
    op: LinearOperator = random_linearop((3, 3), rng)
    vector = rng.complex64_tensor((2, 3))
    expected = tuple(mv + op(v)[0] for mv, v in zip(matrix(*vector), vector, strict=True))
    assert_type(op + matrix, LinearOperatorMatrix)
    assert_type(matrix + op, LinearOperatorMatrix)
    for result in (matrix + op, op + matrix):
        assert isinstance(result, LinearOperatorMatrix)
        torch.testing.assert_close(result(*vector), expected)


def test_linearoperatormatrix_sub_linearoperator():
    """Subtraction of a LinearOperator and a LinearOperatorMatrix, from either side."""
    rng = RandomGenerator(0)
    matrix: LinearOperatorMatrix = random_linearoperatormatrix((2, 2), (3, 3), rng)
    op: LinearOperator = random_linearop((3, 3), rng)
    vector = rng.complex64_tensor((2, 3))
    expected = tuple(mv - op(v)[0] for mv, v in zip(matrix(*vector), vector, strict=True))
    result = matrix - op
    assert_type(result, LinearOperatorMatrix)
    assert isinstance(result, LinearOperatorMatrix)
    torch.testing.assert_close(result(*vector), expected)
    result = op - matrix
    assert_type(result, LinearOperatorMatrix)
    assert isinstance(result, LinearOperatorMatrix)
    torch.testing.assert_close(result(*vector), tuple(-e for e in expected))


def test_linearoperatormatrix_add_linearoperator_nonsquare():
    """Adding a LinearOperator to a non-square matrix is not defined."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((2, 3), (3, 3), rng)
    op = random_linearop((3, 3), rng)
    with pytest.raises(ValueError, match='square'):
        matrix + op
    with pytest.raises(ValueError, match='square'):
        op + matrix


def test_linearoperatormatrix_neg():
    """Negation of a LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix: LinearOperatorMatrix = random_linearoperatormatrix((3, 2), (3, 10), rng)
    vector = rng.complex64_tensor((2, 10))
    result = -matrix
    assert_type(result, LinearOperatorMatrix)
    assert isinstance(result, LinearOperatorMatrix)
    torch.testing.assert_close(result(*vector), tuple(-y for y in matrix(*vector)))


def test_linearoperatormatrix_neg_integer():
    """Matrix negation preserves integer dtype and values."""
    matrix = LinearOperatorMatrix([[IdentityOp()]])
    x = torch.tensor([16777217], dtype=torch.int64)
    torch.testing.assert_close((-matrix)(x), (-x,))


def test_linearoperatormatrix_rmul():
    """Test post multiplication with tensor."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    other = rng.complex64_tensor(3)
    vector = rng.complex64_tensor((3, 10))
    result = (other * matrix1)(*vector)
    expected = tuple(other * el for el in matrix1(*vector))
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_mul():
    """Test pre multiplication with tensor."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    other = rng.complex64_tensor(10)
    vector = rng.complex64_tensor((3, 10))
    result = (matrix1 * other)(*vector)
    expected = matrix1(*(other * el for el in vector))
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_composition():
    """Test composition of LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((1, 5), (2, 3), rng)
    matrix2 = random_linearoperatormatrix((5, 3), (3, 10), rng)
    vector = rng.complex64_tensor((3, 10))
    result = (matrix1 @ matrix2)(*vector)
    expected = matrix1(*(matrix2(*vector)))
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_rmatmul_linearoperator():
    """LinearOperator@LinearOperatorMatrix applies the operator to each output, i.e. composes it with each block."""
    rng = RandomGenerator(0)
    matrix: LinearOperatorMatrix = random_linearoperatormatrix((4, 3), (3, 10), rng)
    op: LinearOperator = random_linearop((2, 3), rng)
    vector = rng.complex64_tensor((3, 10))
    result = op @ matrix
    assert_type(result, LinearOperatorMatrix)
    assert isinstance(result, LinearOperatorMatrix)
    assert result.shape == (4, 3)
    expected = tuple(op(y)[0] for y in matrix(*vector))
    torch.testing.assert_close(result(*vector), expected)
    torch.testing.assert_close(result.H(*expected), matrix.H(*(op.H(y)[0] for y in expected)))


def test_linearoperatormatrix_rmatmul_linearoperator_single_row():
    """For a single-row matrix, LinearOperator@LinearOperatorMatrix is the plain composition."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((1, 3), (3, 10), rng)
    op = random_linearop((2, 3), rng)
    vector = rng.complex64_tensor((3, 10))
    result = op @ matrix
    assert isinstance(result, LinearOperatorMatrix)
    torch.testing.assert_close(result(*vector), op(*matrix(*vector)))


def test_linearoperatormatrix_rmatmul_special_operators():
    """IdentityOp and ZeroOp composed with a LinearOperatorMatrix keep the number of outputs."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((4, 3), (3, 10), rng)
    vector = rng.complex64_tensor((3, 10))
    torch.testing.assert_close((IdentityOp() @ matrix)(*vector), matrix(*vector))
    result = ZeroOp() @ matrix
    assert isinstance(result, LinearOperatorMatrix)
    assert result.shape == (4, 3)
    assert all((y == 0).all() for y in result(*vector))


def test_linearoperatormatrix_rmatmul_zero_keep_shape():
    """Shape-preserving zero composition retains each output's shape."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((2, 3), (4, 5), rng)
    vector = rng.complex64_tensor((3, 5))
    zero = ZeroOp(keep_shape=True)
    torch.testing.assert_close((zero @ matrix)(*vector), tuple(zero(y)[0] for y in matrix(*vector)))


def test_linearoperatormatrix_composition_mismatch():
    """Test composition with mismatching shapes."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((1, 5), (2, 3), rng)
    matrix2 = random_linearoperatormatrix((4, 3), (3, 10), rng)
    vector = rng.complex64_tensor((4, 10))
    with pytest.raises(ValueError, match='shapes do not match'):
        (matrix1 @ matrix2)(*vector)


def test_linearoperatormatrix_adjoint():
    """Test adjointness of Adjoint."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((5, 3), (3, 10), rng)

    class Wrapper(LinearOperator):
        """Stack the output of the matrix operator."""

        def forward(self, x):
            return (torch.stack(matrix(*x), 0),)

        def adjoint(self, x):
            return (torch.stack(matrix.adjoint(*x), 0),)

    dotproduct_adjointness_test(Wrapper(), rng.complex64_tensor((3, 10)), rng.complex64_tensor((5, 3)))


def test_linearoperatormatrix_repr():
    """Test repr of LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((5, 3), (3, 10), rng)
    assert 'LinearOperatorMatrix(shape=(5, 3)' in repr(matrix)


def test_linearoperatormatrix_getitem():
    """Test slicing of LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((12, 6), (3, 10), rng)

    def check(actual, expected):
        assert tuple(tuple(row) for row in actual) == tuple(tuple(row) for row in expected)

    sliced = matrix[1:3, 2]
    assert sliced.shape == (2, 1)
    check(sliced._operators, [row[2:3] for row in matrix._operators[1:3]])

    sliced = matrix[0]
    assert sliced.shape == (1, 6)
    check(sliced._operators, matrix._operators[:1])

    sliced = matrix[..., 0]
    assert sliced.shape == (12, 1)
    check(sliced._operators, [row[:1] for row in matrix._operators])

    sliced = matrix[1:6:2, (3, 4)]
    assert sliced.shape == (3, 2)
    check(sliced._operators, [[matrix._operators[i][j] for j in (3, 4)] for i in range(1, 6, 2)])

    sliced = matrix[-2:-4:-1, -1]
    assert sliced.shape == (2, 1)
    check(sliced._operators, [row[-1:] for row in matrix._operators[-2:-4:-1]])

    sliced = matrix[5, 5]
    assert isinstance(sliced, LinearOperator)
    assert sliced == matrix._operators[5][5]


def test_linearoperatormatrix_getitem_slice_clamp():
    """Out-of-range slice bounds are clamped, as for sequences, numpy arrays and torch tensors."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((4, 3), (3, 10), rng)
    sliced = matrix[:, :5]
    assert sliced.shape == (4, 3)
    sliced = matrix[-100:2, 1:100]
    assert sliced.shape == (2, 2)
    assert tuple(map(tuple, sliced)) == tuple(tuple(row[1:]) for row in matrix._operators[:2])


def test_linearoperatormatrix_getitem_one_tuple():
    """A 1-tuple index selects rows, i.e. ``matrix[(i,)] == matrix[i]``."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((4, 3), (3, 10), rng)
    assert tuple(map(tuple, matrix[(1,)])) == tuple(map(tuple, matrix[1]))
    assert matrix[(slice(1, 3),)].shape == (2, 3)


def test_linearoperatormatrix_getitem_error():
    """Test error when slicing with wrong indices."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((12, 6), (3, 10), rng)

    with pytest.raises(IndexError, match='Too many indices'):
        matrix[1, 1, 1]

    with pytest.raises(IndexError, match='out of range'):
        matrix[20]
    with pytest.raises(IndexError, match='out of range'):
        matrix[-20]
    with pytest.raises(IndexError, match='out of range'):
        matrix[(100, 1)]
    with pytest.raises(IndexError, match='out of range'):
        matrix[..., 20]
    with pytest.raises(IndexError, match='out of range'):
        matrix[..., -20]
    with pytest.raises(IndexError, match='out of range'):
        matrix[[0, 20]]
    with pytest.raises(IndexError, match='empty'):
        matrix[20:]
    with pytest.raises(IndexError, match='empty'):
        matrix[:, 6:]
    with pytest.raises(IndexError, match='empty'):
        matrix[[]]
    with pytest.raises(IndexError, match='index type'):
        matrix[..., 1.0]


def test_linearoperatormatrix_norm_rows():
    """Test norm of LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((3, 1), (3, 10), rng)
    vector = rng.complex64_tensor((1, 10))
    result = matrix.operator_norm(*vector)
    expected = sum(row[0].operator_norm(vector[0], dim=None) ** 2 for row in matrix._operators) ** 0.5
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_norm_cols():
    """Test norm of LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((1, 3), (3, 10), rng)
    vector = rng.complex64_tensor((3, 10))
    result = matrix.operator_norm(*vector)
    expected = (
        sum(op.operator_norm(v, dim=None) ** 2 for op, v in zip(matrix._operators[0], vector, strict=False)) ** 0.5
    )
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_norm_preserves_batch_shape():
    """Test norm of LinearOperatorMatrix preserves operator-norm batch dimensions."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((2, 2), (2, 3, 10), rng)
    vector = rng.complex64_tensor((2, 2, 10))
    result = matrix.operator_norm(*vector, dim=(-1,))
    expected = torch.linalg.matrix_norm(
        torch.stack(
            [
                torch.stack(
                    [op.operator_norm(v, dim=(-1,)) for op, v in zip(row, vector, strict=True)],
                )
                for row in matrix._operators
            ]
        ).movedim((0, 1), (-2, -1)),
        ord=2,
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_linearoperatormatrix_norm(seed):
    """Test norm of LinearOperatorMatrix."""
    rng = RandomGenerator(seed)
    matrix = random_linearoperatormatrix((4, 2), (3, 10), rng)
    vector = rng.complex64_tensor((2, 10))
    result = matrix.operator_norm(*vector)

    class Wrapper(LinearOperator):
        """Stack the output of the matrix operator."""

        def forward(self, x):
            return (torch.stack(matrix(*x), 0),)

        def adjoint(self, x):
            return (torch.stack(matrix.adjoint(*x), 0),)

    real = Wrapper().operator_norm(vector, dim=None)

    assert result >= real


def test_linearoperatormatrix_shorthand_vertical():
    """Test shorthand for vertical stacking."""
    rng = RandomGenerator(0)
    op1 = random_linearop((3, 10), rng)
    op2 = random_linearop((4, 10), rng)
    x1 = rng.complex64_tensor((10,))

    matrix1 = op1 % op2
    assert matrix1.shape == (2, 1)

    actual = matrix1(x1)
    expected = (*op1(x1), *op2(x1))
    torch.testing.assert_close(actual, expected)

    matrix2 = op2 % (matrix1 % op1)
    assert matrix2.shape == (4, 1)

    matrix3 = matrix2 % matrix2
    assert matrix3.shape == (8, 1)

    actual = matrix3(x1)
    expected = 2 * (*op2(x1), *matrix1(x1), *op1(x1))
    torch.testing.assert_close(actual, expected)


def test_linearoperatormatrix_shorthand_horizontal():
    """Test shorthand for horizontal stacking."""
    rng = RandomGenerator(0)
    op1 = random_linearop((3, 4), rng)
    op2 = random_linearop((3, 2), rng)
    x1 = rng.complex64_tensor((4,))
    x2 = rng.complex64_tensor((2,))
    x3 = rng.complex64_tensor((4,))
    x4 = rng.complex64_tensor((2,))

    matrix1 = op1 | op2
    assert matrix1.shape == (1, 2)

    actual1 = matrix1(x1, x2)
    expected1 = (op1(x1)[0] + op2(x2)[0],)
    torch.testing.assert_close(actual1, expected1)

    matrix2 = op2 | (matrix1 | op1)
    assert matrix2.shape == (1, 4)

    matrix3 = matrix2 | matrix2
    assert matrix3.shape == (1, 8)

    expected3 = (2 * (op2(x2)[0] + (matrix1(x3, x4)[0] + op1(x1)[0])),)
    actual3 = matrix3(x2, x3, x4, x1, x2, x3, x4, x1)
    torch.testing.assert_close(actual3, expected3)


def test_linearoperatormatrix_stacking_error():
    """Test error when stacking matrix operators with different shapes."""
    rng = RandomGenerator(0)
    matrix1 = random_linearoperatormatrix((3, 4), (3, 10), rng)
    matrix2 = random_linearoperatormatrix((3, 2), (3, 10), rng)
    matrix3 = random_linearoperatormatrix((2, 4), (3, 10), rng)
    op = random_linearop((3, 10), rng)
    with pytest.raises(ValueError, match='Shape mismatch'):
        matrix1 % matrix2  # vertical stacking needs same columns
    with pytest.raises(ValueError, match='Shape mismatch'):
        matrix1 | matrix3  # horizontal stacking needs same rows
    with pytest.raises(ValueError, match='Shape mismatch'):
        matrix1 | op  # horizontal stacking with operator needs 1-row matrix
    with pytest.raises(ValueError, match='Shape mismatch'):
        matrix1 % op  # vertical stacking with operator needs 1-column matrix


def test_linearoperatormatrix_error_nonlinearop():
    """Test error if trying to create a LinearOperatorMatrix with non linear operator."""
    op: Any = [[MagnitudeOp()]]  # Any is used to hide this error from mypy
    with pytest.raises(ValueError, match='LinearOperator'):
        LinearOperatorMatrix(op)


def test_linearoperatormatrix_error_inconsistent_shapes():
    """Test error if trying to create a LinearOperatorMatrix with inonsistent row lengths."""
    rng = RandomGenerator(0)
    op = random_linearop((3, 4), rng)
    with pytest.raises(ValueError, match='same length'):
        LinearOperatorMatrix([[op, op], [op]])


def test_linearoperatormatrix_from_diagonal():
    """Test creation of LinearOperatorMatrix from diagonal."""
    rng = RandomGenerator(0)
    ops = [random_linearop((2, 4), rng) for _ in range(3)]
    matrix = LinearOperatorMatrix.from_diagonal(*ops)
    xs = rng.complex64_tensor((3, 4))
    actual = matrix(*xs)
    expected = tuple(op(x)[0] for op, x in zip(ops, xs, strict=False))
    torch.testing.assert_close(actual, expected)


def test_linearoperatormatrix_gram():
    """Test gram of LinearOperatorMatrix."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((3, 2), (4, 10), rng)
    vector = tuple(rng.complex64_tensor((2, 10)))
    result = matrix.gram(*vector)
    expected = (matrix.H @ matrix)(*vector)
    torch.testing.assert_close(result, expected)


def test_linearoperatormatrix_operator_composition_single_row():
    """A non-linear operator composed with a single-row matrix is a plain composition."""
    rng = RandomGenerator(0)
    matrix = random_linearoperatormatrix((1, 3), (3, 10), rng)
    vector = rng.complex64_tensor((3, 10))
    torch.testing.assert_close((MagnitudeOp() @ matrix)(*vector), MagnitudeOp()(*matrix(*vector)))
