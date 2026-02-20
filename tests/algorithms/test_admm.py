"""Tests for ADMM variants."""

import pytest
import torch
from mr2.algorithms.optimizers import admm_l2, admm_linear
from mr2.algorithms.optimizers.admm import ADMMLinearStatus
from mr2.operators import IdentityOp, LinearOperatorMatrix
from mr2.operators.functionals import L1Norm, L2NormSquared
from mr2.utils import RandomGenerator


def test_l2_l1_identification_admm_l2() -> None:
    """Set up min_x 1/2*||x - y||_2^2 + lambda * ||x||_1 and compare to soft-thresholding."""
    rng = RandomGenerator(seed=0)

    data = rng.float32_tensor(size=(32, 32))
    regularization_parameter = 0.2
    initial_values = (rng.float32_tensor(size=data.shape),)

    g = regularization_parameter * L1Norm(divide_by_n=False)
    expected = torch.nn.functional.softshrink(data, regularization_parameter)

    (solution,) = admm_l2(
        g=g,
        op=IdentityOp(),
        b=data,
        a=IdentityOp(),
        initial_values=initial_values,
        tau=1.0,
        max_iterations=128,
        cg_max_iterations=32,
    )
    torch.testing.assert_close(solution, expected, rtol=5e-4, atol=5e-4)


def test_l2_l1_identification_admm_linear() -> None:
    """Set up min_x 1/2*||x - y||_2^2 + lambda * ||x||_1 and compare to soft-thresholding."""
    rng = RandomGenerator(seed=0)

    data = rng.float32_tensor(size=(32, 32))
    regularization_parameter = 0.2
    initial_values = (rng.float32_tensor(size=data.shape),)

    f = 0.5 * L2NormSquared(target=data, divide_by_n=False)
    g = regularization_parameter * L1Norm(divide_by_n=False)
    expected = torch.nn.functional.softshrink(data, regularization_parameter)

    (solution,) = admm_linear(
        f=f,
        g=g,
        operator=IdentityOp(),
        initial_values=initial_values,
        tau=1.0,
        mu=0.95,
        max_iterations=192,
    )
    torch.testing.assert_close(solution, expected, rtol=5e-4, atol=5e-4)


def test_admm_linear_tuple_inputs() -> None:
    """Check tuple support using 2x2 diagonal operator matrix."""
    rng = RandomGenerator(seed=0)
    data1 = rng.float32_tensor(size=(16, 16))
    data2 = rng.float32_tensor(size=(16, 16))

    f = (0.5 * L2NormSquared(target=data1, divide_by_n=False)) | (0.5 * L2NormSquared(target=data2, divide_by_n=False))
    g = (0.1 * L1Norm(divide_by_n=False)) | (0.2 * L1Norm(divide_by_n=False))
    operator = LinearOperatorMatrix.from_diagonal(IdentityOp(), IdentityOp())

    initial_values = (rng.float32_tensor(size=data1.shape), rng.float32_tensor(size=data2.shape))
    solution = admm_linear(
        f=f,
        g=g,
        operator=operator,
        initial_values=initial_values,
        tau=1.0,
        mu=0.95,
        max_iterations=192,
    )

    expected1 = torch.nn.functional.softshrink(data1, 0.1)
    expected2 = torch.nn.functional.softshrink(data2, 0.2)
    torch.testing.assert_close(solution[0], expected1, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(solution[1], expected2, rtol=5e-4, atol=5e-4)


def test_admm_linear_callback_early_stop() -> None:
    """Check early stop via callback."""
    rng = RandomGenerator(seed=0)

    data = rng.float32_tensor(size=(16, 16))
    f = 0.5 * L2NormSquared(target=data, divide_by_n=False)
    g = 0.1 * L1Norm(divide_by_n=False)
    initial_values = (rng.float32_tensor(size=data.shape),)

    callback_count = 0

    def callback(admm_status: ADMMLinearStatus):
        nonlocal callback_count
        _, _, _, _, _ = (
            admm_status['iteration_number'],
            admm_status['solution'][0],
            admm_status['z'],
            admm_status['u'],
            admm_status['objective'](*admm_status['solution']),
        )
        callback_count += 1
        return False

    admm_linear(
        f=f,
        g=g,
        operator=IdentityOp(),
        initial_values=initial_values,
        tau=1.0,
        mu=0.95,
        max_iterations=64,
        callback=callback,
    )
    assert callback_count == 1


def test_admm_linear_value_errors() -> None:
    """Check that value-errors are caught."""
    rng = RandomGenerator(seed=0)
    initial_values = (rng.float32_tensor(size=(8, 8)),)

    with pytest.raises(ValueError, match='same number of components'):
        admm_linear(
            f=(L2NormSquared() | L2NormSquared()),
            g=L1Norm(),
            operator=None,
            initial_values=initial_values,
            tau=1.0,
            mu=0.95,
            max_iterations=1,
        )

    with pytest.raises(ValueError, match='rows'):
        admm_linear(
            f=L2NormSquared(),
            g=L1Norm(),
            operator=LinearOperatorMatrix(((IdentityOp(),), (IdentityOp(),))),
            initial_values=initial_values,
            tau=1.0,
            mu=0.95,
            max_iterations=1,
        )

    with pytest.raises(ValueError, match='columns'):
        admm_linear(
            f=(L2NormSquared() | L2NormSquared()),
            g=(L1Norm() | L1Norm()),
            operator=LinearOperatorMatrix(((IdentityOp(),), (IdentityOp(),))),
            initial_values=initial_values,
            tau=1.0,
            mu=0.95,
            max_iterations=1,
        )
