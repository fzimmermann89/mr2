"""Tests for ADMM variants."""

from collections.abc import Callable

import pytest
import torch
from mr2.algorithms.optimizers import admm_l2, admm_linear
from mr2.algorithms.optimizers.admm import ADMML2Status, ADMMLinearStatus
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


def test_admm_l2_callback_early_stop() -> None:
    """Check early stop via callback."""
    rng = RandomGenerator(seed=0)

    data = rng.float32_tensor(size=(16, 16))
    initial_values = (rng.float32_tensor(size=data.shape),)
    callback_count = 0

    def callback(admm_status: ADMML2Status):
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

    admm_l2(
        g=0.1 * L1Norm(divide_by_n=False),
        op=IdentityOp(),
        b=data,
        a=IdentityOp(),
        initial_values=initial_values,
        tau=1.0,
        max_iterations=64,
        callback=callback,
    )
    assert callback_count == 1


@pytest.mark.cuda
@pytest.mark.parametrize('algorithm', [admm_linear, admm_l2], ids=['linear', 'l2'])
def test_cuda_admm_solution(algorithm) -> None:
    """ADMM variants recover a small soft-thresholded CUDA solution."""
    data = torch.tensor([-0.8, -0.1, 0.3, 1.2], device='cuda')
    regularization_parameter = 0.2
    expected = torch.nn.functional.softshrink(data, regularization_parameter)
    initial_values = (torch.zeros_like(data),)
    g = regularization_parameter * L1Norm(divide_by_n=False)

    if algorithm is admm_linear:
        solution = algorithm(
            f=0.5 * L2NormSquared(target=data, divide_by_n=False),
            g=g,
            operator=IdentityOp(),
            initial_values=initial_values,
            tau=1.0,
            mu=0.95,
            max_iterations=128,
        )
    else:
        solution = algorithm(
            g=g,
            op=IdentityOp(),
            b=data,
            a=IdentityOp(),
            initial_values=initial_values,
            tau=1.0,
            max_iterations=128,
            cg_max_iterations=8,
        )

    assert solution[0].is_cuda
    torch.testing.assert_close(solution[0], expected, atol=1e-3, rtol=1e-3)


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


def test_admm_l2_value_errors() -> None:
    """Check that value-errors are caught."""
    rng = RandomGenerator(seed=0)
    initial_values = (rng.float32_tensor(size=(8, 8)),)
    data = rng.float32_tensor(size=(8, 8))

    with pytest.raises(ValueError, match='rows of a'):
        admm_l2(
            g=L1Norm() | L1Norm(),
            op=IdentityOp(),
            b=data,
            a=IdentityOp(),
            initial_values=initial_values,
            tau=1.0,
            max_iterations=1,
        )

    with pytest.raises(ValueError, match='rows of op'):
        admm_l2(
            g=L1Norm(),
            op=LinearOperatorMatrix(((IdentityOp(),), (IdentityOp(),))),
            b=data,
            a=IdentityOp(),
            initial_values=initial_values,
            tau=1.0,
            max_iterations=1,
        )

    with pytest.raises(ValueError, match='initial_z'):
        admm_l2(
            g=L1Norm(),
            op=IdentityOp(),
            b=data,
            a=IdentityOp(),
            initial_values=initial_values,
            tau=1.0,
            initial_z=(initial_values[0], initial_values[0]),
            max_iterations=1,
        )

    with pytest.raises(ValueError, match='initial_u'):
        admm_l2(
            g=L1Norm(),
            op=IdentityOp(),
            b=data,
            a=IdentityOp(),
            initial_values=initial_values,
            tau=1.0,
            initial_u=(initial_values[0], initial_values[0]),
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


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_identity_denoiser(dtype: torch.dtype) -> None:
    """An identity denoiser recovers least squares, preserving the tensor contract."""
    data = torch.tensor([[1, -2], [3, 0]], dtype=dtype)
    (result,) = admm_l2(
        lambda x, **_: (x,),
        2 * IdentityOp(),
        data,
        IdentityOp(),
        torch.zeros_like(data),
        stopping_criterion='residual',
        tau=0.7,
        max_iterations=40,
    )
    torch.testing.assert_close(result, data / 2)
    assert result.shape == data.shape
    assert result.dtype == dtype
    assert result.device == data.device


@pytest.mark.parametrize('tau', [0.2, 1.0, 3.0])
def test_l1_prox(tau: float) -> None:
    """A proximal denoiser matches analytic shrinkage and the convex ADMM API."""
    data = torch.tensor([-3.0, -0.2, 0.0, 0.2, 3.0], dtype=torch.float64)
    g = 0.4 * L1Norm(divide_by_n=False)
    (result,) = admm_l2(
        g.prox,
        IdentityOp(),
        data,
        IdentityOp(),
        data,
        stopping_criterion='residual',
        tau=tau,
        max_iterations=160,
        cg_tolerance=1e-12,
    )
    (convex_result,) = admm_l2(
        g,
        IdentityOp(),
        data,
        IdentityOp(),
        data,
        tau=tau,
        max_iterations=160,
        cg_tolerance=1e-12,
    )
    torch.testing.assert_close(result, torch.nn.functional.softshrink(data, 0.4))
    torch.testing.assert_close(result, convex_result)


def test_primal_unchanged_is_not_convergence() -> None:
    """x=3 stays fixed initially, but z=2 and u=1 require another iteration."""
    data = torch.tensor([3.0], dtype=torch.float64)
    statuses: list[ADMML2Status] = []
    (result,) = admm_l2(
        lambda x, **_: (torch.nn.functional.softshrink(x, 1),),
        IdentityOp(),
        data,
        IdentityOp(),
        data,
        stopping_criterion='residual',
        tau=1,
        max_iterations=80,
        tolerance=1e-7,
        cg_tolerance=1e-12,
        callback=statuses.append,
    )
    assert len(statuses) > 1
    torch.testing.assert_close(statuses[0]['solution'][0], data)
    torch.testing.assert_close(statuses[0]['z'][0], torch.tensor([2.0], dtype=data.dtype))
    torch.testing.assert_close(statuses[0]['primal_residual'][0], torch.ones_like(data))
    torch.testing.assert_close(statuses[0]['dual_residual'][0], -torch.ones_like(data))
    torch.testing.assert_close(result, torch.tensor([2.0], dtype=data.dtype))


@pytest.mark.parametrize(('tolerance', 'expected_iterations'), [(0.0, 3), (1e-5, 1)])
@pytest.mark.parametrize('cg_tolerance', [0.0, 1e-6])
def test_zero_data(tolerance: float, expected_iterations: int, cg_tolerance: float) -> None:
    """Zero residuals terminate safely with zero norm scales; callbacks still run."""
    data = torch.zeros(2, 3, dtype=torch.complex128)
    statuses: list[ADMML2Status] = []
    (result,) = admm_l2(
        lambda x, **_: (x,),
        IdentityOp(),
        data,
        IdentityOp(),
        data,
        stopping_criterion='residual',
        tau=1,
        max_iterations=3,
        tolerance=tolerance,
        cg_tolerance=cg_tolerance,
        callback=statuses.append,
    )
    assert len(statuses) == expected_iterations
    assert [s['iteration_number'] for s in statuses] == list(range(expected_iterations))
    torch.testing.assert_close(result, data)
    assert statuses[-1]['data_fidelity']() == 0
    assert torch.isfinite(result).all()


def test_callback_stop_and_initial_state() -> None:
    """Callbacks see the complete update and may stop; return x rather than z."""
    data = torch.tensor([3.0])
    statuses: list[ADMML2Status] = []

    def callback(status: ADMML2Status) -> bool:
        statuses.append(status)
        return False

    (result,) = admm_l2(
        lambda x, **_: (x * 0.5,),
        IdentityOp(),
        data,
        IdentityOp(),
        data,
        stopping_criterion='residual',
        tau=2,
        initial_z=torch.tensor([1.0]),
        initial_u=torch.tensor([0.5]),
        callback=callback,
    )
    expected_x = (data + 0.5 * (1 - 0.5)) / 1.5
    assert len(statuses) == 1
    assert 'objective' not in statuses[0]
    torch.testing.assert_close(result, expected_x)
    torch.testing.assert_close(statuses[0]['z'][0], (expected_x + 0.5) * 0.5)
    torch.testing.assert_close(statuses[0]['data_fidelity'](), 0.5 * (expected_x - data).square().sum())


def test_autograd() -> None:
    """Unrolled CG and denoiser retain gradients of data, tau and denoiser parameters."""
    data = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    tau = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    strength = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)

    def solve(data_: torch.Tensor, tau_: torch.Tensor, strength_: torch.Tensor) -> torch.Tensor:
        return admm_l2(
            lambda x, **_: (strength_ * x,),
            2 * IdentityOp(),
            data_,
            IdentityOp(),
            torch.zeros_like(data_),
            stopping_criterion='residual',
            tau=tau_,
            max_iterations=3,
            cg_tolerance=1e-12,
        )[0]

    assert torch.autograd.gradcheck(solve, (data, tau, strength))


@pytest.mark.parametrize('tau', [0, -1, float('nan'), float('inf'), 1j, torch.ones(2), torch.ones(1)])
def test_invalid_tau(tau: float | complex | torch.Tensor) -> None:
    """Reject nonpositive, nonfinite, complex and nonscalar inverse penalties."""
    data = torch.ones(2)
    with pytest.raises(ValueError, match='tau'):
        admm_l2(
            lambda x, **_: (x,),
            IdentityOp(),
            data,
            IdentityOp(),
            data,
            stopping_criterion='residual',
            tau=torch.as_tensor(tau),
        )


@pytest.mark.parametrize('denoiser', [lambda x: x[:1], lambda x: x.double(), lambda x: x.to('meta')])
def test_denoiser_contract(denoiser: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """Reject shape, dtype or device changes at the callable boundary."""
    data = torch.ones(2)
    with pytest.raises(ValueError, match='split update output'):
        admm_l2(
            lambda x, **_: (denoiser(x),),
            IdentityOp(),
            data,
            IdentityOp(),
            data,
            stopping_criterion='residual',
            tau=1,
        )


def test_initial_state_contract() -> None:
    """Initial split and dual states must match the image tensor."""
    data = torch.ones(2)
    with pytest.raises(ValueError, match='initial_z'):
        admm_l2(
            lambda x, **_: (x,),
            IdentityOp(),
            data,
            IdentityOp(),
            data,
            stopping_criterion='residual',
            initial_z=data.double(),
            tau=1,
        )
    with pytest.raises(ValueError, match='initial_u'):
        admm_l2(
            lambda x, **_: (x,),
            IdentityOp(),
            data,
            IdentityOp(),
            data,
            stopping_criterion='residual',
            initial_u=data[:1],
            tau=1,
        )
    (result,) = admm_l2(
        lambda x, **_: (x,),
        IdentityOp(),
        data,
        IdentityOp(),
        data,
        stopping_criterion='residual',
        tau=1,
        max_iterations=0,
    )
    torch.testing.assert_close(result, data)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda() -> None:
    """CPU tau moves to the image device without changing complex precision."""
    data = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128, device='cuda')
    (result,) = admm_l2(
        lambda x, **_: (x,),
        IdentityOp(),
        data,
        IdentityOp(),
        torch.zeros_like(data),
        stopping_criterion='residual',
        tau=torch.tensor(0.5),
        max_iterations=64,
        cg_tolerance=1e-12,
    )
    torch.testing.assert_close(result, data)
    assert result.is_cuda
    assert result.dtype == data.dtype


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_block_updates_and_residuals(dtype: torch.dtype) -> None:
    """A rectangular split uses A.H for dual residuals and accepts coupled updates."""
    identity = IdentityOp()
    acquisition = LinearOperatorMatrix.from_diagonal(identity, identity)
    split = LinearOperatorMatrix(((2 * identity, identity), (identity, -1 * identity), (0 * identity, 3 * identity)))
    matrix = torch.tensor([[2, 1], [1, -1], [0, 3]], dtype=dtype)
    data = torch.tensor([[1, -2], [3, 4]], dtype=dtype)
    tau = torch.tensor(0.7, dtype=torch.float64)
    statuses: list[ADMML2Status] = []

    def update(*values: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, ...]:
        torch.testing.assert_close(sigma, tau)
        return values[0] / 2, values[1] / 3 + values[0] / 4, values[2] * 0.8

    result = admm_l2(
        update,
        acquisition,
        tuple(data.unbind()),
        split,
        tuple(torch.zeros_like(data).unbind()),
        tau=tau,
        max_iterations=2,
        cg_tolerance=1e-12,
        stopping_criterion='residual',
        callback=statuses.append,
    )
    x = torch.zeros_like(data)
    z = matrix @ x
    u = torch.zeros_like(z)
    normal = torch.eye(2, dtype=dtype) + matrix.mH @ matrix / tau
    for status in statuses:
        x = torch.linalg.solve(normal, data + matrix.mH @ (z - u) / tau)
        z_new = torch.stack(update(*(matrix @ x + u).unbind(), sigma=tau))
        primal = matrix @ x - z_new
        dual = matrix.mH @ (z_new - z) / tau
        u = u + primal
        torch.testing.assert_close(torch.stack(status['solution']), x)
        torch.testing.assert_close(torch.stack(status['primal_residual']), primal)
        torch.testing.assert_close(torch.stack(status['dual_residual']), dual)
        torch.testing.assert_close(torch.stack(status['u']), u)
        torch.testing.assert_close(status['data_fidelity'](), 0.5 * (x - data).abs().square().sum())
        assert 'objective' not in status
        z = z_new
    assert len(statuses) == 2
    torch.testing.assert_close(torch.stack(result), x)


@pytest.mark.parametrize('as_callable', [False, True])
def test_transformed_l1_residual_stopping(as_callable: bool) -> None:
    """Functional and callable priors converge to the same transformed-L1 solution."""
    data = torch.tensor([-3.0, -0.2, 0.0, 0.2, 3.0], dtype=torch.float64)
    g = 0.4 * L1Norm(divide_by_n=False)
    statuses: list[ADMML2Status] = []
    (result,) = admm_l2(
        g.prox if as_callable else g,
        IdentityOp(),
        data,
        2 * IdentityOp(),
        data,
        tau=1,
        max_iterations=256,
        tolerance=1e-7,
        stopping_criterion='residual',
        cg_tolerance=1e-12,
        callback=statuses.append,
    )
    torch.testing.assert_close(result, torch.nn.functional.softshrink(data, 0.8))
    assert 1 < len(statuses) < 256
    status = statuses[-1]
    assert ('objective' in status) is not as_callable
    if not as_callable:
        torch.testing.assert_close(status['objective'](result), 0.5 * (result - data).square().sum() + g(2 * result)[0])


def test_solution_stopping_compatibility() -> None:
    """The default keeps x-change stopping and its original callback timing."""
    data = torch.tensor([3.0], dtype=torch.float64)
    statuses: list[ADMML2Status] = []
    (result,) = admm_l2(
        L1Norm(divide_by_n=False),
        IdentityOp(),
        data,
        IdentityOp(),
        data,
        tau=1,
        tolerance=1e-7,
        callback=statuses.append,
    )
    torch.testing.assert_close(result, data)
    assert not statuses


@pytest.mark.parametrize('count', [0, 2])
def test_update_component_count(count: int) -> None:
    """Updates cannot remove or add split components."""
    data = torch.ones(2)
    with pytest.raises(ValueError, match='split update output'):
        admm_l2(lambda x, **_: (x,) * count, IdentityOp(), data, IdentityOp(), data, tau=1)


@pytest.mark.parametrize(
    'case',
    [
        'target_dtype',
        'initial_z_shape',
        'initial_z_dtype',
        'initial_u_shape',
        'initial_u_dtype',
        'tau_singleton',
        'tau_broadcast',
    ],
)
def test_functional_broadcast_and_promotion(case: str) -> None:
    """Functional updates retain their established broadcasting and dtype promotion."""
    data = torch.arange(1, 11, dtype=torch.float32).reshape(2, 5)
    target = torch.zeros_like(data, dtype=torch.float64 if case == 'target_dtype' else torch.float32)
    z = torch.zeros((1, 5)) if case == 'initial_z_shape' else torch.zeros_like(data)
    u = torch.zeros((1, 5)) if case == 'initial_u_shape' else torch.zeros_like(data)
    if case == 'initial_z_dtype':
        z = z.double()
    if case == 'initial_u_dtype':
        u = u.double()
    tau = torch.ones(5) if case == 'tau_broadcast' else torch.tensor([1.0] if case == 'tau_singleton' else 1.0)
    statuses: list[ADMML2Status] = []
    (result,) = admm_l2(
        L2NormSquared(target=target, divide_by_n=False),
        IdentityOp(),
        data,
        IdentityOp(),
        torch.zeros_like(data),
        tau=tau,
        initial_z=z,
        initial_u=u,
        max_iterations=2,
        callback=statuses.append,
    )
    first_x = (data + (z - u) / tau) / (1 + 1 / tau)
    first_z = (first_x + u + 2 * tau * target) / (1 + 2 * tau)
    first_u = u + first_x - first_z
    expected = (data + (first_z - first_u) / tau) / (1 + 1 / tau)
    torch.testing.assert_close(result, expected)
    assert len(statuses) == 2
    assert 'objective' in statuses[-1]


@pytest.mark.parametrize('as_callable', [False, True])
def test_callback_data_fidelity_is_lazy(as_callable: bool) -> None:
    """Logging callbacks avoid extra forward calls; saved diagnostics bind their iterate."""
    data = torch.tensor([2.0, 3.0], dtype=torch.float64)
    operator = 2 * IdentityOp()
    regularizer = L1Norm(divide_by_n=False)
    forward_calls: list[None] = []
    hook = operator.register_forward_hook(lambda *_: forward_calls.append(None))
    g = regularizer.prox if as_callable else regularizer
    admm_l2(g, operator, data, IdentityOp(), torch.zeros_like(data), tau=0.7, max_iterations=2, cg_tolerance=1e-12)
    baseline_calls = len(forward_calls)
    forward_calls.clear()
    statuses: list[ADMML2Status] = []
    admm_l2(
        g,
        operator,
        data,
        IdentityOp(),
        torch.zeros_like(data),
        callback=statuses.append,
        tau=0.7,
        max_iterations=2,
        cg_tolerance=1e-12,
    )
    assert len(forward_calls) == baseline_calls
    assert len(statuses) == 2
    for index, status in enumerate(statuses, start=1):
        expected = 0.5 * (2 * status['solution'][0] - data).square().sum()
        torch.testing.assert_close(status['data_fidelity'](), expected)
        assert len(forward_calls) == baseline_calls + index
    hook.remove()
