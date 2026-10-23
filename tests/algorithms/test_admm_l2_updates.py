"""Callable split updates and residual stopping for quadratic ADMM."""

from collections.abc import Callable

import pytest
import torch
from mr2.algorithms.optimizers import admm_l2
from mr2.algorithms.optimizers.admm import ADMML2Status
from mr2.operators import IdentityOp, LinearOperatorMatrix
from mr2.operators.functionals import L1Norm


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
def test_zero_data(tolerance: float, expected_iterations: int) -> None:
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
        callback=statuses.append,
    )
    assert len(statuses) == expected_iterations
    assert [s['iteration_number'] for s in statuses] == list(range(expected_iterations))
    torch.testing.assert_close(result, data)
    assert statuses[-1]['data_fidelity'] == 0
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
    torch.testing.assert_close(statuses[0]['data_fidelity'], 0.5 * (expected_x - data).square().sum())


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
        torch.testing.assert_close(status['data_fidelity'], 0.5 * (x - data).abs().square().sum())
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
