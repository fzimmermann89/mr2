"""Mathematical and interface tests for plug-and-play ADMM."""

from collections.abc import Callable

import pytest
import torch
from mr2.algorithms.optimizers import admm_l2, pnp_admm
from mr2.algorithms.optimizers.pnp_admm import PNPADMMStatus
from mr2.operators import IdentityOp
from mr2.operators.functionals import L1Norm


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_identity_denoiser(dtype: torch.dtype) -> None:
    """An identity denoiser recovers least squares, preserving the tensor contract."""
    data = torch.tensor([[1, -2], [3, 0]], dtype=dtype)
    (result,) = pnp_admm(lambda x: x, 2 * IdentityOp(), data, torch.zeros_like(data), tau=0.7, max_iterations=40)
    torch.testing.assert_close(result, data / 2)
    assert result.shape == data.shape
    assert result.dtype == dtype
    assert result.device == data.device


@pytest.mark.parametrize('tau', [0.2, 1.0, 3.0])
def test_l1_prox(tau: float) -> None:
    """A proximal denoiser matches analytic shrinkage and the convex ADMM API."""
    data = torch.tensor([-3.0, -0.2, 0.0, 0.2, 3.0], dtype=torch.float64)
    g = 0.4 * L1Norm(divide_by_n=False)
    (result,) = pnp_admm(
        lambda x: g.prox(x, sigma=tau)[0],
        IdentityOp(),
        data,
        data,
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
    statuses: list[PNPADMMStatus] = []
    (result,) = pnp_admm(
        lambda x: torch.nn.functional.softshrink(x, 1),
        IdentityOp(),
        data,
        data,
        tau=1,
        max_iterations=80,
        tolerance=1e-7,
        cg_tolerance=1e-12,
        callback=statuses.append,
    )
    assert len(statuses) > 1
    torch.testing.assert_close(statuses[0]['solution'][0], data)
    torch.testing.assert_close(statuses[0]['z'], torch.tensor([2.0], dtype=data.dtype))
    torch.testing.assert_close(statuses[0]['primal_residual'], torch.ones_like(data))
    torch.testing.assert_close(statuses[0]['dual_residual'], -torch.ones_like(data))
    torch.testing.assert_close(result, torch.tensor([2.0], dtype=data.dtype))


@pytest.mark.parametrize(('tolerance', 'expected_iterations'), [(0.0, 3), (1e-5, 1)])
def test_zero_data(tolerance: float, expected_iterations: int) -> None:
    """Zero residuals terminate safely with zero norm scales; callbacks still run."""
    data = torch.zeros(2, 3, dtype=torch.complex128)
    statuses: list[PNPADMMStatus] = []
    (result,) = pnp_admm(
        lambda x: x,
        IdentityOp(),
        data,
        data,
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
    statuses: list[PNPADMMStatus] = []

    def callback(status: PNPADMMStatus) -> bool:
        statuses.append(status)
        return False

    (result,) = pnp_admm(
        lambda x: x * 0.5,
        IdentityOp(),
        data,
        data,
        tau=2,
        initial_z=torch.tensor([1.0]),
        initial_u=torch.tensor([0.5]),
        callback=callback,
    )
    expected_x = (data + 0.5 * (1 - 0.5)) / 1.5
    assert len(statuses) == 1
    torch.testing.assert_close(result, expected_x)
    torch.testing.assert_close(statuses[0]['z'], (expected_x + 0.5) * 0.5)
    torch.testing.assert_close(statuses[0]['data_fidelity'], 0.5 * (expected_x - data).square().sum())


def test_autograd() -> None:
    """Unrolled CG and denoiser retain gradients of data, tau and denoiser parameters."""
    data = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    tau = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    strength = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)

    def solve(data_: torch.Tensor, tau_: torch.Tensor, strength_: torch.Tensor) -> torch.Tensor:
        return pnp_admm(
            lambda x: strength_ * x,
            2 * IdentityOp(),
            data_,
            torch.zeros_like(data_),
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
        pnp_admm(lambda x: x, IdentityOp(), data, data, tau=torch.as_tensor(tau))


@pytest.mark.parametrize('denoiser', [lambda x: x[:1], lambda x: x.double(), lambda x: x.to('meta')])
def test_denoiser_contract(denoiser: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """Reject shape, dtype or device changes at the callable boundary."""
    data = torch.ones(2)
    with pytest.raises(ValueError, match='denoiser output'):
        pnp_admm(denoiser, IdentityOp(), data, data, tau=1)


def test_initial_state_contract() -> None:
    """Initial split and dual states must match the image tensor."""
    data = torch.ones(2)
    with pytest.raises(ValueError, match='initial_z'):
        pnp_admm(lambda x: x, IdentityOp(), data, data, initial_z=data.double(), tau=1)
    with pytest.raises(ValueError, match='initial_u'):
        pnp_admm(lambda x: x, IdentityOp(), data, data, initial_u=data[:1], tau=1)
    (result,) = pnp_admm(lambda x: x, IdentityOp(), data, data, tau=1, max_iterations=0)
    torch.testing.assert_close(result, data)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda() -> None:
    """CPU tau moves to the image device without changing complex precision."""
    data = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128, device='cuda')
    (result,) = pnp_admm(
        lambda x: x, IdentityOp(), data, torch.zeros_like(data), tau=torch.tensor(0.5), max_iterations=64
    )
    torch.testing.assert_close(result, data)
    assert result.is_cuda
    assert result.dtype == data.dtype
