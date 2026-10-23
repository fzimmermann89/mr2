"""Plug-and-play ADMM with a tensor denoiser."""

from collections.abc import Callable

import torch

from mr2.algorithms.optimizers.admm import _admm_l2_x_update
from mr2.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mr2.operators import IdentityOp, LinearOperator, LinearOperatorMatrix


class PNPADMMStatus(OptimizerStatus):
    """Status of plug-and-play ADMM, without an assumed global objective."""

    tau: torch.Tensor
    """Inverse ADMM penalty."""

    z: torch.Tensor
    """Denoised split variable."""

    u: torch.Tensor
    """Scaled dual variable."""

    primal_residual: torch.Tensor
    """Primal residual x - z."""

    dual_residual: torch.Tensor
    """Dual residual (z - z_previous) / tau."""

    data_fidelity: torch.Tensor
    """Current value of 0.5 * ||op(x) - b|| squared."""


def pnp_admm(
    denoiser: Callable[[torch.Tensor], torch.Tensor],
    op: LinearOperator,
    b: torch.Tensor,
    initial_values: torch.Tensor,
    *,
    tau: float | torch.Tensor,
    max_iterations: int = 128,
    tolerance: float = 0.0,
    initial_z: torch.Tensor | None = None,
    initial_u: torch.Tensor | None = None,
    cg_max_iterations: int = 128,
    cg_tolerance: float = 1e-6,
    callback: Callable[[PNPADMMStatus], bool | None] | None = None,
) -> tuple[torch.Tensor]:
    r"""Apply plug-and-play ADMM to quadratic data fidelity and a denoiser.

    Each iteration solves the data-consistency subproblem with CG, denoises,
    and updates the scaled dual variable:

    .. math::

        (Op^H Op + \tau^{-1} I)x_{k+1} &= Op^H b + \tau^{-1}(z_k-u_k)\\
        z_{k+1} &= D(x_{k+1}+u_k)\\
        u_{k+1} &= u_k + x_{k+1}-z_{k+1}.

    As in `~mr2.algorithms.optimizers.admm.admm_l2`, ``tau`` is the inverse
    penalty: an ADMM penalty of rho corresponds to ``tau=1/rho``. The denoiser
    receives only a tensor. Bind its strength explicitly, e.g. use
    ``lambda x: g.prox(x, sigma=tau)[0]`` for a proximable functional ``g``.
    Arbitrary denoisers need not define an objective or converge.

    Parameters
    ----------
    denoiser
        Tensor-to-tensor callable preserving shape, dtype and device. It must
        not modify its input in place. Differentiation requires a differentiable callable.
    op
        Linear acquisition operator.
    b
        Measured data.
    initial_values
        Initial image tensor x. The returned tuple contains the data-consistent
        iterate x, while the callback also exposes the denoised iterate z.
    tau
        Positive finite real scalar inverse penalty, moved to the image device.
    max_iterations
        Maximum number of ADMM iterations.
    tolerance
        Nonnegative finite relative residual tolerance. Stop when both
        ``||x-z|| <= tolerance * max(||x||, ||z||)`` and
        ``||(z-z_previous)/tau|| <= tolerance * ||u/tau||``.
        A zero scale requires an exactly zero residual. Zero disables stopping.
    initial_z
        Initial denoised split variable. Defaults to x.
    initial_u
        Initial scaled dual variable. Defaults to zeros.
    cg_max_iterations
        Maximum CG iterations per data-consistency update.
    cg_tolerance
        Positive finite absolute CG residual tolerance.
    callback
        Called after every completed iteration, including the last one, with
        `PNPADMMStatus`. Returning `False` stops iteration. Iterations start at zero.

    Returns
    -------
        One-element tuple containing the final data-consistent image tensor.

    Raises
    ------
    ValueError
        If parameters or the denoiser output violate the tensor contract.

    Notes
    -----
    Based on the plug-and-play reconstruction proposed in PTB-MR/mrpro PR #1013,
    using mr2's quadratic ADMM/CG mechanics and inverse penalty convention.
    """
    tau_ = torch.as_tensor(tau, device=initial_values.device)
    if tau_.is_complex() or tau_.ndim != 0 or not torch.isfinite(tau_).all() or not (tau_ > 0).all():
        raise ValueError('tau must be a positive finite real scalar')
    if max_iterations < 0 or cg_max_iterations <= 0:
        raise ValueError('max_iterations must be nonnegative and cg_max_iterations positive')
    if not 0 <= tolerance < float('inf') or not 0 < cg_tolerance < float('inf'):
        raise ValueError('tolerance must be nonnegative finite and cg_tolerance positive finite')
    x = initial_values
    z = x if initial_z is None else initial_z
    u = torch.zeros_like(x) if initial_u is None else initial_u

    def check_tensor(value: torch.Tensor, name: str) -> None:
        if not isinstance(value, torch.Tensor) or (
            value.shape != x.shape or value.dtype != x.dtype or value.device != x.device
        ):
            raise ValueError(f'{name} must preserve image shape, dtype and device')

    check_tensor(z, 'initial_z')
    check_tensor(u, 'initial_u')
    inv_tau = tau_.reciprocal()
    identity = LinearOperatorMatrix.from_diagonal(IdentityOp())
    h = LinearOperatorMatrix.from_diagonal(op.gram) + inv_tau * identity
    op_h_b = op.H(b)
    for iteration in range(max_iterations):
        (x,) = _admm_l2_x_update(h, op_h_b, identity, (x,), (z,), (u,), inv_tau, cg_max_iterations, cg_tolerance)
        z_new = denoiser(x + u)
        check_tensor(z_new, 'denoiser output')
        primal_residual = x - z_new
        dual_residual = inv_tau * (z_new - z)
        u = u + primal_residual
        z = z_new
        if callback is not None and (
            callback(
                PNPADMMStatus(
                    solution=(x,),
                    iteration_number=iteration,
                    tau=tau_,
                    z=z,
                    u=u,
                    primal_residual=primal_residual,
                    dual_residual=dual_residual,
                    data_fidelity=0.5 * torch.linalg.vector_norm(op(x)[0] - b).square(),
                )
            )
            is False
        ):
            break
        if tolerance > 0:
            primal_scale = torch.maximum(torch.linalg.vector_norm(x), torch.linalg.vector_norm(z))
            dual_scale = torch.linalg.vector_norm(inv_tau * u)
            if (
                torch.linalg.vector_norm(primal_residual) <= tolerance * primal_scale
                and torch.linalg.vector_norm(dual_residual) <= tolerance * dual_scale
            ):
                break
    return (x,)
