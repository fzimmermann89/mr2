"""Alternating Direction Method of Multipliers variants."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from mr2.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mr2.algorithms.optimizers.cg import cg, vdot
from mr2.operators import (
    IdentityOp,
    LinearOperator,
    LinearOperatorMatrix,
    ProximableFunctional,
    ProximableFunctionalSeparableSum,
)
from mr2.utils.to_tuple import to_tuple


def _norm_squared(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Squared L2 norm of a sequence of tensors."""
    return vdot(values, values).real

@dataclass
class ADMMLinearStatus(OptimizerStatus):
    """Status of linearized ADMM."""

    objective: Callable[[*tuple[torch.Tensor, ...]], torch.Tensor]
    tau: float | torch.Tensor
    mu: float | torch.Tensor
    z: tuple[torch.Tensor, ...]
    u: tuple[torch.Tensor, ...]


@dataclass
class ADMML2Status(OptimizerStatus):
    """Status of ADMM with L2 data term."""

    objective: Callable[[*tuple[torch.Tensor, ...]], torch.Tensor]
    tau: float | torch.Tensor
    z: tuple[torch.Tensor, ...]
    u: tuple[torch.Tensor, ...]


def admm_linear(
    f: ProximableFunctionalSeparableSum | ProximableFunctional,
    g: ProximableFunctionalSeparableSum | ProximableFunctional,
    operator: LinearOperator | LinearOperatorMatrix | None,
    initial_values: Sequence[torch.Tensor] | torch.Tensor,
    *,
    tau: float | torch.Tensor,
    mu: float | torch.Tensor,
    max_iterations: int = 128,
    tolerance: float = 0.0,
    initial_z: Sequence[torch.Tensor] | torch.Tensor | None = None,
    initial_u: Sequence[torch.Tensor] | torch.Tensor | None = None,
    callback: Callable[[ADMMLinearStatus], bool | None] | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""Linearized ADMM for :math:`\min_x f(x) + g(Ax)`."""
    tau_ = torch.as_tensor(tau)
    mu_ = torch.as_tensor(mu)
    if (tau_ <= 0).any() or tau_.dtype.is_complex:
        raise ValueError('tau must be real and positive')
    if (mu_ <= 0).any() or mu_.dtype.is_complex:
        raise ValueError('mu must be real and positive')

    if isinstance(f, ProximableFunctional):
        f_sum = ProximableFunctionalSeparableSum(f)
    else:
        f_sum = f

    if isinstance(g, ProximableFunctional):
        g_sum = ProximableFunctionalSeparableSum(g)
    else:
        g_sum = g

    if operator is None:
        if len(f_sum) != len(g_sum):
            raise ValueError('If operator is None, f and g must have the same number of components')
        operator_matrix = LinearOperatorMatrix.from_diagonal(*((IdentityOp(),) * len(f_sum)))
    elif isinstance(operator, LinearOperator):
        operator_matrix = LinearOperatorMatrix.from_diagonal(operator)
    else:
        operator_matrix = operator

    n_rows, n_columns = operator_matrix.shape
    if len(g_sum) != n_rows:
        raise ValueError('Number of rows in operator does not match number of functionals in g')
    if len(f_sum) != n_columns:
        raise ValueError('Number of columns in operator does not match number of functionals in f')

    x = to_tuple(n_columns, initial_values)

    ax = operator_matrix(*x)
    z = ax if initial_z is None else to_tuple(n_rows, initial_z)
    u = tuple(torch.zeros_like(zi) for zi in z) if initial_u is None else to_tuple(n_rows, initial_u)

    if len(z) != n_rows or len(u) != n_rows:
        raise ValueError('initial_z and initial_u must have same length as operator rows')

    for iteration in range(max_iterations):
        residual = tuple(ax_i - zi + ui for ax_i, zi, ui in zip(ax, z, u, strict=True))
        gradient_step = operator_matrix.H(*residual)
        x_gradient_step = tuple(xi - mu_ / tau_ * gi for xi, gi in zip(x, gradient_step, strict=True))
        x_new = f_sum.prox(*x_gradient_step, sigma=mu_)

        ax = operator_matrix(*x_new)
        z_new = g_sum.prox(*tuple(ax_i + ui for ax_i, ui in zip(ax, u, strict=True)), sigma=tau_)
        u = tuple(ui + ax_i - zi for ui, ax_i, zi in zip(u, ax, z_new, strict=True))

        if tolerance > 0:
            change_squared = _norm_squared(tuple(old - new for old, new in zip(x, x_new, strict=True)))
            x_new_squared = _norm_squared(x_new)
            if change_squared < tolerance**2 * x_new_squared:
                return x_new

        x = x_new
        z = z_new

        if callback is not None:
            continue_iterations = callback(
                ADMMLinearStatus(
                    solution=x,
                    iteration_number=iteration,
                    objective=lambda *x_: f_sum(*x_)[0] + g_sum(*operator_matrix(*x_))[0],
                    tau=tau_,
                    mu=mu_,
                    z=z,
                    u=u,
                )
            )
            if continue_iterations is False:
                break

    return x


def admm_l2(
    g: ProximableFunctionalSeparableSum | ProximableFunctional,
    op: LinearOperator | LinearOperatorMatrix,
    b: Sequence[torch.Tensor] | torch.Tensor,
    a: LinearOperator | LinearOperatorMatrix,
    initial_values: Sequence[torch.Tensor] | torch.Tensor,
    *,
    tau: float | torch.Tensor,
    max_iterations: int = 128,
    tolerance: float = 0.0,
    initial_z: Sequence[torch.Tensor] | torch.Tensor | None = None,
    initial_u: Sequence[torch.Tensor] | torch.Tensor | None = None,
    cg_max_iterations: int = 128,
    cg_tolerance: float = 1e-6,
    callback: Callable[[ADMML2Status], bool | None] | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""ADMM for :math:`\min_x \frac{1}{2}\|Op\,x-b\|_2^2 + g(Ax)`."""
    tau_ = torch.as_tensor(tau)
    if (tau_ <= 0).any() or tau_.dtype.is_complex:
        raise ValueError('tau must be real and positive')
    inv_tau = 1 / tau_

    if isinstance(g, ProximableFunctional):
        g_sum = ProximableFunctionalSeparableSum(g)
    else:
        g_sum = g

    op_matrix = LinearOperatorMatrix.from_diagonal(op) if isinstance(op, LinearOperator) else op
    a_matrix = LinearOperatorMatrix.from_diagonal(a) if isinstance(a, LinearOperator) else a
    n_data_rows, n_x = op_matrix.shape
    n_reg_rows, n_x_a = a_matrix.shape
    if n_x != n_x_a:
        raise ValueError('op and a must have matching number of columns')
    if len(g_sum) != n_reg_rows:
        raise ValueError('Number of functionals in g must match rows of a')

    x = to_tuple(n_x, initial_values)
    b_tuple = to_tuple(n_data_rows, b)
    ax = a_matrix(*x)
    z = ax if initial_z is None else to_tuple(n_reg_rows, initial_z)
    u = tuple(torch.zeros_like(zi) for zi in z) if initial_u is None else to_tuple(n_reg_rows, initial_u)

    h = op_matrix.gram + inv_tau * a_matrix.gram
    op_h_b = op_matrix.H(*b_tuple)

    for iteration in range(max_iterations):
        a_h_z_minus_u = a_matrix.H(*tuple(zi - ui for zi, ui in zip(z, u, strict=True)))
        rhs = tuple(op_rhs + inv_tau * a_rhs for op_rhs, a_rhs in zip(op_h_b, a_h_z_minus_u, strict=True))
        x_new = cg(
            h,
            rhs,
            initial_value=x,
            max_iterations=cg_max_iterations,
            tolerance=cg_tolerance,
        )
        ax = a_matrix(*x_new)
        z_new = g_sum.prox(*tuple(ax_i + ui for ax_i, ui in zip(ax, u, strict=True)), sigma=tau_)
        u = tuple(ui + ax_i - zi for ui, ax_i, zi in zip(u, ax, z_new, strict=True))

        if tolerance > 0:
            change_squared = _norm_squared(tuple(old - new for old, new in zip(x, x_new, strict=True)))
            x_new_squared = _norm_squared(x_new)
            if change_squared < tolerance**2 * x_new_squared:
                return x_new

        x = x_new
        z = z_new

        if callback is not None:
            continue_iterations = callback(
                ADMML2Status(
                    solution=x,
                    iteration_number=iteration,
                    objective=lambda *x_: 0.5
                    * _norm_squared(tuple(op_x_i - b_i for op_x_i, b_i in zip(op_matrix(*x_), b_tuple, strict=True)))
                    + g_sum(*a_matrix(*x_))[0],
                    tau=tau_,
                    z=z,
                    u=u,
                )
            )
            if continue_iterations is False:
                break

    return x
