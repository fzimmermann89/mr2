"""Alternating Direction Method of Multipliers variants.

This module provides ADMM-based optimizers for composite problems using
`~mr2.operators.ProximableFunctional` objects and `~mr2.operators.LinearOperator`
or `~mr2.operators.LinearOperatorMatrix` objects. Quadratic ADMM also accepts
callable split updates, including denoisers without a known objective.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
from typing_extensions import NotRequired

from mr2.algorithms.optimizers.cg import cg, vdot
from mr2.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mr2.operators import (
    IdentityOp,
    LinearOperator,
    LinearOperatorMatrix,
    ProximableFunctional,
    ProximableFunctionalSeparableSum,
)
from mr2.utils.to_tuple import to_tuple

_ADMMSplitUpdate = Callable[..., tuple[torch.Tensor, ...]]
"""Callable ``update(*values, sigma=tau)`` preserving the split tensor contract."""


def _norm_squared(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Squared L2 norm of a sequence of tensors."""
    return vdot(values, values).real


@dataclass
class ADMMLinearStatus(OptimizerStatus):
    """Status of linearized ADMM.

    Attributes
    ----------
    solution
        Current estimate of the primal variable(s).
    iteration_number
        Current iteration count.
    objective
        Callable returning the objective value :math:`f(x) + g(Ax)` for given
        primal variables.
    tau
        ADMM penalty parameter used for the z-update.
    mu
        Proximal/linearization parameter used for the x-update.
    z
        Current auxiliary split variable(s).
    u
        Current scaled dual variable(s).
    """

    objective: Callable[[*tuple[torch.Tensor, ...]], torch.Tensor]
    tau: float | torch.Tensor
    mu: float | torch.Tensor
    z: tuple[torch.Tensor, ...]
    u: tuple[torch.Tensor, ...]


class ADMML2Status(OptimizerStatus):
    r"""Status of ADMM with L2 data term.

    Attributes
    ----------
    solution
        Current estimate of the primal variable(s).
    iteration_number
        Current iteration count.
    objective
        Present only for a functional regularizer. Callable returning the objective value
        :math:`\frac{1}{2}\|Op\,x-b\|_2^2 + g(Ax)`.
    tau
        Inverse ADMM penalty, tau = 1/rho.
    z
        Current auxiliary split variable(s).
    u
        Current scaled dual variable(s).
    primal_residual
        Split residual Ax - z, in the split space.
    dual_residual
        Dual residual A.H(z - z_previous) / tau, in the primal space.
    data_fidelity
        Zero-argument callable returning 0.5 * ||op(x) - b|| squared for this
        iteration. The forward operator is evaluated only when invoked.
    """

    objective: NotRequired[Callable[[*tuple[torch.Tensor, ...]], torch.Tensor]]
    tau: float | torch.Tensor
    z: tuple[torch.Tensor, ...]
    u: tuple[torch.Tensor, ...]

    primal_residual: tuple[torch.Tensor, ...]
    dual_residual: tuple[torch.Tensor, ...]
    data_fidelity: Callable[[], torch.Tensor]


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
    r"""Linearized ADMM for :math:`\min_x f(x) + g(Ax)`.

    This routine solves convex composite problems of the form

        :math:`\min_x f(x) + g(Ax)`,

    where :math:`f` and :math:`g` are proximable and :math:`A` is linear.
    It supports single-variable as well as block-variable formulations by
    operating on tuples of tensors and `LinearOperatorMatrix`.

    The algorithm applies the following updates:

    .. math::

        x_{k+1} = \mathrm{prox}_{\mu f}\left(x_k - \frac{\mu}{\tau} A^H(Ax_k-z_k+u_k)\right)\\
        z_{k+1} = \mathrm{prox}_{\tau g}(Ax_{k+1} + u_k)\\
        u_{k+1} = u_k + Ax_{k+1} - z_{k+1}

    with scaled dual variable :math:`u`.

    For convergence of the linearized method, ``mu / tau`` must be chosen
    sufficiently small relative to the squared operator norm of :math:`A`.
    This is not checked automatically.

    Parameters
    ----------
    f
        Proximable functional :math:`f`. Can be a single functional or a
        `~mr2.operators.ProximableFunctionalSeparableSum`.
    g
        Proximable functional :math:`g`. Can be a single functional or a
        `~mr2.operators.ProximableFunctionalSeparableSum`.
    operator
        Linear operator :math:`A`. If `None`, an identity operator matrix is
        used (requiring matching number of components in `f` and `g`).
    initial_values
        Initial primal variable(s). Single tensor or tuple of tensors.
    tau
        Positive ADMM penalty parameter for the z-update.
    mu
        Positive linearization/proximal parameter for the x-update.
        Together with ``tau``, it must satisfy the convergence condition of
        linearized ADMM.
    max_iterations
        Maximum number of iterations.
    tolerance
        Relative stopping tolerance on the primal update
        :math:`\|x_{k+1}-x_k\|_2 / \|x_{k+1}\|_2`.
        If zero, no tolerance-based early stopping is applied.
    initial_z
        Optional initial split variable(s). If `None`, initialized as
        :math:`z_0 = A x_0`.
    initial_u
        Optional initial scaled dual variable(s). If `None`, initialized to
        zero with shape matching `z`.
    callback
        Optional callback called after each iteration with `ADMMLinearStatus`.
        If it returns `False`, iterations stop early.

    Returns
    -------
        Tuple of tensors representing the final primal variable(s).

    Raises
    ------
    ValueError
        If parameters are inconsistent (non-positive `tau`/`mu`, mismatched
        operator/functional dimensions, or incompatible initial variables).
    """
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
    g: ProximableFunctionalSeparableSum | ProximableFunctional | _ADMMSplitUpdate,
    op: LinearOperator | LinearOperatorMatrix,
    b: Sequence[torch.Tensor] | torch.Tensor,
    a: LinearOperator | LinearOperatorMatrix,
    initial_values: Sequence[torch.Tensor] | torch.Tensor,
    *,
    tau: float | torch.Tensor,
    max_iterations: int = 128,
    tolerance: float = 0.0,
    stopping_criterion: Literal['solution', 'residual'] = 'solution',
    initial_z: Sequence[torch.Tensor] | torch.Tensor | None = None,
    initial_u: Sequence[torch.Tensor] | torch.Tensor | None = None,
    cg_max_iterations: int = 128,
    cg_tolerance: float = 1e-6,
    callback: Callable[[ADMML2Status], bool | None] | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""ADMM for :math:`\min_x \frac{1}{2}\|Op\,x-b\|_2^2 + g(Ax)`.

    This routine targets inverse problems with quadratic data fidelity and
    proximable regularization in transformed coordinates:

        :math:`\min_x \frac{1}{2}\|Op\,x-b\|_2^2 + g(Ax)`.

    Introducing :math:`z = Ax`, ADMM uses:

    .. math::

        x_{k+1} = \arg\min_x \frac{1}{2}\|Op\,x-b\|_2^2 +
        \frac{1}{2\tau}\|Ax-z_k+u_k\|_2^2\\
        z_{k+1} = \mathrm{prox}_{\tau g}(Ax_{k+1}+u_k)\\
        u_{k+1} = u_k + Ax_{k+1} - z_{k+1}

    The x-update is solved with
    `~mr2.algorithms.optimizers.cg.cg` on the normal equations:

    .. math::

        (Op^H Op + \tau^{-1} A^H A)x =
        Op^H b + \tau^{-1}A^H(z-u).

    Parameters
    ----------
    g
        Proximable regularizer. Can be a single functional or a
        `~mr2.operators.ProximableFunctionalSeparableSum`, or a split-update
        callable with signature ``g(*values, sigma=tau) -> tuple[Tensor, ...]``.
        The callable replaces the proximal step and must preserve tensor count,
        shapes, dtypes and devices without modifying inputs in place. It need not
        define an objective or converge. For a tensor denoiser, use
        ``lambda x, **_: (denoiser(x),)`` and bind its strength explicitly.
    op
        Forward operator :math:`Op` in the data term.
    b
        Data tensor(s). Single tensor or tuple of tensors, matching rows of
        `op`.
    a
        Linear operator :math:`A` used for splitting in :math:`g(Ax)`.
    initial_values
        Initial primal variable(s). Single tensor or tuple of tensors.
    tau
        Positive finite real inverse penalty: tau = 1/rho. Complex tensors with
        zero imaginary part are accepted. Callable updates
        require a scalar tensor. Functional updates with solution stopping also
        accept broadcastable tensors.
        Residual stopping requires a single value.
    max_iterations
        Maximum number of ADMM iterations.
    tolerance
        Nonnegative finite relative stopping tolerance. Zero disables stopping.
    stopping_criterion
        ``'solution'`` checks relative x-change:
        :math:`\|x_{k+1}-x_k\|_2 < tolerance \|x_{k+1}\|_2`.
        ``'residual'`` requires both ``||Ax-z|| <= tolerance * max(||Ax||, ||z||)``
        and ``||A.H(z-z_previous)/tau|| <= tolerance * ||A.H(u)/tau||``.
        Norms combine all tuple components; zero scales require zero residuals.
        Residual stopping is recommended for arbitrary split updates, where
        unchanged x alone need not imply convergence.
    initial_z
        Optional initial split variable(s). If `None`, initialized as
        :math:`z_0 = A x_0`.
    initial_u
        Optional initial scaled dual variable(s). If `None`, initialized to
        zero with shape matching `z`.
    cg_max_iterations
        Maximum CG iterations used inside each ADMM x-update.
    cg_tolerance
        Nonnegative finite absolute residual tolerance used by CG for each x-update.
    callback
        Optional callback called after each iteration with `ADMML2Status`.
        If it returns `False`, iterations stop early. With residual stopping,
        callbacks include the converged iteration. With solution stopping, an
        early return precedes the callback.
        Status includes an objective only when ``g`` is a functional.

    Returns
    -------
        Tuple of tensors representing the final primal variable(s).

    Raises
    ------
    ValueError
        If the penalty is invalid, operator or functional dimensions are incompatible,
        or a callable update or its initial split state violates the tensor contract.
    """
    g_sum = ProximableFunctionalSeparableSum(g) if isinstance(g, ProximableFunctional) else g
    if isinstance(g_sum, ProximableFunctionalSeparableSum):
        functional: ProximableFunctionalSeparableSum | None = g_sum
        split_update = g_sum.prox
    else:
        functional = None
        split_update = g_sum

    x_device = initial_values.device if isinstance(initial_values, torch.Tensor) else initial_values[0].device
    tau_ = torch.as_tensor(tau, device=x_device)
    if tau_.is_complex():
        if (tau_.imag != 0).any():
            raise ValueError('tau must be positive finite and real')
        tau_ = tau_.real
    if not torch.isfinite(tau_).all() or not (tau_ > 0).all():
        raise ValueError('tau must be positive finite and real')
    if functional is None and tau_.ndim != 0:
        raise ValueError('tau must be a scalar for callable split updates')
    if stopping_criterion == 'residual':
        if tau_.numel() != 1:
            raise ValueError('tau must contain a single value for residual stopping')
        tau_ = tau_.reshape(())
    inv_tau = tau_.reciprocal()

    op_matrix = LinearOperatorMatrix.from_diagonal(op) if isinstance(op, LinearOperator) else op
    a_matrix = LinearOperatorMatrix.from_diagonal(a) if isinstance(a, LinearOperator) else a
    n_data_rows, n_x = op_matrix.shape
    n_reg_rows, n_x_a = a_matrix.shape
    if n_x != n_x_a:
        raise ValueError('op and a must have matching number of columns')
    if functional is not None and len(functional) != n_reg_rows:
        raise ValueError('Number of functionals in g must match rows of a')
    if (1 if isinstance(b, torch.Tensor) else len(b)) != n_data_rows:
        raise ValueError('b must have same length as rows of op')
    if initial_z is not None and (1 if isinstance(initial_z, torch.Tensor) else len(initial_z)) != n_reg_rows:
        raise ValueError('initial_z must have same length as rows of a')
    if initial_u is not None and (1 if isinstance(initial_u, torch.Tensor) else len(initial_u)) != n_reg_rows:
        raise ValueError('initial_u must have same length as rows of a')

    x = to_tuple(n_x, initial_values)
    b_tuple = to_tuple(n_data_rows, b)
    ax = a_matrix(*x)
    z = ax if initial_z is None else to_tuple(n_reg_rows, initial_z)
    u = tuple(torch.zeros_like(zi) for zi in z) if initial_u is None else to_tuple(n_reg_rows, initial_u)

    if functional is None:
        if any(
            (zi.shape, zi.dtype, zi.device) != (ai.shape, ai.dtype, ai.device) for zi, ai in zip(z, ax, strict=True)
        ):
            raise ValueError('initial_z must match split shape, dtype and device')
        if any(
            (ui.shape, ui.dtype, ui.device) != (ai.shape, ai.dtype, ai.device) for ui, ai in zip(u, ax, strict=True)
        ):
            raise ValueError('initial_u must match split shape, dtype and device')

    def data_fidelity(*values: torch.Tensor) -> torch.Tensor:
        return 0.5 * _norm_squared(
            tuple(predicted - measured for predicted, measured in zip(op_matrix(*values), b_tuple, strict=True))
        )

    def norm(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(value) for value in values]))

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
        z_new = split_update(*tuple(ax_i + ui for ax_i, ui in zip(ax, u, strict=True)), sigma=tau_)
        if functional is None and (
            len(z_new) != len(ax)
            or any(
                (zi.shape, zi.dtype, zi.device) != (ai.shape, ai.dtype, ai.device)
                for zi, ai in zip(z_new, ax, strict=True)
            )
        ):
            raise ValueError('split update output must preserve tensor count, shape, dtype and device')
        primal_residual = tuple(ax_i - zi for ax_i, zi in zip(ax, z_new, strict=True))
        u = tuple(ui + ri for ui, ri in zip(u, primal_residual, strict=True))
        if callback is not None or (stopping_criterion == 'residual' and tolerance > 0):
            dual_residual = tuple(
                inv_tau * value for value in a_matrix.H(*tuple(new - old for new, old in zip(z_new, z, strict=True)))
            )

        if tolerance > 0 and stopping_criterion == 'solution':
            change_squared = _norm_squared(tuple(old - new for old, new in zip(x, x_new, strict=True)))
            x_new_squared = _norm_squared(x_new)
            if change_squared < tolerance**2 * x_new_squared:
                return x_new

        x = x_new
        z = z_new

        if callback is not None:
            status = ADMML2Status(
                solution=x,
                iteration_number=iteration,
                tau=tau_,
                z=z,
                u=u,
                primal_residual=primal_residual,
                dual_residual=dual_residual,
                data_fidelity=partial(data_fidelity, *x),
            )
            if functional is not None:
                status['objective'] = lambda *x_: data_fidelity(*x_) + functional(*a_matrix(*x_))[0]
            if callback(status) is False:
                break

        if stopping_criterion == 'residual' and tolerance > 0:
            primal_scale = torch.maximum(norm(ax), norm(z))
            dual_scale = norm(tuple(inv_tau * value for value in a_matrix.H(*u)))
            if norm(primal_residual) <= tolerance * primal_scale and norm(dual_residual) <= tolerance * dual_scale:
                break

    return x
