"""Reusable Moore--Penrose pseudoinverse actions and implicit differentiation."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.autograd.function import once_differentiable
from typing_extensions import Unpack

from mr2.operators.ConjugateGradientOp import ConjugateGradientOp
from mr2.operators.LinearOperator import (
    LinearOperator,
    LinearOperatorElementwiseProductLeft,
    LinearOperatorElementwiseProductRight,
)
from mr2.operators.LinearOperatorMatrix import LinearOperatorMatrix
from mr2.operators.Operator import Operator

if TYPE_CHECKING:
    from mr2.operators.SensitivityOp import SensitivityOp

Source = LinearOperator | LinearOperatorMatrix
Scaled = LinearOperatorElementwiseProductLeft | LinearOperatorElementwiseProductRight
Vectors = tuple[torch.Tensor, ...]


def _validate_options(tolerance: float, max_iterations: int) -> None:
    if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError('tolerance must be finite and in [0, 1).')
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer.')


def _snapshot(source: Source) -> tuple[Source, Vectors, Vectors]:
    """Copy module containers, replacing tensor state by independent autograd leaves.

    Tensor storage is shared, with the originals saved by autograd for version
    checking. Copying the containers isolates attribute replacement. Detaching
    each coefficient avoids counting its upstream graph twice when both a tensor
    and a derived tensor occur in the source. No source attributes are modified.
    """
    originals: list[torch.Tensor] = []
    leaves: list[torch.Tensor] = []
    memo: dict[int, Any] = {}

    def visit(value: Any) -> Any:  # noqa: ANN401
        result: Any
        if id(value) in memo:
            return memo[id(value)]
        if isinstance(value, torch.Tensor):
            result = value.detach().requires_grad_(value.requires_grad)
            originals.append(value)
            leaves.append(result)
        elif isinstance(value, torch.nn.Module):
            result = copy.copy(value)
            memo[id(value)] = result
            result.__dict__ = {key: visit(item) for key, item in vars(value).items()}
        elif isinstance(value, dict):
            result = value.copy()
            memo[id(value)] = result
            for key, item in value.items():
                result[key] = visit(item)
        elif isinstance(value, list):
            result = []
            memo[id(value)] = result
            result.extend(visit(item) for item in value)
        elif isinstance(value, tuple):
            # Preserve torch.Size and other immutable tuple subclasses.
            result = tuple(visit(item) for item in value) if type(value) is tuple else value
        else:
            return value
        memo[id(value)] = result
        return result

    return cast(Source, visit(source)), tuple(originals), tuple(leaves)


def _check_vectors(values: Vectors, reference: Vectors | None = None) -> None:
    if not values or any(not (v.is_floating_point() or v.is_complex()) for v in values):
        raise ValueError('Pseudoinverse actions require floating-point or complex tensors.')
    if any(not torch.isfinite(v).all() for v in values):
        raise RuntimeError('Nonfinite value in pseudoinverse action.')
    if reference is not None and (
        len(values) != len(reference) or any(a.shape != b.shape for a, b in zip(values, reference, strict=True))
    ):
        raise ValueError('Pseudoinverse requires a fixed domain and a shape-compatible true adjoint.')


def _norm(values: Vectors) -> torch.Tensor:
    # Scale before summation to avoid squaring large or tiny entries.
    scale = torch.stack([v.abs().amax() if v.numel() else v.real.new_zeros(()) for v in values]).amax()
    if scale == 0:
        return scale
    return scale * sum(((v / scale).abs().square().sum() for v in values), start=scale.new_zeros(())).sqrt()


def _solve(source: Source, rhs: Vectors, tolerance: float, max_iterations: int) -> Vectors:
    """Apply a pseudoinverse from zero, accepting only a fresh normal residual."""
    _check_vectors(rhs)
    normal_rhs = source.adjoint(*rhs)
    _check_vectors(normal_rhs)
    if all(not torch.count_nonzero(v) for v in normal_rhs):
        zeros = tuple(torch.zeros_like(v) for v in normal_rhs)
        _check_vectors(source(*zeros), rhs)
        return zeros
    norm = _norm(normal_rhs)
    # CGOp currently computes its own norm by squaring; reject scales it cannot
    # represent, rather than silently treating a nonzero problem as zero.
    squared = sum((v.abs().square().sum() for v in normal_rhs), start=norm.new_zeros(()))
    if not torch.isfinite(squared) or squared == 0:
        raise RuntimeError('Pseudoinverse right-hand side exceeds the numerical range of CG.')
    normal = source.gram
    solver = ConjugateGradientOp(
        lambda *_: normal,
        lambda *_: normal_rhs,
        implicit_backward=False,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    solution = solver()
    _check_vectors(solution, normal_rhs)
    prediction = source(*solution)
    _check_vectors(prediction, rhs)
    residual = source.adjoint(*(b - ax for b, ax in zip(rhs, prediction, strict=True)))
    _check_vectors(residual, normal_rhs)
    relative_residual = (_norm(residual) / norm).item()
    if relative_residual > tolerance:
        raise RuntimeError(
            f'Pseudoinverse did not meet tolerance {tolerance:g}: '
            f'normal residual {relative_residual:g}, iteration budget {max_iterations}.'
        )
    return solution


class _PseudoInverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401
        source: Source,
        leaves: Vectors,
        tolerance: float,
        max_iterations: int,
        n_rhs: int,
        *tensors: torch.Tensor,
    ) -> Vectors:
        rhs = tensors[:n_rhs]
        solution = _solve(source, rhs, tolerance, max_iterations)
        ctx.source, ctx.leaves = source, leaves
        ctx.tolerance, ctx.max_iterations, ctx.n_rhs = tolerance, max_iterations, n_rhs
        ctx.n_solution = len(solution)
        ctx.save_for_backward(*solution, *tensors)
        ctx.set_materialize_grads(True)
        return solution

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *grad_output: torch.Tensor) -> tuple:  # noqa: ANN401
        saved = ctx.saved_tensors  # also checks versions of fixed source buffers
        solution = saved[: ctx.n_solution]
        rhs = saved[ctx.n_solution : ctx.n_solution + ctx.n_rhs]
        source = ctx.source
        with torch.no_grad():
            z = _solve(source.H, grad_output, ctx.tolerance, ctx.max_iterations)
            active = [(i, t) for i, t in enumerate(ctx.leaves) if t.requires_grad]
            if active:
                # For P=A^+, the constant-rank VJP is
                # dL/dA = -z x.H + residual q.H + t h.H.
                # Evaluate these outer products through source VJPs below.
                residual = tuple(b - ax for b, ax in zip(rhs, source(*solution), strict=True))
                q = _solve(source, z, ctx.tolerance, ctx.max_iterations)
                t = _solve(source.H, solution, ctx.tolerance, ctx.max_iterations)
                h = tuple(g - ahz for g, ahz in zip(grad_output, source.adjoint(*z), strict=True))
        coefficient_grads: list[torch.Tensor | None] = [None] * len(ctx.leaves)
        if active:
            with torch.enable_grad():
                outputs = (*source(*solution), *source(*q), *source(*h))
                cotangents = (*(-v for v in z), *residual, *t)
                # A source may contain unused tensors or an input-only custom
                # autograd implementation. Preserve its existing gradient rules.
                pairs = [(out, cot) for out, cot in zip(outputs, cotangents, strict=True) if out.requires_grad]
                if pairs:
                    grads = torch.autograd.grad(
                        tuple(out for out, _ in pairs),
                        tuple(v for _, v in active),
                        tuple(cot for _, cot in pairs),
                        allow_unused=True,
                    )
                    for (index, _), grad in zip(active, grads, strict=True):
                        coefficient_grads[index] = grad
        return (None, None, None, None, None, *z, *coefficient_grads)


def _apply(source: Source, rhs: Vectors, tolerance: float, max_iterations: int) -> Vectors:
    snapshot, originals, leaves = _snapshot(source)
    return _PseudoInverseFunction.apply(snapshot, leaves, tolerance, max_iterations, len(rhs), *rhs, *originals)


class PseudoInverseOp(LinearOperator):
    r"""Numerical Moore--Penrose pseudoinverse of a fixed-space linear operator.

    Zero-start CG targets the minimum-norm least-squares solution. Finite
    iterations approximate linearity and the adjoint; convergence of the normal
    residual is not a solution-error or numerical-rank guarantee. Normal equations
    square the positive-spectrum condition number. No damping or rank cutoff is used.

    Backward uses the constant-rank pseudoinverse derivative, including range and
    nullspace terms for coefficient gradients. Source tensors must be stored in
    module attributes or tensor containers, not closures or opaque objects.
    The source must provide a genuine adjoint and differentiable coefficient
    actions. Existing input-only custom backward rules cannot supply coefficient
    gradients. Geometry must remain fixed through backward. Higher derivatives,
    JVP and vmap are unsupported for the numerical fallback.
    """

    def __init__(self, source: LinearOperator, tolerance: float = 1e-6, max_iterations: int = 100) -> None:
        """Initialize a pseudoinverse.

        Parameters
        ----------
        source
            Linear operator with a true adjoint on fixed input and output spaces.
        tolerance
            Relative normal-equation residual tolerance, not a rank cutoff.
        max_iterations
            Iteration budget for each forward or backward solve.
        """
        super().__init__()
        _validate_options(tolerance, max_iterations)
        self.source = source
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the pseudoinverse with implicit first-order differentiation."""
        return cast(tuple[torch.Tensor], _apply(self.source, (x,), self.tolerance, self.max_iterations))

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Approximate the adjoint of the target pseudoinverse."""
        return cast(tuple[torch.Tensor], _apply(self.source.H, (x,), self.tolerance, self.max_iterations))

    def _pinv(self, *, tolerance: float, max_iterations: int) -> LinearOperator:  # noqa: ARG002
        """Use the Moore--Penrose identity (A^+)^+ = A."""
        return self.source


class PseudoInverseMatrixOp(Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]):
    """Coupled pseudoinverse action on a tuple of tensor spaces.

    This is the multiple-input counterpart of `PseudoInverseOp`, with the same
    numerical and differentiation contract. Its blocks cannot generally be
    evaluated independently without knowing every component's shape.
    """

    def __init__(self, source: LinearOperatorMatrix, tolerance: float, max_iterations: int) -> None:
        """Initialize the tuple action from a source and per-solve CG settings."""
        super().__init__()
        _validate_options(tolerance, max_iterations)
        self.source = source
        self.tolerance, self.max_iterations = tolerance, max_iterations

    def forward(self, *x: torch.Tensor) -> Vectors:
        """Apply the coupled pseudoinverse."""
        return _apply(self.source, x, self.tolerance, self.max_iterations)

    @property
    def H(self) -> PseudoInverseMatrixOp:  # noqa: N802
        """Return the paired target-adjoint action."""
        return PseudoInverseMatrixOp(self.source.H, self.tolerance, self.max_iterations)


def _safe_reciprocal(x: torch.Tensor) -> torch.Tensor:
    """Reciprocate nonzero entries without evaluating division by zero."""
    nonzero = x != 0
    denominator = torch.where(nonzero, x, torch.ones_like(x))
    return torch.where(nonzero, denominator.reciprocal(), torch.zeros_like(x))


class _GatherPseudoInverse(LinearOperator):
    """Normalize a coordinate extraction adjoint by its actual multiplicities."""

    def __init__(self, source: LinearOperator) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        assembled = self.source.adjoint(x)[0]
        counts = self.source.adjoint(torch.ones_like(x))[0]
        return (assembled * _safe_reciprocal(counts),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        template = self.source(x)[0]
        counts = self.source.adjoint(torch.ones_like(template))[0]
        return self.source(x * _safe_reciprocal(counts))


class _SensitivityPseudoInverse(LinearOperator):
    """Combine coil images using live sensitivity power, including zero power."""

    def __init__(self, source: SensitivityOp) -> None:
        super().__init__()
        self.source = source

    def _weights(self, x: torch.Tensor) -> torch.Tensor:
        csm = self.source.csm_tensor
        if (
            csm.ndim < 4
            or x.ndim < 4
            or x.shape[-4] != 1
            or torch.broadcast_shapes(x.shape, csm.shape)[-4:] != (csm.shape[-4], *x.shape[-3:])
            or torch.broadcast_shapes(x.shape, csm.shape)[:-4] != x.shape[:-4]
        ):
            raise ValueError('Sensitivity pseudoinverse requires a singleton coil and fully expanded noncoil domain.')
        return _safe_reciprocal(csm.abs().square().sum(-4, keepdim=True))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        csm = self.source.csm_tensor
        if (
            x.ndim < 4
            or csm.ndim < 4
            or x.shape[-4] != csm.shape[-4]
            or torch.broadcast_shapes(x.shape, csm.shape) != x.shape
        ):
            raise ValueError('Sensitivity pseudoinverse input must have the fully expanded coil-image shape.')
        combined = self.source.adjoint(x)[0]
        return (combined * self._weights(combined),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return self.source(x * self._weights(x))


class _DiagonalPseudoInverse(LinearOperator):
    """Reciprocate a live coefficient on a fixed, unexpanded tensor space."""

    def __init__(self, source: LinearOperator, attribute: str) -> None:
        super().__init__()
        self.source, self.attribute = source, attribute

    def _weights(self, x: torch.Tensor) -> torch.Tensor:
        weights = getattr(self.source, self.attribute)
        if weights is None:
            return x.new_ones(())
        if not isinstance(weights, torch.Tensor):
            weights = x.new_tensor(weights)
        if torch.broadcast_shapes(x.shape, weights.shape) != x.shape:
            raise ValueError('Diagonal pseudoinverse coefficients must not expand the input shape.')
        return _safe_reciprocal(weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x * self._weights(x),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x * self._weights(x).conj(),)


class _ScalarPseudoInverse(LinearOperator):
    """Apply a source pseudoinverse with live scalar normalization."""

    def __init__(self, source: Scaled, tolerance: float, max_iterations: int) -> None:
        super().__init__()
        self.source = source
        self.tolerance, self.max_iterations = tolerance, max_iterations
        self.inverse = source._operator.pinv(tolerance=tolerance, max_iterations=max_iterations)

    def _factor(self) -> torch.Tensor | complex:
        scalar = self.source._scalar
        if isinstance(scalar, torch.Tensor):
            return _safe_reciprocal(scalar)
        return 1 / scalar if scalar != 0 else 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        factor = self._factor()
        if factor == 0:
            return cast(tuple[torch.Tensor], _apply(self.source, (x,), self.tolerance, self.max_iterations))
        return (self.inverse(x)[0] * factor,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        factor = self._factor()
        if factor == 0:
            return cast(tuple[torch.Tensor], _apply(self.source.H, (x,), self.tolerance, self.max_iterations))
        conjugate = factor.conj() if isinstance(factor, torch.Tensor) else factor.conjugate()
        return (self.inverse.adjoint(x)[0] * conjugate,)


class _PartialIsometryPseudoInverse(LinearOperator):
    """Use an adjoint on declared spaces without restricting ordinary source calls."""

    def __init__(
        self, source: LinearOperator, dim: Sequence[int], domain_shape: Sequence[int], range_shape: Sequence[int]
    ) -> None:
        super().__init__()
        self.source = source
        self.dim = tuple(dim)
        self.domain_shape, self.range_shape = tuple(domain_shape), tuple(range_shape)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        if tuple(x.shape[d] for d in self.dim) != self.range_shape:
            raise ValueError('Pseudoinverse input shape must match the declared encoding/padded shape.')
        return self.source.adjoint(x)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        if tuple(x.shape[d] for d in self.dim) != self.domain_shape:
            raise ValueError('Pseudoinverse adjoint input shape must match the declared reconstruction/original shape.')
        return self.source(x)
