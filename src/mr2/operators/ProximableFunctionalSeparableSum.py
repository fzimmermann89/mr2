"""Separable Sum of Proximable Functionals."""

from __future__ import annotations

from typing import cast

import torch
from typing_extensions import TypeVarTuple, Unpack, overload

from mr2.operators.Functional import ProximableFunctional
from mr2.operators.FunctionalSeparableSum import FunctionalSeparableSum
from mr2.operators.Operator import Operator

T = TypeVarTuple('T')


class ProximableFunctionalSeparableSum(FunctionalSeparableSum[Unpack[T]]):
    r"""Separable Sum of Proximable Functionals.

    This is a separable sum of the functionals. The forward method returns the sum of the functionals
    evaluated at the inputs, :math:`\sum_i f_i(x_i)`.
    Use ``f | g`` to build a separable sum from two proximable functionals.
    """

    functionals: tuple[ProximableFunctional, ...]

    @overload
    def __init__(self: ProximableFunctionalSeparableSum[torch.Tensor], f1: ProximableFunctional, /) -> None: ...

    @overload
    def __init__(
        self: ProximableFunctionalSeparableSum[torch.Tensor, torch.Tensor],
        f1: ProximableFunctional,
        f2: ProximableFunctional,
        /,
    ) -> None: ...

    @overload
    def __init__(
        self: ProximableFunctionalSeparableSum[torch.Tensor, torch.Tensor, Unpack[tuple[torch.Tensor, ...]]],
        f1: ProximableFunctional,
        f2: ProximableFunctional,
        /,
        *f: ProximableFunctional,
    ) -> None: ...

    def __init__(self, *functionals: ProximableFunctional) -> None:
        """Initialize the separable sum of proximable functionals."""
        super().__init__(*cast(tuple[Operator[torch.Tensor, tuple[torch.Tensor]], ...], functionals))  # type: ignore[misc]

    def prox(self, *x: Unpack[T], sigma: float | torch.Tensor = 1) -> tuple[Unpack[T]]:
        """Apply the proximal operators of the functionals to the inputs.

        Parameters
        ----------
        x
            The inputs to the proximal operators
        sigma
            The scaling factor for the proximal operators

        Returns
        -------
            A tuple of the proximal operators applied to the inputs
        """
        prox_x = tuple(
            f.prox(xi, sigma)[0] for f, xi in zip(self.functionals, cast(tuple[torch.Tensor, ...], x), strict=True)
        )
        return cast(tuple[Unpack[T]], prox_x)

    def prox_convex_conj(self, *x: Unpack[T], sigma: float | torch.Tensor = 1) -> tuple[Unpack[T]]:
        """Apply the proximal operators of the convex conjugate of the functionals to the inputs.

        Parameters
        ----------
        x
            The inputs to the proximal operators
        sigma
            The scaling factor for the proximal operators

        Returns
        -------
            A tuple of the proximal convex conjugate operators applied to the inputs
        """
        prox_convex_conj_x = tuple(
            f.prox_convex_conj(xi, sigma)[0]
            for f, xi in zip(self.functionals, cast(tuple[torch.Tensor, ...], x), strict=True)
        )
        return cast(tuple[Unpack[T]], prox_convex_conj_x)

    @overload  # type: ignore[override]
    def __or__(
        self: ProximableFunctionalSeparableSum[Unpack[T]], other: ProximableFunctional
    ) -> ProximableFunctionalSeparableSum[Unpack[T], torch.Tensor]: ...

    @overload
    def __or__(
        self: ProximableFunctionalSeparableSum[Unpack[T]], other: ProximableFunctionalSeparableSum[torch.Tensor]
    ) -> ProximableFunctionalSeparableSum[Unpack[T], torch.Tensor]: ...

    @overload
    def __or__(
        self: ProximableFunctionalSeparableSum[Unpack[T]],
        other: ProximableFunctionalSeparableSum[torch.Tensor, torch.Tensor],
    ) -> ProximableFunctionalSeparableSum[Unpack[T], torch.Tensor, torch.Tensor]: ...

    @overload
    def __or__(
        self: ProximableFunctionalSeparableSum[Unpack[T]],
        other: ProximableFunctionalSeparableSum[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> ProximableFunctionalSeparableSum[Unpack[T], torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __or__(
        self: ProximableFunctionalSeparableSum[Unpack[T]],
        other: ProximableFunctionalSeparableSum[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> ProximableFunctionalSeparableSum[Unpack[T], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __or__(
        self: ProximableFunctionalSeparableSum[Unpack[T]],
        other: Operator[torch.Tensor, tuple[torch.Tensor]],
    ) -> FunctionalSeparableSum[Unpack[T], torch.Tensor]: ...

    def __or__(  # type: ignore[misc]
        self: ProximableFunctionalSeparableSum[Unpack[T]], other: Operator[torch.Tensor, tuple[torch.Tensor]]
    ) -> FunctionalSeparableSum[Unpack[T], torch.Tensor]:
        """Separable sum of functionals.

        ``f | g`` is a ~mr2.operators.ProximableFunctionalSeparableSum,
        with ``(f|g)(x,y) == f(x) + g(y)``.
        """
        if isinstance(other, ProximableFunctionalSeparableSum):
            return cast(
                ProximableFunctionalSeparableSum[Unpack[T], torch.Tensor],
                self.__class__(*self.functionals, *other.functionals),
            )
        elif isinstance(other, ProximableFunctional):
            return cast(
                ProximableFunctionalSeparableSum[Unpack[T], torch.Tensor], self.__class__(*self.functionals, other)
            )
        elif isinstance(other, FunctionalSeparableSum):
            return cast(
                FunctionalSeparableSum[Unpack[T], torch.Tensor],
                FunctionalSeparableSum(*self.functionals, *other.functionals),
            )
        else:
            return NotImplemented

    def __ror__(
        self: ProximableFunctionalSeparableSum[Unpack[T]], other: Operator[torch.Tensor, tuple[torch.Tensor]]
    ) -> FunctionalSeparableSum[torch.Tensor, Unpack[T]]:
        """Separable sum of functionals."""
        if isinstance(other, ProximableFunctional):
            return cast(
                ProximableFunctionalSeparableSum[torch.Tensor, Unpack[T]], self.__class__(other, *self.functionals)
            )
        elif isinstance(other, FunctionalSeparableSum):
            return cast(
                FunctionalSeparableSum[torch.Tensor, Unpack[T]],
                FunctionalSeparableSum(*other.functionals, *self.functionals),
            )
        elif isinstance(other, Operator):
            return cast(
                FunctionalSeparableSum[torch.Tensor, Unpack[T]], FunctionalSeparableSum(other, *self.functionals)
            )
        else:
            return NotImplemented
