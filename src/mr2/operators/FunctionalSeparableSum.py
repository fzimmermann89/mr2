"""Separable Sum of Functionals."""

from __future__ import annotations

import operator
from collections.abc import Iterator
from functools import reduce
from typing import cast

import torch
from typing_extensions import TypeVarTuple, Unpack, overload

from mr2.operators.Operator import Operator

T = TypeVarTuple('T')


class FunctionalSeparableSum(Operator[Unpack[T], tuple[torch.Tensor]]):
    r"""Separable Sum of functionals.

    This is a separable sum of functionals. The forward method returns the sum of the functionals
    evaluated at the inputs, :math:`\sum_i f_i(x_i)`.
    """

    functionals: tuple[Operator[torch.Tensor, tuple[torch.Tensor]], ...]

    @overload
    def __init__(
        self: FunctionalSeparableSum[torch.Tensor], f1: Operator[torch.Tensor, tuple[torch.Tensor]], /
    ) -> None: ...

    @overload
    def __init__(
        self: FunctionalSeparableSum[torch.Tensor, torch.Tensor],
        f1: Operator[torch.Tensor, tuple[torch.Tensor]],
        f2: Operator[torch.Tensor, tuple[torch.Tensor]],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self: FunctionalSeparableSum[torch.Tensor, torch.Tensor, torch.Tensor],
        f1: Operator[torch.Tensor, tuple[torch.Tensor]],
        f2: Operator[torch.Tensor, tuple[torch.Tensor]],
        f3: Operator[torch.Tensor, tuple[torch.Tensor]],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self: FunctionalSeparableSum[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        f1: Operator[torch.Tensor, tuple[torch.Tensor]],
        f2: Operator[torch.Tensor, tuple[torch.Tensor]],
        f3: Operator[torch.Tensor, tuple[torch.Tensor]],
        f4: Operator[torch.Tensor, tuple[torch.Tensor]],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self: FunctionalSeparableSum[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        f1: Operator[torch.Tensor, tuple[torch.Tensor]],
        f2: Operator[torch.Tensor, tuple[torch.Tensor]],
        f3: Operator[torch.Tensor, tuple[torch.Tensor]],
        f4: Operator[torch.Tensor, tuple[torch.Tensor]],
        f5: Operator[torch.Tensor, tuple[torch.Tensor]],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self: FunctionalSeparableSum[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Unpack[tuple[torch.Tensor, ...]]
        ],
        f1: Operator[torch.Tensor, tuple[torch.Tensor]],
        f2: Operator[torch.Tensor, tuple[torch.Tensor]],
        f3: Operator[torch.Tensor, tuple[torch.Tensor]],
        f4: Operator[torch.Tensor, tuple[torch.Tensor]],
        f5: Operator[torch.Tensor, tuple[torch.Tensor]],
        /,
        *f: Operator[torch.Tensor, tuple[torch.Tensor]],
    ) -> None: ...

    def __init__(self, *functionals: Operator[torch.Tensor, tuple[torch.Tensor]]) -> None:
        """Initialize the separable sum of functionals.

        Parameters
        ----------
        functionals
            The functionals to be summed.
        """
        super().__init__()
        self.functionals = functionals

    def __call__(self, *x: Unpack[T]) -> tuple[torch.Tensor]:
        """Evaluate the sum of separable functionals."""
        return super().__call__(*x)

    def forward(self, *x: Unpack[T]) -> tuple[torch.Tensor]:
        """Apply forward of FunctionalSeparableSum."""
        if len(x) != len(self.functionals):
            raise ValueError('The number of inputs must match the number of functionals.')
        result = reduce(
            operator.add, (f(xi)[0] for f, xi in zip(self.functionals, cast(tuple[torch.Tensor, ...], x), strict=True))
        )
        return (result,)

    def __or__(
        self: FunctionalSeparableSum[Unpack[T]], other: Operator[torch.Tensor, tuple[torch.Tensor]]
    ) -> FunctionalSeparableSum[Unpack[T], torch.Tensor]:
        """Separable sum of functionals."""
        if isinstance(other, FunctionalSeparableSum):
            return cast(
                FunctionalSeparableSum[Unpack[T], torch.Tensor], self.__class__(*self.functionals, *other.functionals)
            )
        elif isinstance(other, Operator):
            return cast(FunctionalSeparableSum[Unpack[T], torch.Tensor], self.__class__(*self.functionals, other))
        else:
            return NotImplemented

    def __ror__(
        self: FunctionalSeparableSum[Unpack[T]], other: Operator[torch.Tensor, tuple[torch.Tensor]]
    ) -> FunctionalSeparableSum[torch.Tensor, Unpack[T]]:
        """Separable sum of functionals."""
        if isinstance(other, Operator):
            return cast(FunctionalSeparableSum[torch.Tensor, Unpack[T]], self.__class__(other, *self.functionals))
        else:
            return NotImplemented

    def __iter__(self) -> Iterator[Operator[torch.Tensor, tuple[torch.Tensor]]]:
        """Iterate over the functionals."""
        return iter(self.functionals)

    def __len__(self) -> int:
        """Return the number of functionals."""
        return len(self.functionals)
