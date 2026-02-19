"""General matrix of operators."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import reduce
from typing import cast

import torch
from typing_extensions import Unpack

from mr2.operators.Operator import Operator


class OperatorMatrix(Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]):
    r"""Matrix of (possibly non-linear) operators.

    A matrix of operators where each element is an `~mr2.operators.Operator` taking
    a single tensor input and returning one or more tensors.

    The i-th row output is the tuple-sum over columns j of `operators[i][j](x[j])`.
    Outputs of all rows are concatenated.

    Use ``A | B`` for horizontal stacking and ``A % B`` for vertical stacking.
    """

    _operators: list[list[Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]]]

    def __init__(
        self, operators: Sequence[Sequence[Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]]]
    ):
        """Initialize operator matrix from rows."""
        if not all(isinstance(op, Operator) for row in operators for op in row):
            raise ValueError('All elements should be Operators.')
        if not all(len(row) == len(operators[0]) for row in operators):
            raise ValueError('All rows should have the same length.')
        super().__init__()
        self._operators = cast(
            list[list[Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]]],
            torch.nn.ModuleList(torch.nn.ModuleList(row) for row in operators),
        )
        self._shape = (len(operators), len(operators[0]) if operators else 0)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the operator matrix as (rows, columns)."""
        return self._shape

    def __call__(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply operator matrix to input tensors."""
        return super().__call__(*x)

    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply forward of OperatorMatrix."""
        if len(x) != self.shape[1]:
            raise ValueError('Input should be the same number of tensors as the OperatorMatrix has columns.')

        def _add(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            return tuple(aa + bb for aa, bb in zip(a, b, strict=True))

        row_results = [reduce(_add, (op(xi) for op, xi in zip(row, x, strict=True))) for row in self._operators]
        return tuple(out for row_out in row_results for out in row_out)

    def __iter__(self) -> Iterator[Sequence[Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]]]:
        """Iterate over matrix rows."""
        return iter(self._operators)

    def __repr__(self):
        """Representation of the operator matrix."""
        return f'OperatorMatrix(shape={self._shape}, operators={self._operators})'

    def __or__(
        self,
        other: Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]] | OperatorMatrix,
    ) -> OperatorMatrix:
        """Horizontal stacking."""
        if isinstance(other, OperatorMatrix):
            if (rows_self := self.shape[0]) != (rows_other := other.shape[0]):
                raise ValueError(
                    'Shape mismatch in horizontal stacking: '
                    f'cannot stack matrices with {rows_self} and {rows_other} rows.'
                )
            return OperatorMatrix([[*self_row, *other_row] for self_row, other_row in zip(self, other, strict=True)])
        elif isinstance(other, Operator) and not isinstance(other, OperatorMatrix):
            if (rows := self.shape[0]) > 1:
                raise ValueError(
                    f'Shape mismatch in horizontal stacking: cannot stack Operator and matrix with {rows} rows.'
                )
            return OperatorMatrix([[*self._operators[0], other]])
        else:
            return NotImplemented

    def __mod__(
        self,
        other: Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]] | OperatorMatrix,
    ) -> OperatorMatrix:
        """Vertical stacking."""
        if isinstance(other, OperatorMatrix):
            if (cols_self := self.shape[1]) != (cols_other := other.shape[1]):
                raise ValueError(
                    'Shape mismatch in vertical stacking: '
                    f'cannot stack matrices with {cols_self} and {cols_other} columns.'
                )
            return OperatorMatrix([*self._operators, *other._operators])
        elif isinstance(other, Operator) and not isinstance(other, OperatorMatrix):
            if (cols := self.shape[1]) > 1:
                raise ValueError(
                    f'Shape mismatch in vertical stacking: cannot stack Operator and matrix with {cols} columns.'
                )
            return OperatorMatrix([*self._operators, [other]])
        else:
            return NotImplemented
