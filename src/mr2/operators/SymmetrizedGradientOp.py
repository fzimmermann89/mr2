"""Symmetrized gradient operator."""

from collections.abc import Sequence
from typing import Literal

import torch

from mr2.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mr2.operators.LinearOperator import LinearOperator
from mr2.operators.RearrangeOp import RearrangeOp


class SymmetrizedGradientOp(LinearOperator):
    r"""Symmetrized gradient operator.

    This operator computes the symmetric part of the discrete gradient.
    The first axis of the input tensor indexes components and must satisfy
    ``v.shape[0] == len(dim)``.

    Directional finite differences are computed with
    `~mr2.operators.FiniteDifferenceOp` along the axes in ``dim`` and then
    symmetrized over the first two axes:

    .. math::
        E(v) = \tfrac{1}{2}\,(\nabla v + (\nabla v)^{\mathsf T}),
        \qquad
        E(v)_{i,j} = \tfrac{1}{2}\,((\nabla v)_{i,j} + (\nabla v)_{j,i}).

    For input shape ``(len(dim), ...)``, the output shape is ``(len(dim), len(dim), ...)``.
    """

    def __init__(
        self,
        dim: Sequence[int],
        mode: Literal['central', 'forward', 'backward'] = 'backward',
        pad_mode: Literal['zeros', 'circular'] = 'zeros',
    ) -> None:
        """Symmetrized gradient operator.

        Parameters
        ----------
        dim
            Axes along which finite differences are computed.
            Axis ``0`` is reserved for vector components and must not be part of ``dim``.
        mode
            Finite-difference scheme (`'forward'`, `'backward'`, or `'central'`).
        pad_mode
            Boundary handling used by finite differences (`'zeros'` or `'circular'`).

        Raises
        ------
        ValueError
            If ``dim`` contains axis ``0``.
        """
        super().__init__()

        if 0 in dim:
            raise ValueError('dim must not contain axis 0, which indexes vector components.')

        self._n_dim = len(dim)
        finite_difference_op = FiniteDifferenceOp(dim=dim, mode=mode, pad_mode=pad_mode)
        transpose_op = RearrangeOp('sym_grad_dim grad_dim ... -> grad_dim sym_grad_dim ...')
        self._operator = 0.5 * (1 + transpose_op) @ finite_difference_op

    def __call__(self, v: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the symmetrized gradient.

        Parameters
        ----------
        v
            Input tensor with shape ``(len(dim), ...)``.

        Returns
        -------
            Symmetrized gradient with shape ``(len(dim), len(dim), ...)``.
        """
        return super().__call__(v)

    def forward(self, v: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of SymmetrizedGradientOp.

        .. note::
            Prefer calling the instance of the SymmetrizedGradientOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if v.shape[0] != self._n_dim:
            raise ValueError('First dimension of input tensor has to match the number of finite difference directions.')
        return self._operator(v)

    def adjoint(self, w: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the symmetrized gradient.

        Parameters
        ----------
        w
            Symmetrized-gradient tensor with shape ``(len(dim), len(dim), *S)``.

        Returns
        -------
            Tensor with shape ``(len(dim), ...)``.

        Raises
        ------
        ValueError
            If the first two dimensions of ``w`` do not equal ``len(dim)``.
        """
        if w.shape[0] != self._n_dim or w.shape[1] != self._n_dim:
            raise ValueError(
                'First two dimensions of input tensor must match the number of finite difference directions.'
            )
        return self._operator.adjoint(w)
