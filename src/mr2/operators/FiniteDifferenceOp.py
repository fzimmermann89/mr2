"""Class for Finite Difference Operator."""

from collections.abc import Sequence
from typing import Literal

import torch

from mr2.operators.LinearOperator import LinearOperator
from mr2.utils.filters import filter_separable


class FiniteDifferenceOp(LinearOperator):
    r"""Finite-difference operator.

    Computes directional discrete derivatives along the axes in ``dim`` (for
    example ``dim=(-2, -1)`` for spatial ``(y, x)`` gradients). One directional
    derivative is computed per axis and stacked along a new leading dimension.

    Each output channel ``i`` computes finite differences along axis
    ``dim[i]``. For ``mode='forward'``, values follow

    .. math::
        y_i[k] = x[k + e_i] - x[k],

    where :math:`e_i` is a one-step shift along axis ``dim[i]``. ``'central'``
    and ``'backward'`` use the stencils :math:`\\tfrac{1}{2}(-1,0,1)` and
    :math:`(-1,1,0)`.

    For input ``x`` with shape ``S``, ``op(x)`` returns a tensor with shape
    ``(len(dim), *S)`` where channel ``i`` corresponds to axis ``dim[i]``.
    Supported schemes are ``'forward'``, ``'backward'``, and ``'central'``.
    Boundary handling is controlled by ``pad_mode`` (``'zeros'`` or
    ``'circular'``).
    """

    @staticmethod
    def finite_difference_kernel(mode: Literal['central', 'forward', 'backward']) -> torch.Tensor:
        """Return the 1D finite-difference kernel for a given mode.

        Parameters
        ----------
        mode
            Difference scheme: `'forward'`, `'backward'`, or `'central'`.

        Returns
        -------
            1D kernel tensor of length 3.

        Raises
        ------
        `ValueError`
            If `mode` is not one of `'central'`, `'forward'`, or `'backward'`.
        """
        match mode:
            case 'forward':
                kernel = torch.tensor((0, -1, 1))
            case 'backward':
                kernel = torch.tensor((-1, 1, 0))
            case 'central':
                kernel = torch.tensor((-1, 0, 1)) / 2
            case _:
                raise ValueError(f'mode should be one of (central, forward, backward), not {mode}')
        return kernel

    def __init__(
        self,
        dim: Sequence[int],
        mode: Literal['central', 'forward', 'backward'] = 'forward',
        pad_mode: Literal['zeros', 'circular'] = 'zeros',
    ) -> None:
        """Initialize a finite-difference operator.

        Parameters
        ----------
        dim
            Axes along which finite differences are computed.
            The order defines the order of directional channels in the output.
        mode
            Finite-difference scheme used to construct the 1D kernel.
        pad_mode
            Boundary handling used during filtering.
            `'zeros'` maps to constant-zero padding, `'circular'` to periodic padding.
        """
        super().__init__()
        self.dim = dim
        self.pad_mode: Literal['constant', 'circular'] = 'constant' if pad_mode == 'zeros' else pad_mode
        self.kernel = self.finite_difference_kernel(mode)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply directional finite differences.

        Computes one directional derivative per axis in `dim` and stacks the
        results along a new leading dimension.

        Parameters
        ----------
        x
            Input tensor from the operator domain.

        Returns
        -------
            Single tensor of shape `(len(dim), *x.shape)` containing directional
            finite differences in the same order as `dim`.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of FiniteDifferenceOp.

        .. note::
            Prefer calling the instance of the FiniteDifferenceOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return (
            torch.stack(
                [
                    filter_separable(x, (self.kernel,), dim=(dim,), pad_mode=self.pad_mode, pad_value=0.0)
                    for dim in self.dim
                ]
            ),
        )

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the finite-difference operator.

        Expects stacked directional components (as returned by `forward`), applies
        the per-direction adjoint filter (kernel flipped in 1D), and sums across
        directions.

        Parameters
        ----------
        y
            Directional components stacked along the leading axis.
            `y.shape[0]` must equal `len(dim)`.

        Returns
        -------
            Tensor in the original domain shape.

        Raises
        ------
        ValueError
            If the leading dimension of `y` does not match `len(dim)`.
        """
        if y.shape[0] != len(self.dim):
            raise ValueError('First dimension of input tensor has to match the number of finite difference directions.')
        return (
            torch.sum(
                torch.stack(
                    [
                        filter_separable(
                            yi,
                            (torch.flip(self.kernel, dims=(-1,)),),
                            dim=(dim,),
                            pad_mode=self.pad_mode,
                            pad_value=0.0,
                        )
                        for dim, yi in zip(self.dim, y, strict=False)
                    ]
                ),
                dim=0,
            ),
        )
