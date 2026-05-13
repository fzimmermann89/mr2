"""Downsampling operator."""

from collections.abc import Sequence

import torch
from torchnd import adjoint_linear_interpolation_nd, linear_interpolation_nd

from mr2.operators.LinearOperator import LinearOperator
from mr2.utils.reshape import normalize_indices


class DownSamplingOp(LinearOperator, adjoint_as_backward=True):
    """Downsample a tensor using linear interpolation.

    The forward operation maps from the high-resolution domain to the
    low-resolution range. The adjoint maps from the low-resolution range back to
    the high-resolution domain using the adjoint of linear interpolation.
    """

    def __init__(
        self,
        dim: Sequence[int] | int,
        domain_shape: Sequence[int],
        range_shape: Sequence[int],
        align_corners: bool = False,
    ) -> None:
        """Initialize the DownSamplingOp.

        Parameters
        ----------
        dim
            Dimension(s) to downsample.
        domain_shape
            High-resolution shape along ``dim``.
        range_shape
            Low-resolution shape along ``dim``.
        align_corners
            Geometric convention passed to ``torch.nn.functional.interpolate``.
        """
        super().__init__()
        self.dim = (dim,) if isinstance(dim, int) else tuple(dim)
        self.domain_shape = tuple(domain_shape)
        self.range_shape = tuple(range_shape)
        self.align_corners = align_corners

        if len(self.dim) != len(self.domain_shape) or len(self.dim) != len(self.range_shape):
            raise ValueError('dim, domain_shape, and range_shape must have the same length.')
        if any(size <= 0 for size in (*self.domain_shape, *self.range_shape)):
            raise ValueError('domain_shape and range_shape entries must be positive.')

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Downsample ``x`` from ``domain_shape`` to ``range_shape`` along ``dim``.

        Parameters
        ----------
        x
            Input tensor with high-resolution shape ``domain_shape`` along ``dim``.

        Returns
        -------
            Downsampled tensor with low-resolution shape ``range_shape`` along ``dim``.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward of DownSamplingOp.

        .. note::
            Prefer calling the instance of the DownSamplingOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        dim = normalize_indices(x.ndim, self.dim)
        for d, size in zip(dim, self.domain_shape, strict=True):
            if x.shape[d] != size:
                raise ValueError(f'Expected domain size {size} along dimension {d}, got {x.shape[d]}.')
        return (linear_interpolation_nd(x, self.range_shape, dims=dim, align_corners=self.align_corners),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply adjoint of DownSamplingOp."""
        dim = normalize_indices(x.ndim, self.dim)
        for d, size in zip(dim, self.range_shape, strict=True):
            if x.shape[d] != size:
                raise ValueError(f'Expected range size {size} along dimension {d}, got {x.shape[d]}.')
        return (adjoint_linear_interpolation_nd(x, self.domain_shape, dims=dim, align_corners=self.align_corners),)
