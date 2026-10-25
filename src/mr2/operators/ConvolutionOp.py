"""N-dimensional convolution with a Hermitian adjoint."""

from collections.abc import Sequence
from typing import Literal

import torch
from torchnd import adjoint_pad_nd, conv_nd, pad_nd

from mr2.operators.LinearOperator import LinearOperator
from mr2.utils import normalize_index, normalize_indices


class ConvolutionOp(LinearOperator):
    """Apply a convolutional filter bank along selected spatial axes.

    The forward follows PyTorch's cross-correlation convention, with stride one
    and symmetric padding to preserve spatial sizes. The channel axis is retained
    in the output, with its size changed from input to output channels. All other
    axes are treated as batch axes.

    The Hermitian adjoint conjugates the kernel, applies transposed convolution,
    and folds padded boundary contributions back into the original image. It is
    generally different from a forward convolution with exchanged channels and
    is not an inverse.
    """

    def __init__(
        self,
        kernel: torch.Tensor,
        *,
        dim: Sequence[int] | None = None,
        channel_dim: int = 0,
        pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'circular',
    ) -> None:
        """Initialize the convolution operator.

        Parameters
        ----------
        kernel
            Filter bank of shape ``(out_channels, in_channels, *kernel_size)``.
            Channel sizes must be positive and spatial sizes must be positive
            and odd. The tensor is stored directly, preserving gradients to
            caller-owned kernels. Prefer a Parameter in this layout for a
            trainable kernel; a view of a Parameter is not registered as a
            parameter of this operator.
        dim
            Spatial axes, in the order of the kernel's spatial dimensions.
            Defaults to the last ``kernel.ndim - 2`` axes.
        channel_dim
            Input and output channel axis. Defaults to the leading axis.
            Must be distinct from the spatial axes.
        pad_mode
            Boundary padding. ``'constant'`` pads with zeros; the other modes
            use reflected, repeated-edge, or periodic values.

        Notes
        -----
        Input and kernel must have compatible devices and dtypes; no explicit
        conversion is performed. Real and complex tensors are supported.
        For nonconstant padding, the adjoint uses torchnd's autograd-based
        padding adjoint and requires gradient recording to be enabled.
        """
        super().__init__()
        if kernel.ndim < 3 or any(size == 0 for size in kernel.shape[:2]):
            raise ValueError('kernel must have shape (out_channels, in_channels, *kernel_size) with positive channels')
        if any(size % 2 == 0 for size in kernel.shape[2:]):
            raise ValueError('Spatial kernel sizes must be positive and odd')
        if pad_mode not in ('constant', 'reflect', 'replicate', 'circular'):
            raise ValueError("pad_mode must be 'constant', 'reflect', 'replicate', or 'circular'")
        self.dim = tuple(range(2 - kernel.ndim, 0)) if dim is None else tuple(dim)
        if len(self.dim) != kernel.ndim - 2:
            raise ValueError('dim must contain one axis per spatial kernel dimension')
        self.channel_dim = channel_dim
        self.pad_mode = pad_mode
        self.kernel = kernel
        self._pad = tuple(p for size in kernel.shape[2:] for p in (size // 2, size // 2))

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the filter bank.

        Parameters
        ----------
        x
            Tensor with ``kernel.shape[1]`` channels along ``channel_dim``.

        Returns
        -------
            Single tensor with ``kernel.shape[0]`` output channels, retaining
            the spatial sizes and axis layout of the input.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Pad the input and apply convolution."""
        dim = normalize_indices(x.ndim, self.dim)
        channel_dim = normalize_index(x.ndim, self.channel_dim)
        if channel_dim in dim:
            raise ValueError('channel_dim must be distinct from the spatial axes')
        if x.shape[channel_dim] != self.kernel.shape[1]:
            raise ValueError('Input channel size must match kernel.shape[1]')
        padded = pad_nd(x, pad=self._pad, dims=dim, mode=self.pad_mode)
        return (conv_nd(padded, self.kernel, dim=dim, channel_dim=channel_dim),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply conjugated transposed convolution and the padding adjoint."""
        dim = normalize_indices(x.ndim, self.dim)
        channel_dim = normalize_index(x.ndim, self.channel_dim)
        if channel_dim in dim:
            raise ValueError('channel_dim must be distinct from the spatial axes')
        if x.shape[channel_dim] != self.kernel.shape[0]:
            raise ValueError('Adjoint input channel size must match kernel.shape[0]')
        padded = conv_nd(x, self.kernel.conj(), dim=dim, channel_dim=channel_dim, transposed=True)
        return (adjoint_pad_nd(padded, pad=self._pad, dims=dim, mode=self.pad_mode),)
