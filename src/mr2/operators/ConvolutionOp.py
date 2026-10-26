"""N-dimensional convolution with a Hermitian adjoint."""

from collections.abc import Sequence
from typing import Literal

import torch
from torchnd import ConvNd, adjoint_pad_nd, pad_nd

from mr2.operators.LinearOperator import LinearOperator
from mr2.utils import normalize_index, normalize_indices


class ConvolutionOp(LinearOperator):
    r"""Apply an N-dimensional convolution filter bank.

    Let :math:`k_{oi}[m]` be the kernel, :math:`s` its spatial size,
    :math:`p=(s-1)/2`, and :math:`P` symmetric padding by :math:`p`.
    For output indices :math:`n\in\Omega`, the discrete forward and Hermitian
    adjoint are

    .. math::

        (Cx)_o[n] &= \sum_i\sum_{0\le m<s} k_{oi}[m]\,(Px)_i[n+s-1-m], \\
        (C^H y)_i &= (P^H z)_i, \qquad
        z_i[t] = \sum_o\sum_{\substack{n\in\Omega,\,0\le m<s\\n+s-1-m=t}}
                     \overline{k_{oi}[m]}\,y_o[n].

    Indices and inequalities are componentwise over the axes in ``dim``;
    other axes are batch axes. Spatial sizes are unchanged. :math:`P^H` crops
    zero padding or sums contributions from repeated boundary samples for
    circular, reflect, and replicate padding. The adjoint is not an inverse.
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
            Channel sizes must be positive; spatial sizes must be positive and odd.
        dim
            Spatial axes, in the order of the kernel's spatial dimensions.
            Defaults to the last ``kernel.ndim - 2`` axes.
        channel_dim
            Input and output channel axis. Defaults to the leading axis.
            Must be distinct from the spatial axes.
        pad_mode
            Boundary padding. ``'constant'`` pads with zeros; the other modes
            use reflected, repeated-edge, or periodic values.
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
        # Keep padding outside ConvNd: its checkpointed padding is incompatible with torch.func.grad.
        self._convolution = ConvNd(
            in_channels=kernel.shape[1],
            out_channels=kernel.shape[0],
            kernel_size=tuple(kernel.shape[2:]),
            dim=self.dim,
            channel_dim=channel_dim,
            bias=False,
            convolution=True,
            weight=kernel.detach() if isinstance(kernel, torch.nn.Parameter) else kernel,
        )
        if isinstance(kernel, torch.nn.Parameter) or kernel.requires_grad:
            # ConvNd initializes supplied Parameters; install the original only after construction.
            del self._convolution.weight
            self._convolution.weight = kernel  # type: ignore[assignment]
        self._pad = tuple(p for size in kernel.shape[2:] for p in (size // 2, size // 2))

    @property
    def kernel(self) -> torch.Tensor:
        """Convolution kernel in output-channel, input-channel, spatial order."""
        return self._convolution.weight

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
        """Apply forward of ConvolutionOp.

        .. note::
            Prefer calling the operator as ``operator(x)`` over directly calling this method.
            See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        dim = normalize_indices(x.ndim, self.dim)
        channel_dim = normalize_index(x.ndim, self.channel_dim)
        if channel_dim in dim:
            raise ValueError('channel_dim must be distinct from the spatial axes')
        if x.shape[channel_dim] != self.kernel.shape[1]:
            raise ValueError('Input channel size must match kernel.shape[1]')
        padded = pad_nd(x, pad=self._pad, dims=dim, mode=self.pad_mode)
        return (self._convolution(padded),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the Hermitian adjoint defined in the class docstring."""
        dim = normalize_indices(x.ndim, self.dim)
        channel_dim = normalize_index(x.ndim, self.channel_dim)
        if channel_dim in dim:
            raise ValueError('channel_dim must be distinct from the spatial axes')
        if x.shape[channel_dim] != self.kernel.shape[0]:
            raise ValueError('Adjoint input channel size must match kernel.shape[0]')
        padded = self._convolution.adjoint(x)
        return (adjoint_pad_nd(padded, pad=self._pad, dims=dim, mode=self.pad_mode),)
