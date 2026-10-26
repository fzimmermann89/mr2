"""N-dimensional convolution with a Hermitian adjoint."""

from collections.abc import Sequence
from typing import Literal

import torch
from torchnd import ConvNd, adjoint_pad_nd, pad_nd

from mr2.operators.LinearOperator import LinearOperator
from mr2.utils import normalize_index, normalize_indices


class ConvolutionOp(LinearOperator):
    r"""Apply an N-dimensional convolution or cross-correlation filter bank.

    For input channels :math:`x_1,\ldots,x_I` and convolutional filters
    :math:`k_{oi}`, the operator is defined by

    .. math::

        Cx = \left[\sum_{i=1}^I k_{1i}\ast x_i,\ldots,
                    \sum_{i=1}^I k_{Oi}\ast x_i\right]^T.

    Here :math:`\ast` denotes PyTorch-style cross-correlation by default,
    or mathematical convolution when ``correlation=False``.
    One input channel gives an analysis filter bank; one output channel sums
    filtered coefficient channels as a synthesis filter bank.
    Both operations are complex-linear in the input:

    .. math::

        k\ast x = \operatorname{Re}(k)\ast\operatorname{Re}(x)
                  -\operatorname{Im}(k)\ast\operatorname{Im}(x)
                  +\mathrm{i}\big(\operatorname{Re}(k)\ast\operatorname{Im}(x)
                  +\operatorname{Im}(k)\ast\operatorname{Re}(x)\big).

    Thus, a real filter acts separately on the real and imaginary parts.
    The forward operation does not conjugate the kernel.

    More precisely, let :math:`s` be the spatial kernel size,
    :math:`p=(s-1)/2`, and :math:`E` extend each input channel by :math:`p`
    samples on both sides according to ``pad_mode``, retaining the original
    spatial coordinates. Define the kernel offset

    .. math::

        d(m) = \begin{cases}
            m-p & \text{if correlation=True}, \\
            p-m & \text{if correlation=False}.
        \end{cases}

    For output indices :math:`n\in\Omega`,

    .. math::

        (Cx)_o[n] = \sum_{i=1}^I\sum_{0\le m<s}
                     k_{oi}[m]\,(Ex_i)[n+d(m)].

    Indices and inequalities are
    componentwise over the axes in ``dim``; other nonchannel axes are batch
    axes. Spatial sizes are unchanged.

    Writing :math:`\widetilde y` for :math:`y` extended by zero outside
    :math:`\Omega`, the Hermitian adjoint is

    .. math::

        (C^H y)_i &= E^H z_i, \\
        z_i[t] &= \sum_{o=1}^O\sum_{0\le m<s}
                    \overline{k_{oi}[m]}\,\widetilde y_o[t-d(m)].

    Here :math:`t` ranges over the extended input domain. :math:`E^H` crops
    constant zero padding or sums contributions from repeated boundary
    samples for circular, reflect, and replicate padding. The adjoint is
    generally not an inverse.
    """

    def __init__(
        self,
        kernel: torch.Tensor,
        *,
        dim: Sequence[int] | None = None,
        channel_dim: int = 0,
        correlation: bool = True,
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
        correlation
            Use PyTorch-style cross-correlation (default). Set to ``False``
            for mathematical convolution, reversing the spatial kernel axes.
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
            convolution=not correlation,
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
