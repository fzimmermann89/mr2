"""Spatial and temporal filters."""

import math
import warnings
from collections.abc import Sequence
from functools import reduce
from math import ceil
from typing import Literal

import numpy as np
import torch
from einops import repeat

from mr2.utils.reshape import normalize_indices


def filter_separable(
    x: torch.Tensor,
    kernels: Sequence[torch.Tensor],
    dim: Sequence[int],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular', 'none'] = 'constant',
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Apply one-dimensional kernels successively along the selected axes.

    Each step uses PyTorch-style cross-correlation without conjugating the
    kernel. Padding preserves the input shape unless ``pad_mode='none'``.

    Parameters
    ----------
    x
        Tensor to filter
    kernels
        List of 1D kernels to apply to the tensor x
    dim
        Axes to filter over. Must have the same length as kernels.
    pad_mode
        Padding mode
    pad_value
        Padding value for pad_mode = constant

    Returns
    -------
        Filtered tensor with the promoted dtype of the input and kernels.
        The shape is unchanged unless ``pad_mode='none'``, which keeps only
        positions where the kernel fits entirely inside the input.
    """
    if len(dim) != len(kernels):
        raise ValueError('Must provide matching length kernels and dim arguments.')

    dim = normalize_indices(x.ndim, dim)

    if pad_mode == 'constant' and pad_value == 0:
        # padding is done inside the convolution
        padding_conv = 'same'
    else:
        # padding is done with pad() before the convolution
        padding_conv = 'valid'

    # output will be of the promoted type of the input and the kernels
    target_dtype = reduce(torch.promote_types, [k.dtype for k in kernels], x.dtype)
    x = x.to(target_dtype)

    for kernel, d in zip(kernels, dim, strict=False):
        kernel = kernel.to(device=x.device, dtype=target_dtype)
        # moveaxis is not implemented for batched tensors, so vmap would fail.
        # thus we use permute.
        idx = list(range(x.ndim))
        # swapping the last axis and the axis to filter over
        idx[d], idx[-1] = idx[-1], idx[d]
        x = x.permute(idx)
        # flatten first to allow for circular, replicate and reflection padding for arbitrary tensor size
        x_flat = x.flatten(end_dim=-2)
        if padding_conv == 'valid' and pad_mode != 'none':
            left_pad = (len(kernel) - 1) // 2
            right_pad = (len(kernel) - 1) - left_pad
            x_flat = torch.nn.functional.pad(x_flat, pad=(left_pad, right_pad), mode=pad_mode, value=pad_value)
        x = torch.nn.functional.conv1d(
            repeat(x_flat, 'batch x -> batch channels x', channels=1),
            repeat(kernel, 'x -> batch channels x', batch=1, channels=1),
            padding=padding_conv,
        ).reshape(*x.shape[:-1], -1)
        # for a single permutation, this undoes the permutation
        x = x.permute(idx)
    return x


def gaussian_filter(
    x: torch.Tensor,
    sigmas: float | Sequence[float] | torch.Tensor,
    dim: int | Sequence[int] | None = None,
    truncate: int = 3,
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    pad_value: float = 0.0,
) -> torch.Tensor:
    r"""Smooth a tensor with a separable N-dimensional Gaussian filter.

    Each axis uses samples of :math:`\exp(-t^2/(2\sigma^2))`, truncated at
    ``ceil(truncate * sigma)`` pixels on either side and normalized to sum to one.

    Parameters
    ----------
    x
        Tensor to filter
    sigmas
        Standard deviation for Gaussian kernel. If iterable, must have length equal to the number of axes.
    dim
        Axis or axes to filter over. If None, filters over all axes.
    truncate
        Truncate the filter at this many standard deviations.
    pad_mode
        Padding mode
    pad_value
        Padding value for pad_mode = constant

    Returns
    -------
        Smoothed tensor with the same shape as the input.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    sigmas = torch.as_tensor(sigmas) if np.iterable(sigmas) else torch.tensor([sigmas] * len(dim))
    if not torch.all(sigmas > 0):
        raise ValueError('`sigmas` must be positive')

    if len(sigmas) != len(dim):
        raise ValueError('Must provide matching length sigmas and dim arguments. ')

    kernels = tuple(
        [
            torch.exp(-0.5 * (torch.arange(-ceil(truncate * sigma), ceil(truncate * sigma) + 1) / sigma) ** 2)
            for sigma in sigmas
        ]
    )
    kernels = tuple([(k / k.sum()).to(device=x.device) for k in kernels])
    x_filtered = filter_separable(x, kernels, dim, pad_mode, pad_value)
    return x_filtered


def uniform_filter(
    x: torch.Tensor,
    width: int | Sequence[int] | torch.Tensor,
    dim: int | Sequence[int] | None = None,
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Smooth a tensor by averaging a rectangular neighborhood.

    Each selected axis uses a constant kernel whose coefficients sum to one.
    Window widths larger than the corresponding input axis are clipped to its length.

    Parameters
    ----------
    x
        Tensor to filter
    width
        Width of uniform kernel. If iterable, must have length equal to the number of axes.
    dim
        Axis or axes to filter over. If None, filters over all axes.
    pad_mode
        Padding mode
    pad_value
        Padding value for pad_mode = constant

    Returns
    -------
        Smoothed tensor with the same shape as the input.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    width = torch.as_tensor(width) if np.iterable(width) else torch.tensor([width] * len(dim))
    if not torch.all(width > 0):
        raise ValueError('width must be positive.')
    if torch.any(width % 2 != 1):
        warnings.warn('width should be odd.', stacklevel=2)
    if len(width) != len(dim):
        raise ValueError('Must provide matching length width and dim arguments. ')
    width = torch.minimum(width, torch.tensor(x.shape)[(dim), ...])

    kernels = tuple([torch.ones(width, device=x.device) / width for width in width])
    x_filtered = filter_separable(x, kernels, dim, pad_mode, pad_value)
    return x_filtered


def gabor_filters_2d(
    n_filters: int,
    kernel_size: tuple[int, int],
    sigma: float = 2.0,
    frequency: float = 0.25,
) -> torch.Tensor:
    r"""Create oriented cosine filters under an isotropic Gaussian envelope.

    On the pixel grid centered at ``((ky - 1) / 2, (kx - 1) / 2)``, filter
    :math:`i` samples

    .. math::

        g_i(x,y) = \exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right)
                   \cos\!\left(2\pi f(x\cos\theta_i+y\sin\theta_i)\right),
        \qquad \theta_i = \frac{i\pi}{n_\mathrm{filters}}.

    The orientations cover :math:`[0,\pi)` and specify the direction in which
    the cosine varies. Each sampled filter has its mean subtracted and is
    divided by its L2 norm when nonzero.

    Parameters
    ----------
    n_filters
        Number of Gabor filters to generate.
    kernel_size
        Spatial shape ``(ky, kx)`` in pixels.
    sigma
        Positive finite standard deviation of the Gaussian envelope, in pixels.
    frequency
        Spatial frequency of the sinusoidal carrier, in cycles per pixel.

    Returns
    -------
        Real Gabor filter bank of shape ``(n_filters, ky, kx)``, using the
        default floating dtype. Filters have zero mean; nonzero filters have
        unit L2 norm.
    """
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be positive and finite')
    ky, kx = kernel_size
    x = torch.linspace(-(kx - 1) / 2, (kx - 1) / 2, kx)
    y = torch.linspace(-(ky - 1) / 2, (ky - 1) / 2, ky)

    yy, xx = torch.meshgrid(y, x, indexing='ij')
    gaussian = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

    filters = torch.zeros(n_filters, ky, kx)

    for i in range(n_filters):
        theta = i * math.pi / n_filters
        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        carrier = torch.cos(2.0 * math.pi * frequency * x_theta)
        gabor = gaussian * carrier
        gabor = gabor - gabor.mean()
        norm = torch.sqrt(torch.sum(gabor**2))

        if norm > 0:
            gabor = gabor / norm

        filters[i] = gabor

    return filters


def dct_matrix(
    n: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Create the orthonormal discrete cosine transform of type II.

    The rows are cosine basis vectors, indexed by increasing frequency.
    For :math:`k,m=0,\ldots,n-1`,

    .. math::

        C_{k,m} = \alpha_k\cos\!\left(\frac{\pi k(m+1/2)}{n}\right),
        \qquad \alpha_0 = n^{-1/2},\quad \alpha_{k>0} = (2/n)^{1/2}.

    ``C @ x`` computes the DCT-II of a length-``n`` vector, and ``C.T``
    applies its inverse. The first row is the constant (DC) basis vector.

    Parameters
    ----------
    n
        Number of samples and basis vectors.
    dtype
        Floating dtype of the matrix.

    Returns
    -------
        Orthonormal DCT-II matrix of shape ``(n, n)``.
    """
    k = torch.arange(n, dtype=dtype)[:, None]
    x = torch.arange(n, dtype=dtype)[None, :]

    matrix = torch.cos(torch.pi / n * (x + 0.5) * k)

    matrix[0] *= math.sqrt(1.0 / n)
    matrix[1:] *= math.sqrt(2.0 / n)

    return matrix


def dct_filters(
    kernel_size: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Create a separable orthonormal cosine basis for an N-dimensional patch.

    For spatial shape :math:`(s_1,\ldots,s_d)`, the filter indexed by
    :math:`(k_1,\ldots,k_d)` is the product of one-dimensional DCT-II rows:

    .. math::

        h_{k_1,\ldots,k_d}[m_1,\ldots,m_d]
            = \prod_{j=1}^d C^{(s_j)}_{k_j,m_j}.

    Frequency indices are flattened with the last axis varying fastest.
    The first filter is constant. Flattening each filter gives the rows of
    an orthonormal patch-transform matrix; its transpose is the inverse.

    Parameters
    ----------
    kernel_size
        Positive spatial sizes, for example ``(5, 7)`` or ``(5, 7, 9)``.
    dtype
        Floating dtype of the filters.

    Returns
    -------
        DCT basis filters of shape ``(prod(kernel_size), *kernel_size)``.
    """
    if len(kernel_size) == 0:
        raise ValueError('kernel_size must contain at least one dimension.')

    if any(n <= 0 for n in kernel_size):
        raise ValueError('All dimensions in kernel_size must be positive.')

    bases = [dct_matrix(n, dtype=dtype) for n in kernel_size]

    filters = reduce(torch.kron, bases)
    return filters.reshape(math.prod(kernel_size), *kernel_size)


def fft_filters(
    kernel_size: tuple[int, ...],
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    r"""Create the complex Fourier modes of an N-dimensional patch.

    For spatial shape :math:`(s_1,\ldots,s_d)` and :math:`N=\prod_j s_j`,
    the filter indexed by :math:`(k_1,\ldots,k_d)` is

    .. math::

        h_{k_1,\ldots,k_d}[m_1,\ldots,m_d]
            = N^{-1/2}\exp\!\left(2\pi\mathrm{i}\sum_{j=1}^d
                                   \frac{k_jm_j}{s_j}\right).

    These are positive-exponent (inverse-DFT) modes. Their Hermitian inner
    products with a patch give its orthonormal forward-DFT coefficients.
    Frequency indices use FFT order, with the last axis varying fastest;
    the first filter is constant. Spatial indices start at zero, without shifting.

    Parameters
    ----------
    kernel_size
        Spatial shape of the Fourier filters, e.g. ``(7, 7)``
        or ``(5, 7, 9)``.
    dtype
        Complex data type of the filters.

    Returns
    -------
        Fourier basis filters of shape ``(prod(kernel_size), *kernel_size)``.
    """
    n_filters = math.prod(kernel_size)

    basis = torch.eye(
        n_filters,
        dtype=dtype,
    ).reshape(n_filters, *kernel_size)

    filters = torch.fft.ifftn(
        basis,
        dim=tuple(range(1, len(kernel_size) + 1)),
        norm='ortho',
    )

    return filters


def hann_window(coordinate: torch.Tensor, width: float) -> torch.Tensor:
    """Evaluate a continuous Hann window with unit gain at the origin.

    Parameters
    ----------
    coordinate
        Real floating-point coordinates, centered at zero.
    width
        Finite positive full support width in coordinate units. Weights are
        ``(1 + cos(2*pi*coordinate/width)) / 2`` inside the support and zero outside.

    Returns
    -------
        Real weights with the coordinate shape, dtype and device, without energy normalization.
    """
    if not math.isfinite(width) or width <= 0:
        raise ValueError('width must be finite and positive.')
    normalized = coordinate / (width / 2)
    return torch.where(normalized.abs() < 1, (1 + torch.cos(torch.pi * normalized.clamp(-1, 1))) / 2, 0)


def gaussian_window(coordinate: torch.Tensor, width: float) -> torch.Tensor:
    """Evaluate a continuous Gaussian window with unit gain at the origin.

    Parameters
    ----------
    coordinate
        Real floating-point coordinates, centered at zero.
    width
        Finite positive full width at half maximum in coordinate units.
        Weights are ``exp(-4*log(2)*(coordinate/width)**2)`` without a hard cutoff.

    Returns
    -------
        Real weights with the coordinate shape, dtype and device, without energy normalization.
    """
    if not math.isfinite(width) or width <= 0:
        raise ValueError('width must be finite and positive.')
    return torch.exp(-math.log(2) * (coordinate / (width / 2)).square())


def radial_hann_window(coordinates: Sequence[torch.Tensor], widths: Sequence[float]) -> torch.Tensor:
    """Evaluate a Hann window on the radius of scaled continuous coordinates.

    Parameters
    ----------
    coordinates
        Nonempty sequence of broadcastable real floating-point coordinate tensors
        with the same dtype and device, one per spatial direction.
    widths
        Finite positive full support widths in the corresponding coordinate units.
        Must have the same length as coordinates; unequal widths give ellipsoidal support.

    Returns
    -------
        Real weights with the broadcast coordinate shape, unit gain and zero
        coordinate gradient at the origin, without energy normalization.
    """
    if not coordinates or len(coordinates) != len(widths):
        raise ValueError('Must provide nonempty coordinates and matching length widths.')
    if any(not math.isfinite(width) or width <= 0 for width in widths):
        raise ValueError('width must be finite and positive in every direction.')
    radius_squared = coordinates[0].new_zeros(())
    for coordinate, width in zip(coordinates, widths, strict=True):
        radius_squared = radius_squared + (coordinate / (width / 2)).square()
    at_dc = radius_squared == 0
    radius = torch.where(at_dc, 1, radius_squared).sqrt()
    return torch.where(at_dc, 1, hann_window(radius, 2))
