"""Wavelet denoising using soft thresholding."""

from __future__ import annotations

from typing import overload

import torch
from pywt import Wavelet

from mr2.data.IData import IData
from mr2.operators.functionals import L1NormViewAsReal
from mr2.operators.WaveletOp import WaveletOp, WaveletType
from mr2.utils import normalize_index


@overload
def wavelet_denoising(
    idata: IData,
    regularization_dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    regularization_weight: float | torch.Tensor,
    *,
    wavelet_name: WaveletType = 'db4',
    level: int | None = None,
) -> IData: ...


@overload
def wavelet_denoising(
    idata: torch.Tensor,
    regularization_dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    regularization_weight: float | torch.Tensor,
    *,
    wavelet_name: WaveletType = 'db4',
    level: int | None = None,
) -> torch.Tensor: ...


def wavelet_denoising(
    idata: IData | torch.Tensor,
    regularization_dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    regularization_weight: float | torch.Tensor,
    *,
    wavelet_name: WaveletType = 'db4',
    level: int | None = None,
) -> IData | torch.Tensor:
    r"""Soft threshold wavelet coefficients and reconstruct the image.

    All coefficients, including the low-frequency approximation band, are thresholded.
    Complex coefficients are shrunk separately in their real and imaginary parts using
    `~mr2.operators.functionals.L1NormViewAsReal`. This penalizes
    :math:`|\mathrm{Re}(c)| + |\mathrm{Im}(c)|`, rather than complex magnitude, and is not phase invariant.

    This is the exact proximal map of the weighted wavelet L1 penalty only when the
    transform is orthonormal/unitary (for example, Haar with axis lengths divisible by
    :math:`2^{\mathrm{level}}`). `~mr2.operators.WaveletOp` uses zero padding and usually
    produces redundant boundary coefficients for longer filters. Thresholding followed
    by reconstruction is then not generally the proximal map of the analysis penalty.

    Parameters
    ----------
    idata
        Input image with float32, float64, complex64 or complex128 dtype.
    regularization_dim
        One, two or three distinct axes along which to transform. Negative axes are supported.
        Selected axis lengths must be positive and even, as required by `~mr2.operators.WaveletOp`.
    regularization_weight
        Finite, nonnegative real coefficient threshold, not a noise standard deviation.
        A tensor must broadcast to exactly the packed coefficient shape produced by
        `~mr2.operators.WaveletOp`; there is no separate scale axis. The threshold is moved
        to the image device and converted to its real dtype. It is shared by real and imaginary parts.
    wavelet_name
        Orthogonal discrete wavelet supported by `~mr2.operators.WaveletOp`, default ``'db4'``.
        Biorthogonal wavelets are not supported by this convenience function.
    level
        Nonnegative number of levels. If `None`, use the maximum level without boundary effects
        according to PyWavelets. A level of zero thresholds the image directly.

    Returns
    -------
        Denoised tensor or `~mr2.data.IData`, matching the input type, dtype and device.
        Tensor outputs preserve the input shape. For IData input, the header is preserved;
        repeated data values may be stored as singleton dimensions, as usual for IData.
    """
    img_tensor = idata if isinstance(idata, torch.Tensor) else idata.data
    if img_tensor.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise ValueError('Input must have float32, float64, complex64 or complex128 dtype')
    if not 1 <= len(regularization_dim) <= 3:
        raise ValueError('Only 1D, 2D and 3D wavelet transforms are supported')
    if any(not isinstance(d, int) or isinstance(d, bool) for d in regularization_dim):
        raise ValueError('regularization_dim must contain integer axes')
    dim = tuple(normalize_index(img_tensor.ndim, d) for d in regularization_dim)
    if len(set(dim)) != len(dim):
        raise ValueError('Repeated values are not allowed in regularization_dim')
    if any(img_tensor.shape[d] == 0 for d in dim):
        raise ValueError('Selected axes must have positive lengths')
    if level is not None and (not isinstance(level, int) or isinstance(level, bool) or level < 0):
        raise ValueError('level must be a nonnegative integer or None')
    if not Wavelet(wavelet_name).orthogonal:
        raise ValueError('wavelet_denoising only supports orthogonal wavelets')

    threshold = torch.as_tensor(
        regularization_weight,
        device=img_tensor.device,
        dtype=img_tensor.real.dtype if isinstance(regularization_weight, float) else None,
    )
    if threshold.is_complex() or not torch.isfinite(threshold).all() or (threshold < 0).any():
        raise ValueError('regularization_weight must be finite, real and nonnegative')
    threshold = threshold.to(img_tensor.real.dtype)
    if not torch.isfinite(threshold).all():
        raise ValueError('regularization_weight must be finite in the image dtype')

    wavelet_op = WaveletOp(
        domain_shape=tuple(img_tensor.shape[d] for d in dim),
        dim=regularization_dim,
        wavelet_name=wavelet_name,
        level=level,
    )
    (coefficients,) = wavelet_op(img_tensor)
    if torch.broadcast_shapes(threshold.shape, coefficients.shape) != coefficients.shape:
        raise ValueError('regularization_weight must broadcast to exactly the packed coefficient shape')
    (thresholded_coefficients,) = L1NormViewAsReal().prox(coefficients, threshold)
    (img_tensor,) = wavelet_op.H(thresholded_coefficients)
    return img_tensor if isinstance(idata, torch.Tensor) else IData(img_tensor, idata.header)
