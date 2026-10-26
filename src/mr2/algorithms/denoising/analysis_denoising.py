"""Analysis dictionary denoising using PDHG."""

import math
from typing import TypeVar

import torch

from mr2.algorithms.optimizers.pdhg import pdhg
from mr2.data import IData
from mr2.operators import IdentityOp, LinearOperator
from mr2.operators.functionals import L1Norm, L2NormSquared

Image = TypeVar('Image', torch.Tensor, IData)


def analysis_denoising(
    idata: Image,
    analysis_operator: LinearOperator,
    regularization_weight: float | torch.Tensor,
    *,
    initial_image: torch.Tensor | None = None,
    max_iterations: int = 100,
    tolerance: float = 0,
) -> Image:
    r"""Denoise by minimizing an analysis sparsity objective with PDHG.

    Solves :math:`\min_x \frac12\|x-y\|_2^2 + \sum_i w_i |(Ax)_i|`.
    Complex coefficients are penalized by magnitude, preserving phase under shrinkage.
    For convolutional denoising, compose a `~mr2.operators.ConvolutionOp` with
    ``RearrangeOp('... -> 1 ...')`` to add a singleton input channel.
    The convolution's padding mode determines the boundary behavior.
    This analysis model is generally different from synthesis sparse coding.

    Operators are used as supplied and must support the image's device and dtype.
    The initial image is promoted to the image and coefficient dtypes when necessary.

    Parameters
    ----------
    idata
        Noisy image, as a tensor or IData. The output retains its shape and header.
    analysis_operator
        Linear map from images to coefficients, with a true Hermitian adjoint.
    regularization_weight
        Nonnegative finite scalar or map broadcasting to, without expanding, the coefficient shape.
    initial_image
        Initial image of exactly the input shape. Defaults to the noisy image.
    max_iterations
        Maximum number of PDHG iterations (positive).
    tolerance
        Nonnegative relative primal-change tolerance. Zero disables early stopping.

    Returns
    -------
        Denoised image of the same container type as the input.
    """
    if max_iterations < 1:
        raise ValueError('max_iterations must be positive')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    image = idata if isinstance(idata, torch.Tensor) else idata.data
    if not (image.is_floating_point() or image.is_complex()) or image.numel() == 0:
        raise ValueError('image must be a nonempty floating point or complex tensor')
    coefficients = analysis_operator(image)[0]
    weight = torch.as_tensor(
        regularization_weight,
        device=coefficients.device,
        dtype=torch.float64 if isinstance(regularization_weight, int | float) else None,
    )
    if weight.is_complex() or not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('regularization_weight must be real, finite and nonnegative')
    if torch.broadcast_shapes(weight.shape, coefficients.shape) != coefficients.shape:
        raise ValueError('regularization_weight must not expand the coefficient shape')
    weight = weight.to(dtype=coefficients.real.dtype)
    if not torch.isfinite(weight).all():
        raise ValueError('regularization_weight must be finite in the coefficient dtype')
    dtype = torch.promote_types(image.dtype, coefficients.dtype)
    if initial_image is None:
        initial_image = image.to(dtype=dtype)
    else:
        if initial_image.shape != image.shape:
            raise ValueError('initial value must have exactly the expected shape')
        if initial_image.is_complex() and not dtype.is_complex:
            raise ValueError('initial value must not introduce a complex dtype')
        initial_image = initial_image.to(device=image.device, dtype=dtype)
    (result,) = pdhg(
        f=0.5 * L2NormSquared(target=image) | L1Norm(weight=weight),
        g=None,
        operator=IdentityOp() % analysis_operator,
        initial_values=(initial_image,),
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return result if isinstance(idata, torch.Tensor) else IData(result, idata.header)
