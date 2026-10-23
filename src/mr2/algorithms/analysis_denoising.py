"""Analysis dictionary denoising using PDHG."""

from typing import TypeVar

import torch

from mr2.algorithms._dictionary_denoising import _initial, _prepare, _validate_controls, _weight
from mr2.algorithms.optimizers.pdhg import pdhg
from mr2.data import IData
from mr2.operators import IdentityOp, LinearOperator, LinearOperatorMatrix, ProximableFunctionalSeparableSum
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
    For convolutional denoising, pass a `~mr2.operators.ConvAnalysisDictionaryOp`;
    its padding mode determines the boundary behavior. Only odd spatial kernels are supported.
    This analysis model is generally different from synthesis sparse coding.

    The image determines the execution device. Its dtype is promoted with the operator's
    registered floating point/complex tensors. Operators are reused directly when compatible,
    otherwise a converted copy is used; the supplied operator is not modified.

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

    Notes
    -----
    Adapted from the convolutional denoising recipe in PTB-MR/mrpro PR #1018
    (https://github.com/PTB-MR/mrpro/pull/1018), using mr2's existing operators and solvers.
    """
    _validate_controls(max_iterations, tolerance)
    image, (analysis_operator,) = _prepare(idata if isinstance(idata, torch.Tensor) else idata.data, analysis_operator)
    coefficients = analysis_operator(image)[0]
    if analysis_operator.H(coefficients)[0].shape != image.shape:
        raise ValueError('analysis adjoint must return exactly the image shape')
    weight = _weight(regularization_weight, coefficients)
    (result,) = pdhg(
        f=ProximableFunctionalSeparableSum(0.5 * L2NormSquared(target=image), L1Norm(weight=weight)),
        g=None,
        operator=LinearOperatorMatrix(((IdentityOp(),), (analysis_operator,))),
        initial_values=(_initial(initial_image, image),),
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return result if isinstance(idata, torch.Tensor) else IData(result, idata.header)
