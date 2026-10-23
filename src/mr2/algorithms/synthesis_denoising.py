"""Synthesis dictionary denoising using PGD, with optional low-pass residual splitting."""

from typing import TypeVar

import torch

from mr2.algorithms._dictionary_denoising import _image, _initial, _validate_controls, _weight
from mr2.algorithms.optimizers.cg import cg
from mr2.algorithms.optimizers.pgd import PGDStatus, pgd
from mr2.data import IData
from mr2.operators import IdentityOp, LinearOperator
from mr2.operators.functionals import L1Norm, L2NormSquared

Image = TypeVar('Image', torch.Tensor, IData)


def synthesis_denoising(
    idata: Image,
    synthesis_operator: LinearOperator,
    regularization_weight: float | torch.Tensor,
    *,
    initial_codes: torch.Tensor | None = None,
    max_iterations: int = 100,
    tolerance: float = 0,
    low_pass_operator: LinearOperator | None = None,
    max_iterations_low_pass: int = 32,
    tolerance_low_pass: float = 1e-6,
) -> Image:
    r"""Denoise by synthesis sparse coding with backtracked accelerated PGD.

    Without a low-pass operator, solves
    :math:`c^* = \arg\min_c \frac12\|Dc-y\|_2^2 + \sum_i w_i |c_i|`
    and returns :math:`Dc^*`. Complex coefficients use magnitude sparsity.
    For convolutional denoising, pass a `~mr2.operators.ConvSynthesisDictionaryOp`;
    its padding mode determines boundary behavior.

    With a low-pass operator :math:`B`, first computes the baseline
    :math:`b = \arg\min_b \frac12\|b-y\|_2^2 + \frac12\|Bb\|_2^2`
    using CG on :math:`(I+B^HB)b=y`. It then codes the residual :math:`y-b`
    using the above sparsity objective and returns :math:`b+Dc^*`.
    For example, use ``low_pass_operator=beta**0.5 * FiniteDifferenceOp(dim=(-2, -1))``.
    This is an explicit two-stage model, generally different from analysis denoising.

    Operators are used as supplied and must support the image's device and dtype.
    No operators or images are copied or converted; dtype promotion follows the underlying operations.
    PGD starts at 0.9 divided by the estimated squared dictionary norm, with backtracking
    as a safeguard. A zero norm uses a unit step.

    Parameters
    ----------
    idata
        Noisy tensor or IData. Output retains its shape and header.
    synthesis_operator
        Linear map from coefficients to images, with a true Hermitian adjoint.
        The adjoint applied to the image determines the coefficient shape and dtype.
    regularization_weight
        Nonnegative finite scalar or map broadcasting to, without expanding, the coefficient shape.
    initial_codes
        Initial coefficients of exactly the expected shape.
        Defaults to zeros with the adjoint output's dtype and device.
    max_iterations
        Maximum number of accelerated PGD iterations (positive).
    tolerance
        Nonnegative relative coefficient-change tolerance. Zero disables early stopping.
    low_pass_operator
        Optional image-to-penalty map defining the baseline, including its strength and boundaries.
    max_iterations_low_pass
        Maximum CG iterations for the optional baseline (positive).
    tolerance_low_pass
        Positive absolute residual-norm tolerance for CG; choose it according to the image scale.

    Returns
    -------
        Denoised image of the same container type as the input.

    Notes
    -----
    Adapted from PTB-MR/mrpro PR #1018 (https://github.com/PTB-MR/mrpro/pull/1018).
    mr2's existing CG and PGD implement the baseline and sparse coding stages.
    """
    _validate_controls(max_iterations, tolerance)
    _validate_controls(max_iterations_low_pass, tolerance_low_pass)
    if tolerance_low_pass == 0:
        raise ValueError('tolerance_low_pass must be positive')
    image = _image(idata if isinstance(idata, torch.Tensor) else idata.data)
    baseline = torch.zeros_like(image)
    if low_pass_operator is not None:
        (baseline,) = cg(
            IdentityOp() + low_pass_operator.gram,
            image,
            initial_value=image,
            max_iterations=max_iterations_low_pass,
            tolerance=tolerance_low_pass,
        )
    reference = synthesis_operator.H(image)[0]
    weight = _weight(regularization_weight, reference)
    codes = _initial(initial_codes, torch.zeros_like(reference))
    # A random probe avoids a zero starting vector, including for zero images.
    norm = float(synthesis_operator.operator_norm(torch.randn_like(reference), dim=None).detach())
    stepsize = 0.9 / norm**2 if norm > 0 else 1.0
    previous = codes

    def stop(status: PGDStatus) -> bool:
        nonlocal previous
        current = status['solution'][0]
        change = torch.linalg.vector_norm(current - previous)
        scale = torch.linalg.vector_norm(current)
        previous = current
        return bool(change > tolerance * scale)

    (codes,) = pgd(
        f=0.5 * (L2NormSquared(target=image - baseline) @ synthesis_operator),
        g=L1Norm(weight=weight),
        initial_value=codes,
        max_iterations=max_iterations,
        stepsize=stepsize,
        backtrack_factor=0.5,
        convergent_iterates_variant=True,
        callback=stop if tolerance else None,
    )
    result = synthesis_operator(codes)[0] + baseline
    return result if isinstance(idata, torch.Tensor) else IData(result, idata.header)
