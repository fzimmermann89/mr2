"""Differentiable Rician negative log likelihood."""

import math
from collections.abc import Sequence

import torch

from mr2.operators.Functional import throw_if_negative_or_complex
from mr2.operators.Operator import Operator


def _rician_nll(amplitude: torch.Tensor, target: torch.Tensor, noise_std: torch.Tensor) -> torch.Tensor:
    """Evaluate the reduced likelihood without Bessel derivative cancellation."""
    z = (amplitude / noise_std) * (target / noise_std)
    # At these switches, asymptotic truncation errors are < 5e-8 / 3e-17.
    switch, terms = (10, 8) if z.dtype == torch.float32 else (32, 14)
    large = z >= switch
    # I0 is smooth at zero; i0e's absolute-value cusp gives the wrong gradient
    # when combined with the residual form there. Mask before evaluating I0.
    u = torch.where(large, 0, amplitude) / noise_std
    v = torch.where(large, 0, target) / noise_std
    small_value = (u.square() + v.square()) / 2 - torch.special.i0(u * v).log()

    # DLMF 10.40.1: i0e(z) ~ P(1/z)/sqrt(2πz). Differentiating this
    # correction preserves the small Bessel deficit without subtracting ratios.
    # Safe inactive inputs keep higher derivatives finite at zero.
    a = torch.where(large, amplitude, noise_std)
    y = torch.where(large, target, noise_std)
    inverse_z = (noise_std / a) * (noise_std / y)
    correction = torch.zeros_like(inverse_z)
    for k in range(terms, 0, -1):
        correction = (1 + correction) * (inverse_z * ((2 * k - 1) ** 2 / (8 * k)))
    log_z = a.log() + y.log() - 2 * noise_std.log()
    residual = torch.where(large, amplitude - target, 0) / noise_std
    large_value = residual.square() / 2 + (math.log(2 * math.pi) + log_z) / 2 - correction.log1p()
    return torch.where(large, large_value, small_value) + 2 * noise_std.log()


class RicianNegLogLikelihood(Operator[torch.Tensor, tuple[torch.Tensor]]):
    r"""Reduced Rician negative log likelihood for nonnegative amplitudes.

    For predicted amplitude :math:`a`, observed magnitude :math:`y` and noise
    standard deviation :math:`s`, with :math:`z=ay/s^2`, the per-observation objective is

    .. math::
        \ell(a,y,s) = \frac{a^2+y^2}{2s^2} - \log I_0(ay/s^2) + 2\log s.

    This omits the data-only term :math:`-\log y` from the full negative log
    density, allowing zero observations. Observation derivatives refer to this
    reduced objective: :math:`\partial_y\ell=(y-a I_1(z)/I_0(z))/s^2`,
    including its zero limit at :math:`y=0`. Amplitude derivatives likewise
    vanish at :math:`a=0`. The noise normalization is retained for fitting
    learnable noise. This likelihood is generally nonconvex.

    Inputs must be finite, real, nonnegative amplitudes; ``noise_std`` must be
    strictly positive. Float32 and float64 are supported. The caller must keep
    amplitudes in their physical domain. Values and derivatives are limited by
    the representable range of the chosen dtype. Ordinary PyTorch operations
    support double backward, Hessian-vector products and ``torch.func``.
    """

    def __init__(
        self,
        target: torch.Tensor | float,
        noise_std: torch.Tensor | float,
        weight: torch.Tensor | float = 1.0,
        dim: int | Sequence[int] | None = None,
        divide_by_n: bool = False,
        keepdim: bool = False,
    ) -> None:
        """Initialize the likelihood.

        Parameters
        ----------
        target
            Nonnegative observed magnitude, broadcastable with the input.
        noise_std
            Standard deviation of EACH real/imaginary Gaussian noise component
            (complex noise variance is twice its square). Strictly positive,
            broadcastable with the input; may require gradients. Pass a float64
            tensor for float64 noise precision; Python floats use the default dtype.
        weight
            Nonnegative likelihood weights, broadcastable with the observations.
            Multiplies the per-observation loss, rather than the residual.
        dim
            Dimensions to reduce; ``None`` reduces all dimensions.
        divide_by_n
            Use the mean instead of the sum over the broadcast loss dimensions.
            The divisor counts elements, including zero weights.
        keepdim
            Retain reduced dimensions with size one.
        """
        super().__init__()
        self.target = torch.as_tensor(target)
        self.weight = torch.as_tensor(weight)
        self.dim = (dim,) if isinstance(dim, int) else tuple(dim) if dim is not None else None
        self.divide_by_n = divide_by_n
        self.keepdim = keepdim
        self.noise_std = torch.as_tensor(noise_std)
        throw_if_negative_or_complex(self.target, 'target must be real and nonnegative')
        throw_if_negative_or_complex(self.weight, 'weight must be real and nonnegative')
        if self.noise_std.is_complex() or not (self.noise_std > 0).all():
            raise ValueError('noise_std must be real and strictly positive')

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Return the weighted reduced likelihood for physical amplitude ``x``."""
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError('RicianNegLogLikelihood supports float32 and float64 amplitudes')
        value = self.weight * _rician_nll(x, self.target, self.noise_std)
        reduction = torch.mean if self.divide_by_n else torch.sum
        return (reduction(value, dim=self.dim, keepdim=self.keepdim),)
