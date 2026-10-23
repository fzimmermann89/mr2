"""L2,1 norm."""

import math
from collections.abc import Sequence

import torch

from mr2.operators.Functional import ElementaryProximableFunctional, throw_if_negative_or_complex
from mr2.utils.reshape import normalize_indices


class L21Norm(ElementaryProximableFunctional):
    r"""Sum or mean of Euclidean group norms of ``weight * (x - target)``.

    This implements :math:`f(x) = \frac{1}{n}\sum_g |w_g|\|x_g-b_g\|_2`,
    where :math:`b` is `target`, :math:`w_g` is the scalar `weight` for group g,
    and n is the number of outer groups for a mean, or one for a sum.

    The inner L2 norm groups components along `group_dim`. The outer sum (or mean)
    reduces the group norms along `dim`; remaining axes are batch axes. Weights
    act through their magnitude, just as in `L1Norm`, without a square root.
    Forward evaluation supports broadcasting that expands the input. Proximal
    mappings require target and weight to broadcast without expanding its shape.
    """

    def __init__(
        self,
        target: torch.Tensor | complex | None = None,
        weight: torch.Tensor | complex = 1.0,
        dim: int | Sequence[int] | None = None,
        divide_by_n: bool = False,
        keepdim: bool = False,
        *,
        group_dim: int | Sequence[int],
    ) -> None:
        """Initialize the group norm.

        Parameters
        ----------
        target
            Target tensor, broadcast with the input and weight.
        weight
            Scalar per group, possibly negative or complex. Must have singleton
            sizes along group axes after right alignment to the broadcast shape.
        dim
            Outer reduction axes in the broadcast input rank, disjoint from
            `group_dim`. None reduces all non-group axes; () performs no outer
            reduction. Negative axes are supported; duplicates and aliases fail.
        divide_by_n
            Average over outer groups instead of summing. Group components are
            never counted. Averaging over zero outer groups raises ValueError.
        keepdim
            Retain both group and outer reduction axes with size one. Otherwise
            remove both sets of axes. Proximal mappings always retain input axes.
        group_dim
            Required nonempty axis or sequence of axes for each Euclidean group.
            All axes may be grouped, yielding a single Euclidean norm.
        """
        super().__init__(target=target, weight=weight, dim=dim, divide_by_n=divide_by_n, keepdim=keepdim)
        self.group_dim = (group_dim,) if isinstance(group_dim, int) else tuple(group_dim)
        if not self.group_dim:
            raise ValueError('group_dim must be nonempty')

    @staticmethod
    def _check_group_scalar(value: torch.Tensor, shape: torch.Size, group: tuple[int, ...], name: str) -> None:
        """Require a broadcastable scalar per group for the radial closed form."""
        if torch.broadcast_shapes(value.shape, shape) != shape:
            raise ValueError(f'{name} must broadcast to the input shape')
        aligned_shape = (1,) * (len(shape) - value.ndim) + value.shape
        if any(aligned_shape[d] != 1 for d in group):
            raise ValueError(f'{name} must have singleton sizes along group_dim')

    def _prepare(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], torch.Tensor]:
        """Resolve broadcast shape, axes and normalized group weights."""
        diff, _ = torch.broadcast_tensors(x - self.target, self.weight)
        group = normalize_indices(diff.ndim, self.group_dim)
        outer = (
            tuple(d for d in range(diff.ndim) if d not in group)
            if self.dim is None
            else normalize_indices(diff.ndim, self.dim)
        )
        if set(group) & set(outer):
            raise ValueError('dim and group_dim must be disjoint')
        self._check_group_scalar(self.weight, diff.shape, group, 'weight')
        count = math.prod(diff.shape[d] for d in outer) if self.divide_by_n else 1
        if count == 0:
            raise ValueError('Cannot average over zero outer groups')
        return diff, group, outer, self.weight.abs() / count

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Compute the sum or mean of weighted group norms."""
        diff, group, outer, _ = self._prepare(x)
        value = torch.linalg.vector_norm(self.weight * diff, dim=group, keepdim=True)
        if outer:
            value = value.mean(dim=outer, keepdim=True) if self.divide_by_n else value.sum(dim=outer, keepdim=True)
        if not self.keepdim:
            for d in sorted(group + outer, reverse=True):
                value = value.squeeze(d)
        return (value,)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """Shrink each group toward target by ``sigma * abs(weight) / n``.

        Here n counts outer groups for a mean, and is one for a sum. Sigma must
        be real, nonnegative and broadcastable with singleton group axes.

        Parameters
        ----------
        x
            Input tensor. Target and weight must broadcast without expanding its shape.
        sigma
            Nonnegative scaling factor, scalar per group.

        Returns
        -------
            Proximal mapping applied to the input tensor, with the same shape.

        Raises
        ------
        ValueError
            If target or weight expands the input shape.
        """
        throw_if_negative_or_complex(sigma)
        diff, group, _, radius = self._prepare(x)
        if diff.shape != x.shape:
            raise ValueError('Proximal mappings cannot expand the input shape')
        sigma = sigma.to(device=x.device) if isinstance(sigma, torch.Tensor) else x.real.new_tensor(sigma)
        self._check_group_scalar(sigma, diff.shape, group, 'sigma')
        norm = torch.linalg.vector_norm(diff, dim=group, keepdim=True)
        threshold = sigma * radius
        denominator = torch.where(norm > 0, norm, 1)
        factor = torch.where(norm > threshold, 1 - threshold / denominator, 0)
        # Zero steps and zero weights are the identity, including at zero groups.
        factor = torch.where(threshold == 0, 1, factor)
        return (self.target + diff * factor,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """Project ``x - sigma * target`` onto balls of radius ``abs(weight) / n``.

        Sigma obeys the same restrictions as in `prox`. Zero steps return the
        input, including outside the conjugate domain. Zero weights project to
        zero for positive steps.

        Parameters
        ----------
        x
            Input tensor. Target and weight must broadcast without expanding its shape.
        sigma
            Nonnegative scaling factor, scalar per group.

        Returns
        -------
            Proximal mapping of the convex conjugate, with the input shape.

        Raises
        ------
        ValueError
            If target or weight expands the input shape.
        """
        throw_if_negative_or_complex(sigma)
        diff, group, _, radius = self._prepare(x)
        if diff.shape != x.shape:
            raise ValueError('Proximal mappings cannot expand the input shape')
        sigma = sigma.to(device=x.device) if isinstance(sigma, torch.Tensor) else x.real.new_tensor(sigma)
        self._check_group_scalar(sigma, diff.shape, group, 'sigma')
        shifted = x - sigma * self.target
        norm = torch.linalg.vector_norm(shifted, dim=group, keepdim=True)
        denominator = torch.maximum(norm, radius)
        factor = radius / torch.where(denominator > 0, denominator, 1)
        return (torch.where(sigma == 0, x, shifted * factor),)
