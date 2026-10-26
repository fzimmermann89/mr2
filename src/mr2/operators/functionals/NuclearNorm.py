"""Nuclear norm."""

import torch

from mr2.operators.Functional import ElementaryProximableFunctional, throw_if_negative_or_complex


class NuclearNorm(ElementaryProximableFunctional):
    r"""Weighted nuclear norm on the last two input axes.

    For each matrix in an input of shape ``(..., m, n)``, compute
    :math:`f(X) = |w| \sum_i s_i(X-b) / N`, where :math:`s_i` are singular
    values, :math:`b` is ``target``, and :math:`N=mn` if ``divide_by_n=True``
    (otherwise :math:`N=1`). Batch axes are retained without reduction.
    Use `~mr2.operators.RearrangeOp` to arrange other matrix axes externally.

    Weights and proximal steps must broadcast to ``(..., 1, 1)`` without
    expanding the input batch shape. For input ``(B, m, n)``, use ``(B, 1, 1)``
    for per-matrix parameters. Entrywise weights or steps are unsupported.
    Complex and negative weights use their magnitude, as in `L1Norm`.
    """

    def __init__(
        self,
        target: torch.Tensor | complex | None = None,
        weight: torch.Tensor | complex = 1.0,
        divide_by_n: bool = False,
        keepdim: bool = False,
    ) -> None:
        """Initialize the nuclear norm.

        Parameters
        ----------
        target
            Translation, broadcastable to the input without expanding its shape.
        weight
            Scalar or per-matrix weight. Its magnitude multiplies the norm.
        divide_by_n
            Divide by the number of matrix entries, ``m*n``, in all methods.
        keepdim
            Retain both matrix axes with size one in the forward result.
        """
        super().__init__(target, weight, dim=(-2, -1), divide_by_n=divide_by_n, keepdim=keepdim)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Return one weighted sum of singular values per matrix.

        Parameters
        ----------
        x
            Real or complex input of shape ``(..., m, n)``.

        Returns
        -------
            Batch of norms, with two trailing singleton axes if ``keepdim=True``.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward of NuclearNorm.

        .. note::
            Prefer calling the instance of the NuclearNorm as ``operator(x)`` over directly calling this method.
            See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if x.ndim < 2:
            raise ValueError('NuclearNorm requires at least two input dimensions')
        weight = torch.broadcast_to(self.weight.abs().to(device=x.device), (*x.shape[:-2], 1, 1))
        weight = self._divide_by_n(weight, x.shape)
        target = torch.broadcast_to(self.target.to(device=x.device), x.shape)
        value = torch.linalg.svdvals(x - target).sum(-1, keepdim=True).unsqueeze(-1) * weight
        return (value if self.keepdim else value[..., 0, 0],)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """Apply singular-value soft thresholding about ``target``.

        Parameters
        ----------
        x
            Real or complex input of shape ``(..., m, n)``.
        sigma
            Nonnegative real scalar or per-matrix step, with trailing singleton
            matrix axes. Zero steps return the corresponding input matrices.

        Returns
        -------
            Proximal mapping with the same shape as the input.
        """
        throw_if_negative_or_complex(sigma)
        if x.ndim < 2:
            raise ValueError('NuclearNorm requires at least two input dimensions')
        shape = (*x.shape[:-2], 1, 1)
        weight = torch.broadcast_to(self.weight.abs().to(device=x.device), shape)
        sigma = sigma.to(device=x.device) if isinstance(sigma, torch.Tensor) else x.real.new_tensor(sigma)
        step = torch.broadcast_to(sigma, shape)
        threshold = self._divide_by_n(weight * step, x.shape)
        target = torch.broadcast_to(self.target.to(device=x.device), x.shape)
        diff = x - target
        u, s, vh = torch.linalg.svd(diff.to(torch.result_type(diff, threshold)), full_matrices=False)
        result = (u * torch.relu(s - threshold[..., 0]).unsqueeze(-2)) @ vh + target
        return (torch.where(threshold == 0, x, result),)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        r"""Project ``x - sigma * target`` onto the spectral-norm ball.

        The ball radius is :math:`|w|/N`. Zero steps return the input,
        including outside the conjugate domain, as in `L21Norm`.

        Parameters
        ----------
        x
            Real or complex input of shape ``(..., m, n)``.
        sigma
            Nonnegative real scalar or per-matrix step, with trailing singleton
            matrix axes.

        Returns
        -------
            Proximal mapping of the convex conjugate, with the input shape.
        """
        throw_if_negative_or_complex(sigma)
        if x.ndim < 2:
            raise ValueError('NuclearNorm requires at least two input dimensions')
        shape = (*x.shape[:-2], 1, 1)
        weight = torch.broadcast_to(self.weight.abs().to(device=x.device), shape)
        radius = self._divide_by_n(weight, x.shape)
        sigma = sigma.to(device=x.device) if isinstance(sigma, torch.Tensor) else x.real.new_tensor(sigma)
        step = torch.broadcast_to(sigma, shape)
        target = torch.broadcast_to(self.target.to(device=x.device), x.shape)
        diff = x - step * target
        u, s, vh = torch.linalg.svd(diff.to(torch.result_type(diff, radius)), full_matrices=False)
        projected = (u * torch.clamp_max(s, radius[..., 0]).unsqueeze(-2)) @ vh
        return (torch.where(step == 0, x, projected),)
