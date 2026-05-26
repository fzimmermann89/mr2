"""Sampling from von Mises-Fisher distribution."""

# based on: https://github.com/jasonlaska/spherecluster/blob/701b0b1909088a56e353b363b2672580d4fe9d93/spherecluster/util.py
# http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python
# https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
# http://www.stat.pitt.edu/sungkyu/software/randvonMisesFisher3.pdf

from math import log, sqrt

import torch

from mr2.utils.RandomGenerator import RandomGenerator


def sample_vmf(mu: torch.Tensor, kappa: float, n_samples: int, rng: RandomGenerator | None = None) -> torch.Tensor:
    """
    Draws samples from the von Mises–Fisher distribution on the unit hypersphere centered at `mu`.
    
    Parameters:
        mu (torch.Tensor): Mean direction on the unit hypersphere. Shape: (..., dim) or (dim,).
        kappa (float): Concentration parameter; larger values concentrate samples closer to `mu`.
        n_samples (int): Number of samples to draw per leading entry of `mu`.
        rng (RandomGenerator | None): Optional random generator. If `None`, a new generator is created on `mu`'s device.
    
    Returns:
        torch.Tensor: Unit-length samples with shape (n_samples, ..., dim), where `...` matches `mu`'s leading dimensions; for a 1-D `mu` the shape is (n_samples, dim).
    """
    mu_ = mu.unsqueeze(0) if mu.dim() == 1 else mu
    total_samples = n_samples * mu_[..., 0].numel()
    mu_ = mu_.expand((n_samples, *mu_.shape))
    dim = mu_.shape[-1]
    rng = RandomGenerator(device=mu_.device) if rng is None else rng

    b = (dim - 1) / (sqrt(4.0 * kappa**2 + (dim - 1) ** 2) + 2 * kappa)
    x = (1.0 - b) / (1.0 + b)
    c = kappa * x + (dim - 1) * log(1 - x**2)

    ws: list[torch.Tensor] = []

    while sum(len(w) for w in ws) < total_samples:
        # rejection sampling
        z = rng.beta_tensor((total_samples,), (dim - 1) / 2.0, (dim - 1) / 2.0, dtype=mu_.dtype, device=mu_.device)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = rng.rand_tensor((total_samples,), mu_.dtype, device=mu_.device)
        accepted = kappa * w + (dim - 1) * torch.log(1.0 - x * w) - c >= torch.log(u)
        ws.append(w[accepted])
    weights = torch.cat(ws)[:total_samples].reshape(mu_.shape[:-1])

    v = rng.randn_tensor(mu_.shape, mu_.dtype, device=mu_.device)
    orthogonal_vectors = v - (mu_ * v).sum(-1, keepdim=True) * mu_ / mu_.norm(dim=-1, keepdim=True)
    orthonormal_vectors = orthogonal_vectors / orthogonal_vectors.norm(dim=-1, keepdim=True)
    samples = orthonormal_vectors * (1.0 - weights**2).sqrt().unsqueeze(-1) + weights.unsqueeze(-1) * mu_
    if mu.dim() == 1:
        samples = samples.squeeze(-2)
    return samples
