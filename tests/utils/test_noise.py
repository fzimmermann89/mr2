"""Statistical and algebraic checks for noise augmentation."""

import torch
from mr2.utils import RandomGenerator
from mr2.utils.noise import filter_noise, gaussian_kspace_window, noise_pair, tapered_kspace_window


def test_noise_covariance() -> None:
    """Combined asymmetric windows preserve variance and impose the analytic complex covariance."""
    rng = RandomGenerator(42)
    noise = 2**0.5 * rng.randn_tensor((8192, 32), dtype=torch.complex128)
    gaussian = gaussian_kspace_window(32, 2, dtype=torch.float64)
    partial = tapered_kspace_window(32, 8, 32, 3, dtype=torch.float64)
    filtered = filter_noise(noise, (gaussian, partial), (-1, -1))
    combined = gaussian * partial
    power = combined.square() / combined.square().mean()
    expected = 2 * torch.fft.ifft(torch.fft.ifftshift(power))
    # Circular complex noise has equal component variances and zero covariance at one pixel.
    torch.testing.assert_close(
        filtered.real.square().mean(), torch.tensor(1.0, dtype=torch.float64), atol=0.015, rtol=0
    )
    torch.testing.assert_close(
        filtered.imag.square().mean(), torch.tensor(1.0, dtype=torch.float64), atol=0.015, rtol=0
    )
    torch.testing.assert_close(
        (filtered.real * filtered.imag).mean(), torch.tensor(0.0, dtype=torch.float64), atol=0.015, rtol=0
    )
    for lag in (0, 1, 3):
        observed = (filtered.roll(-lag, -1) * filtered.conj()).mean()
        torch.testing.assert_close(observed, expected[lag], atol=0.02, rtol=0)


def test_noise_pair() -> None:
    """Scale the target and noise once, retaining a spatially varying g-map."""
    rng = RandomGenerator(7)
    image = rng.randn_tensor((2, 8, 8), dtype=torch.complex128)
    noise = 2**0.5 * rng.randn_like(image)
    sigma = torch.tensor([0.5, 2.0], dtype=torch.float64).reshape(2, 1, 1)
    gmap = torch.linspace(1, 3, 64, dtype=torch.float64).reshape(8, 8)
    noisy, target = noise_pair(image, noise, sigma, gmap)
    scale = (1 + sigma.square()).sqrt()
    torch.testing.assert_close(target * scale, image)
    torch.testing.assert_close((noisy - target) * scale, sigma * gmap * noise)
