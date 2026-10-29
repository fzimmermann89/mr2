"""Statistical and algebraic checks for noise augmentation."""

import pytest
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
    moments = torch.stack(
        (
            filtered.real.square().mean(),
            filtered.imag.square().mean(),
            (filtered.real * filtered.imag).mean(),
        )
    )
    torch.testing.assert_close(moments, torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64), atol=0.015, rtol=0)
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


@pytest.mark.parametrize('amplitude', [1e-30, 1e30])
def test_filter_noise_window_scale(amplitude: float) -> None:
    """Tiny and large constant windows remain identity filters, including repeated axes."""
    noise = RandomGenerator(9).randn_tensor((4, 32), dtype=torch.complex64)
    window = torch.full((32,), amplitude, dtype=torch.float32)
    torch.testing.assert_close(filter_noise(noise, (window, window), (-1, -1)), noise)


def test_filter_noise_small_window_overlap() -> None:
    """Complementary windows with tiny overlapping support still normalize to identity."""
    noise = RandomGenerator(9).randn_tensor((4, 2), dtype=torch.complex64)
    window = torch.tensor([1.0, 1e-24])
    torch.testing.assert_close(filter_noise(noise, (window, window.flip(0)), (-1, -1)), noise)


def test_noise_windows() -> None:
    """Windows peak at FFT DC and taper only retained cut boundaries."""
    torch.testing.assert_close(gaussian_kspace_window(4, 2), torch.tensor([-2.0, -0.5, 0.0, -0.5]).exp())
    torch.testing.assert_close(gaussian_kspace_window(1, 2), torch.ones(1))
    torch.testing.assert_close(gaussian_kspace_window(5, 0), torch.ones(5))
    torch.testing.assert_close(
        tapered_kspace_window(8, 2, 8, 2), torch.tensor([0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0, 1.0])
    )
    with pytest.raises(ValueError, match='overlapping'):
        tapered_kspace_window(8, 2, 6, 3)
