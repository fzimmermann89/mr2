"""Pre-built denoising algorithms."""

from mr2.algorithms.denoising.total_variation_denoising import total_variation_denoising
from mr2.algorithms.denoising.wavelet_denoising import wavelet_denoising

__all__ = ['total_variation_denoising', 'wavelet_denoising']
