# %% [markdown]
# # MoDL-style reconstruction with a small CNN
#
# Here we train an unrolled reconstruction network using a convolutional neural
# network (CNN) and conjugate-gradient data consistency. We use simulated
# brain images from BrainWeb and run the reconstruction on CPU.
#
# The default run downloads one BrainWeb volume (about 4 MB) and uses eight 96 x 96 slices
# for training and two held-out slices for evaluation, with 40 training updates.
# Training takes a few seconds on the development CPU; download time depends on
# the connection. Install ``mrtwo[notebooks]`` to run the notebook.
#
# This toy example demonstrates the reconstruction and training steps. It does
# not reproduce the full MoDL paper or evaluate reconstruction of real MRI data.

# %%
import gzip
import time

import matplotlib.pyplot as plt
import mr2
import numpy as np
import requests
import torch
from torch import nn
from torch.nn import functional

torch.manual_seed(7)
torch.set_num_threads(2)  # Avoid thread overhead for these tiny convolutions.
device = torch.device('cpu')

# %% [markdown]
# ## Create images and simulate undersampled k-space data
#
# We use the T1-weighted, 1 mm normal-brain simulation with zero noise and
# zero intensity non-uniformity from [BrainWeb](https://brainweb.bic.mni.mcgill.ca/brainweb/selection_normal.html).
# The volume contains magnitude images, so we add a smooth phase ourselves
# before simulating complex k-space data. These are simulated data, not acquired
# multi-coil measurements.
#
# We take eight axial slices for training and two higher slices for evaluation.
# Both sets come from one anatomical model: the held-out loss does not measure
# generalization to another subject. We pad each slice to a square before
# resizing to 96 x 96 pixels, preserving its aspect ratio, and normalize each
# slice by its maximum magnitude.
#
# The image tensors have shape ``(batch, coils, z, y, x)``. We use one slice and
# one coil with sensitivity one. The forward model $A = MF$ combines a centered,
# orthonormal Fourier transform $F$ with a Cartesian sampling mask $M$.
# We keep every fourth phase-encoding line and four central lines, storing
# unsampled entries as zeros. We use the same mask for all images.

# %% tags=["hide-input"] mystnb={"code_prompt_show": "Show BrainWeb download and preparation"}
# BrainWeb raw-byte volumes have axes (z, y, x) and shape (181, 217, 181).
response = requests.get(
    'https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1',
    params={
        'do_download_alias': 'T1 ICBM normal 1mm pn0 rf0',
        'format_value': 'raw_byte',
        'zip_value': 'gnuzip',
        'download_for_real': '[Start download!]',
    },
    timeout=60,
)
response.raise_for_status()
volume = torch.from_numpy(
    np.frombuffer(gzip.decompress(response.content), dtype=np.uint8).copy().reshape(181, 217, 181)
)

size = 96
slice_indices = [50, 58, 66, 74, 82, 90, 98, 106, 120, 128]
magnitude = volume[slice_indices, None].float()
magnitude = functional.pad(magnitude, (18, 18, 0, 0))
magnitude = functional.interpolate(magnitude, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
magnitude /= magnitude.amax(dim=(-2, -1), keepdim=True)
y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
phase_slopes = 0.6 * torch.rand(10, 2, 1, 1)
phase = phase_slopes[:, :1] * x + phase_slopes[:, 1:] * y
truth = (magnitude * torch.exp(1j * phase)).unsqueeze(-3).to(device)
train_truth, test_truth = truth[:8], truth[8:]

# %%
mask = torch.zeros(1, 1, 1, size, 1, device=device)
mask[..., ::4, :] = 1
mask[..., size // 2 - 2 : size // 2 + 2, :] = 1
sampling = mr2.operators.IdentityOp() * mask
fourier = mr2.operators.FastFourierOp(dim=(-2, -1))
acquisition = sampling @ fourier
(measured,) = acquisition(truth)
train_data, test_data = measured[:8], measured[8:]
(adjoint,) = acquisition.H(test_data)
print(f'Sampled phase-encoding lines: {int(mask.sum())}/{size}')

# %% [markdown]
# ## Denoising followed by data consistency
#
# We initialize the reconstruction with the adjoint image $x_0=A^H b$, where
# $b$ denotes the undersampled k-space data. At each of three iterations we
# denoise the current image and solve a regularized least-squares problem:
#
# $$z_k = D_\theta(x_k), \qquad
# x_{k+1} = \operatorname*{arg\,min}_x
# \left(\|Ax-b\|_2^2 + \lambda\|x-z_k\|_2^2\right).$$
#
# The corresponding normal equation is
#
# $$(A^H A + \lambda I)x_{k+1} = A^H b + \lambda z_k.$$
#
# We use `~mr2.nn.data_consistency.ConjugateGradientDC` to solve this system.
# It computes gradients by implicit differentiation. The trainable parameter
# ``log_weight`` defines $\lambda=\exp(\mathrm{log\_weight})>0$, keeping the
# system positive definite. Increasing $\lambda$ gives the denoised image more
# weight relative to the measured data. The solution can therefore differ from
# the measurements even at sampled positions.
#
# Our denoiser is $D_\theta(x)=x+N_\theta(x)$, where $N_\theta$ is a
# `~mr2.nn.nets.BasicCNN`. Before applying the 2D CNN, we remove the singleton
# z axis to obtain shape ``(batch, 1, y, x)``.
# `~mr2.nn.ComplexAsChannel` splits the complex channel into real and imaginary
# channels and converts the CNN output back to a complex tensor. We restore
# the z axis before the data-consistency step.
#
# For independent slices, we could combine the slice and batch axes before
# applying the CNN. For multiple coils, we would also need coil sensitivity
# maps and a forward model that maps a coil-combined image to coil data.
#
# We share the CNN and regularization weight across all three iterations.
# To use different parameters at each iteration, we would create a separate
# CNN and data-consistency module for each step, stored in a ``ModuleList``.


# %%
class TinyMoDL(nn.Module):
    """Three shared denoising and data-consistency steps."""

    def __init__(self):
        """Create the shared CNN and positive regularization weight."""
        super().__init__()
        self.correction = mr2.nn.ComplexAsChannel(
            mr2.nn.nets.BasicCNN(n_dim=2, n_channels_in=2, n_channels_out=2, n_features=(8, 8))
        )
        self.dc = mr2.nn.data_consistency.ConjugateGradientDC(0.1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Reconstruct masked k-space tensors."""
        (image,) = acquisition.H(data)
        for _ in range(3):
            network_image = image.squeeze(-3)
            denoised = (network_image + self.correction(network_image)).unsqueeze(-3)
            # DC supports composed linear operators; its overloads currently name only FourierOp.
            image = self.dc(denoised, data, acquisition)  # type: ignore[call-overload]
        return image


model = TinyMoDL().to(device)

# %% [markdown]
# ## Train the reconstruction network
#
# The supervised loss is mean squared complex error,
# $\operatorname{mean}|x-x_{\mathrm{true}}|^2$, which includes both real and
# imaginary errors. We apply ``torch.nn.functional.mse_loss`` to the real and
# imaginary components using ``torch.view_as_real``. Multiplying by two gives
# the mean squared complex error, since MSE averages over both components.
# We update the CNN and the positive regularization weight
# together. We monitor the held-out loss without using it to update the parameters.
# This short run uses one fixed mask, noiseless data and no augmentation.

# %%
start = time.perf_counter()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
train_losses, test_losses = [], []
for _ in range(40):
    model.train()
    optimizer.zero_grad()
    prediction = model(train_data)
    loss = 2 * functional.mse_loss(torch.view_as_real(prediction), torch.view_as_real(train_truth))
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        train_losses.append(loss.item())
        test_prediction = model(test_data)
        test_loss = 2 * functional.mse_loss(torch.view_as_real(test_prediction), torch.view_as_real(test_truth))
        test_losses.append(test_loss.item())

with torch.no_grad():
    reconstruction = model(test_data)
    adjoint_loss = 2 * functional.mse_loss(torch.view_as_real(adjoint), torch.view_as_real(test_truth))
print(f'Training and evaluation time: {time.perf_counter() - start:.1f} s')
print(f'Learned regularization weight: {model.dc.log_weight.exp().item():.3f}')
print(f'Held-out complex MSE: adjoint={adjoint_loss.item():.5f}, unrolled={test_losses[-1]:.5f}')

# %% [markdown]
# ## Compare with the adjoint reconstruction
#
# We display the first held-out image, its adjoint reconstruction and the
# trained network output. The magnitude images share an intensity scale;
# the absolute complex errors share a separate scale. The loss curves show
# the mean squared complex error over all training and held-out images.
#
# We train and evaluate on slices of the same simulated brain with one fixed
# mask and no added noise. These results do not measure performance on other
# subjects, sampling masks or acquired multi-coil data.

# %% tags=["hide-input"] mystnb={"code_prompt_show": "Show plotting code"}
fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
for axis, image, title in zip(
    axes[0],
    [test_truth, adjoint, reconstruction],
    ['Reference magnitude', 'Adjoint magnitude', 'Unrolled magnitude'],
    strict=True,
):
    axis.imshow(image[0, 0, 0].abs().numpy(), cmap='gray', vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis('off')
error_max = max((adjoint - test_truth).abs().max().item(), (reconstruction - test_truth).abs().max().item())
for axis, image, title in zip(
    axes[1, :2], [adjoint, reconstruction], ['Adjoint complex error', 'Unrolled complex error'], strict=True
):
    plot = axis.imshow((image - test_truth)[0, 0, 0].abs().numpy(), cmap='magma', vmin=0, vmax=error_max)
    axis.set_title(title)
    axis.axis('off')
fig.colorbar(plot, ax=list(axes[1, :2]), shrink=0.7)
axes[1, 2].plot(range(1, 41), train_losses, label='Training (before update)')
axes[1, 2].plot(range(1, 41), test_losses, label='Held out (after update)')
axes[1, 2].set(xlabel='Update', ylabel='Complex MSE', yscale='log')
axes[1, 2].legend(fontsize='small')
plt.show()
