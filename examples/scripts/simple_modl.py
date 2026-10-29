# %% [markdown]
# # MoDL-style reconstruction with a small CNN
#
# We train a small unrolled network with a convolutional neural network (CNN)
# and conjugate-gradient data consistency on simulated BrainWeb images.
# Training runs for 60 seconds on CPU and requires a 4 MB download.
# Install ``mrtwo[notebooks]`` to run the notebook.

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
# We use the T1-weighted, 1 mm [BrainWeb](https://brainweb.bic.mni.mcgill.ca/brainweb/selection_normal.html)
# volume without noise or intensity non-uniformity. We pad and resize the
# slices to 96 x 96 pixels, normalize their magnitudes and add a smooth phase.
# Eight axial slices are used for training and two higher slices for evaluation.
# Both sets come from the same simulated brain.
#
# For this toy single-coil acquisition, the tensors have shape
# ``(batch, coils=1, z=1, y, x)``. The forward model $A=MF$ combines a centered,
# orthonormal Fourier transform with a mask that keeps every fourth
# phase-encoding line and four central lines. Unsampled entries remain zero.

# %% tags=["hide-input"] mystnb={"code_prompt_show": "Show BrainWeb download and preparation"}
# BrainWeb raw-byte volumes have axes (z, y, x) and shape (181, 217, 181).
response = requests.get(
    'https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1',
    params={
        'do_download_alias': 'T1 ICBM normal 1mm pn0 rf0',  # typos: ignore
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
# Starting from the adjoint image $x_0=A^H b$, we alternate denoising and
# data consistency for three iterations:
#
# $$z_k = D_\theta(x_k),$$
#
# $$(A^H A + \lambda I)x_{k+1} = A^H b + \lambda z_k.$$
#
# `~mr2.nn.data_consistency.ConjugateGradientDC` solves the second equation
# and computes gradients by implicit differentiation. Its trainable parameter
# ``log_weight`` gives $\lambda=\exp(\mathrm{log\_weight})>0$.
# Larger weights pull the reconstruction toward the denoised image.
#
# We use a residual denoiser $D_\theta(x)=x+N_\theta(x)$ with a
# `~mr2.nn.nets.BasicCNN`. Removing the singleton z axis gives the CNN input
# shape ``(batch, 1, y, x)``. `~mr2.nn.ComplexAsChannel` splits the complex
# channel into real and imaginary channels and converts the output back to
# complex values. We restore z before data consistency.
#
# The CNN and regularization weight are shared across iterations. Unshared
# iterations would each have their own CNN and data-consistency module.


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

    def forward(self, data: torch.Tensor, acquisition: mr2.operators.LinearOperator) -> torch.Tensor:
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
# We minimize the complex MSE, $\operatorname{mean}|x-x_{\mathrm{true}}|^2$.
# ``mse_loss`` averages over real and imaginary components, so we multiply
# by two. Adam updates the CNN and regularization weight together; we compute
# the held-out loss without gradients.

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
train_losses, test_losses = [], []
training_seconds = 60
start = time.perf_counter()
while time.perf_counter() - start < training_seconds:
    model.train()
    optimizer.zero_grad()
    prediction = model(train_data, acquisition)
    loss = 2 * functional.mse_loss(torch.view_as_real(prediction), torch.view_as_real(train_truth))
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        train_losses.append(loss.item())
        test_prediction = model(test_data, acquisition)
        test_loss = 2 * functional.mse_loss(torch.view_as_real(test_prediction), torch.view_as_real(test_truth))
        test_losses.append(test_loss.item())

with torch.no_grad():
    reconstruction = model(test_data, acquisition)
    adjoint_loss = 2 * functional.mse_loss(torch.view_as_real(adjoint), torch.view_as_real(test_truth))
print(f'Training and evaluation time: {time.perf_counter() - start:.1f} s')
print(f'Learned regularization weight: {model.dc.log_weight.exp().item():.3f}')
print(f'Held-out complex MSE: adjoint={adjoint_loss.item():.5f}, unrolled={test_losses[-1]:.5f}')

# %% [markdown]
# ## Compare with the adjoint reconstruction
#
# We compare the first held-out slice with the adjoint reconstruction.
# Magnitudes and absolute complex errors each use a shared color scale.

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
fig.colorbar(plot, ax=axes[1, 1], shrink=0.7, label='Absolute error')
axes[1, 2].plot(range(1, len(train_losses) + 1), train_losses, label='Training (before update)')
axes[1, 2].plot(range(1, len(train_losses) + 1), test_losses, label='Held out (after update)')
axes[1, 2].set(xlabel='Update', ylabel='Complex MSE', yscale='log')
axes[1, 2].legend(fontsize='small')
plt.show()
