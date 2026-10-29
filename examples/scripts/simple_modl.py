# %% [markdown]
# # MoDL-style reconstruction with a small CNN
#
# Here we train an unrolled reconstruction network using a convolutional neural
# network (CNN) and conjugate-gradient data consistency. We use synthetic complex
# images so that the example runs on CPU without downloading data.
#
# The default run uses eight 32 x 32 training images, two held-out images and
# 40 training updates. It took about two seconds on the development machine;
# runtime depends on the CPU. Install ``mrtwo[notebooks]`` to run the notebook.
#
# This toy example demonstrates the reconstruction and training steps. It does
# not reproduce the full MoDL paper or evaluate reconstruction of real MRI data.

# %%
import time

import matplotlib.pyplot as plt
import mr2
import torch
from torch import nn

torch.manual_seed(7)
torch.set_num_threads(2)  # Avoid thread overhead for these tiny convolutions.
device = torch.device('cpu')
start = time.perf_counter()

# %% [markdown]
# ## Create images and simulate undersampled k-space data
#
# We form each magnitude image by summing five elliptical Gaussian functions
# with random positions, widths and amplitudes, then add a linear phase.
# We reserve the last two images for evaluation and train on the first eight.
# All ten images follow the same distribution.
#
# The image tensors have shape ``(batch, coils, z, y, x)``. We use one slice and
# one coil with sensitivity one. The forward model $A = MF$ combines a centered,
# orthonormal Fourier transform $F$ with a Cartesian sampling mask $M$.
# We keep every fourth phase-encoding line and four central lines, storing
# unsampled entries as zeros. We use the same mask for all images.

# %%
size = 32
y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
images = []
for _ in range(10):
    magnitude = torch.zeros(size, size)
    for _ in range(5):
        cx, cy = (torch.rand(2) - 0.5) * 1.2
        wx, wy = 0.12 + torch.rand(2) * 0.3
        magnitude += (0.2 + torch.rand(())) * torch.exp(-(((x - cx) / wx) ** 2) - ((y - cy) / wy) ** 2)
    magnitude /= magnitude.max()
    phase = 0.6 * (torch.rand(()) * x + torch.rand(()) * y)
    images.append(magnitude * torch.exp(1j * phase))
truth = torch.stack(images)[:, None, None].to(device)
train_truth, test_truth = truth[:8], truth[8:]

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

# Check the CG solve against the closed-form solution available for this
# particular binary mask and unitary FFT. General coil models need CG.
with torch.no_grad():
    prior = adjoint
    consistent = model.dc(prior, test_data, acquisition)  # type: ignore[call-overload]
    weight = model.dc.log_weight.exp()
    (prior_kspace,) = fourier(prior)
    (expected,) = fourier.H((test_data + weight * prior_kspace) / (mask + weight))
    torch.testing.assert_close(consistent, expected, atol=2e-5, rtol=2e-5)

# %% [markdown]
# ## Train the reconstruction network
#
# The supervised loss is mean squared complex error,
# $\operatorname{mean}|x-x_{\mathrm{true}}|^2$, which includes both real and
# imaginary errors. We update the CNN and the positive regularization weight
# together. We monitor the held-out loss without using it to update the parameters.
# This short run uses one fixed mask, noiseless data and no augmentation.

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
train_losses, test_losses = [], []
for _ in range(40):
    model.train()
    optimizer.zero_grad()
    prediction = model(train_data)
    loss = (prediction - train_truth).abs().square().mean()
    loss.backward()
    assert model.dc.log_weight.grad is not None and torch.isfinite(model.dc.log_weight.grad)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        train_losses.append(loss.item())
        test_losses.append((model(test_data) - test_truth).abs().square().mean().item())

with torch.no_grad():
    reconstruction = model(test_data)
assert reconstruction.shape == test_truth.shape and torch.isfinite(reconstruction).all()
print(f'Elapsed CPU time: {time.perf_counter() - start:.1f} s')
print(f'Learned regularization weight: {model.dc.log_weight.exp().item():.3f}')
print(
    f'Held-out complex MSE: adjoint={(adjoint - test_truth).abs().square().mean().item():.5f}, '
    f'unrolled={test_losses[-1]:.5f}'
)

# %% [markdown]
# ## Compare with the adjoint reconstruction
#
# We display the first held-out image, its adjoint reconstruction and the
# trained network output. The magnitude images share an intensity scale;
# the absolute complex errors share a separate scale. The loss curves show
# the mean squared complex error over all training and held-out images.
#
# We use noiseless data from a small set of smooth synthetic images. These
# results do not measure performance on anatomical images, other sampling
# masks or multi-coil acquisitions.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting code"}
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
