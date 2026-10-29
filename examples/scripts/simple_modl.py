# %% [markdown]
# # A tiny MoDL-style reconstruction
#
# This toy demonstration trains a small unrolled network on synthetic complex
# images. It needs no downloads or GPU: the default uses CPU, eight 32 x 32
# training images, two held-out images and 40 full-batch updates (typically under
# a minute on a laptop). Install ``mrtwo[notebooks]`` to run it.
# This is a teaching example, not a full paper reproduction or evidence of
# reconstruction performance on real MRI data.

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
# ## Synthetic images and a known acquisition model
#
# We sum smooth ellipses with random positions, widths and amplitudes, then add
# a smooth phase. This deliberately narrow image distribution makes a small
# supervised exercise possible. The last two images are held out from training;
# they are still from the same toy distribution.
#
# MR images have axes ``(batch, coils, z, y, x)`` here. We use one coil with
# sensitivity one and one slice. No metadata objects are needed for this
# synthetic tensor example. The forward model $A = MF$ combines a centered,
# orthonormal Fourier transform with a Cartesian sampling mask. Unsampled
# entries are stored as zeros; the same mask is used for all images.

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
# Starting at $x_0=A^H b$, each of three iterations computes
#
# $$z_k = D_\theta(x_k), \qquad
# (A^H A + \lambda I)x_{k+1} = A^H b + \lambda z_k.$$
#
# `~mr2.nn.data_consistency.ConjugateGradientDC` solves this positive-definite
# system and differentiates through the solve implicitly. Its trainable
# ``log_weight`` gives $\lambda=\exp(\mathrm{log\_weight})>0$. Larger weights
# pull the solution toward the denoiser. This is a soft consistency penalty,
# so measured samples need not be reproduced exactly.
#
# The residual denoiser is $D_\theta(x)=x+N_\theta(x)$. To use a 2D CNN we
# remove only the singleton z axis: ``(B, 1, 1, H, W) -> (B, 1, H, W)``.
# `~mr2.nn.ComplexAsChannel` maps that complex channel to two real channels,
# real then imaginary, and converts the CNN output back to complex values.
# We restore z before applying the MR operator. For multiple slices one could
# fold slices into the batch; multiple coils require a suitable sensitivity
# model and coil combination rather than silently treating coils as slices.
#
# The same CNN and regularization weight are shared across all iterations.
# Unshared iterations would use a separate CNN and DC module per step (for
# example in a ``ModuleList``), increasing the number of learned parameters.


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
# ## A plain PyTorch training loop
#
# The supervised loss is mean squared complex error,
# $\operatorname{mean}|x-x_{\mathrm{true}}|^2$, which includes both real and
# imaginary errors. We update the CNN and the positive regularization weight
# together. The held-out loss is only monitored, never used for gradients.
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
# ## Inspect the result
#
# Compare magnitudes on the same intensity scale and absolute complex errors
# on a separate shared scale. Loss curves describe this tiny training exercise;
# they do not establish generalization to other masks, anatomy, noise or coils.
# A practical study needs representative data, independent evaluation and
# careful choices of network size, number of iterations and acquisition model.

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
