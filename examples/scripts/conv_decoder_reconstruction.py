# %% [markdown]
# # Untrained reconstruction with a convolutional decoder
#
# Here, we use `~mr2.nn.nets.ConvDecoder` to reconstruct a synthetic single-coil
# image from noisy, undersampled Cartesian k-space. We fit the network to this
# measurement without a training dataset, keeping its random latent input fixed.
# The example runs on CPU and requires no data downloads.
#
# With the acquisition operator $A$, measurement $y$ and fixed input $z$, we solve
#
# $$\min_\theta \|A G_\theta(z)-y\|_2^2.$$
#
# We compare the reconstructed image with a zero-filled adjoint reconstruction
# and track the data fidelity during optimization. The synthetic reference also
# lets us inspect image error, which would be unavailable for a measured scan.
# %%
import matplotlib.pyplot as plt
import mr2
import torch

# A fixed seed makes the phantom noise, initialization and latent reproducible.
# One CPU thread avoids overhead for these small convolutions.
torch.set_num_threads(1)
torch.manual_seed(7)
size = 32
iterations = 300
stop_iteration = 100

# %% [markdown]
# ## Create the phantom and simulate k-space
#
# We create an elliptical phantom with a smooth phase and unit coil sensitivity.
# Its layout is ``(other, coils, z, y, x) = (1, 1, 1, 32, 32)``.
# We retain every second phase-encode line and six central lines, then add complex
# Gaussian noise at the acquired locations. Unacquired entries remain zero.
#
# The acquisition operator $A=MF$ combines the Fourier transform $F$ and binary
# mask $M$. We compose these operators with ``@`` and unpack their output tuples.
# %%
coordinate = torch.linspace(-1, 1, size)
y_grid, x_grid = torch.meshgrid(coordinate, coordinate, indexing='ij')
magnitude = 0.7 * ((x_grid / 0.75).square() + (y_grid / 0.9).square() < 1).float()
magnitude += 0.3 * (((x_grid + 0.2) / 0.25).square() + ((y_grid - 0.1) / 0.4).square() < 1)
magnitude -= 0.3 * (((x_grid - 0.25) / 0.15).square() + ((y_grid + 0.25) / 0.2).square() < 1)
reference = (magnitude * torch.exp(0.5j * x_grid))[None, None, None]
mask = torch.zeros(1, 1, 1, size, size)
mask[..., ::2, :] = 1
mask[..., size // 2 - 3 : size // 2 + 3, :] = 1
fourier = mr2.operators.FastFourierOp(dim=(-2, -1))
sampling = mr2.operators.CartesianMaskingOp(mask)
acquisition = sampling @ fourier
(clean_measurement,) = acquisition(reference)
noise = 0.05 * torch.randn_like(clean_measurement)
(measurement,) = sampling(clean_measurement + noise)
(zero_filled,) = acquisition.H(measurement)

# %% [markdown]
# We apply the adjoint with ``.H`` to obtain the zero-filled inverse FFT baseline.
# Undersampling prevents this adjoint from inverting the acquisition.
# We can also apply the normal operator $A^H A$ using ``.gram``:

# %%
(normal_image,) = acquisition.gram(reference)
(explicit_normal_image,) = acquisition.H(clean_measurement)
torch.testing.assert_close(normal_image, explicit_normal_image)

# %% [markdown]
# ## Set up the decoder
#
# `~mr2.nn.nets.ConvDecoder` takes real tensors with layout
# ``(batch, channels, height, width)``. We use 16 features and two upsampling stages.
# `~mr2.nn.ComplexAsChannel` combines the two real output channels into one complex
# channel. Adding a singleton z axis gives the MRtwo image layout used above.
# These neural network modules return tensors; the acquisition operator returns
# a tuple containing its output tensor.
#
# We draw the latent input once and optimize only the network parameters with Adam.
# %%
decoder = mr2.nn.ComplexAsChannel(
    mr2.nn.nets.ConvDecoder(
        n_dim=2,
        n_channels_in=16,
        n_channels_out=2,
        output_shape=(size, size),
        n_features=16,
        n_levels=2,
    )
)
latent = torch.rand(1, 16, 4, 4)
optimizer = torch.optim.Adam(decoder.parameters(), lr=0.008)

# %% [markdown]
# ## Reconstruct the image
#
# We normalize the squared residual by the number of acquired complex samples.
# After each Adam update, we record this data fidelity and the relative complex
# image error $\|x-x_\mathrm{ref}\|_2/\|x_\mathrm{ref}\|_2$.
# We use the reference only to evaluate image error, not in the loss or to select
# a checkpoint. We save the image after 100 updates and continue fitting to 300.
# %%
fidelity_history = []
error_history = []
checkpoints = {}
for iteration in range(iterations + 1):
    if iteration:
        optimizer.zero_grad(set_to_none=True)
        image = decoder(latent).unsqueeze(-3)
        (prediction,) = acquisition(image)
        loss = (prediction - measurement).abs().square().sum() / mask.sum()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        image = decoder(latent).unsqueeze(-3)
        (prediction,) = acquisition(image)
        fidelity_history.append(float((prediction - measurement).abs().square().sum() / mask.sum()))
        error_history.append(float(torch.linalg.vector_norm(image - reference) / torch.linalg.vector_norm(reference)))
        if iteration in (0, stop_iteration, iterations):
            checkpoints[iteration] = image.clone()

assert torch.isfinite(torch.tensor(fidelity_history)).all()
assert fidelity_history[-1] < fidelity_history[0]

# %% [markdown]
# ## Compare the reconstructions
#
# We plot the data fidelity and reference error against the number of updates.
# Further fitting can reduce the measurement residual by fitting acquired noise;
# the loss does not constrain unmeasured frequencies. An increase in reference
# error can indicate overfitting, but need not occur within this short run.
# The 100-update checkpoint illustrates a fixed stopping budget. For measured
# data, we would need to choose the budget or a stopping criterion without the
# reference image, for example using held-out samples or a noise estimate.
#
# All magnitude images use the same intensity scale. The complex image error
# includes phase differences as well as magnitude differences. To explore the
# effect of stopping, change the iteration budget or noise level and rerun the
# notebook from initialization.
# %% tags=["hide-input"] mystnb={"code_prompt_show": "Show plotting details"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
axes[0].semilogy(fidelity_history)
axes[0].set(xlabel='Adam updates', ylabel='Mean squared acquired residual')
axes[1].plot(error_history)
axes[1].set(xlabel='Adam updates', ylabel='Relative complex image error')
for axis in axes:
    axis.axvline(stop_iteration, color='tab:orange', linestyle='--', label='Saved checkpoint')
    axis.legend()
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
for axis, image, title in zip(
    axes,
    (reference, zero_filled, checkpoints[stop_iteration], checkpoints[iterations]),
    ('Reference', 'Zero-filled adjoint', f'Decoder: {stop_iteration} updates', f'Decoder: {iterations} updates'),
    strict=True,
):
    axis.imshow(image.squeeze().abs(), cmap='gray', vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis('off')
plt.show()

# %% [markdown]
# For measured `~mr2.data.KData`, `~mr2.algorithms.reconstruction.ConvDecoderReconstruction`
# provides acquisition setup, intensity scaling and callback-controlled stopping.
# Here, we wrote the fitting loop explicitly to work with tensors and show each
# optimization step. The decoder uses instance normalization in place of
# the batch normalization in the original
# [ConvDecoder paper](https://doi.org/10.1109/TCI.2021.3097596).
#
# This example assumes known sampling and unit sensitivity for one synthetic coil.
# Applying the method to measured data requires a suitable acquisition model and
# noise handling. The small phantom and short fitting budget do not establish
# reconstruction quality on clinical scans.
