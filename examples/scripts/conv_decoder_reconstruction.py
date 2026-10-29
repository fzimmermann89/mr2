# %% [markdown]
# # Untrained reconstruction with a convolutional decoder
#
# Fit one randomly initialized `~mr2.nn.nets.ConvDecoder` to one noisy,
# undersampled measurement. There is no training dataset: the network architecture
# and a fixed small latent tensor parameterize the reconstructed image.
# We minimize $\|A G_\theta(z)-y\|_2^2$ over weights $\theta$, keeping $z$ fixed.
#
# This small CPU example uses a synthetic single-coil object, known Cartesian
# sampling and unit sensitivity. It illustrates fitting and stopping, rather than
# image quality on clinical data. Real scans need an appropriate trajectory,
# coil sensitivities, noise handling and a separately chosen stopping rule.

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
# ## One measurement and its acquisition operator
#
# The tensor layout is ``(other, coils, z, y, x) = (1, 1, 1, 32, 32)``.
# The object has a smooth phase, so reconstructing only a real magnitude would
# impose the wrong forward model. The binary mask keeps every second phase-encode
# line plus six central lines. Unacquired entries stay zero on the full grid.
# Compose $A=MF$ with ``@``; operators return tuples, unpacked below.

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
# ``.H`` applies the adjoint: here it is the zero-filled inverse FFT baseline.
# It is not an inverse of the undersampled acquisition. ``.gram`` represents
# $A^H A$, the normal operator used by many iterative reconstructions.

# %%
(normal_image,) = acquisition.gram(reference)
(explicit_normal_image,) = acquisition.H(clean_measurement)
torch.testing.assert_close(normal_image, explicit_normal_image)

# %% [markdown]
# ## A fixed latent tensor and complex output
#
# `~mr2.nn.nets.ConvDecoder` takes real ``(batch, channels, height, width)``
# tensors. `~mr2.nn.ComplexAsChannel` combines its two real output channels into
# one complex channel. Insert the singleton z axis to obtain the MRtwo image
# layout. Unlike operators, these neural network modules return a tensor.
# The latent tensor has no gradients and is never updated; Adam sees only the
# network parameters. This is a small instance of the existing MRtwo decoder,
# with 16 features and two upsampling stages.

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
# ## Fit weights and record the trajectory
#
# Normalize the squared residual by the number of acquired complex samples.
# Each recorded value is evaluated after the corresponding update. The reference
# error is a simulation diagnostic only: the reference is never used in the loss
# or for selecting a checkpoint. Save the fixed-budget checkpoint at 100 updates,
# then continue to 300 to inspect how further fitting changes the image.

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
# ## Stopping and interpretation
#
# Lower measurement error need not mean a better image: later updates can fit
# acquired noise, while unmeasured frequencies remain unconstrained by the loss.
# The curves let us inspect this possibility; a visible error upturn is not
# guaranteed for every noise level, seed or budget. The 100-update checkpoint is
# an illustrative budget, not an optimal stopping rule. With real data, choose a
# budget or validation/noise-based criterion without access to a reference.
#
# Magnitudes below share a fixed intensity scale. The trace uses complex relative
# image error, so it also includes phase discrepancies. No superiority over the
# adjoint baseline is assumed. Try changing the noise, mask or iteration budget
# and rerun from initialization; each run fits a fresh network to one measurement.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show plotting details"}
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
# The explicit loop here exposes the same scan-specific fitting idea without raw
# data headers. This decoder variant uses instance normalization, differing from
# the batch normalization in the original
# [ConvDecoder paper](https://doi.org/10.1109/TCI.2021.3097596).
