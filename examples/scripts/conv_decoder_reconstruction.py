# %% [markdown]
# # Untrained reconstruction with a convolutional decoder
#
# Here, we use `~mr2.nn.nets.ConvDecoder` to reconstruct a knee MRI slice from
# undersampled Cartesian k-space. We fit the network to one measurement without
# a training dataset, keeping its random latent input fixed.
#
# With acquisition operator $A$, measurement $y$ and fixed input $z$, we solve
#
# $$\min_\theta \|A G_\theta(z)-y\|_2^2.$$
#
# We use a public single-coil acquisition from the
# [i3M A4IM dataset](https://doi.org/10.5281/zenodo.20700288)
# by Borreguero, Fernández-García and Alonso (CC BY 4.0).
# The example downloads one 24 MB subject archive and fits a 120 x 160 slice on CPU.

# %% tags=["hide-cell"] mystnb={"code_prompt_show": "Show download details"}
import os
import tempfile
from pathlib import Path
from zipfile import ZipFile

import zenodo_get

# Keep the archive and extracted raw file in a temporary directory.
tmp = tempfile.TemporaryDirectory()
data_folder = Path(tmp.name)
zenodo_get.download(
    record='20700288',
    file_glob='Subject 1.zip',
    output_dir=data_folder,
    retry_attempts=5,
    access_token=os.environ.get('ZENODO_TOKEN'),
)
raw_name = 'Subject 1/Qualitative protocol/Subject1-LeftKnee/ismrmrd/Subject1-LeftKnee-PD (684).h5'
raw_path = data_folder / 'knee_pd.h5'
with ZipFile(data_folder / 'Subject 1.zip') as archive:
    raw_path.write_bytes(archive.read(raw_name))

# %% [markdown]
# ## Load one measured knee slice
#
# We load the proton-density-weighted left-knee acquisition from subject 1,
# acquired at 72 mT. We reconstruct the source volume with its Cartesian Fourier
# operator and select the central slice (zero-based index 16). Its tensor layout
# is ``(other, coils, z, y, x) = (1, 1, 1, 120, 160)``.
#
# This complex coil image is our reference for retrospective undersampling.
# It retains the source acquisition's noise and coil sensitivity profile; it is
# not a noise-free ground truth. We model the coil image directly, so no separate
# sensitivity operator is needed. We do not add synthetic noise or resize the image.

# %%
import matplotlib.pyplot as plt
import mr2
import torch

# Two CPU threads avoid excessive overhead for these small convolutions.
torch.set_num_threads(2)
torch.manual_seed(7)
iterations = 1000
stop_iteration = 300
kdata = mr2.data.KData.from_file(raw_path, mr2.data.traj_calculators.KTrajectoryCartesian())
source_fourier = mr2.operators.FourierOp.from_kdata(kdata)
(source_volume,) = source_fourier.H(kdata.data)
reference = source_volume[:, :, 16:17].clone()
height, width = reference.shape[-2:]

# %% [markdown]
# ## Undersample the slice
#
# We draw half the phase-encode lines at random and retain 16 central lines.
# The union of these sets defines the actual sampling fraction, shown below.
# We Fourier transform the complex reference slice and keep only these lines.
# This is a simulated 2D acquisition derived from the measured 3D volume, rather
# than a prospectively accelerated scan.
#
# The acquisition operator $A=MF$ combines the Fourier transform $F$ and binary
# mask $M$. We compose these operators with ``@`` and unpack their output tuples.

# %%
mask = torch.zeros(1, 1, 1, height, 1)
mask[..., torch.randperm(height)[: height // 2], :] = 1
mask[..., height // 2 - 8 : height // 2 + 8, :] = 1
fourier = mr2.operators.FastFourierOp(dim=(-2, -1))
sampling = mr2.operators.CartesianMaskingOp(mask)
acquisition = sampling @ fourier
(measurement,) = acquisition(reference)
(zero_filled,) = acquisition.H(measurement)

# Use measured data for scaling, without needing the reference image.
scale = zero_filled.abs().square().mean().sqrt()
measurement = measurement / scale
zero_filled = zero_filled / scale
reference = reference / scale
acquired_samples = mask.sum() * width

# %% tags=["hide-input"] mystnb={"code_prompt_show": "Show sampling plot details"}
fig, axis = plt.subplots(figsize=(6, 3))
axis.imshow(mask[0, 0, 0].expand(-1, width), cmap='gray', aspect='auto', vmin=0, vmax=1)
axis.set(
    xlabel='Readout samples',
    ylabel='Phase-encode lines',
    title=f'{int(mask.sum())}/{height} lines acquired (R = {height / float(mask.sum()):.2f})',
)
plt.show()

# %% [markdown]
# We apply the adjoint with ``.H`` to obtain the zero-filled inverse FFT baseline.
# Undersampling prevents this adjoint from inverting the acquisition.
# We can also apply the normal operator $A^H A$ using ``.gram``:

# %%
(normal_image,) = acquisition.gram(reference)
(explicit_normal_image,) = acquisition.H(measurement)
torch.testing.assert_close(normal_image, explicit_normal_image)

# %% [markdown]
# ## Set up the decoder
#
# `~mr2.nn.nets.ConvDecoder` takes real tensors with layout
# ``(batch, channels, height, width)``. We use 32 features and three upsampling stages.
# `~mr2.nn.ComplexAsChannel` combines the two real output channels into one complex
# channel. Adding a singleton z axis gives the MRtwo image layout used above.
# These neural network modules return tensors; the acquisition operator returns
# a tuple containing its output tensor.
#
# We draw the latent input once and optimize only the network parameters with Adam.
# Scaling the data by the adjoint image's RMS keeps the fitting problem near unit
# intensity. Multiplying the reconstructed image by ``scale`` restores input units.
# %%
decoder = mr2.nn.ComplexAsChannel(
    mr2.nn.nets.ConvDecoder(
        n_dim=2,
        n_channels_in=32,
        n_channels_out=2,
        output_shape=(height, width),
        n_features=32,
        n_levels=3,
    )
)
latent = torch.rand(1, 32, 8, 10)
optimizer = torch.optim.Adam(decoder.parameters(), lr=0.005)

# %% [markdown]
# ## Reconstruct the image
#
# We normalize the squared residual by the number of acquired complex samples.
# After each Adam update, we record this data fidelity and the relative complex
# image error $\|x-x_\mathrm{ref}\|_2/\|x_\mathrm{ref}\|_2$.
# We use the reference only to evaluate image error, not in the loss or to select
# a checkpoint. We save the image after 300 updates and continue fitting to 1000.
# %%
fidelity_history = []
error_history = []
checkpoints = {}
for iteration in range(iterations + 1):
    if iteration:
        optimizer.zero_grad(set_to_none=True)
        image = decoder(latent).unsqueeze(-3)
        (prediction,) = acquisition(image)
        loss = (prediction - measurement).abs().square().sum() / acquired_samples
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        image = decoder(latent).unsqueeze(-3)
        (prediction,) = acquisition(image)
        fidelity_history.append(float((prediction - measurement).abs().square().sum() / acquired_samples))
        error_history.append(float(torch.linalg.vector_norm(image - reference) / torch.linalg.vector_norm(reference)))
        if iteration in (0, stop_iteration, iterations):
            checkpoints[iteration] = image.clone()

assert torch.isfinite(torch.tensor(fidelity_history)).all()
assert fidelity_history[-1] < fidelity_history[0]

# %% [markdown]
# ## Compare the reconstructions
#
# We plot the data fidelity and relative complex error against the number of
# updates. The error compares each estimate with the source-volume reference,
# including its noise. Further fitting can fit acquired noise; unmeasured
# frequencies remain unconstrained by the loss. An error increase can indicate
# overfitting, but need not occur within this fitting budget.
#
# The 300-update checkpoint shows an earlier stopping point. We chose the decoder
# settings and final budget while developing this example on this slice; these
# are not independently validated reconstruction parameters. For a new scan,
# choose a stopping criterion without using its reference image, for example
# using held-out samples or a noise estimate.
#
# All magnitude images share the same intensity scale, set by the 99.5th
# percentile of the reference magnitude for display only. The image titles show
# relative complex error (NRMSE). We retain the zero-filled baseline so we can
# compare both anatomy and error with the fitted decoder.

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

fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
vmax = float(reference.abs().quantile(0.995))
for axis, image, title in zip(
    axes.flat,
    (reference, zero_filled, checkpoints[stop_iteration], checkpoints[iterations]),
    (
        'Measured reference',
        'Zero-filled adjoint',
        f'Decoder: {stop_iteration} updates',
        f'Decoder: {iterations} updates',
    ),
    strict=True,
):
    error = float(torch.linalg.vector_norm(image - reference) / torch.linalg.vector_norm(reference))
    axis.imshow(image.squeeze().abs(), cmap='gray', vmin=0, vmax=vmax)
    axis.set_title(f'{title}\nNRMSE = {error:.3f}')
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
# This example reconstructs one retrospectively undersampled slice. It does not
# evaluate prospective acceleration, multiple coils or diagnostic performance.
# The source-volume reconstruction is a measured reference, so the reported
# error includes differences in noise as well as anatomical structure.
