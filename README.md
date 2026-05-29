<h1 align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/fzimmermann89/mr2/refs/heads/main/docs/source/_static/logo_white.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/fzimmermann89/mr2/refs/heads/main/docs/source/_static/logo.svg">
  <img src="https://raw.githubusercontent.com/fzimmermann89/mr2/refs/heads/main/docs/source/_static/logo.svg" alt="MRtwo logo" width="50%">
</picture>

</h1><br>

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/mrtwo/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Coverage Bagde](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/fzimmermann89/f688ae5c6e8daec44ac7f8fc4067e93f/raw/coverage.json)](https://github.com/fzimmermann89/mr2/actions?query=workflow%3A%22%22Report+PyTest%22%22)
[![arXiv](https://img.shields.io/badge/arXiv-2507.23129-b31b1b.svg)](https://arxiv.org/abs/2507.23129)
[![FAIR checklist badge](https://fairsoftwarechecklist.net/badge.svg)](https://fairsoftwarechecklist.net/v0.2?f=31&a=32113&i=21322&r=133)

MR data processing and image reconstruction.

This project is a faster moving continuation/fork of [MRpro](https://github.com/PTB-MR/mrpro/).
Some of the additional features of MRtwo will eventually get backported to MRpro, all new features of MRpro will be included in MRtwo.

In most cases, you can replace `mrpro` by `mr2` in you code and everything works.

- **Source code:** <https://github.com/fzimmermann89/mr2>
- **Documentation:** <https://fzimmermann89.github.io/mr2/>
- **Bug reports:** <https://github.com/fzimmermann89/mr2/issues>
- **Try it out:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fzimmermann89/mr2)

## Changes compared to MRpro

MRtwo keeps the MRpro reconstruction and data model, but adds several larger extensions. Some of these features or fixes have already been backported to MRpro, and others might be backported in the future in the same or similar form.

- **Bloch-McConnell simulation:** MRtwo includes a Bloch-McConnell simulation framework for multi-pool magnetization models with exchange, relaxation, chemical shift, off-resonance, and RF pulse effects. This enables quantitative MRI and CEST-style workflows that need coupled exchanging pools rather than single-pool signal models.
- **MR2 file format:** MRtwo dataclasses can be saved and restored with the MR2 HDF5-based format using `save_as_mr2` and `from_mr2`. This gives native round-tripping for objects such as `KData`, `IData`, headers, trajectories, tensors, rotations, and nested dataclasses without converting through ISMRMRD or DICOM.
- **Image registration:** MRtwo adds an image-registration module with affine, spline, and correlation-based registration utilities, including subpixel correlation registration, for building registration and motion-estimation workflows directly in the MRtwo operator/data ecosystem.
- **Neural network module:** `mr2.nn` provides MRI-oriented PyTorch components, including N-dimensional convolution and pooling helpers, normalization layers, residual blocks, FiLM conditioning, Fourier features, positional encodings, attention blocks, data-consistency layers, and ready-to-use models such as UNet, BasicCNN, MLP, VAE, Restormer, SwinIR, Uformer, HourglassTransformer, and DiT.
- **Additional reconstruction and qMRI tools:** MRtwo adds differentiable dictionary matching, ADMM and BiCG solvers, ROVir coil compression, improved coil-sensitivity estimation with optional extrapolation, and smaller helper operators such as lookup-table interpolation.
- **Bug fixes and robustness improvements:** MRtwo includes fixes and follow-up improvements for CUDA execution, reconstruction setup, slice projection, interpolation, dictionary matching, PCA compression, tensor/device handling, and compatibility with newer PyTorch and ISMRMRD versions.

## Main features
- **Data handling** Custom dataclasses for fast data subsetting, sorting, rearranging
- **Neural Network Block** Common blocks and networks used for ML based MRI reconstruction
- **ISMRMRD support** MRtwo supports [ismrmrd-format](https://ismrmrd.readthedocs.io/en/latest/) for MR raw data.
- **PyTorch** All data containers utilize PyTorch tensors to ensure easy integration in PyTorch-based network schemes.
- **Cartesian and non-Cartesian trajectories** MRtwo can reconstruct data obtained with Cartesian and non-Cartesian (e.g. radial, spiral...) sapling schemes. MRtwo automatically detects if FFT or nuFFT is required to reconstruct the k-space data.
- **Pulseq support** If the data acquisition was carried out using a [pulseq-based](http://pulseq.github.io/) sequence, the seq-file can be provided to MRtwo and the used trajectory is automatically calculated.
- **Signal models** A range of different MR signal models are implemented (e.g. T1 recovery, WASABI).
- **Regularized image reconstruction** Regularized image reconstruction algorithms including Wavelet-based compressed sensing or total variation regularized image reconstruction are available.

## Examples

In the following, we show some code snippets to highlight the use of MRtwo. Each code snippet only shows the main steps. A complete working notebook can be found in the provided link.

### Simple reconstruction

Read the data and trajectory and reconstruct an image by applying a density compensation function and then the adjoint of the Fourier operator and the adjoint of the coil sensitivity operator.

```python
# Read the trajectory from the ISMRMRD file
trajectory = mr2.data.traj_calculators.KTrajectoryIsmrmrd()
# Load in the Data from the ISMRMRD file
kdata = mr2.data.KData.from_file(data_file.name, trajectory)
# Perform the reconstruction
reconstruction = mr2.algorithms.reconstruction.DirectReconstruction(kdata)
img = reconstruction(kdata)
```

Full example: <https://github.com/fzimmermann89/mr2/blob/main/examples/scripts/direct_reconstruction.py>

### Estimate quantitative parameters

Quantitative parameter maps can be obtained by creating a functional to be minimized and calling a non-linear solver such as ADAM.

```python
# Define signal model
model = MagnitudeOp() @ InversionRecovery(ti=idata_multi_ti.header.ti)
# Define loss function and combine with signal model
mse = MSE(idata_multi_ti.data.abs())
functional = mse @ model
[...]
# Run optimization
params_result = adam(functional, [m0_start, t1_start], n_iterations=n_iterations, learning_rate=learning_rate)
```

Full example: <https://github.com/fzimmermann89/mr2/blob/main/examples/scripts/qmri_sg_challenge_2024_t1.py>

### Pulseq support

The trajectory can be calculated directly from a provided pulseq-file.

```python
# Read raw data and calculate trajectory using KTrajectoryPulseq
kdata = KData.from_file(data_file.name, KTrajectoryPulseq(seq_path=seq_file.name))
```

Full example: <https://github.com/fzimmermann89/mr2/blob/main/examples/scripts/comparison_trajectory_calculators.py>

## Development
 ``` pip install -e ".[dev]" ```
