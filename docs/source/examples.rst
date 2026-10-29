Examples
========

Each notebook includes data acquisition and plotting cells, with a Colab
badge for running it in a browser. Downloads need network access and local
storage. The B0 correction notebook generates a phantom instead.
The Colab install cell installs the PyPI release if ``mr2`` is absent. For
APIs only available on main, replace it with the GitHub installation from
:doc:`user_guide`.

Start with a direct reconstruction on CPU. Before running an iterative
reconstruction or parameter fit, check the iteration counts and tensor sizes;
runtime and memory use depend on those settings and your machine.

Getting started
---------------
The direct example reconstructs a public radial ISMRMRD acquisition,
estimating coil maps and density compensation. Use the Cartesian example to
work through the Fourier model, or the trajectory comparison to adapt a
loader to stored coordinates or a calculated trajectory. The latter includes
Pulseq and needs the associated sequence files as well as raw data.
These notebooks assume basic PyTorch knowledge.

.. toctree::
   :maxdepth: 1

   _notebooks/direct_reconstruction
   _notebooks/cartesian_reconstruction
   _notebooks/comparison_trajectory_calculators

Reconstruction and regularization
---------------------------------
Compare iterative SENSE with wavelet or TV penalties after trying the direct
reconstruction. PGD uses proximal steps, while PDHG uses primal-dual splitting.
The motion and B0 examples add corrections to the acquisition model; read
:doc:`operators` before adapting them. Most examples here download raw
acquisitions and run iterative solvers, which cost more than a single adjoint
reconstruction.

.. toctree::
   :maxdepth: 1

   _notebooks/iterative_sense_reconstruction_radial2D
   _notebooks/iterative_sense_reconstruction_with_regularization
   _notebooks/pgd_wavelet_reconstruction
   _notebooks/tv_minimization_reconstruction_pdhg
   _notebooks/motion_corrected_reconstruction
   _notebooks/b0_correction

Quantitative imaging
--------------------
The T1 challenge starts from DICOM images and fits a signal model. Cardiac
fingerprinting and continuous gradient-echo T1 mapping require familiarity
with signal models, autograd and iterative optimization. Check the download
settings and optimization loops before running these fits.

.. toctree::
   :maxdepth: 1

   _notebooks/qmri_sg_challenge_2024_t1
   _notebooks/qmri_cardiac_fingerprinting
   _notebooks/qmri_t1_mapping_with_grad_acq

Learning and differentiable optimization
----------------------------------------
The quantitative imaging notebooks use autograd to fit signal models through
a composed loss. For learned reconstruction, :doc:`operators` explains how
to keep gradients through an acquisition model and manage tuple outputs.
``mr2.nn`` provides neural network components; check each module's return
type, since some return a tensor and others return tuples.
