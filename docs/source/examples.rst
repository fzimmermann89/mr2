Examples
========

Start with a complete notebook, then adapt its acquisition model or solver.
Each notebook includes its own data acquisition and plotting cells and a
Colab launch badge. Downloads need network access on first use; allow local
storage for the selected record. Small direct reconstructions are a useful
CPU starting point. Iterative, motion and parameter-fitting examples take
longer; check their iteration counts and tensor sizes before scaling up.
The B0 correction example generates its own phantom rather than downloading
a raw dataset. There is no fixed runtime across machines, and Colab GPU
metadata alone does not mean a GPU is required.

Getting started
---------------
The direct example uses a public radial ISMRMRD acquisition and estimates
coil maps and density compensation. The Cartesian example introduces the
Fourier model step by step. Start here with basic PyTorch knowledge.
The trajectory comparison is for adapting a loader to stored, calculated or
Pulseq trajectories; it requires the associated raw data and sequence files.

.. toctree::
   :maxdepth: 1

   _notebooks/direct_reconstruction
   _notebooks/cartesian_reconstruction
   _notebooks/comparison_trajectory_calculators

Reconstruction and regularization
---------------------------------
After the first reconstruction, compare iterative SENSE with wavelet or
TV penalties. Most of these examples download raw acquisitions and run iterative
solvers, so expect more work than an adjoint reconstruction. PGD introduces
proximal steps; PDHG introduces primal-dual splitting. Motion and B0 examples
add acquisition-model corrections and are best read after :doc:`operators`.

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
with signal models, autograd and iterative optimization. Their datasets and
parameter-fitting loops are larger than the introductory examples; review
the download and optimization settings before running them.

.. toctree::
   :maxdepth: 1

   _notebooks/qmri_sg_challenge_2024_t1
   _notebooks/qmri_cardiac_fingerprinting
   _notebooks/qmri_t1_mapping_with_grad_acq

Learning and differentiable optimization
----------------------------------------
Signal-model fitting above demonstrates autograd with a composed loss and
an optimizer. Read :doc:`operators` for tuple outputs and gradient/device
usage, and `mr2.nn` for tensor-returning neural network components. Learned
reconstruction and differentiable-solver tutorials are being developed
separately; they will be linked here when available.
