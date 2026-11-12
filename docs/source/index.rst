:hide-toc:

.. MRtwo documentation

.. role:: mr2-brand

.. rst-class:: mr2-landing

MR image reconstruction and processing with :mr2-brand:`MRtwo`
==============================================================

.. container:: mr2-hero

   .. rst-class:: mr2-eyebrow

   PyTorch + MRI

   .. rst-class:: mr2-lead

   An open-source Python package for MR image reconstruction and processing with PyTorch.
   Data, physics models, optimizers and neural networks share one interface:
   start with a high-level reconstruction and replace its components as your application requires.

   .. container:: mr2-buttons

      .. button-ref:: user_guide
         :ref-type: doc
         :color: primary

         Get started →

      .. button-ref:: examples
         :ref-type: doc
         :color: secondary
         :outline:

         Examples

      .. button-ref:: api
         :ref-type: doc
         :color: secondary
         :outline:

         API

      .. button-link:: https://github.com/fzimmermann89/mr2
         :color: secondary
         :outline:

         GitHub

      .. button-link:: https://colab.research.google.com/github/fzimmermann89/mr2
         :color: secondary
         :outline:

         Try in Colab

   .. card:: Example: direct reconstruction
      :class-card: mr2-code-card

      .. code-block:: python

         import mr2

         # Tell mr2 the type of trajectory
         trajectory = mr2.data.traj_calculators.KTrajectoryIsmrmrd()

         # Load the data from the ISMRMRD file
         kdata = mr2.data.KData.from_file(filename, trajectory)

         # Perform the reconstruction
         recon = mr2.algorithms.reconstruction.DirectReconstruction(kdata)
         img = recon(kdata)

      +++

      :doc:`View full example → <_notebooks/direct_reconstruction>`

Start here
----------

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: mr2-cards

   .. grid-item-card:: Install and reconstruct
      :link: user_guide
      :link-type: doc

      Install the package, download an acquisition and display your first reconstruction.

   .. grid-item-card:: Work with your data
      :link: data
      :link-type: doc

      Axes, units, broadcasting, indexing and file formats of the data containers.

   .. grid-item-card:: Learn by application
      :link: examples
      :link-type: doc

      Complete notebooks for reconstruction, quantitative imaging and learning, grouped by task.

   .. grid-item-card:: Find an API
      :link: api
      :link-type: doc

      Classes, functions and call signatures, generated from the docstrings.

   .. grid-item-card:: Contribute
      :link: contributor_guide
      :link-type: doc

      Code, tests and documentation conventions.

   .. grid-item-card:: Questions
      :link: faq
      :link-type: doc

      Frequently asked questions and known pitfalls.

Main features
-------------

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: mr2-cards mr2-features

   .. grid-item-card:: Standard file formats
      :link: _autosummary/mr2.data
      :link-type: doc

      ISMRMRD raw data, DICOM images and BART files are read into data containers that keep the MR metadata.

   .. grid-item-card:: PyTorch integration
      :link: _autosummary/mr2.data
      :link-type: doc

      All data containers hold PyTorch tensors, so reconstructions run on the GPU and support autograd.

   .. grid-item-card:: Cartesian and non-Cartesian
      :link: _autosummary/mr2.operators.FourierOp
      :link-type: doc

      Radial, spiral and other trajectories. The Fourier operator chooses FFT or NUFFT automatically.

   .. grid-item-card:: Pulseq support
      :link: _autosummary/mr2.data.traj_calculators.KTrajectoryPulseq
      :link-type: doc

      For pulseq-based sequences, the trajectory is calculated directly from the seq-file.

   .. grid-item-card:: Signal models
      :link: _autosummary/mr2.operators.models
      :link-type: doc

      Inversion and saturation recovery, MOLLI, WASABI, EPG, Bloch-McConnell and MR fingerprinting models.

   .. grid-item-card:: Regularized reconstruction
      :link: _autosummary/mr2.algorithms.reconstruction
      :link-type: doc

      Iterative SENSE, wavelet and total-variation regularization and plug-and-play ADMM, built on PGD, PDHG, ADMM, CG and LBFGS solvers.

   .. grid-item-card:: Coil sensitivities
      :link: _autosummary/mr2.algorithms.csm
      :link-type: doc

      Walsh, Inati and ESPIRiT estimation of coil sensitivity maps.

   .. grid-item-card:: Neural network building blocks
      :link: _autosummary/mr2.nn
      :link-type: doc

      U-Net, transformer and data-consistency layers for learned and unrolled reconstruction.

   .. grid-item-card:: Datasets and phantoms
      :link: _autosummary/mr2.phantoms
      :link-type: doc

      fastMRI, M4Raw and BrainWeb datasets and analytic phantoms for training and testing.

Installation
------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: mr2-cards

   .. grid-item-card:: Install from PyPI

      .. code-block:: bash

         pip install mrtwo

      The PyPI distribution is ``mrtwo``, the Python import is ``mr2``.
      Use ``pip install "mrtwo[notebooks]"`` to run the example notebooks locally.
      MRtwo is a continuation of `MRpro <https://github.com/PTB-MR/mrpro/>`_ with a faster-moving development API.

   .. grid-item-card:: Try in Colab
      :link: https://colab.research.google.com/github/fzimmermann89/mr2

      Every example notebook can be opened in Google Colab, without installing anything on your machine.

Resources
---------

.. grid:: 1 2 4 4
   :gutter: 3
   :class-container: mr2-cards

   .. grid-item-card:: User guide
      :link: user_guide
      :link-type: doc

      Installation, first steps and citation.

   .. grid-item-card:: Source code
      :link: https://github.com/fzimmermann89/mr2

      Browse the code and contribute on GitHub.

   .. grid-item-card:: Issue tracker
      :link: https://github.com/fzimmermann89/mr2/issues

      Report bugs and request features.

   .. grid-item-card:: Cite MRtwo
      :link: https://doi.org/10.1002/nbm.70405

      MRpro framework paper, NMR in Biomedicine (2026).

.. toctree::
   :hidden:
   :maxdepth: 2

   user_guide
   data
   examples
   contributor_guide
   api
   faq
