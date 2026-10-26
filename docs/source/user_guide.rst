==========
User Guide
==========

MRtwo is a MR image reconstruction and processing framework specifically developed to work well with pytorch.
The data classes utilize `torch.Tensor` for storing data such as MR raw data or reconstructed image data.
Operators are implemented as `torch.nn.Module` where possible batch parallelisation of pytorch is utilized to speed up image reconstruction.

Installation
============

MRtwo is available on `pypi <https://pypi.org/project/mrtwo/>`_ and can be installed with::

    pip install mrtwo

To install additional dependencies used in our example notebooks, use::

    pip install mrtwo[notebooks]

You can also install the latest development directly from github using::

    pip install "git+https://github.com/fzimmermann89/mr2"


Usage
=====
MRtwo is designed to work directly from MR raw data using the `MRD <https://ismrmrd.readthedocs.io/en/latest/>`_ data format.

A basic pipeline would contain the following steps:

* Reading in raw data
* Preparation for reconstruction
* Data reconstruction
* Image processing

.. |colab-badge| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/fzimmermann89/mr2

The following provides some basic information about these steps.
For more detailed information please have a look at the :doc:`examples`.
You can easily launch notebooks via the |colab-badge| badge and give the notebooks a try!

Reading in raw data
-------------------
Reading in raw data from a MRD file works by creating a `mr2.data.KData` object and using the class method `~mr2.data.KData.from_file`.
`~mr2.data.KData` contains the raw k-space data, the header information obtained from the MRD file and the k-space trajectory.
To ensure the trajectory is calculated correctly, a `~mr2.data.traj_calculators.KTrajectoryCalculator` needs to be provided.
The trajectory can either be calculated based on MRtwo functionality (e.g. for a 2D radial sampling scheme), read out
from MRD or calculated from a `pulseq <http://pulseq.github.io/>`_ file. See `~mr2.data.traj_calculators`
for available trajectory calculators and :doc:`_notebooks/comparison_trajectory_calculators` for an example.


.. note::
    In MRtwo, we use the convention ``(z, y, x)`` for spatial dimensions and ``(k2, k1, k0)`` for k-space dimensions.
    Here, `k0` is the readout direction, `k1` and `k2` are phase encoding directions.
    The full shape of a multi-slice 2D k-space data for example is ``(other, coils, 1, k1, k0)`` where `other` will be the different slices.
    In general, `other` can be any number of additional dimensions. All our data tensors will contain at least 5 dimensions, including
    at least one `other` and always including a `coils` and `k0` dimension -- even for properties that must be the same for all coils, such
    as the trajectory. For these tensors, the `coils` dimension will thus always be ``1`` (but still be present).

.. note::
    The trajectory is expected to be defined within the space of the `encoding_matrix`, e.g. if the
    `encoding_matrix` is defined as ``(z=1, y=256, x=256)``, then a fully sampled Cartesian trajectory without partial
    echo or partial Fourier is expected to be within ``[-128, 127]`` along both readout and phase encoding.

Preparation for reconstruction
------------------------------
MRtwo provides a range of functionality to prepare the data for image reconstruction such as:

* Noise prewhiting
* Removal of oversampling along readout direction
* Calculation of the density compensation function
* Estimation of coil sensitivity maps
* Fourier transformation

Data reconstruction
-------------------
MRtwo provides a flexible framework for MR image reconstruction. We provide some high level functions for commonly used
reconstruction algorithms in `mr2.algorithms.reconstruction`, such as
`~mr2.algorithms.reconstruction.RegularizedIterativeSENSEReconstruction`. We also provide all building blocks to
create custom reconstruction algorithms and do manual reconstructions.

As a first step for a new reconstruction, an acquisition model consisting of linear and non-linear operators can be created.
A simply acquisition model could consist of a `~mr2.operators.SensitivityOp` describing the effect of different
receiver coils and `~mr2.operators.FourierOp` describing the transform from image space to k-space taking the sampling scheme
(trajectory) into account. Additional operators describing transformations due to physiological motion or
MR signal models can be added. See `~mr2.operators` for a list of available operators.
All operators take one or more tensors as input and return a tuple of one or more tensors as output.
Operators can be chained using ``@`` to form a full acquisition model. We also support addition, multiplication, etc.
between operators.

Based on the acquisition model either a suitable optimizer from `mr2.algorithms.optimizers` can be selected
or a new optimizer using pytorch functions can be created.

See for examples  :doc:`_notebooks/cartesian_reconstruction`, :doc:`_notebooks/direct_reconstruction`, and :doc:`_notebooks/iterative_sense_reconstruction_radial2D`

Untrained ConvDecoder reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`~mr2.algorithms.reconstruction.ConvDecoderReconstruction` fits a randomly initialized
convolutional generator directly to acquired complex k-space, using the existing Fourier
and sensitivity operators. For example, with ``kdata`` and estimated ``csm``:

.. code-block:: python

    reconstruction = mr2.algorithms.reconstruction.ConvDecoderReconstruction(
        kdata, csm=csm, max_iterations=1000, seed=0
    )
    image = reconstruction(kdata)

Each call fits fresh generators and returns complex
``IData`` with a header derived from the input. The random latent stays fixed while
Adam updates the weights. The seed controls initialization without changing global
random state. The default latent shape is ``(8, 4)``; smaller images need a smaller
``latent_shape``. Its length selects the generator dimensionality: one entry for
1D, two for slice-wise 2D (the default), or three for 3D volumes. The first
upsampling stage needs at least two pixels. Memory use grows with the number of generators.

The reusable `~mr2.nn.nets.ConvDecoder` network accepts and returns ordinary real
channels and supports 1D, 2D and 3D. Wrap it with `~mr2.nn.ComplexAsChannel` for
complex output. To resume custom fitting, restore the network ``state_dict``,
optimizer ``state_dict`` and fixed latent; reconstruct the network with the same
configuration. The reconstruction class intentionally starts fresh on each call.

This variant uses nearest-neighbor upsampling, convolution, ReLU and stateless affine
InstanceNorm, followed by a linear complex object output. The authors' ConvDecoder
uses BatchNorm; their paper describes both object-output and coil-output formulations:
`Darestani and Heckel (2021) <https://doi.org/10.1109/TCI.2021.3097596>`_.
Sensitivity maps are required for multicoil object reconstruction. With ``noise``
prewhitening enabled, supplied maps must already match the whitened coil basis;
automatic estimation uses whitened data.

``max_iterations`` sets the fitting budget. An optional ``callback`` receives
``iteration_number`` (zero-based) and ``solution=(image_tensor,)`` after each update;
returning ``False`` stops fitting. Fitting loss alone does not protect against noise
overfitting, so the stopping rule must be chosen for the acquisition and noise level.

Low-field raw datasets
----------------------
``mr2.phantoms.IsmrmrdDataset`` loads local HDF5 ISMRMRD collections using
``KData.from_file``. Files are naturally ordered, and the default trajectory
assumes Cartesian sampling. Pass a trajectory explicitly for other sampling schemes.
Serialized ISMRMRD streams are not supported by this dataset interface.

``mr2.phantoms.a4im_lowfield`` provides ``OSI2Dataset``, ``IBTDataset``,
``I3MDataset``, and ``FreeMaxDataset`` with scanner-specific file selectors.
Construction never downloads data. For example::

    from mr2.phantoms.a4im_lowfield import download_ibt, IBTDataset

    path = download_ibt(subjects=1)  # explicit network access
    dataset = IBTDataset(path, acquisition='t2w', individual_averages=True)
    kdata = dataset[0]

Each item is one file, except IBT individual averages, where matching files become
one scan with an added leading average axis. Filename repetition labels, including
noncontiguous labels, populate ``header.acq_info.idx.average`` even when the
in-file indices are zero. With ``individual_averages=True``, only ``averageN``
files are selected: T1 mapping scans and other files without that suffix are
excluded. T2 mapping contrasts with individual averages remain separate items.
Shared geometry, sampling and timing must agree within an average group.
The known truncated FreeMax Subject9 right PD scan is excluded with a warning;
other loading failures propagate to the caller.

Use ``torch.utils.data.DataLoader(dataset, batch_size=None)`` for unbatched
iteration, or supply a custom collation function. Download helpers check Zenodo
checksums and cache extracted paths and sizes. ``verify=True`` additionally checks
cached file hashes. Downloader calls require network access for record metadata;
local dataset selection and loading work offline. Concurrent downloads to the same
output directory fail explicitly. If a process is forcibly terminated, remove
``.mr2-downloads/lock`` in that output directory before retrying. Readers should use
completed downloads; installation of a multi-file archive is not atomic.


Image processing
----------------
Further processing of the reconstructed data such as quantitative parameter estimation is available.
Our examples contain a notebook showing how to read in DICOM images and perform qMRI parameter estimation using
a non-linear optimizer: :doc:`_notebooks/qmri_sg_challenge_2024_t1`,


Citation
========
We are currently preparing a manuscript for MRtwo. In the meantime, please cite:

Zimmermann, F. F., Schuenke, P., Brahma, S., Guastini, M., Hammacher, J., Kofler, A., Redshaw Kranich, C., Lunin, L., Martin, S., Schote, D., & Kolbitsch, C. (2024).
MRtwo - PyTorch-based MR image reconstruction and processing package
`10.5281/zenodo.14509598 <https://doi.org/10.5281/zenodo.14509598>`_
