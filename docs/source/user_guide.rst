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

Creating KData from received acquisitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`~mr2.data.KData.from_ismrmrd` accepts a parsed header or XML string/bytes and an iterable of
acquisition objects. Reconstruction servers such as OpenRecon can use their received metadata
and acquisitions directly, without writing an intermediate MRD file:

.. code-block:: python

    from mr2.data import KData
    from mr2.data.traj_calculators import KTrajectoryCartesian

    kdata = KData.from_ismrmrd(metadata, acquisitions, KTrajectoryCartesian())

The constructor filters, crops and sorts acquisitions without modifying the input objects.
Servers can pass each reconstruction batch with the same header, choosing an appropriate
``acquisition_filter_criterion``. Fallback metadata can be supplied through ``header_defaults``;
``header_overwrites`` takes precedence over defaults and measurement metadata.

Reading serialized streams
~~~~~~~~~~~~~~~~~~~~~~~~~~
`~mr2.data.KData.from_file` accepts HDF5 paths/file-like objects and serialized ISMRMRD stream
paths. Use `~mr2.data.KData.from_ismrmrd_stream` to select one measurement from a binary stream,
or `~mr2.data.KData.iter_ismrmrd_stream` to process successive measurements. For example,
``python read_stream.py < data.ismrmrd`` can run this script:

.. code-block:: python

    import sys

    from mr2.data import KData
    from mr2.data.traj_calculators import KTrajectoryIsmrmrd

    for kdata in KData.iter_ismrmrd_stream(sys.stdin.buffer, KTrajectoryIsmrmrd()):
        print(kdata.data.shape)

Measurements end at the next header or CLOSE; acquisition flags do not define batch boundaries.
Caller-owned streams remain open. See the reader docstrings for selection and completion requirements.


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
