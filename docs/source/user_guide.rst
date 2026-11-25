==========
User Guide
==========

MRtwo reconstructs and processes MR images with PyTorch. Data containers
store tensors with MR metadata; operators inherit from `torch.nn.Module`
and use PyTorch's batch operations.

Installation
============

Install MRtwo from `PyPI <https://pypi.org/project/mrtwo/>`_::

    pip install mrtwo

For the example notebooks, install their dependencies too::

    pip install mrtwo[notebooks]

For the development version on git/main, install from GitHub::

    pip install "git+https://github.com/fzimmermann89/mr2"


Your first reconstruction
=========================
The distribution is named ``mrtwo``; Python code imports ``mr2``.
The website follows git/main, which can contain features not yet released on
PyPI. Use the development install above when following a main-only API, or
check the documentation and examples in your installed release's source tag.

Run this example on CPU to download and reconstruct the public radial
acquisition from :doc:`_notebooks/direct_reconstruction`. It needs
``mrtwo[notebooks]`` for downloading and plotting. Allow network access and
space for the raw-data record; reconstruction time and memory use depend
on your machine. The ``with`` block keeps the temporary directory available
until reconstruction finishes.

.. code-block:: python

    import tempfile
    from pathlib import Path

    import matplotlib.pyplot as plt
    import zenodo_get
    import mr2

    with tempfile.TemporaryDirectory() as directory:
        data_folder = Path(directory)
        zenodo_get.download(record='14617082', output_dir=data_folder)
        trajectory = mr2.data.traj_calculators.KTrajectoryIsmrmrd()
        kdata = mr2.data.KData.from_file(
            data_folder / 'radial2D_402spokes_golden_angle_with_traj.h5',
            trajectory,
        )
        reconstruction = mr2.algorithms.reconstruction.DirectReconstruction(kdata)
        img = reconstruction(kdata)

    plt.imshow(img.rss()[0, 0].detach().cpu(), cmap='gray')
    plt.axis('off')
    plt.show()

The plot displays the reconstructed magnitude image. Direct reconstruction
estimates density compensation and coil sensitivity maps, then applies the
adjoint Fourier transform and combines coils. The returned `mr2.data.IData`
stores the complex image in ``img.data`` and its metadata in ``img.header``.
Here coil combination has already produced a singleton coil axis. For
display, ``rss()`` takes the magnitude and removes that axis; ``[0, 0]``
selects the first image and z plane.

Compare the result with
:doc:`_notebooks/iterative_sense_reconstruction_radial2D`, or read :doc:`data`
before selecting samples from your own acquisition. The :doc:`operators`
guide covers custom acquisition models.

Reading in raw data
-------------------
Load ISMRMRD raw data with `mr2.data.KData.from_file`, passing a
`~mr2.data.traj_calculators.KTrajectoryCalculator` for the trajectory.
The returned `~mr2.data.KData` stores k-space data with its header and
sampling coordinates. Choose a calculator that reads stored coordinates,
calculates them for a known sampling scheme, or uses a
`Pulseq <http://pulseq.github.io/>`_ sequence file. See
`~mr2.data.traj_calculators` and
:doc:`_notebooks/comparison_trajectory_calculators` for the choices.

`~mr2.data.KData.from_file` also reads MR2 storage, which includes its
trajectory. Reconstruction servers can pass an XML header and received
acquisitions to `~mr2.data.KData.from_ismrmrd` without an intermediate file.


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
Before reconstruction, you can prewhiten k-space using a noise acquisition,
remove readout oversampling, and estimate density compensation or coil
sensitivities. Fourier operators map between image space and the sampled
k-space coordinates.

Data reconstruction
-------------------
`mr2.algorithms.reconstruction` provides reconstruction classes such as
`~mr2.algorithms.reconstruction.RegularizedIterativeSENSEReconstruction`.
For a custom reconstruction, combine a `~mr2.operators.SensitivityOp` with
a `~mr2.operators.FourierOp` to model the coil signals at the measured
coordinates. You can extend the model with motion operators or signal models
and choose a solver from `mr2.algorithms.optimizers`.

See :doc:`operators` for composition and solver assumptions. The
:doc:`_notebooks/cartesian_reconstruction` and
:doc:`_notebooks/iterative_sense_reconstruction_radial2D` notebooks show
complete reconstructions.

For reconstruction with BART, use ``to_bart`` to export k-space, images, coil sensitivity maps or density compensation
weights as ``.hdr``/``.cfl`` pairs or ``.ra`` files, and ``from_file`` to import images, maps or weights without installing BART
(image and map imports require a reference ``header``).

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
checksums and cache extracted paths and sizes. ``verify=True`` also checks
cached file hashes. Downloader calls require network access for record metadata;
local dataset selection and loading work offline. Concurrent downloads to the same
output directory fail explicitly. If a process is forcibly terminated, remove
``.mr2-downloads/lock`` in that output directory before retrying. Readers should use
completed downloads; installation of a multi-file archive is not atomic.


Image processing
----------------
For quantitative parameter estimation from DICOM images, see
:doc:`_notebooks/qmri_sg_challenge_2024_t1`. The notebook fits a signal model
using a nonlinear optimizer.


Citation
========
MRtwo is a continuation of MRpro. If you use MRtwo, please cite the MRpro framework paper:

F. F. Zimmermann, P. Schuenke, C. S. Aigner, B. A. Bernhardt, M. Guastini, J. Hammacher,
H. Herthum, N. Jaitner, A. Kofler, L. Lunin, S. Martin, C. Redshaw Kranich, J. Schattenfroh,
D. Schote, Y. Wu, and C. Kolbitsch (2026).
MRpro: Open Framework for Model-Based, Learned, and Quantitative MR Imaging—Application to Low-Field MRI.
*NMR in Biomedicine*, 39(11), e70405. Zimmermann and Schuenke contributed equally.
`doi:10.1002/nbm.70405 <https://doi.org/10.1002/nbm.70405>`_
