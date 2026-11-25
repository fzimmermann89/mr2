Data containers, dimensions and indexing
========================================

`mr2.data.KData` stores raw data with its header and trajectory;
`mr2.data.IData` stores an image tensor and image header. Both inherit from
`~mr2.data.Dataclass`. Inspect their fields with ``print(obj)`` or
``obj.items()``. Container methods apply the same selection or transformation
to nested tensor fields.

Axes, broadcasting and units
----------------------------
K-space tensors use ``(*other, coils, k2, k1, k0)``; image tensors use
``(*other, coils, z, y, x)``. Keep singleton axes, including at least one
``other`` axis. Additional batch axes go in front. For a 2D acquisition,
``k2`` is commonly singleton; ``k0`` is the readout axis.

``obj.shape`` is the joint broadcast shape of its fields, which can have
different singleton axes. For a Cartesian trajectory, ``kx`` can vary along
``k0`` while ``ky`` varies along ``k1``. Dataclasses can reduce repeated values
to singleton axes to save storage.

Physical quantities use SI units: meters, seconds and radians. Matrix sizes
and acquisition indices are counts. Trajectory coordinates use encoding-grid
units: a full Cartesian axis of length 128 normally spans -64 through 63.
`mr2.data.SpatialDimension` is a dataclass with ``z``, ``y`` and ``x`` fields.
Use `mr2.data.Rotation` for orientations; it supports rotation matrices,
quaternions and Euler angles.

A small, self-contained acquisition
-----------------------------------
Create a small container with random k-space values to try the operations
below without downloading data. These values don't simulate an MR signal.

.. code-block:: python

    import torch
    import mr2

    matrix = mr2.data.SpatialDimension(z=1, y=8, x=16)
    fov = mr2.data.SpatialDimension(z=0.005, y=0.2, x=0.2)
    header = mr2.data.KHeader(
        recon_matrix=matrix, encoding_matrix=matrix,
        recon_fov=fov, encoding_fov=fov,
    )
    header.acq_info.idx.k1 = torch.arange(8).reshape(1, 1, 1, 8, 1)
    traj = mr2.data.KTrajectory(
        kz=torch.zeros(1, 1, 1, 1, 1),
        ky=torch.arange(-4, 4).reshape(1, 1, 1, 8, 1),
        kx=torch.arange(-8, 8).reshape(1, 1, 1, 1, 16),
    )
    kdata = mr2.data.KData(
        header=header, traj=traj,
        data=torch.randn(2, 4, 1, 8, 16, dtype=torch.complex64),
    )
    print(kdata.shape)                         # torch.Size([2, 4, 1, 8, 16])
    print(kdata.traj.shape)                    # torch.Size([1, 1, 1, 8, 16])
    print(kdata.header.acq_info.idx.k1.shape)   # torch.Size([1, 1, 1, 8, 1])

Indexing the whole object
-------------------------
Indexing selects nested tensor fields in the header and trajectory as if
each field had the parent's broadcast shape. Shared axes can remain singleton,
so selecting a coil doesn't duplicate the trajectory. The result retains
non-tensor scalar metadata. Check the geometry after selection: indexing
neither recalculates the field of view nor checks whether the samples still
form a suitable acquisition.

.. code-block:: python

    first = kdata[0]                 # (1, 4, 1, 8, 16): integer keeps its axis
    cropped = kdata[..., 2:6, 4:12]   # (2, 4, 1, 4, 8)
    print(cropped.header.acq_info.idx.k1)  # selected lines 2, 3, 4, 5
    selected = kdata[..., [1, 3], :]  # (2, 4, 1, 2, 16)
    mask = torch.tensor([True, False, True, False, False, False, False, False])
    masked = kdata[..., mask, :]     # (2, 4, 1, 2, 16)
    indexed = kdata[..., torch.tensor([[1, 3], [2, 4]]), :]
    # indexed.shape is (2, 2, 4, 1, 2, 16)
    paired = kdata[..., [1, 3], [2, 4]]  # (2, 2, 4, 1, 1, 1)
    points = kdata[..., torch.eye(8, 16, dtype=torch.bool)]
    # points.shape is (8, 2, 4, 1, 1, 1): selected y/x pairs move to the front
    leading = kdata[None]            # (1, 2, 4, 1, 8, 16)

Integers and positive-step slices return tensor views. Negative slice steps
are unsupported. Advanced indexing can allocate new tensors. Multiple integer
sequences must have the same shape and select paired positions, rather than
an outer product. The selection axis goes at the front; indexed axes become
singleton. For a single integer tensor, its last axis replaces the indexed
axis and its other axes move to the front.

Only one boolean mask is allowed. Singleton mask axes mean full slices.
A mask spanning multiple non-singleton axes creates a leading selection axis
and leaves singleton axes at the indexed positions. ``None`` is allowed only
at the beginning of an index; one ellipsis fills omitted axes.

Sharing, rearranging and joining
--------------------------------
A slice can share tensor storage with the original container. Call ``clone``
before editing it in place if you need to keep the original values. Avoid
writes through broadcast views, whose elements can share storage.

.. code-block:: python

    independent = cropped.clone()
    independent.data.zero_()
    phases = kdata.rearrange(
        'batch coils k2 (phase k1) k0 -> batch phase coils k2 k1 k0', phase=2
    )
    windows = kdata.split(dim=-2, size=4, overlap=2)
    halves = kdata.split(dim=-2, size=4)
    joined = halves[0].concatenate(halves[1], dim=-2)
    other = kdata.clone()
    other.data = other.data + 1
    stacked = kdata.stack(other)  # new first axis of length 2

``rearrange`` uses einops patterns and transforms nested tensor fields too;
it may return views or allocate. ``split`` returns full windows only, dropping
an incomplete trailing window. With dilation, the span is
``(size - 1) * dilation + 1`` and stride is ``span - overlap``.
``stack`` adds a first axis; identical repeated fields can reduce back to
singleton, as elsewhere in dataclasses.
``concatenate`` joins an existing axis, broadcasts tensor fields on other axes,
joins lists, and concatenates fields that support it, such as
`mr2.data.Rotation`. It keeps remaining fields from the first object.
Check geometry, timing and other metadata yourself before joining acquisitions.

Device, precision and gradients
-------------------------------

.. code-block:: python

    double = kdata.to(device='cpu', dtype=torch.float64)
    assert double.data.dtype == torch.complex128
    assert double.traj.kx.dtype == torch.float64
    assert double.header.acq_info.idx.k1.dtype == torch.int64
    detached = kdata.detach()
    copied = kdata.clone()
    contiguous = kdata.apply(
        lambda value: value.contiguous() if isinstance(value, torch.Tensor) else value
    )

``to`` moves nested tensors and changes floating-point precision while
preserving real, complex and integer kinds. A real dtype argument sets the
precision of complex fields too, preserving their imaginary parts. ``to`` may share unchanged
tensors; ``clone`` copies tensor storage and preserves autograd connections.
``detach`` removes gradient tracking but can share storage. ``apply`` clones
and transforms fields recursively; ``apply_`` changes the object in place.
The callback receives every nested field, including trajectory and header
tensors. Return non-tensor fields unchanged when appropriate.

Storage and interchange
-----------------------

.. code-block:: python

    kdata.save_as_mr2('acquisition.mr2')
    restored = mr2.data.KData.from_file('acquisition.mr2')

MR2 stores dataclass fields in HDF5, including the trajectory and nested
headers. Reload the file with the matching class's ``from_file`` method.
The file contains tensor values without their computation graph.

For scanner raw data, use ``KData.from_file(path, trajectory_calculator)`` for
ISMRMRD HDF5, or ``KData.from_ismrmrd`` for an XML header and acquisitions.
``KData.to_file`` writes ISMRMRD. ``IData.from_file`` reads MR2, DICOM files or
folders, and BART arrays; BART image imports require a reference header.
``IData.to_dicom_folder`` and ``IData.to_nifti`` export images. NIfTI import is
not provided by ``IData.from_file``. K-space, images, sensitivity maps and density
weights also support BART export through ``to_bart``. Consult each method for
format-specific shape and metadata requirements.
