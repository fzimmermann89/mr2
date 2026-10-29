"""Reading and writing BART arrays.

BART [BART]_ stores complex64 arrays either as a pair of files, a text header (``.hdr``) and the
column-major data (``.cfl``), or as a single raw array file (``.ra``) [BARTFMT]_.
These functions do not require BART to be installed.

Notes on using the files with the BART command line:

- BART expects the name without suffix for ``.hdr``/``.cfl`` pairs, e.g. ``bart pics kspace sens recon``
  for files written as ``kspace.cfl`` and ``sens.cfl``. ``.ra`` files are passed with their suffix.
- ``bart pics`` does not accept data along `BartDimension.BATCH`, the default for flattened `other`
  dimensions, and ``bart ecalib`` returns a single set of maps for all non-spatial dimensions.
  Use BART's loop option, e.g. ``bart -l 32768 -e <n> pics ...`` loops over `n` entries of the batch
  dimension (flag ``1 << 15``), or choose other BART dimensions with `other_dims`.

References
----------
.. [BART] https://codeberg.org/mrirecon/bart
.. [BARTFMT] https://codeberg.org/mrirecon/bart/src/branch/master/src/misc/io.c
"""

import math
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import numpy as np
import torch

from mr2.data.enums import BartDimension

BART_SUFFIXES = ('.cfl', '.hdr', '.ra')
"""File suffixes of BART arrays."""

_N_BART_DIMS = len(BartDimension)
_RA_MAGIC = 0x7961727261776172
_RA_FLAG_BIG_ENDIAN = 1
_RA_TYPE_COMPLEX = 4
_RA_HEADER_DTYPE = np.dtype('<u8')
_RA_HEADER_ELEMENTS = 6
_RA_MAX_DIMS = 100
_ELEMENT_DTYPE = np.dtype('<c8')

BART_IMAGE_DIMS = (BartDimension.COIL, BartDimension.PHS2, BartDimension.PHS1, BartDimension.READ)
"""BART dimensions of images `(coils, z, y, x)` and Cartesian k-space `(coils, k2, k1, k0)`."""

BART_NONCARTESIAN_DIMS = (BartDimension.COIL, BartDimension.PHS2, BartDimension.PHS1)
"""BART dimensions of non-Cartesian k-space samples `(coils, readouts, k0)`."""

BART_TRAJECTORY_DIMS = (BartDimension.PHS2, BartDimension.PHS1, BartDimension.READ)
"""BART dimensions of non-Cartesian trajectories `(readouts, k0, kxyz)`."""

_SPATIAL_AND_COIL = frozenset({BartDimension.READ, BartDimension.PHS1, BartDimension.PHS2, BartDimension.COIL})


def is_bart_file(filename: str | PathLike) -> bool:
    """Check if a filename refers to a BART array by its suffix."""
    return Path(filename).suffix.lower() in BART_SUFFIXES


def _cfl_paths(filename: Path) -> tuple[Path, Path]:
    """Get the header and data path of a ``.hdr``/``.cfl`` pair."""
    return filename.with_suffix('.hdr'), filename.with_suffix('.cfl')


def _check_suffix(filename: Path) -> None:
    if filename.suffix.lower() not in BART_SUFFIXES:
        raise ValueError(f'BART files must end in one of {BART_SUFFIXES}, got {filename.name}.')


def read_bart(filename: str | PathLike) -> torch.Tensor:
    """Read a BART array.

    Parameters
    ----------
    filename
        Path to a ``.cfl`` or ``.hdr`` file of a ``.hdr``/``.cfl`` pair, or to a ``.ra`` file.

    Returns
    -------
        complex64 tensor with the dimensions in BART order, i.e., ``data[i0, i1, ...]``
        with ``i0`` being BART dimension 0. Trailing singleton dimensions are kept as stored.
    """
    filename = Path(filename)
    _check_suffix(filename)
    if filename.suffix.lower() == '.ra':
        with filename.open('rb') as file:
            header = np.fromfile(file, dtype=_RA_HEADER_DTYPE, count=_RA_HEADER_ELEMENTS)
            if header.size != _RA_HEADER_ELEMENTS:
                raise ValueError(f'Incomplete .ra header in {filename}.')
            magic, flags, element_type, element_size, data_size, n_dims = (int(value) for value in header)
            if magic != _RA_MAGIC:
                raise ValueError(f'{filename} is not a .ra file.')
            if flags & _RA_FLAG_BIG_ENDIAN:
                raise ValueError(f'Big endian .ra files are not supported: {filename}.')
            if element_type != _RA_TYPE_COMPLEX or element_size != _ELEMENT_DTYPE.itemsize:
                raise ValueError(f'Only complex float .ra files are supported: {filename}.')
            if n_dims > _RA_MAX_DIMS:
                raise ValueError(f'Too many dimensions ({n_dims}) in {filename}.')
            dims = [int(dim) for dim in np.fromfile(file, dtype=_RA_HEADER_DTYPE, count=n_dims)]
            if len(dims) != n_dims or data_size != math.prod(dims) * _ELEMENT_DTYPE.itemsize:
                raise ValueError(f'Inconsistent .ra header in {filename}.')
            data = np.fromfile(file, dtype=_ELEMENT_DTYPE, count=math.prod(dims))
            if data.size != math.prod(dims) or file.read(1):
                raise ValueError(f'Data size does not match the dimensions {dims} in {filename}.')
    else:
        header_path, data_path = _cfl_paths(filename)
        lines = header_path.read_text().splitlines()
        try:
            dims = [int(dim) for dim in lines[lines.index('# Dimensions') + 1].split()]
        except (ValueError, IndexError):
            raise ValueError(f'No valid dimensions found in {header_path}.') from None
        data = np.fromfile(data_path, dtype=_ELEMENT_DTYPE)
        if data.size != math.prod(dims):
            raise ValueError(
                f'{data_path} contains {data.size} values, but the dimensions {dims} require {math.prod(dims)}.'
            )
    # column-major storage: the first dimension varies fastest
    tensor = torch.from_numpy(data.astype(np.complex64, copy=False)).reshape(dims[::-1])
    return tensor.permute(*range(tensor.ndim - 1, -1, -1))


def write_bart(filename: str | PathLike, data: torch.Tensor) -> None:
    """Write a tensor as BART array.

    Parameters
    ----------
    filename
        A ``.cfl`` or ``.hdr`` suffix writes a ``.hdr``/``.cfl`` pair, a ``.ra`` suffix writes a single file.
    data
        Tensor with dimensions in BART order. It is converted to complex64.
    """
    filename = Path(filename)
    _check_suffix(filename)
    dims = tuple(data.shape)
    # column-major storage: the first dimension varies fastest
    values = data.detach().to(device='cpu', dtype=torch.complex64).resolve_conj().resolve_neg()
    values_np = values.permute(*range(values.ndim - 1, -1, -1)).contiguous().numpy()
    if filename.suffix.lower() == '.ra':
        header = np.array(
            [_RA_MAGIC, 0, _RA_TYPE_COMPLEX, _ELEMENT_DTYPE.itemsize, values_np.nbytes, len(dims), *dims],
            dtype=_RA_HEADER_DTYPE,
        )
        with filename.open('wb') as file:
            file.write(header.tobytes())
            file.write(values_np.astype(_ELEMENT_DTYPE, copy=False).tobytes())
    else:
        header_path, data_path = _cfl_paths(filename)
        header_path.write_text('# Dimensions\n' + ' '.join(str(dim) for dim in dims) + '\n')
        values_np.astype(_ELEMENT_DTYPE, copy=False).tofile(data_path)


def _validate_other_dims(
    other_dims: Sequence[BartDimension], reserved: Sequence[BartDimension]
) -> tuple[BartDimension, ...]:
    """Check that the BART dimensions for the MR2 `other` dimensions are valid and unique."""
    other_dims = tuple(other_dims)
    if not all(isinstance(dim, BartDimension) for dim in other_dims):
        raise TypeError('other_dims must contain BartDimension members.')
    if invalid := [dim.name for dim in other_dims if dim in _SPATIAL_AND_COIL or dim in reserved]:
        raise ValueError(f'BART dimensions {invalid} are reserved and cannot be used in other_dims.')
    if len(set(other_dims)) != len(other_dims):
        raise ValueError(f'other_dims contains duplicates: {[dim.name for dim in other_dims]}.')
    return other_dims


def to_bart_layout(
    data: torch.Tensor,
    trailing_dims: Sequence[BartDimension],
    other_dims: Sequence[BartDimension] | None = None,
) -> torch.Tensor:
    """Arrange an MR2 tensor in BART dimension order.

    Parameters
    ----------
    data
        Tensor with shape `(*other, *trailing)`.
    trailing_dims
        BART dimensions of the trailing MR2 dimensions.
    other_dims
        BART dimensions of the leading MR2 `other` dimensions, in tensor order.
        `None` flattens all `other` dimensions into `BartDimension.BATCH`.

    Returns
    -------
        Tensor with all BART dimensions.
    """
    trailing_dims = tuple(trailing_dims)
    n_other = data.ndim - len(trailing_dims)
    if n_other < 0:
        raise ValueError(f'Expected at least {len(trailing_dims)} dimensions, got shape {tuple(data.shape)}.')
    if other_dims is None:
        if n_other > 0:
            data = data.reshape(-1, *data.shape[n_other:])
        other_dims = (BartDimension.BATCH,) if n_other > 0 else ()
    else:
        other_dims = _validate_other_dims(other_dims, trailing_dims)
        if len(other_dims) != n_other:
            raise ValueError(
                f'other_dims has {len(other_dims)} entries, but the data has {n_other} other dimensions '
                f'with shape {tuple(data.shape[:n_other])}.'
            )
    dims = (*other_dims, *trailing_dims)
    order = sorted(range(len(dims)), key=lambda axis: dims[axis])
    shape = [1] * _N_BART_DIMS
    for axis in order:
        shape[dims[axis]] = data.shape[axis]
    return data.permute(order).reshape(shape)


def from_bart_layout(
    data: torch.Tensor,
    trailing_dims: Sequence[BartDimension],
    other_dims: Sequence[BartDimension] | None = None,
) -> torch.Tensor:
    """Arrange a tensor in BART dimension order as MR2 tensor.

    Parameters
    ----------
    data
        Tensor with dimensions in BART order.
    trailing_dims
        BART dimensions of the trailing MR2 dimensions.
    other_dims
        BART dimensions to use as leading MR2 `other` dimensions, in tensor order.
        `None` uses `BartDimension.BATCH` as single `other` dimension.
        All BART dimensions that are neither in `trailing_dims` nor in `other_dims` must be singleton.

    Returns
    -------
        Tensor with shape `(*other, *trailing)`.
    """
    trailing_dims = tuple(trailing_dims)
    other_dims = (BartDimension.BATCH,) if other_dims is None else _validate_other_dims(other_dims, trailing_dims)
    if data.ndim > _N_BART_DIMS:
        if any(size != 1 for size in data.shape[_N_BART_DIMS:]):
            raise ValueError(f'Only {_N_BART_DIMS} BART dimensions are supported, got shape {tuple(data.shape)}.')
        data = data.reshape(data.shape[:_N_BART_DIMS])
    data = data.reshape(*data.shape, *[1] * (_N_BART_DIMS - data.ndim))
    dims = (*other_dims, *trailing_dims)
    unused = [axis for axis in range(_N_BART_DIMS) if axis not in dims]
    if unmapped := [BartDimension(axis).name for axis in unused if data.shape[axis] != 1]:
        raise ValueError(f'BART dimensions {unmapped} are not singleton. Include them in other_dims to keep them.')
    return data.permute(*dims, *unused).reshape([data.shape[dim] for dim in dims])
