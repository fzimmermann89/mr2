"""Tests for reading and writing BART files."""

from pathlib import Path

import numpy as np
import pytest
import torch
from mr2.data import CsmData, DcfData, IData, IHeader, KData, KHeader, KTrajectory, SpatialDimension
from mr2.data.bart import from_bart_layout, read_bart, to_bart_layout, write_bart
from mr2.data.enums import BartDimension
from mr2.utils import RandomGenerator

IMAGE_DIMS = (BartDimension.COIL, BartDimension.PHS2, BartDimension.PHS1, BartDimension.READ)


def bart_shape(**sizes: int) -> list[int]:
    """Shape with all 16 BART dimensions, given non-singleton sizes by dimension name."""
    shape = [1] * 16
    for name, size in sizes.items():
        shape[BartDimension[name.upper()]] = size
    return shape


@pytest.mark.parametrize('suffix', ['.cfl', '.hdr', '.ra'])
def test_bart_roundtrip(tmp_path: Path, suffix: str) -> None:
    """Write and read a BART array."""
    data = RandomGenerator(0).complex64_tensor((3, 4, 1, 5))
    write_bart(tmp_path / f'array{suffix}', data)
    torch.testing.assert_close(read_bart(tmp_path / f'array{suffix}'), data, rtol=0, atol=0)


def test_bart_cfl_column_major(tmp_path: Path) -> None:
    """The .hdr/.cfl pair stores dimensions as text and the first dimension varies fastest."""
    data = RandomGenerator(0).complex64_tensor((2, 3, 4))
    write_bart(tmp_path / 'array.cfl', data)
    assert (tmp_path / 'array.hdr').read_text().splitlines() == ['# Dimensions', '2 3 4']
    stored = np.fromfile(tmp_path / 'array.cfl', dtype='<c8')
    np.testing.assert_array_equal(stored, data.numpy().ravel(order='F'))


def test_bart_ra_layout(tmp_path: Path) -> None:
    """The .ra file contains the header, the dimensions and the data in column-major order."""
    data = RandomGenerator(0).complex64_tensor((2, 3))
    write_bart(tmp_path / 'array.ra', data)
    raw = (tmp_path / 'array.ra').read_bytes()
    header = np.frombuffer(raw[:64], dtype='<u8')
    np.testing.assert_array_equal(header, [0x7961727261776172, 0, 4, 8, 48, 2, 2, 3])
    np.testing.assert_array_equal(np.frombuffer(raw[64:], dtype='<c8'), data.numpy().ravel(order='F'))


def test_bart_read_header_written_by_bart(tmp_path: Path) -> None:
    """Read a header with the additional entries and trailing spaces written by BART."""
    data = RandomGenerator(0).complex64_tensor((4, 3))
    data.numpy().ravel(order='F').astype('<c8').tofile(tmp_path / 'array.cfl')
    (tmp_path / 'array.hdr').write_text('# Dimensions\n4 3 1 1 \n# Command\nbart phantom array\n')
    assert read_bart(tmp_path / 'array.cfl').shape == (4, 3, 1, 1)


def test_bart_read_size_mismatch(tmp_path: Path) -> None:
    """Data and header have to match."""
    write_bart(tmp_path / 'array.cfl', torch.zeros(2, 3, dtype=torch.complex64))
    (tmp_path / 'array.hdr').write_text('# Dimensions\n2 4\n')
    with pytest.raises(ValueError, match='require 8'):
        read_bart(tmp_path / 'array.cfl')


def test_bart_invalid_suffix(tmp_path: Path) -> None:
    """Only BART suffixes are accepted."""
    with pytest.raises(ValueError, match='must end in'):
        write_bart(tmp_path / 'array.npy', torch.zeros(1))


def test_bart_layout_flattens_other_into_batch() -> None:
    """By default, all other dimensions are flattened into the batch dimension."""
    data = RandomGenerator(0).complex64_tensor((2, 3, 4, 1, 5, 6))
    bart = to_bart_layout(data, IMAGE_DIMS)
    assert list(bart.shape) == bart_shape(read=6, phs1=5, coil=4, batch=6)
    torch.testing.assert_close(bart[..., 4].reshape(6, 5, 1, 4), data[1, 1].permute(3, 2, 1, 0))
    assert from_bart_layout(bart, IMAGE_DIMS).shape == (6, 4, 1, 5, 6)


def test_bart_layout_other_dims() -> None:
    """Map each other dimension to the given BART dimension and back."""
    data = RandomGenerator(0).complex64_tensor((2, 3, 4, 1, 5, 6))
    other_dims = (BartDimension.SLICE, BartDimension.TE)
    bart = to_bart_layout(data, IMAGE_DIMS, other_dims)
    assert list(bart.shape) == bart_shape(read=6, phs1=5, coil=4, te=3, slice=2)
    torch.testing.assert_close(from_bart_layout(bart, IMAGE_DIMS, other_dims), data)


@pytest.mark.parametrize(
    ('other_dims', 'message'),
    [
        ((BartDimension.TE,), 'has 1 entries'),
        ((BartDimension.TE, BartDimension.TE), 'duplicates'),
        ((BartDimension.ITER, BartDimension.MOTION), 'duplicates'),
        ((BartDimension.TE, BartDimension.COIL), 'reserved'),
    ],
)
def test_bart_layout_invalid_other_dims(other_dims: tuple[BartDimension, ...], message: str) -> None:
    """Reject invalid mappings of the other dimensions."""
    with pytest.raises(ValueError, match=message):
        to_bart_layout(torch.zeros(2, 3, 4, 1, 5, 6), IMAGE_DIMS, other_dims)


def test_bart_layout_rejects_unmapped_dimensions() -> None:
    """Non-singleton BART dimensions have to be mapped on import."""
    data = torch.zeros(bart_shape(read=4, phs1=4, maps=2))
    with pytest.raises(ValueError, match='MAPS'):
        from_bart_layout(data, IMAGE_DIMS)
    assert from_bart_layout(data, IMAGE_DIMS, (BartDimension.MAPS,)).shape == (2, 1, 1, 4, 4)


@pytest.mark.parametrize('suffix', ['.cfl', '.ra'])
def test_idata_bart_roundtrip(tmp_path: Path, random_iheader: IHeader, suffix: str) -> None:
    """Save IData as BART file and load it again."""
    data = RandomGenerator(0).complex64_tensor((2, 3, 4, 1, 5, 6))
    idata = IData(data=data, header=random_iheader)
    other_dims = (BartDimension.TIME, BartDimension.SLICE)
    idata.to_bart(tmp_path / f'image{suffix}', other_dims=other_dims)
    assert list(read_bart(tmp_path / f'image{suffix}').shape) == bart_shape(read=6, phs1=5, coil=4, time=2, slice=3)
    loaded = IData.from_file(tmp_path / f'image{suffix}', header=random_iheader, other_dims=other_dims)
    torch.testing.assert_close(loaded.data, data)
    assert loaded.header is random_iheader


def test_csmdata_from_bart_maps(tmp_path: Path, random_kheader: KHeader) -> None:
    """Load ESPIRiT maps from BART as CsmData using a KHeader."""
    data = RandomGenerator(0).complex64_tensor(bart_shape(read=6, phs1=5, coil=4, maps=2))
    write_bart(tmp_path / 'sens.ra', data)
    csm = CsmData.from_file(tmp_path / 'sens.ra', header=random_kheader, other_dims=(BartDimension.MAPS,))
    assert isinstance(csm, CsmData)
    torch.testing.assert_close(csm.data, data[:, :, :, :, :, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].permute(4, 3, 2, 1, 0))


def test_idata_from_bart_requires_header(tmp_path: Path) -> None:
    """BART files contain no meta data."""
    write_bart(tmp_path / 'image.cfl', torch.zeros(4, 4))
    with pytest.raises(ValueError, match='header'):
        IData.from_file(tmp_path / 'image.cfl')


def cartesian_kdata(n_k1: int = 8, n_k0: int = 6, k1_lines: tuple[int, ...] = (-4, 2, 0, -1)) -> KData:
    """Undersampled, unsorted Cartesian KData with two other dimensions."""
    matrix = SpatialDimension(z=1, y=n_k1, x=n_k0)
    header = KHeader(recon_matrix=matrix, encoding_matrix=matrix, recon_fov=matrix, encoding_fov=matrix)
    ky = torch.tensor(k1_lines, dtype=torch.float32)[:, None]
    kx = torch.arange(n_k0, dtype=torch.float32) - n_k0 // 2
    traj = KTrajectory(kz=torch.zeros(1, 1, 1, 1, 1), ky=ky[None, None, None], kx=kx[None, None, None, None])
    data = RandomGenerator(0).complex64_tensor((2, 3, 4, 1, len(k1_lines), n_k0))
    return KData(header=header, data=data, traj=traj)


def test_kdata_to_bart_cartesian(tmp_path: Path) -> None:
    """Cartesian data is sorted into the encoding matrix with the k-space center at n // 2."""
    kdata = cartesian_kdata()
    kdata.to_bart(tmp_path / 'kspace.cfl', other_dims=(BartDimension.SLICE, BartDimension.TE))
    kspace = read_bart(tmp_path / 'kspace.cfl')
    assert list(kspace.shape) == bart_shape(read=6, phs1=8, coil=4, te=3, slice=2)
    kspace = from_bart_layout(kspace, IMAGE_DIMS, (BartDimension.SLICE, BartDimension.TE))
    for line_idx, k1 in enumerate((-4, 2, 0, -1)):
        torch.testing.assert_close(kspace[..., 0, k1 + 4, :], kdata.data[..., 0, line_idx, :])
    assert not kspace[..., 0, [1, 2, 5, 7], :].any()


def test_kdata_to_bart_cartesian_repeated_points(tmp_path: Path) -> None:
    """Repeated k-space points cannot be stored on the grid."""
    with pytest.raises(ValueError, match='Repeated'):
        cartesian_kdata(k1_lines=(0, 1, 0)).to_bart(tmp_path / 'kspace.cfl')


def test_kdata_to_bart_cartesian_trajectory_filename(tmp_path: Path) -> None:
    """Cartesian data has no trajectory file."""
    with pytest.raises(ValueError, match='Cartesian'):
        cartesian_kdata().to_bart(tmp_path / 'kspace.cfl', trajectory_filename=tmp_path / 'traj.cfl')


def radial_kdata(n_k2: int = 2, n_k1: int = 3, n_k0: int = 5) -> KData:
    """Stack-of-stars KData with a broadcasted trajectory."""
    matrix = SpatialDimension(z=n_k2, y=n_k0, x=n_k0)
    header = KHeader(recon_matrix=matrix, encoding_matrix=matrix, recon_fov=matrix, encoding_fov=matrix)
    angle = torch.arange(n_k1)[:, None] * torch.pi / n_k1
    radius = torch.arange(n_k0) - n_k0 / 2 + 0.3
    traj = KTrajectory(
        kz=(torch.arange(n_k2, dtype=torch.float32) - n_k2 // 2)[None, None, :, None, None],
        ky=(torch.sin(angle) * radius)[None, None, None],
        kx=(torch.cos(angle) * radius)[None, None, None],
    )
    data = RandomGenerator(0).complex64_tensor((2, 4, n_k2, n_k1, n_k0))
    return KData(header=header, data=data, traj=traj)


@pytest.mark.parametrize('suffix', ['.cfl', '.ra'])
def test_kdata_to_bart_noncartesian(tmp_path: Path, suffix: str) -> None:
    """Non-Cartesian samples and trajectory are saved with matching readout order."""
    kdata = radial_kdata()
    kdata.to_bart(tmp_path / f'kspace{suffix}', trajectory_filename=tmp_path / f'traj{suffix}')
    kspace = read_bart(tmp_path / f'kspace{suffix}')
    traj = read_bart(tmp_path / f'traj{suffix}')
    assert list(kspace.shape) == bart_shape(phs1=5, phs2=6, coil=4, batch=2)
    assert list(traj.shape) == bart_shape(read=3, phs1=5, phs2=6, batch=2)
    assert not traj.imag.any()
    for k2 in range(2):
        for k1 in range(3):
            readout = k2 * 3 + k1
            torch.testing.assert_close(kspace[0, :, readout, :, ..., 1].reshape(5, 4), kdata.data[1, :, k2, k1, :].T)
            torch.testing.assert_close(traj[0, :, readout, ..., 1].real.reshape(5), kdata.traj.kx[0, 0, 0, k1])
            torch.testing.assert_close(traj[1, :, readout, ..., 1].real.reshape(5), kdata.traj.ky[0, 0, 0, k1])
            kz = kdata.traj.kz[0, 0, k2, 0].expand(5)
            torch.testing.assert_close(traj[2, :, readout, ..., 1].real.reshape(5), kz)


def test_kdata_to_bart_noncartesian_requires_trajectory_filename(tmp_path: Path) -> None:
    """Non-Cartesian samples cannot be saved without the trajectory."""
    with pytest.raises(ValueError, match='trajectory_filename'):
        radial_kdata().to_bart(tmp_path / 'kspace.cfl')


@pytest.mark.parametrize('suffix', ['.cfl', '.ra'])
def test_dcfdata_bart_roundtrip(tmp_path: Path, suffix: str) -> None:
    """DCF uses the layout of non-Cartesian samples; readouts are restored along k1."""
    data = RandomGenerator(0).float32_tensor((2, 1, 2, 3, 5), low=0.1)
    DcfData(data=data).to_bart(tmp_path / f'dcf{suffix}')
    assert list(read_bart(tmp_path / f'dcf{suffix}').shape) == bart_shape(phs1=5, phs2=6, batch=2)
    loaded = DcfData.from_file(tmp_path / f'dcf{suffix}')
    torch.testing.assert_close(loaded.data, data.reshape(2, 1, 1, 6, 5))


def test_dcfdata_from_bart_complex(tmp_path: Path) -> None:
    """DCF has to be real."""
    write_bart(tmp_path / 'dcf.cfl', torch.ones(1, 5, 3) * (1 + 1j))
    with pytest.raises(ValueError, match='imaginary'):
        DcfData.from_file(tmp_path / 'dcf.cfl')


def test_dcfdata_mr2_roundtrip(tmp_path: Path) -> None:
    """Non-BART files are still loaded from MR2 storage."""
    dcf = DcfData(data=torch.ones(1, 1, 1, 3, 5))
    dcf.save_as_mr2(tmp_path / 'dcf.h5')
    torch.testing.assert_close(DcfData.from_file(tmp_path / 'dcf.h5').data, dcf.data)


def test_kdata_to_bart_cartesian_outside_encoding_matrix(tmp_path: Path) -> None:
    """All k-space points have to lie inside the encoding matrix."""
    with pytest.raises(ValueError, match='outside'):
        cartesian_kdata(k1_lines=(0, 1, 4)).to_bart(tmp_path / 'kspace.cfl')
