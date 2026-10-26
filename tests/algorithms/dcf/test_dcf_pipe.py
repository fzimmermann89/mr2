"""Mathematical and reconstruction checks for Pipe DCF."""

import pytest
import torch
from mr2.algorithms.dcf import dcf_2d3d_voronoi, dcf_pipe


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('ndim', [1, 2, 3])
@pytest.mark.parametrize('size', [7, 8])
def test_cartesian(dtype: torch.dtype, ndim: int, size: int) -> None:
    """A periodic Cartesian grid has approximately unit density, including in 3D."""
    k = torch.stack(torch.meshgrid(*[torch.arange(size, dtype=dtype) - size // 2] * ndim, indexing='ij'))
    w = dcf_pipe(k, (size,) * ndim, n_iterations=5)
    assert w.dtype == dtype
    # The 32-point grid introduces a small ripple for seven-point lattices.
    tolerance = 0.002 if size == 7 else 1e-4
    torch.testing.assert_close(w, torch.ones_like(w), rtol=0, atol=tolerance)


def test_cartesian_density() -> None:
    """A uniform twofold undersampling has weight two in encoded-grid units."""
    k = torch.arange(-16, 16, 2, dtype=torch.float32)[None]
    torch.testing.assert_close(dcf_pipe(k, (32,)), torch.full((16,), 2.0))


def test_duplicates() -> None:
    """Repeating each location halves its weight without changing its total contribution."""
    k = torch.arange(-16, 16, dtype=torch.float64)[None]
    reference = dcf_pipe(k, (32,))
    repeated = dcf_pipe(k.repeat_interleave(2, -1), (32,))
    torch.testing.assert_close(repeated.reshape(-1, 2).sum(-1), reference)
    # Unequal multiplicities should also share the same total contribution.
    counts = torch.arange(1, 33) % 3 + 1
    repeated = dcf_pipe(k.repeat_interleave(counts, -1), (32,))
    sums = torch.stack([part.sum() for part in repeated.split(counts.tolist())])
    torch.testing.assert_close(sums, reference, rtol=0.02, atol=0.02)


def make_trajectory(kind: str) -> torch.Tensor:
    """Return synthetic radial or spiral coordinates in a 32 by 32 encoded region."""
    if kind == 'radial':
        r = torch.linspace(-15.5, 15.5, 64)
        theta = torch.arange(32)[:, None] * torch.pi / 32
    else:
        r = torch.linspace(0, 15.5, 256)
        theta = torch.arange(8)[:, None] * 2 * torch.pi / 8 + r * 2 * torch.pi / 4
    return torch.stack(torch.broadcast_tensors(r * theta.sin(), r * theta.cos()))


def central_psf(k: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Evaluate the weighted Fourier impulse response on a central image patch."""
    x = torch.stack(torch.meshgrid(torch.arange(-8, 9) / 32, torch.arange(-8, 9) / 32, indexing='ij'))
    fourier = torch.exp(2j * torch.pi * (k.flatten(1).T @ x.flatten(1)))
    return weight.flatten().to(torch.complex64) @ fourier / 32**2


@pytest.mark.parametrize('kind', ['radial', 'spiral'])
def test_noncartesian_psf(kind: str) -> None:
    """The normalized PSF agrees with Voronoi and suppresses central oversampling artifacts."""
    k = make_trajectory(kind)
    w = dcf_pipe(k, (32, 32))
    assert w.shape == k.shape[1:]
    assert (w.isfinite() & (w > 0)).all()
    v = dcf_2d3d_voronoi(k)
    v = v * (32**2 / v.sum())
    psf = central_psf(k, w)
    reference = central_psf(k, v)
    uniform = central_psf(k, torch.full_like(w, 32**2 / w.numel()))
    assert (psf - reference).norm() / reference.norm() < 0.15
    assert (psf - reference).norm() < (uniform - reference).norm() / 2
    torch.testing.assert_close(psf[psf.numel() // 2], torch.tensor(1 + 0j))
    if kind == 'radial':
        radius = k.square().sum(0).sqrt()
        interior = (radius[0] > 3) & (radius[0] < 12)
        ratio = (w / radius).mean(0)[interior]
        assert ratio.std() / ratio.mean() < 0.01


@pytest.mark.cuda
@pytest.mark.parametrize('kind', ['radial', 'spiral'])
def test_cuda_parity(kind: str) -> None:
    """Compare normalized DCF and PSF, allowing CPU and CUDA kernel differences."""
    k = make_trajectory(kind)
    cpu = dcf_pipe(k, (32, 32))
    gpu = dcf_pipe(k.cuda(), (32, 32))
    assert gpu.is_cuda
    assert (gpu.isfinite() & (gpu > 0)).all()
    gpu = gpu.cpu()
    assert (cpu - gpu).norm() / cpu.norm() < 0.03
    psf = central_psf(k, cpu)
    assert (psf - central_psf(k, gpu)).norm() / psf.norm() < 0.01


def test_no_coordinate_gradient() -> None:
    """Do not expose the wrapper's invalid gridding-coordinate derivative."""
    k = torch.linspace(-8, 8, 32, requires_grad=True)[None]
    assert not dcf_pipe(k, (32,)).requires_grad


@pytest.mark.parametrize(
    ('k', 'sizes', 'iterations', 'oversampling'),
    [
        (torch.ones(2, 4), (8,), 2, 2.0),
        (torch.ones(1, 4), (0,), 2, 2.0),
        (torch.ones(1, 4), (8,), 0, 2.0),
        (torch.ones(1, 4), (8,), 2, 0.5),
        (torch.full((1, 4), float('nan')), (8,), 2, 2.0),
    ],
)
def test_invalid_input(k: torch.Tensor, sizes: tuple[int], iterations: int, oversampling: float) -> None:
    """Reject unsupported geometry and invalid iteration settings."""
    with pytest.raises(ValueError):
        dcf_pipe(k, sizes, iterations, oversampling)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_periodic_coordinates(device: str) -> None:
    """Adding encoded periods preserves DCF even outside older backend bounds."""
    k = torch.tensor([[-16.0, -5.25, 0.0, 3.5, 15.0]], dtype=torch.float64, device=device)
    shifts = k.new_tensor([[-4, 3, 0, 5, -3]]) * 32
    torch.testing.assert_close(dcf_pipe(k + shifts, (32,)), dcf_pipe(k, (32,)))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_multiple_batch_dimensions(device: str) -> None:
    """Leading batch axes preserve independent weights and their normalization."""
    radial = make_trajectory('radial').to(device).unsqueeze(-3)
    spiral = make_trajectory('spiral').to(device).reshape(2, 1, 32, 64)
    geometries = torch.stack((radial, spiral))
    batch = torch.stack((geometries, geometries * 0.8)).transpose(0, 1)
    result = dcf_pipe(batch, (32, 32))
    assert result.shape == (2, 2, 1, 32, 64)
    for i in range(2):
        for j in range(2):
            torch.testing.assert_close(result[i, j], dcf_pipe(batch[i, j], (32, 32)))
    torch.testing.assert_close(result.sum((-3, -2, -1)), torch.full((2, 2), 32**2, device=device).float())
