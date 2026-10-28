"""PyTest fixtures for reconstruction tests."""

from collections.abc import Callable

import pytest
import torch
from mr2.data import AcqInfo, KData, KHeader, KTrajectory, SpatialDimension
from mr2.data.traj_calculators import KTrajectoryCartesian
from mr2.operators import FastFourierOp
from mr2.phantoms.coils import birdcage_2d
from tests.data import IsmrmrdRawTestData
from tests.phantoms import EllipsePhantomTestData


@pytest.fixture(scope='session')
def cartesian_kdata(ismrmrd_cart: IsmrmrdRawTestData) -> KData:
    """Create Cartesian KData from shared ISMRMRD test fixture."""
    return KData.from_file(ismrmrd_cart.filename, KTrajectoryCartesian())


@pytest.fixture
def grappa_kspace(
    ellipse_phantom: EllipsePhantomTestData,
) -> Callable[[tuple[int, int, int], torch.dtype], torch.Tensor]:
    """Sample the shared ellipse phantom with simulated birdcage coils."""

    def create(shape: tuple[int, int, int], dtype: torch.dtype) -> torch.Tensor:
        """Sample the shared ellipse phantom with simulated birdcage coil sensitivities."""
        dimensions = SpatialDimension(*shape)
        image = ellipse_phantom.phantom.image_space(dimensions).to(dtype)
        image = image * torch.linspace(0.5, 1.0, shape[0]).reshape(1, 1, -1, 1, 1)
        coils = birdcage_2d(8, dimensions).to(dtype)
        (kspace,) = FastFourierOp()(image * coils)
        return kspace[0]

    return create


@pytest.fixture
def grappa_kdata(random_kheader: KHeader) -> Callable[[torch.Tensor, torch.Tensor], KData]:
    """Pack a Cartesian acquisition mask into KData using the shared header fixture."""

    def create(full: torch.Tensor, acquired: torch.Tensor) -> KData:
        """Store samples compactly with an explicit Cartesian acquisition trajectory."""
        shape = full.shape[-3:]
        coordinates = torch.meshgrid(*(torch.arange(n, device=full.device) - n // 2 for n in shape), indexing='ij')
        kz, ky, kx = (coordinate[acquired].reshape(1, 1, 1, 1, -1) for coordinate in coordinates)
        trajectory = KTrajectory(kz, ky, kx)
        header = random_kheader.clone()
        header.recon_matrix = SpatialDimension(*shape)
        header.encoding_matrix = SpatialDimension(*shape)
        header.recon_fov = SpatialDimension(1.0, 1.0, 1.0)
        header.encoding_fov = SpatialDimension(1.0, 1.0, 1.0)
        header.acq_info = AcqInfo()
        header.trajectory = None
        return KData(data=full[..., acquired].unsqueeze(-2).unsqueeze(-2), header=header, traj=trajectory)

    return create


@pytest.fixture
def grappa_data(grappa_kspace: Callable, grappa_kdata: Callable) -> tuple[torch.Tensor, torch.Tensor, KData]:
    """Shared ellipse acquisition with alternate phase-encoding lines missing."""
    full = grappa_kspace((1, 16, 18), torch.complex128)[None]
    acquired = torch.ones(full.shape[-3:], dtype=torch.bool)
    acquired[..., 1::2, :] = False
    return full, acquired, grappa_kdata(full, acquired)
