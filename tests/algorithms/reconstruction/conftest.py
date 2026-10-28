"""PyTest fixtures for reconstruction tests."""

import pytest
import torch
from mr2.data import AcqInfo, KData, KHeader, KTrajectory, SpatialDimension
from mr2.data.traj_calculators import KTrajectoryCartesian
from mr2.operators import FourierOp, SensitivityOp
from tests.data import IsmrmrdRawTestData


@pytest.fixture(scope='session')
def cartesian_kdata(ismrmrd_cart: IsmrmrdRawTestData) -> KData:
    """Create Cartesian KData from shared ISMRMRD test fixture."""
    return KData.from_file(ismrmrd_cart.filename, KTrajectoryCartesian())


@pytest.fixture
def convdecoder_acquisition(random_kheader: KHeader) -> tuple[KData, FourierOp, SensitivityOp]:
    """Create undersampled multicoil data with three time points and three z planes."""
    size = SpatialDimension(3, 5, 7)
    header = random_kheader
    header.recon_matrix = size
    header.encoding_matrix = size
    header.recon_fov = SpatialDimension(0.03, 0.05, 0.07)
    header.encoding_fov = SpatialDimension(0.03, 0.05, 0.07)
    header.acq_info = AcqInfo()
    header.trajectory = None
    traj = KTrajectory(
        torch.arange(-1, 2).reshape(1, 1, 3, 1, 1),
        torch.arange(-2, 3, 2).reshape(1, 1, 1, 3, 1),
        torch.arange(-3, 4).reshape(1, 1, 1, 1, 7),
    )
    fourier = FourierOp(size, size, traj)
    image = torch.rand((3, 1, 3, 5, 7), generator=torch.Generator().manual_seed(9)).to(torch.complex64)
    csm = SensitivityOp(torch.tensor([1, 0.5j]).reshape(1, 2, 1, 1, 1))
    return KData(header=header, data=(fourier @ csm)(image)[0], traj=traj), fourier, csm
