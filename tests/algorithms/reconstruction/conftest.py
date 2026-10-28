"""PyTest fixtures for reconstruction tests."""

import pytest
import torch
from mr2.data import KData, KHeader, SpatialDimension
from mr2.data.traj_calculators import KTrajectoryCartesian
from mr2.operators import FastFourierOp
from mr2.phantoms import EllipsePhantom
from mr2.phantoms.coils import birdcage_2d
from tests.data import IsmrmrdRawTestData


@pytest.fixture(scope='session')
def cartesian_kdata(ismrmrd_cart: IsmrmrdRawTestData) -> KData:
    """Create Cartesian KData from shared ISMRMRD test fixture."""
    return KData.from_file(ismrmrd_cart.filename, KTrajectoryCartesian())


@pytest.fixture(params=[((1, 16, 18), torch.complex128)])
def fully_sampled_kdata(request: pytest.FixtureRequest) -> KData:
    """Fully sampled Cartesian ellipse data; parameters are (spatial shape, dtype)."""
    shape, dtype = request.param
    matrix = SpatialDimension(*shape)
    profile = torch.linspace(0.5, 1.0, shape[0])[:, None, None]
    image = (EllipsePhantom().image_space(matrix) * birdcage_2d(8, matrix) * profile).to(dtype)
    fov = SpatialDimension(1.0, 1.0, 1.0)
    (data,) = FastFourierOp()(image)
    return KData(header=KHeader(matrix, matrix, fov, fov), data=data, traj=KTrajectoryCartesian.fullysampled(matrix))
