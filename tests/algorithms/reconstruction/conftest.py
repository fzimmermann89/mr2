"""PyTest fixtures for reconstruction tests."""

import pytest
import torch
from mr2.data import KData, KHeader, SpatialDimension
from mr2.data.traj_calculators import KTrajectoryCartesian
from mr2.operators import FastFourierOp
from tests.data import IsmrmrdRawTestData


@pytest.fixture(scope='session')
def cartesian_kdata(ismrmrd_cart: IsmrmrdRawTestData) -> KData:
    """Create Cartesian KData from shared ISMRMRD test fixture."""
    return KData.from_file(ismrmrd_cart.filename, KTrajectoryCartesian())


def fully_sampled_kdata(image: torch.Tensor) -> KData:
    """Create fully sampled Cartesian KData of a multi-coil image."""
    matrix = SpatialDimension(*image.shape[-3:])
    fov = SpatialDimension(1.0, 1.0, 1.0)
    (data,) = FastFourierOp()(image)
    return KData(header=KHeader(matrix, matrix, fov, fov), data=data, traj=KTrajectoryCartesian.fullysampled(matrix))
