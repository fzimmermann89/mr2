"""Test fixtures for data consistency tests."""

import pytest
import torch
from mr2.data.KData import KData
from mr2.data.SpatialDimension import SpatialDimension
from mr2.data.traj_calculators.KTrajectoryCartesian import KTrajectoryCartesian
from mr2.operators.FourierOp import FourierOp
from mr2.phantoms.EllipsePhantom import EllipsePhantom
from mr2.utils import RandomGenerator


@pytest.fixture
def kdata():
    matrix = SpatialDimension(x=128, y=128, z=1)
    kdata = EllipsePhantom().kdata(KTrajectoryCartesian.fullysampled(matrix), matrix)
    return kdata


@pytest.fixture
def kdata_noisy(kdata: KData):
    kdata_noisy = kdata.clone()
    kdata_noisy.data += 0.1 * RandomGenerator(123).randn_like(kdata_noisy.data)
    return kdata_noisy


@pytest.fixture
def kdata_us(kdata: KData):
    return kdata[..., ::2, :].clone()


@pytest.fixture
def image_noisy(kdata_noisy: KData):
    fourier_op = FourierOp.from_kdata(kdata_noisy)
    return fourier_op.adjoint(kdata_noisy.data)[0]


@pytest.fixture
def image(kdata: KData):
    fourier_op = FourierOp.from_kdata(kdata)
    return fourier_op.adjoint(kdata.data)[0]


@pytest.fixture
def image_us(kdata_us: KData):
    fourier_op = FourierOp.from_kdata(kdata_us)
    return fourier_op.adjoint(kdata_us.data)[0]


@pytest.fixture
def csm_problem() -> tuple[torch.Tensor, torch.Tensor, KData]:
    """Small multicoil data-consistency problem with explicit sensitivity maps."""
    matrix = SpatialDimension(z=1, y=16, x=16)
    kdata = EllipsePhantom().kdata(KTrajectoryCartesian.fullysampled(matrix), matrix)
    rng = RandomGenerator(123)
    image = rng.complex64_tensor((1, 1, 1, 16, 16))
    csm = rng.complex64_tensor((1, 3, 1, 16, 16), low=0.3, high=1.3)
    kdata.data = rng.complex64_tensor(csm.shape)
    return image, csm, kdata
