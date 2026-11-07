"""Native NUFFT plan reuse and differentiable execution."""

import copy
import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from mr2.data import KTrajectory, SpatialDimension
from mr2.operators import FourierOp, NonUniformFastFourierOp
from mr2.operators.NonUniformFastFourierOp import _NUFFT
from pytorch_finufft.functional import finufft_type1, finufft_type2

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)]


def make_operator(
    points: torch.Tensor, shape: tuple[int, ...], fourier: bool = False
) -> NonUniformFastFourierOp | FourierOp:
    """Build an MR2 operator from explicit coordinates in encoding-grid units."""
    coordinates = [points[:, dim].reshape(points.shape[0], 1, 1, 1, -1) for dim in range(len(shape))]
    coordinates = [torch.zeros_like(coordinates[0])] * (3 - len(shape)) + coordinates
    trajectory = KTrajectory(*coordinates, repeat_detection_tolerance=None)
    if fourier:
        matrix = SpatialDimension(*([1] * (3 - len(shape)) + list(shape)))
        return FourierOp(matrix, matrix, trajectory)
    return NonUniformFastFourierOp(tuple(range(-len(shape), 0)), shape, shape, trajectory)


def direct(points: torch.Tensor, data: torch.Tensor, kind: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Small exact centered Fourier matrix, independent of the NUFFT backend."""
    modes = torch.stack(
        torch.meshgrid(
            *[torch.arange(-(n // 2), n - n // 2, device=points.device, dtype=points.dtype) for n in shape],
            indexing='ij',
        )
    ).flatten(1)
    matrix = torch.exp(-1j * torch.einsum('bdp,dm->bpm', points, modes))
    if kind == 2:
        return torch.einsum('bpm,bcm->bcp', matrix, data.flatten(2))
    return torch.einsum('bpm,bcp->bcm', matrix.conj(), data).reshape(*data.shape[:2], *shape)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shape', [(7,), (5, 6), (3, 4, 5)])
@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_native_plans_match_functional(device: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """Both directions retain existing precision, ordering, scaling and batching."""
    torch.manual_seed(12)
    points = torch.rand(3, len(shape), 13, device=device, dtype=dtype.to_real()) * 5 - 2.5
    operator = make_operator(points, shape)
    omega = operator._omega
    images = torch.randn(3, 2, *shape, device=device, dtype=dtype)
    samples = torch.randn(3, 2, 13, device=device, dtype=dtype)
    for kind, values in ((2, images), (1, samples)):
        actual = _NUFFT.apply(omega, values, operator, kind, shape)
        if kind == 2:
            expected = torch.vmap(lambda p, x: finufft_type2(p, x, isign=-1, modeord=0, upsampfac=2.0))(omega, values)
        else:
            expected = torch.vmap(lambda p, x: finufft_type1(p, x, shape, isign=1, modeord=0, upsampfac=2.0))(
                omega, values
            )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual, direct(omega, values, kind, shape), atol=4e-5, rtol=4e-5)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kind', [1, 2])
def test_native_plan_derivatives(device: str, kind: int) -> None:
    """Check coordinate/data gradients, mixed second derivatives, vmap and JVP."""
    torch.manual_seed(31)
    shape = (5, 6)
    points = (torch.rand(2, 2, 9, device=device, dtype=torch.float64) * 5 - 2.5).requires_grad_()
    operator = make_operator(points, shape)
    size = (2, 2, *shape) if kind == 2 else (2, 2, 9)
    data = torch.randn(size, device=device, dtype=torch.complex128, requires_grad=True)
    expected = direct(points, data, kind, shape)
    actual = _NUFFT.apply(points, data, operator, kind, shape)
    cotangent = torch.randn_like(expected).conj()
    gradients = torch.autograd.grad(actual, (points, data), cotangent, create_graph=True)
    reference_gradients = torch.autograd.grad(expected, (points, data), cotangent, create_graph=True)
    for result, reference in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(result, reference, atol=8e-5, rtol=8e-5)
    second = torch.autograd.grad(sum(g.abs().square().sum() for g in gradients), (points, data))
    reference_second = torch.autograd.grad(sum(g.abs().square().sum() for g in reference_gradients), (points, data))
    for result, reference in zip(second, reference_second, strict=True):
        torch.testing.assert_close(result, reference, atol=3e-3, rtol=2e-4)

    def function(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return _NUFFT.apply(p, x, operator, kind, shape)

    def reference_function(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return direct(p, x, kind, shape)

    tangent = (torch.randn_like(points), torch.randn_like(data))
    _, actual_jvp = torch.func.jvp(function, (points, data), tangent)
    _, expected_jvp = torch.func.jvp(reference_function, (points, data), tangent)
    torch.testing.assert_close(actual_jvp, expected_jvp, atol=1e-4, rtol=1e-4)
    for tangents in ((tangent[0], torch.zeros_like(data)), (torch.zeros_like(points), tangent[1])):
        _, actual_jvp = torch.func.jvp(function, (points, data), tangents)
        _, expected_jvp = torch.func.jvp(reference_function, (points, data), tangents)
        torch.testing.assert_close(actual_jvp, expected_jvp, atol=1e-4, rtol=1e-4)

    batched_points = torch.stack((points.detach(), points.detach() + 0.01))
    batched_data = torch.stack((data.detach(), data.detach() * 0.7))
    for in_dims, p, x in (
        (0, batched_points, batched_data),
        ((None, 0), points.detach(), batched_data),
        ((0, None), batched_points, data.detach()),
    ):
        actual_mapped = torch.vmap(function, in_dims=in_dims)(p, x)
        expected_mapped = torch.vmap(reference_function, in_dims=in_dims)(p, x)
        torch.testing.assert_close(actual_mapped, expected_mapped, atol=8e-5, rtol=8e-5)
    actual_mapped_grad = torch.vmap(torch.func.grad(lambda p, x: function(p, x).abs().square().sum(), argnums=(0, 1)))(
        batched_points, batched_data
    )
    expected_mapped_grad = torch.vmap(
        torch.func.grad(lambda p, x: reference_function(p, x).abs().square().sum(), argnums=(0, 1))
    )(batched_points, batched_data)
    for result, reference in zip(actual_mapped_grad, expected_mapped_grad, strict=True):
        torch.testing.assert_close(result, reference, atol=3e-3, rtol=1e-4)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('fourier', [False, True])
@pytest.mark.parametrize('kind', [1, 2])
def test_public_trajectory_gradients(device: str, fourier: bool, kind: int) -> None:
    """Gradients reach original trajectory tensors through both public operators."""
    torch.manual_seed(27)
    shape = (5, 6)
    points = (torch.rand(2, 2, 11, device=device, dtype=torch.float64) * 4 - 2).requires_grad_()
    operator = make_operator(points, shape, fourier)
    size = (2, 2, 1, *shape) if kind == 2 else (2, 2, 1, 1, 11)
    data = torch.randn(size, device=device, dtype=torch.complex128, requires_grad=True)
    for _ in range(2):
        output = (operator(data) if kind == 2 else operator.adjoint(data))[0]
        flat = data[:, :, 0] if kind == 2 else data.flatten(2)
        scale = torch.tensor(shape).prod().sqrt().reciprocal().to(device)
        omega = points * 2 * torch.pi / torch.tensor(shape, device=device)[:, None]
        reference = (direct(omega, flat, kind, shape) * scale).reshape_as(output)
        cotangent = torch.randn_like(output).conj()
        gradients = torch.autograd.grad(output, (points, data), cotangent)
        reference_gradients = torch.autograd.grad(reference, (points, data), cotangent)
        torch.testing.assert_close(output, reference, atol=2e-5, rtol=2e-5)
        for result, expected in zip(gradients, reference_gradients, strict=True):
            torch.testing.assert_close(result, expected, atol=3e-5, rtol=3e-5)
        with torch.no_grad():
            points.add_(0.02)


@pytest.mark.parametrize('device', DEVICES)
def test_plan_cache_lifetime(device: str) -> None:
    """Reuse across points/calls, replace for batching, and invalidate on conversion."""
    points = torch.rand(3, 2, 13, device=device)
    operator = make_operator(points, (8, 8))
    images = torch.randn(3, 2, 1, 8, 8, device=device, dtype=torch.complex64)
    first = operator(images)[0]
    forward = operator._plan_fwd
    operator.adjoint(first)
    adjoint = operator._plan_adj
    second = operator(images * 2)[0]
    assert operator._plan_fwd is forward
    assert operator._plan_adj is adjoint
    torch.testing.assert_close(second, first * 2)
    operator(images[:, :1])
    assert operator._plan_fwd is not forward
    assert '_plan_adj' not in operator.__dict__
    operator.double()
    assert '_plan_fwd' not in operator.__dict__
    operator(images.to(torch.complex128))
    assert operator._plan_fwd.dtype.name == 'complex128'
    restored = pickle.loads(pickle.dumps(operator))  # noqa: S301
    for duplicate in (copy.copy(operator), copy.deepcopy(operator), restored):
        assert '_plan_fwd' not in duplicate.__dict__
        assert '_plan_adj' not in duplicate.__dict__
        torch.testing.assert_close(duplicate(images.to(torch.complex128))[0], operator(images.to(torch.complex128))[0])
    assert not any('plan' in key for key in operator.state_dict())
    assert not any('lock' in key for key in operator.state_dict())


@pytest.mark.cuda
def test_plan_cuda_streams() -> None:
    """Switch streams safely and use the tensor's device even if it is not current."""
    points = torch.rand(3, 2, 13, device='cuda')
    operator = make_operator(points, (8, 8))
    images = torch.randn(3, 2, 1, 8, 8, device='cuda', dtype=torch.complex64)
    expected = operator(images)[0]
    plan = operator._plan_fwd
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = operator(images)[0]
        assert operator._plan_fwd is not plan
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, expected)
    assert operator._plan_config[-1] == stream
    del plan
    if torch.cuda.device_count() > 1:
        with torch.cuda.device(1):
            torch.testing.assert_close(operator(images)[0], expected)


@pytest.mark.parametrize('device', DEVICES)
def test_plan_concurrent_calls(device: str) -> None:
    """A shared operator serializes native workspace access across threads."""
    points = torch.rand(3, 2, 13, device=device)
    operator = make_operator(points, (8, 8))
    images = torch.randn(3, 2, 1, 8, 8, device=device, dtype=torch.complex64)
    expected = operator(images)[0]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(lambda: operator(images)[0]) for _ in range(4)]
        for future in futures:
            torch.testing.assert_close(future.result(), expected)
