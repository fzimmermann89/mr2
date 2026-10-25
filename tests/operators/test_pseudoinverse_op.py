"""Pseudoinverse actions, exact formulas and constant-rank implicit gradients."""

import pytest
import torch
from mr2.data import KTrajectory, SpatialDimension
from mr2.operators import (
    AveragingOp,
    CartesianMaskingOp,
    CartesianSamplingOp,
    DensityCompensationOp,
    DownSamplingOp,
    EinsumOp,
    FastFourierOp,
    FiniteDifferenceOp,
    IdentityOp,
    LinearOperator,
    LinearOperatorMatrix,
    PatchOp,
    PCACompressionOp,
    PseudoInverseOp,
    RearrangeOp,
    SensitivityOp,
    WaveletOp,
    ZeroOp,
    ZeroPadOp,
)


def matrix(m: int, n: int, rank: int, dtype: torch.dtype) -> torch.Tensor:
    """Build a matrix with a controlled nonzero spectrum and a known rank."""
    generator = torch.Generator().manual_seed(19)
    u = torch.linalg.qr(torch.randn(m, rank, dtype=dtype, generator=generator)).Q
    v = torch.linalg.qr(torch.randn(n, rank, dtype=dtype, generator=generator)).Q
    return (u * torch.linspace(1, 2, rank, dtype=u.real.dtype)) @ v.mH


@pytest.mark.parametrize(('m', 'n', 'rank'), [(5, 3, 3), (3, 5, 3), (5, 4, 2), (4, 6, 2), (3, 3, 0)])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_dense_reference(m: int, n: int, rank: int, dtype: torch.dtype) -> None:
    """Converged actions satisfy the four MP identities, including singular cases."""
    a = matrix(m, n, rank, dtype)
    op = EinsumOp(a)
    p = op.pinv(tolerance=1e-10)
    b = torch.randn(m, dtype=dtype, requires_grad=True)
    u = torch.randn(n, dtype=dtype)
    x = p(b)[0]
    torch.testing.assert_close(x, torch.linalg.pinv(a) @ b, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(op(p(op(u)[0])[0])[0], op(u)[0], atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(p(op(x)[0])[0], x, atol=1e-9, rtol=1e-9)
    z = p.H(u)[0]
    torch.testing.assert_close(torch.vdot(x, u), torch.vdot(b, z), atol=1e-9, rtol=1e-9)
    # Both projectors must be Hermitian; these are actions, not a presumed
    # matrix representation of the finite iterative algorithm.
    c, v = torch.randn_like(b), torch.randn_like(u)
    torch.testing.assert_close(torch.vdot(op(x)[0], c), torch.vdot(b, op(p(c)[0])[0]), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(torch.vdot(p(op(u)[0])[0], v), torch.vdot(u, p(op(v)[0])[0]), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(torch.autograd.grad(x, b, u)[0], z, atol=1e-9, rtol=1e-9)
    assert isinstance(p @ op, LinearOperator)
    assert p.H.H is p


@pytest.mark.parametrize('shape', [(5, 3, 3), (3, 5, 3), (5, 4, 2)])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_parameter_gradients(shape: tuple[int, int, int], dtype: torch.dtype) -> None:
    """Constant-rank factor gradients include residual and moving-nullspace terms."""
    m, n, rank = shape
    a = matrix(m, n, rank, dtype)
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    left = (u[:, :rank] * s[:rank]).detach().requires_grad_()
    right = vh[:rank].detach().requires_grad_()
    b = torch.randn(m, dtype=dtype, requires_grad=True)
    g = torch.randn(n, dtype=dtype)
    value = left @ right
    actual = EinsumOp(value).pinv(tolerance=1e-10)(b)[0]
    expected = torch.linalg.pinv(value) @ b
    grads = torch.autograd.grad(actual, (left, right, b), g, retain_graph=True)
    refs = torch.autograd.grad(expected, (left, right, b), g)
    for grad, ref in zip(grads, refs, strict=True):
        torch.testing.assert_close(grad, ref, atol=1e-8, rtol=1e-8)


def test_shared_and_derived_coefficients() -> None:
    """Shared ancestors and derived attributes receive each partial derivative once."""
    a = matrix(3, 3, 3, torch.float64).requires_grad_()
    b = torch.randn(3, dtype=a.dtype)
    g = torch.randn_like(b)
    op = EinsumOp(a) + EinsumOp(2 * a)
    actual = op.pinv(tolerance=1e-11)(b)[0]
    expected = torch.linalg.solve(3 * a, b)
    grad = torch.autograd.grad(actual, a, g, retain_graph=True)[0]
    ref = torch.autograd.grad(expected, a, g)[0]
    torch.testing.assert_close(grad, ref, atol=1e-8, rtol=1e-8)


def test_moving_nullspace() -> None:
    """A(t)=[1,t] has a smooth pseudoinverse, including at t=0."""
    t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    a = torch.stack((torch.ones_like(t), t))[None]
    x = EinsumOp(a).pinv()(torch.ones(1, dtype=t.dtype))[0]
    torch.testing.assert_close(torch.autograd.grad(x[1], t)[0], torch.ones_like(t))


def test_reuse_and_snapshot() -> None:
    """An invocation preserves its coefficients; subsequent calls see replacements."""
    a = matrix(3, 3, 3, torch.float64).requires_grad_()
    op = EinsumOp(a)
    p = op.pinv(tolerance=1e-10)
    b = torch.randn(3, dtype=a.dtype, requires_grad=True)
    old = p(b)[0]
    op.matrix = (2 * a).detach().requires_grad_()
    old.sum().backward()
    torch.testing.assert_close(b.grad, torch.linalg.solve(a.mH.detach(), torch.ones_like(b)), atol=1e-8, rtol=1e-8)
    for _ in range(2):
        op.matrix.grad = None
        p(b.detach())[0].sum().backward()
        assert op.matrix.grad is not None
    torch.testing.assert_close(p(b.detach())[0], old.detach() / 2, atol=1e-8, rtol=1e-8)


def test_mutation_is_rejected() -> None:
    """In-place changes to fixed buffers between forward/backward are detected."""
    a = torch.eye(3, dtype=torch.float64)
    b = torch.ones(3, dtype=a.dtype, requires_grad=True)
    x = EinsumOp(a).pinv()(b)[0]
    a.add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        x.sum().backward()


@pytest.mark.parametrize('zero_input', [False, True])
def test_singular_backward(zero_input: bool) -> None:
    """An incompatible CG cotangent still has a well-defined input gradient."""
    a = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    b = torch.tensor([0.0 if zero_input else 2.0, 3.0], dtype=a.dtype, requires_grad=True)
    x = EinsumOp(a).pinv()(b)[0]
    torch.testing.assert_close(torch.autograd.grad(x.sum(), b)[0], torch.tensor([1.0, 0.0], dtype=a.dtype))


def test_numerical_failures() -> None:
    """Failures are explicit and tolerance is never replaced by damping."""
    op = EinsumOp(torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64)))
    with pytest.raises(RuntimeError, match='did not meet tolerance'):
        op.pinv(max_iterations=1)(torch.ones(2, dtype=torch.float64))
    for value in [float('nan'), float('inf'), 1e-200, 1e200]:
        with pytest.raises(RuntimeError, match=r'Nonfinite|numerical range'):
            op.pinv()(torch.full((2,), value, dtype=torch.float64))
    for tolerance in [-1.0, float('nan')]:
        with pytest.raises(ValueError):
            IdentityOp().pinv(tolerance=tolerance)
    for iterations in [0, True]:
        with pytest.raises(ValueError):
            IdentityOp().pinv(max_iterations=iterations)


@pytest.mark.parametrize(('original', 'encoded'), [((3,), (6,)), ((6,), (3,)), ((3, 4), (4, 3))])
@pytest.mark.parametrize('fft', [False, True])
def test_pad_fft_shortcut(original: tuple[int, ...], encoded: tuple[int, ...], fft: bool) -> None:
    """Padding, cropping and mixed axes have adjoint pseudoinverses."""
    dims = tuple(range(-len(original), 0))
    op = FastFourierOp(dims, original, encoded) if fft else ZeroPadOp(dims, original, encoded)
    basis = torch.eye(torch.Size(original).numel(), dtype=torch.complex128).reshape(-1, *original)
    dense = torch.stack([op(v)[0].flatten() for v in basis], -1)
    b = torch.randn(encoded, dtype=torch.complex128)
    torch.testing.assert_close(op.pinv()(b)[0].flatten(), torch.linalg.pinv(dense) @ b.flatten())


@pytest.mark.parametrize('flatten', [True, False])
def test_patch_shortcut(flatten: bool) -> None:
    """Unequal overlap and uncovered coordinates agree with a dense reference."""
    op = PatchOp(-1, 3, stride=2, dilation=2, domain_size=8, flatten_patches=flatten)
    dense = torch.stack([op(v)[0].flatten() for v in torch.eye(8, dtype=torch.float64)], -1)
    b = torch.randn(op(torch.ones(8))[0].shape, dtype=torch.float64, requires_grad=True)
    p = op.pinv()
    torch.testing.assert_close(p(b)[0], torch.linalg.pinv(dense) @ b.flatten())
    g = torch.randn(8, dtype=b.dtype)
    torch.testing.assert_close(torch.autograd.grad(p(b)[0], b, g)[0], p.H(g)[0])
    with pytest.raises(ValueError, match='domain_size'):
        PatchOp(-1, 3).pinv()


@pytest.mark.parametrize('rotate', [False, True])
def test_pca_rearrange(rotate: bool) -> None:
    """PCA rows remain orthonormal under varimax; rearrangements are unitary."""
    op = PCACompressionOp(torch.randn(20, 5, dtype=torch.complex128), 3, rotate=rotate)
    y = torch.randn(4, 3, dtype=torch.complex128)
    torch.testing.assert_close(op(op.pinv()(y)[0])[0], y)
    rearrange = RearrangeOp('a b -> (a b)', {'a': 2, 'b': 3})
    y = torch.randn(6)
    torch.testing.assert_close(rearrange(rearrange.pinv()(y)[0])[0], y)


def test_sensitivity_gradients() -> None:
    """Live normalized sensitivities support repeated parameter and input gradients."""
    csm = torch.randn(3, 1, 1, 4, dtype=torch.complex128, requires_grad=True)
    op = SensitivityOp(csm)
    p = op.pinv()
    y = torch.randn_like(csm, requires_grad=True)
    assert torch.autograd.gradcheck(lambda c, b: SensitivityOp(c).pinv()(b), (csm, y), fast_mode=True)
    for _ in range(2):
        torch.autograd.grad(p(y)[0].abs().square().sum(), (csm, y))
    with torch.no_grad():
        csm[..., 0] = 0
    x = p(y)[0]
    assert x[..., 0].count_nonzero() == 0
    assert torch.isfinite(torch.autograd.grad(x.real.sum(), y)[0]).all()
    with pytest.raises(ValueError, match='singleton coil'):
        p.H(torch.ones(1, 1, 1, 1, dtype=csm.dtype))


@pytest.mark.parametrize('kind', ['identity', 'dcf', 'mask'])
def test_diagonal_shortcuts(kind: str) -> None:
    """Nonbinary and zero diagonals use live zero-safe reciprocals."""
    weights = torch.tensor([2.0, 0.0, -3.0], dtype=torch.float64, requires_grad=True)
    op = {
        'identity': lambda: IdentityOp() * weights,
        'dcf': lambda: DensityCompensationOp(weights),
        'mask': lambda: CartesianMaskingOp(weights),
    }[kind]()
    p = op.pinv()
    y = torch.randn(3, dtype=torch.float64, requires_grad=True)
    x = p(y)[0]
    torch.testing.assert_close(x, y * torch.tensor([0.5, 0, -1 / 3], dtype=y.dtype))
    for _ in range(2):
        # The mask constructor has its existing float conversion graph lifecycle.
        torch.autograd.grad(p(y)[0].sum(), y)
    grad = torch.autograd.grad(x.sum(), weights)[0]
    assert torch.isfinite(grad).all()
    with pytest.raises(ValueError, match='expand'):
        p(torch.ones(1))


def test_scalar_and_zero() -> None:
    """Complex scalar inversion and shaped zeros preserve data derivatives."""
    op = EinsumOp(matrix(3, 2, 2, torch.complex128))
    c = torch.tensor(2 + 1j, dtype=torch.complex128, requires_grad=True)
    y = torch.randn(3, dtype=c.dtype)
    torch.testing.assert_close((c * op).pinv()(y)[0], op.pinv()(y)[0] / c)
    assert torch.autograd.gradcheck(lambda scale: (scale * op).pinv(tolerance=1e-10)(y), (c,), fast_mode=True)
    y = torch.ones(3, requires_grad=True)
    x = ZeroOp(keep_shape=True).pinv()(y)[0]
    torch.testing.assert_close(torch.autograd.grad(x.sum(), y)[0], torch.zeros_like(y))
    with pytest.raises(ValueError, match='shape-free'):
        ZeroOp().pinv()


def test_cartesian_duplicates() -> None:
    """Duplicate samples sum in the adjoint and average in the pseudoinverse."""
    trajectory = KTrajectory(
        kz=torch.zeros(1, 1, 1, 1, 1),
        ky=torch.zeros(1, 1, 1, 1, 1),
        kx=torch.tensor([-1.0, -1.0, 0.0, 3.0]).reshape(1, 1, 1, 1, 4),
    )
    with pytest.warns(UserWarning, match='outside'):
        op = CartesianSamplingOp(SpatialDimension(1, 1, 2), trajectory)
    y = torch.tensor([3.0, 7.0, 11.0, 99.0]).reshape(1, 1, 1, 1, 4)
    x = torch.tensor([2.0, 5.0]).reshape(1, 1, 1, 1, 2)
    torch.testing.assert_close(op.H(y)[0].flatten(), torch.tensor([10.0, 11.0]))
    torch.testing.assert_close(op.pinv()(y)[0].flatten(), torch.tensor([5.0, 11.0]))
    torch.testing.assert_close(op.gram(x)[0].flatten(), torch.tensor([4.0, 5.0]))
    torch.testing.assert_close(
        torch.vdot(op(x)[0].flatten(), y.flatten()), torch.vdot(x.flatten(), op.H(y)[0].flatten())
    )


def test_tuple_actions() -> None:
    """Coupled tuple solutions and both composition orders retain all components."""
    a = IdentityOp() | IdentityOp()
    p = a.pinv(tolerance=1e-10)
    b = torch.randn(4, dtype=torch.float64, requires_grad=True)
    x = p(b)
    for xi in x:
        torch.testing.assert_close(xi, b / 2)
    torch.testing.assert_close(
        torch.autograd.grad(sum((v.sum() for v in x), start=b.new_zeros(())), b)[0], torch.ones_like(b)
    )
    torch.testing.assert_close((a @ p)(b)[0], b)
    result = (p @ a)(b, b)
    for xi in result:
        torch.testing.assert_close(xi, b)
    diagonal = LinearOperatorMatrix.from_diagonal(IdentityOp(), 2 * IdentityOp())
    result = diagonal.pinv()(b, b)
    torch.testing.assert_close(result[0], b)
    torch.testing.assert_close(result[1], b / 2)


def test_invalid_adjoints() -> None:
    """Known invalid adjoint configurations fail even when nested in algebra."""
    wavelet = WaveletOp((16,), dim=(-1,), wavelet_name='bior2.2', level=1)
    with pytest.raises(NotImplementedError, match='Biorthogonal'):
        (IdentityOp() @ wavelet).pinv()(wavelet(torch.ones(16))[0])
    average = AveragingOp(-1, [[0, 0, 1]], domain_size=2)
    with pytest.raises(ValueError, match='repeated'):
        average.pinv()(torch.ones(1))


def test_exact_transforms_and_movement() -> None:
    """Exact diagonal paths support JVP/vmap and fixed-source dtype movement."""
    op = IdentityOp() * torch.tensor([2.0, 0.0, 4.0])
    p = op.pinv().to(dtype=torch.float64)
    b = torch.randn(5, 3, dtype=torch.float64)
    torch.testing.assert_close(torch.vmap(lambda x: p(x)[0])(b), p(b)[0])
    tangent = torch.func.jvp(lambda x: p(x)[0], (b,), (torch.ones_like(b),))[1]
    torch.testing.assert_close(tangent, p(torch.ones_like(b))[0])
    generic = PseudoInverseOp(IdentityOp())
    with pytest.raises(RuntimeError, match=r'setup_context|jvp'):
        torch.func.jvp(lambda x: generic(x)[0], (b,), (b,))


def test_tuple_parameter_gradients() -> None:
    """Coupled blocks propagate shared coefficient gradients with different spaces."""
    scale = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    left = EinsumOp(torch.tensor([[1.0], [2.0]], dtype=scale.dtype))
    right = EinsumOp(torch.tensor([[2.0], [-1.0]], dtype=scale.dtype))
    a = (scale * left) | right
    b = torch.tensor([3.0, 4.0], dtype=scale.dtype, requires_grad=True)
    result = a.pinv(tolerance=1e-10)(b)
    actual = torch.cat(result)
    dense = torch.stack((scale * left.matrix[:, 0], right.matrix[:, 0]), -1)
    expected = torch.linalg.solve(dense, b)
    g = torch.tensor([2.0, -1.0], dtype=scale.dtype)
    grads = torch.autograd.grad(actual, (scale, b), g, retain_graph=True)
    refs = torch.autograd.grad(expected, (scale, b), g)
    for grad, ref in zip(grads, refs, strict=True):
        torch.testing.assert_close(grad, ref)


def test_first_order_only() -> None:
    """The generic implicit derivative explicitly excludes higher derivatives."""
    b = torch.ones(3, dtype=torch.float64, requires_grad=True)
    x = EinsumOp(torch.eye(3, dtype=b.dtype)).pinv()(b)[0]
    grad = torch.autograd.grad(x.square().sum(), b, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(grad.sum(), b)


def test_float32_and_small_rhs() -> None:
    """Ordinary single precision and small representable data need no absolute floor."""
    for dtype, scale in [(torch.float32, 1.0), (torch.complex64, 1e-8), (torch.float64, 1e-20)]:
        a = matrix(5, 3, 3, dtype).to(dtype)
        b = torch.randn(5, dtype=dtype) * scale
        actual = EinsumOp(a).pinv()(b)[0]
        expected = torch.linalg.pinv(a) @ b
        torch.testing.assert_close(actual / scale, expected / scale, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('unequal_valid_counts', [False, True])
def test_batched_sampling(unequal_valid_counts: bool) -> None:
    """Different outside positions preserve per-batch counts or fail explicitly."""
    coordinates = torch.tensor([[-1.0, -1.0, 0.0, 3.0], [3.0, -1.0, 0.0, 0.0]])
    if unequal_valid_counts:
        coordinates[1, 1] = 3
    trajectory = KTrajectory(
        kz=torch.zeros(1, 1, 1, 1, 1), ky=torch.zeros(1, 1, 1, 1, 1), kx=coordinates.reshape(2, 1, 1, 1, 4)
    )
    if unequal_valid_counts:
        with pytest.warns(UserWarning, match='outside'), pytest.raises(NotImplementedError, match='equal numbers'):
            CartesianSamplingOp(SpatialDimension(1, 1, 2), trajectory)
        return
    with pytest.warns(UserWarning, match='outside'):
        op = CartesianSamplingOp(SpatialDimension(1, 1, 2), trajectory)
    y = torch.tensor([[3.0, 7.0, 11.0, 99.0], [99.0, 5.0, 3.0, 7.0]]).reshape(2, 1, 1, 1, 4)
    torch.testing.assert_close(op.pinv()(y)[0].reshape(2, 2), torch.tensor([[5.0, 11.0], [5.0, 5.0]]))


def test_zero_scalar_live_update() -> None:
    """A live scalar becoming zero avoids a pointless nonzero inverse solve."""
    scale = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    op = scale * EinsumOp(torch.diag(torch.tensor([1.0, 2.0], dtype=scale.dtype)))
    p = op.pinv(max_iterations=1)
    with torch.no_grad():
        scale.zero_()
    b = torch.ones(2, dtype=scale.dtype, requires_grad=True)
    torch.testing.assert_close(p(b)[0], torch.zeros_like(b))
    torch.testing.assert_close(torch.autograd.grad(p(b)[0].sum(), b)[0], torch.zeros_like(b))


@pytest.mark.parametrize('interpolation', [False, True])
def test_structured_fallback(interpolation: bool) -> None:
    """Interpolation and periodic finite differences agree with their dense maps."""
    op = DownSamplingOp(-1, (6,), (3,)) if interpolation else FiniteDifferenceOp((-1,), pad_mode='circular')
    dense = torch.stack([op(v.reshape(1, 6))[0].flatten() for v in torch.eye(6, dtype=torch.float64)], -1)
    b = torch.randn(op(torch.ones(1, 6, dtype=torch.float64))[0].shape, dtype=torch.float64, requires_grad=True)
    p = op.pinv(tolerance=1e-10)
    torch.testing.assert_close(p(b)[0].flatten(), torch.linalg.pinv(dense) @ b.flatten(), atol=1e-9, rtol=1e-9)
    g = torch.randn(1, 6, dtype=b.dtype)
    torch.testing.assert_close(torch.autograd.grad(p(b)[0], b, g)[0], p.H(g)[0], atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('fft', [False, True])
def test_partial_input_compatibility(fft: bool) -> None:
    """Legacy partial-input adjoints still work; only pinv enforces fixed spaces."""
    op = FastFourierOp((-2, -1), (4, 4), (4, 8)) if fft else ZeroPadOp((-2, -1), (4, 4), (4, 8))
    partial = torch.randn(3, 6, dtype=torch.complex128)
    result = op.adjoint(partial)[0]
    expected = partial
    if fft:
        expected = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(partial), norm='ortho'))
    cropped = expected[:, 1:5]
    expected = partial.new_zeros(4, 4)
    expected[1:] = cropped
    torch.testing.assert_close(result, expected)
    with pytest.raises(ValueError, match='declared encoding'):
        op.pinv()(partial)
    with pytest.raises(ValueError, match='declared reconstruction'):
        op.pinv().H(partial)
