"""ConvDecoder dimensionality, initialization and checkpoint tests."""

from collections.abc import Iterator
from copy import deepcopy

import pytest
import torch
from mr2.nn import ComplexAsChannel
from mr2.nn.nets import ConvDecoder


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    """Keep small convolution tests inexpensive."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('n_dim', [1, 2, 3, 4, 5])
def test_shape_and_gradients(n_dim: int) -> None:
    """Arbitrary real channels and non-power-of-two sizes work in each dimension."""
    shape = (3, 4, 5, 6, 7)[-n_dim:]
    model = ConvDecoder(n_dim, 3, 4, shape, n_features=5, n_levels=2).double()
    latent = torch.rand((2, 3, *((2,) * n_dim)), dtype=torch.float64, requires_grad=True)
    output = model(latent)
    assert output.shape == (2, 4, *shape)
    assert output.dtype == torch.float64
    output.square().mean().backward()
    assert latent.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())
    complex_output = ComplexAsChannel(model)(latent.detach())
    assert complex_output.shape == (2, 2, *shape)
    assert complex_output.dtype == torch.complex128


@pytest.mark.parametrize('n_dim', [1, 2, 3, 4, 5])
def test_resume_fitting(n_dim: int) -> None:
    """Restoring network, Adam and latent state continues the same optimization."""
    shape = (5,) * n_dim
    model = ConvDecoder(n_dim, 2, 3, shape, n_features=4, n_levels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    latent = torch.rand((1, 2, *((2,) * n_dim)))
    target = torch.rand((1, 3, *shape))

    def step(network: ConvDecoder, optim: torch.optim.Adam, z: torch.Tensor) -> None:
        optim.zero_grad(set_to_none=True)
        (network(z) - target).square().mean().backward()
        optim.step()

    step(model, optimizer, latent)
    weights, optimizer_state, saved_latent = deepcopy((model.state_dict(), optimizer.state_dict(), latent))
    step(model, optimizer, latent)
    restored = ConvDecoder(n_dim, 2, 3, shape, n_features=4, n_levels=2)
    restored.load_state_dict(weights)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(optimizer_state)
    step(restored, restored_optimizer, saved_latent)
    torch.testing.assert_close(restored(latent), model(latent), rtol=0, atol=0)


@pytest.mark.parametrize('n_dim', [1, 2, 3, 4, 5])
def test_local_initialization(n_dim: int) -> None:
    """Meta construction and explicit reset do not consume global RNG state."""
    state = torch.random.get_rng_state().clone()
    with torch.device('meta'):
        model = ConvDecoder(n_dim, 3, 2, (5,) * n_dim, n_features=4, n_levels=2)
    model.to_empty(device='cpu')
    model.reset_parameters(torch.Generator().manual_seed(42))
    expected = deepcopy(model.state_dict())
    model.reset_parameters(torch.Generator().manual_seed(42))
    torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
