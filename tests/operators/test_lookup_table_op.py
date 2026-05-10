"""Tests for the LookupTableOp."""

import pytest
import torch
from mr2.operators import LookupTableOp
from mr2.operators.SignalModel import SignalModel


class DummySignalModel(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Simple signal model for lookup-table tests."""

    def forward(self, m0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> tuple[torch.Tensor,]:
        """Return a simple signal depending on m0, p1, and p2."""
        return (m0 * torch.stack((1 + p1 + 2 * p2.square(), 2 - p1 + p2.square()), dim=0),)


def test_lookup_table_op_interpolation() -> None:
    """LookupTableOp approximates the underlying signal model on a fine grid."""

    model = DummySignalModel()

    operator = LookupTableOp(
        model,
        parameter_ranges=((0.0, 1.0, 3), (0.0, 2.0, 1001)),
        index_of_scaling_parameter=0,
    )

    m0 = torch.tensor([[[2.0], [3.0]]])
    p1 = torch.tensor([[[0.25], [0.75]]])
    p2 = torch.tensor([[[0.5, 1.5]]])

    (result,) = operator(m0, p1, p2)
    (expected,) = model(m0, p1, p2)

    torch.testing.assert_close(result, expected, atol=5e-5, rtol=5e-5)


@pytest.mark.cuda
def test_lookup_table_op_cuda() -> None:
    """LookupTableOp works on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available.')

    model = DummySignalModel()
    operator = LookupTableOp(
        model,
        parameter_ranges=((0.0, 1.0, 4), (0.0, 2.0, 1001)),
        index_of_scaling_parameter=0,
    )
    operator.cuda()

    m0 = torch.tensor([[2.0, 3.0]], device='cuda')
    p1 = torch.tensor([[0.25, 0.75]], device='cuda')
    p2 = torch.tensor([[0.5, 1.5]], device='cuda')

    (result,) = operator(m0, p1, p2)
    (expected,) = model(m0, p1, p2)

    assert result.is_cuda
    torch.testing.assert_close(result, expected)


def test_lookup_table_op_autograd() -> None:
    """LookupTableOp supports autograd through the interpolated parameters."""
    model = DummySignalModel()
    operator = LookupTableOp(
        model,
        parameter_ranges=((0.0, 1.0, 4), (0.0, 2.0, 1001)),
        index_of_scaling_parameter=0,
    )

    m0 = torch.tensor([[2.0, 3.0]], requires_grad=True)
    p1 = torch.tensor([[0.25, 0.75]], requires_grad=True)
    p2 = torch.tensor([[0.5, 1.5]], requires_grad=True)

    (result,) = operator(m0, p1, p2)
    loss = result.square().sum()
    gradients = torch.autograd.grad(loss, (m0, p1, p2))

    m0_ref = m0.detach().clone().requires_grad_(True)
    p1_ref = p1.detach().clone().requires_grad_(True)
    p2_ref = p2.detach().clone().requires_grad_(True)
    (expected,) = model(m0_ref, p1_ref, p2_ref)
    expected_loss = expected.square().sum()
    expected_gradients = torch.autograd.grad(expected_loss, (m0_ref, p1_ref, p2_ref))

    (expected_result,) = model(m0, p1, p2)
    torch.testing.assert_close(result, expected_result, atol=5e-5, rtol=5e-5)
    for gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, expected_gradient, atol=2e-3, rtol=2e-3)
