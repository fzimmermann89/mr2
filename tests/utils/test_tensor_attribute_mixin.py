"""Tests for tensor attribute replacement."""

import torch
from mr2.utils.TensorAttributeMixin import TensorAttributeMixin


def test_replace_gradient_tensor() -> None:
    """Replacing a gradient tensor with a detached tensor registers a movable buffer."""
    module = TensorAttributeMixin()
    module.tensor = torch.ones(2, requires_grad=True)
    module.tensor = module.tensor.detach()
    module.double()
    assert module.tensor.dtype == torch.float64
    assert module.get_buffer('tensor') is module.tensor
    module.tensor = module.tensor.clone().requires_grad_()
    module.tensor = module.tensor.detach()
    assert not module.tensor.requires_grad
