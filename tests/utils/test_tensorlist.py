"""Tests for TensorList."""

from collections.abc import Sequence

import pytest
import torch
from mr2.utils import TensorList


def test_tensor_list_tensor_registration() -> None:
    """Test TensorList tensor registration."""
    tensor = torch.ones(1, dtype=torch.float32)
    tensor_with_grad = torch.ones(1, dtype=torch.float32, requires_grad=True)
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

    tensor_list = TensorList([tensor, tensor_with_grad, parameter])

    buffers = dict(tensor_list.named_buffers()).values()
    parameters = dict(tensor_list.named_parameters()).values()
    assert any(buffer is tensor_list[0] for buffer in buffers)
    assert all(buffer is not tensor_list[1] for buffer in buffers)
    assert all(param is not tensor_list[1] for param in parameters)
    assert any(param is tensor_list[2] for param in parameters)

    tensor_list.to(dtype=torch.float64)
    assert tensor_list[0].dtype == torch.float64
    assert tensor_list[1].dtype == torch.float32
    assert tensor_list[2].dtype == torch.float64

    assert isinstance(tensor_list[0:1], TensorList)
    assert tensor_list[0:1][0] is tensor_list[0]


def test_tensor_list_len() -> None:
    """Test TensorList length."""
    tensor_list = TensorList([torch.ones(1), torch.zeros(1)])

    assert len(tensor_list) == 2


def test_tensor_list_iter() -> None:
    """Test TensorList iteration."""
    tensors = [torch.ones(1), torch.zeros(1)]
    tensor_list = TensorList(tensors)

    for actual, expected in zip(tensor_list, tensors, strict=True):
        torch.testing.assert_close(actual, expected)


def test_tensor_list_sequence_typing() -> None:
    """Test TensorList is typed as a sequence."""
    tensor_list: Sequence[torch.Tensor] = TensorList([torch.ones(1)])

    torch.testing.assert_close(tensor_list[0], torch.ones(1))


def test_tensor_list_iadd() -> None:
    """Test TensorList in-place addition."""
    tensor_list = TensorList([torch.ones(1)])
    tensor_list_id = id(tensor_list)

    tensor_list += [torch.zeros(1)]

    assert id(tensor_list) == tensor_list_id
    assert len(tensor_list) == 2
    torch.testing.assert_close(tensor_list[1], torch.zeros(1))


def test_tensor_list_dir() -> None:
    """Test TensorList directory listing."""
    tensor_list = TensorList([torch.ones(1)])

    assert '0' not in dir(tensor_list)


def test_tensor_list_append() -> None:
    """Test TensorList append."""
    tensor_list = TensorList()
    tensor = torch.ones(1)

    result = tensor_list.append(tensor)

    assert result is tensor_list
    torch.testing.assert_close(tensor_list[0], tensor)


def test_tensor_list_extend() -> None:
    """Test TensorList extend."""
    tensor_list = TensorList()
    tensors = [torch.ones(1), torch.zeros(1)]

    result = tensor_list.extend(tensors)

    assert result is tensor_list
    assert len(tensor_list) == len(tensors)
    torch.testing.assert_close(tensor_list[0], tensors[0])
    torch.testing.assert_close(tensor_list[1], tensors[1])


def test_tensor_list_call() -> None:
    """Test TensorList is not callable."""
    tensor_list = TensorList()

    with pytest.raises(RuntimeError, match='BufferList should not be called'):
        tensor_list()
