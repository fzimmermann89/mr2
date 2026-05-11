"""Mixin for smarter tensor attributes."""

import operator
from collections.abc import Iterable, Iterator
from typing import Any, Never, TypeVar, overload

import torch
from typing_extensions import Self

T = TypeVar('T', bound='TensorList')


class TensorAttributeMixin(torch.nn.Module):
    """Create tensor attributes as buffer."""

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Set attribute.

        Set tensors not requiring gradients as buffer.

        Parameters
        ----------
        name
            name of the attribute.
        value
            attribute to set.
        """
        if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter) and not value.requires_grad:
            self.register_buffer(name, value)
        elif isinstance(value, list | tuple) and all(isinstance(v, torch.Tensor) for v in value):
            tensor_list = TensorList(value)
            super().__setattr__(name, tensor_list)
        else:
            super().__setattr__(name, value)


class TensorList(torch.nn.Module):
    """Holds tensors in a list, correctly registering buffers.

    Mimics PyTorch ParameterList, but registers tensors as buffers if they
    do not require gradients and are not Parameters. Tensors requiring
    gradients are registered as standard attributes.
    """

    def __init__(self, tensors: Iterable[torch.Tensor] | None = None) -> None:
        """Initialize TensorList.

        Parameters
        ----------
        tensors
            An iterable of tensors to add to the list.
        """
        super().__init__()
        self._size = 0
        if tensors is not None:
            self.extend(tensors)

    def _get_abs_string_index(self, idx: int) -> str:
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f'index {idx} is out of range')
        if idx < 0:
            idx += len(self)
        return str(idx)

    @overload
    def __getitem__(self, idx: int) -> torch.Tensor: ...

    @overload
    def __getitem__(self: T, idx: slice) -> T: ...

    def __getitem__(self, idx: int | slice) -> torch.Tensor | T:
        """Get item from TensorList."""
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            out = self.__class__()
            for i in range(start, stop, step):
                out.append(self[i])
            return out
        idx_str = self._get_abs_string_index(idx)
        return getattr(self, idx_str)

    def __setitem__(self, idx: int, tensor: torch.Tensor) -> None:
        """Set item in TensorList."""
        idx_str = self._get_abs_string_index(idx)

        # Core Logic: Register buffer vs standard attribute
        if not isinstance(tensor, torch.nn.Parameter) and not tensor.requires_grad:
            self.register_buffer(idx_str, tensor)
        else:
            setattr(self, idx_str, tensor)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self[i] for i in range(len(self)))

    def __iadd__(self, tensors: Iterable[torch.Tensor]) -> Self:
        """Add tensors to TensorList."""
        return self.extend(tensors)

    def __dir__(self) -> list[str]:
        """Get directory of TensorList."""
        keys = super().__dir__()
        return [key for key in keys if not key.isdigit()]

    def append(self, value: torch.Tensor) -> Self:
        """Append a tensor to the end of the list.

        Parameters
        ----------
        value
            Tensor to append.
        """
        new_idx = len(self)
        self._size += 1
        self[new_idx] = value
        return self

    def extend(self, values: Iterable[torch.Tensor]) -> Self:
        """Append tensors from an iterable to the end of the list.

        Parameters
        ----------
        values
            Iterable of tensors to append.
        """
        if not isinstance(values, Iterable) or isinstance(values, torch.Tensor):
            raise TypeError(f'TensorList.extend requires an iterable, got {type(values).__name__}')
        for value in values:
            self.append(value)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Never:
        """TensorList should not be called."""
        raise RuntimeError('BufferList should not be called.')
