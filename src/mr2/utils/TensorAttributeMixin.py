"""Mixin for smarter tensor attributes."""

from typing import Any

import torch

from mr2.utils.TensorList import TensorList


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
        elif isinstance(value, list | tuple) and len(value) and all(isinstance(v, torch.Tensor) for v in value):
            tensor_list = TensorList(value)
            super().__setattr__(name, tensor_list)
        else:
            super().__setattr__(name, value)
