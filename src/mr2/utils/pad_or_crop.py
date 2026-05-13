"""Zero pad and crop data tensor."""

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torchnd import pad_nd

from mr2.utils.reshape import normalize_index


def pad_or_crop(
    data: torch.Tensor,
    new_shape: Sequence[int] | torch.Size,
    dim: None | Sequence[int] = None,
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    value: float = 0.0,
) -> torch.Tensor:
    """Change shape of data by center cropping or symmetric padding.

    Parameters
    ----------
    data
        Data to pad or crop.
    new_shape
        Desired shape of data.
    dim
        Dimensions the `new_shape` corresponds to.
        `None` is interpreted as last ``len(new_shape)`` dimensions.
    mode
        Mode to use for padding.
    value
        Value to use for constant padding.

    Returns
    -------
        Data zero padded or cropped to shape.
    """
    if len(new_shape) > data.ndim:
        raise ValueError('length of new shape should not exceed dimensions of data')

    if dim is None:
        new_shape = (*data.shape[: -len(new_shape)], *new_shape)
    else:
        if len(new_shape) != len(dim):
            raise ValueError('length of shape should match length of dim')
        dim = tuple(normalize_index(data.ndim, idx) for idx in dim)
        if len(dim) != len(set(dim)):
            raise ValueError('repeated values are not allowed in dims')

        shape = list(data.shape)
        for d, size in zip(dim, new_shape, strict=True):
            shape[d] = size
        new_shape = tuple(shape)

    pad: list[int] = []

    for old, new in zip(data.shape, new_shape, strict=True):
        diff = new - old
        after = math.trunc(diff / 2)
        before = diff - after
        pad.extend((before, after))

    return pad_nd(data, pad, dims=range(data.ndim), mode=mode, value=value) if any(pad) else data
