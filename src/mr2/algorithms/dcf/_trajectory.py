"""Identify independent components of a single trajectory geometry."""

from collections.abc import Sequence

import torch


def _trajectory_groups(coordinates: Sequence[torch.Tensor]) -> list[tuple[int, ...]]:
    """Group coordinate directions connected through non-singleton sample axes.

    Coordinates must have three trailing sample axes, with constant axes reduced
    to singletons. Each returned tuple contains indices of directions that must
    be processed together. Directions constant over all samples are omitted.
    Independent groups can be compensated separately and their weights multiplied.
    """
    groups: list[set[int]] = []
    for axis, coordinate in enumerate(coordinates):
        if all(size == 1 for size in coordinate.shape[-3:]):
            continue
        group = {axis}
        for previous in groups.copy():
            if any(coordinates[i].shape[d] > 1 and coordinate.shape[d] > 1 for i in previous for d in (-3, -2, -1)):
                group |= previous
                groups.remove(previous)
        groups.append(group)
    return [tuple(sorted(group)) for group in groups]
