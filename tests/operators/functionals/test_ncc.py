"""Test the NCC functional."""

from collections.abc import Sequence

import numpy as np
import pytest
import torch
from mr2.operators.functionals.NCC import NCC
from mr2.utils import RandomGenerator


def test_ncc() -> None:
    """Test the NCC functional."""
    rng = RandomGenerator(0)
    target = rng.float32_tensor((1, 8, 32, 32), low=0.0, high=1.0)
    ncc = NCC(target)
    perfect = target.clone()
    assert torch.isclose(ncc(perfect)[0], torch.tensor(1.0))
    bad = rng.float32_tensor((1, 8, 32, 32), low=0.0, high=1.0)
    assert ncc(bad)[0].abs() < 0.1


@pytest.mark.parametrize('shape', [(2, 1, 32, 32), (2, 12, 32, 32)])
def test_ncc_mask(shape: Sequence[int]) -> None:
    """Test masking in the local NCC functional."""
    rng = RandomGenerator(0)
    target = rng.float32_tensor(shape, low=0.0, high=1.0)
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[..., 4:-4, 4:-4] = True
    test = rng.rand_like(target) + 0.5 * target + rng.rand_like(target, high=100) * (~mask).float()
    (masked,) = NCC(target, mask, window_size=7)(test)
    (cropped,) = NCC(target[..., 4:-4, 4:-4], window_size=7)(test[..., 4:-4, 4:-4])
    torch.testing.assert_close(masked, cropped)
    assert 0.40 < masked.item() < 0.5


def test_ncc_reduction() -> None:
    """Test the reduction argument of the local NCC functional."""
    rng = RandomGenerator(0)
    target = rng.complex64_tensor((2, 3, 10, 10, 10), low=0.0, high=1.0)
    test = rng.complex64_tensor(target.shape) + rng.float32_tensor((2, 3, 1, 1, 1), low=0.2, high=0.8) * target
    (ncc_volume,) = NCC(target, window_size=7, reduction='volume')(test)
    (ncc_full,) = NCC(target, window_size=7, reduction='full')(test)
    (ncc_none,) = NCC(target, window_size=7, reduction='none')(test)
    torch.testing.assert_close(ncc_volume.mean(), ncc_full)
    torch.testing.assert_close(ncc_none.mean(), ncc_full)
    assert ncc_volume.shape == (2, 3)
    assert ncc_full.shape == ()
    assert ncc_none.shape == (2, 3, 4, 4, 4)


def test_ncc_global_matches_numpy_corrcoef() -> None:
    """Test global NCC against NumPy Pearson correlation."""
    target = torch.tensor([[[[1.0, 2.0], [3.0, 5.0]]]], dtype=torch.float32)
    prediction = torch.tensor([[[[3.0, 7.0], [9.0, 12.0]]]], dtype=torch.float32)

    (ncc_value,) = NCC(target, window_size=None, reduction='volume')(prediction)

    x = target.detach().cpu().numpy().reshape(-1)
    y = prediction.detach().cpu().numpy().reshape(-1)
    expected = np.corrcoef(x, y)[0, 1]

    torch.testing.assert_close(ncc_value, torch.tensor([expected], dtype=ncc_value.dtype))


def test_ncc_local_masked_matches_bruteforce_reference() -> None:
    """Test local masked NCC against a brute-force sliding-window implementation."""
    target = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, 1.0], [0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]]]],
        dtype=torch.float32,
    )
    prediction = torch.tensor(
        [[[[1.0, 1.5, 2.0, 2.5], [2.0, 0.5, 0.0, 1.0], [0.5, 1.0, 2.5, 2.0], [2.5, 2.0, 1.0, 0.0]]]],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [[[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )

    window_size = 3
    eps = 1e-12

    (ncc_none,) = NCC(target, weight=weight, window_size=window_size, reduction='none', eps=eps)(prediction)
    (ncc_volume,) = NCC(target, weight=weight, window_size=window_size, reduction='volume', eps=eps)(prediction)
    (ncc_full,) = NCC(target, weight=weight, window_size=window_size, reduction='full', eps=eps)(prediction)

    # Brute-force local NCC with the same "drop window if any masked entry" rule.
    z, y, x = target.shape[-3:]
    wz, wy, wx = tuple(window_size if s > 1 else 1 for s in (z, y, x))
    ncc_values = []
    win_weights = []
    for z_idx in range(z - wz + 1):
        for iy in range(y - wy + 1):
            for ix in range(x - wx + 1):
                tw = target[0, z_idx : z_idx + wz, iy : iy + wy, ix : ix + wx]
                pw = prediction[0, z_idx : z_idx + wz, iy : iy + wy, ix : ix + wx]
                ww = weight[0, z_idx : z_idx + wz, iy : iy + wy, ix : ix + wx]

                if torch.any(ww <= 0):
                    ncc_values.append(torch.tensor(0.0))
                    win_weights.append(torch.tensor(0.0))
                    continue

                tw_mean = tw.mean()
                pw_mean = pw.mean()
                tw_c = tw - tw_mean
                pw_c = pw - pw_mean
                cov = (tw_c * pw_c).mean()
                var_t = (tw_c.square()).mean()
                var_p = (pw_c.square()).mean()
                ncc_values.append(cov / (torch.sqrt(var_t * var_p) + eps))
                win_weights.append(ww.mean())

    ref_map = torch.stack(ncc_values).reshape_as(ncc_none)
    ref_weights = torch.stack(win_weights).reshape_as(ncc_none)
    ref_weights = ref_weights / ref_weights.sum().clamp_min(eps)
    ref_volume = (ref_map * ref_weights).sum()

    torch.testing.assert_close(ncc_none, ref_map)
    torch.testing.assert_close(ncc_volume, ref_volume[None])
    torch.testing.assert_close(ncc_full, ref_volume)
