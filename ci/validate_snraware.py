"""Compare native SNRAware outputs with locally downloaded official TorchScript weights.

Run from the repository root with ``PYTHONPATH=src python ci/validate_snraware.py
--variant small --checkpoint /path/to/snraware_small_model.pts``. Only load trusted
TorchScript artifacts. No files are downloaded by this script.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from mr2.nn.nets import SNRAware


def main() -> None:
    """Validate conversion and output parity at one and sixteen frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('small', 'medium', 'large'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    official = torch.jit.load(args.checkpoint, map_location='cpu').eval()
    model = SNRAware(args.variant).eval()
    model.load_state_dict(model.convert_state_dict(official.state_dict()))
    comparisons: list[dict[str, int | float]] = []
    record = {
        'variant': args.variant,
        'sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'parameters': sum(p.numel() for p in model.parameters()),
        'comparisons': comparisons,
    }
    with torch.inference_mode():
        for frames in (1, 16):
            generator = torch.Generator().manual_seed(2026)
            x = torch.randn(1, 3, frames, 64, 64, generator=generator)
            x[:, 2] = x[:, 2].abs() + 1
            native, reference = model(x), official(x)
            error = native - reference
            torch.testing.assert_close(native, reference, atol=5e-5, rtol=5e-5)
            comparisons.append(
                {
                    'frames': frames,
                    'max_error': error.abs().max().item(),
                    'rms_error': error.square().mean().sqrt().item(),
                }
            )
    print(json.dumps(record, indent=2))  # noqa: T201


if __name__ == '__main__':
    main()
