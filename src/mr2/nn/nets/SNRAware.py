"""Released SNRAware imaging transformers, ported from Microsoft (MIT; see LICENSE-SNRAware)."""

from collections.abc import Mapping
from typing import Literal

import torch
from torch.nn import LayerNorm, Module, PReLU

from mr2.nn.attention.ConvolutionalAttention import ConvolutionalAttention
from mr2.nn.DropPath import DropPath
from mr2.nn.ndmodules import convND
from mr2.nn.PermutedBlock import PermutedBlock
from mr2.nn.PixelShuffle import PixelUnshuffle
from mr2.nn.Residual import Residual
from mr2.nn.Sequential import Sequential
from mr2.nn.Upsample import Upsample


def _conv(n_channels_in: int, n_channels_out: int, bias: bool = True) -> Module:
    return PermutedBlock(
        (-2, -1),
        convND(2)(
            n_channels_in,
            n_channels_out,
            3,
            padding=1,
            padding_mode='reflect',
            bias=bias,
        ),
    )


class ImagingTransformerBlock(Sequential):
    """Pre-normalized convolutional attention and spatial mixer with residual connections."""

    def __init__(
        self,
        n_channels: int,
        n_heads: int,
        image_size: int,
        mode: Literal['temporal', 'local', 'global'],
        p_dropout: float,
    ) -> None:
        """Initialize a network-specific transformer block.

        Parameters
        ----------
        n_channels
            Feature channels.
        n_heads
            Attention heads.
        image_size
            Spatial normalization and attention size.
        mode
            Attention grouping.
        p_dropout
            Stochastic-depth probability for each residual branch.
        """
        super().__init__(
            Residual(
                Sequential(
                    PermutedBlock((-2, -1), LayerNorm((n_channels, image_size, image_size))),
                    ConvolutionalAttention(n_channels, n_heads, image_size, mode),
                    DropPath(p_dropout, scale_by_keep=True),
                )
            ),
            Residual(
                Sequential(
                    PermutedBlock((-2, -1), LayerNorm((n_channels, image_size, image_size))),
                    _conv(n_channels, 4 * n_channels),
                    PReLU(),
                    _conv(4 * n_channels, n_channels),
                    DropPath(p_dropout, scale_by_keep=True),
                )
            ),
        )


class SNRAware(Module):
    """Two-resolution HRNet imaging transformer used by released SNRAware models.

    Inputs contain real image, imaginary image and g-factor map channels in SNR
    units; no intensity or noise normalization is performed. The output contains
    the denoised real and imaginary channels. The spatially learned normalization
    fixes the patch size. Published weights use 64 x 64 patches and 64 heads.

    References
    ----------
    .. [XUE25] Xue et al. SNRAware: Improved Deep Learning MRI Denoising with SNR
       Unit Training and G-factor Map Augmentation. https://arxiv.org/abs/2503.18162
    """

    def __init__(
        self,
        variant: Literal['small', 'medium', 'large'] = 'small',
        n_features: int = 64,
        n_heads: int = 64,
        image_size: int = 64,
        p_dropout: float = 0.1,
    ) -> None:
        """Initialize SNRAware.

        Parameters
        ----------
        variant
            Released depth configuration: one, two or four TLG repetitions per stage.
        n_features
            Feature channels in the full-resolution branch; doubled in the lower branch.
        n_heads
            Attention heads at both resolutions.
        image_size
            Square spatial patch size. Must be divisible by sixteen.
        p_dropout
            Stochastic-depth probability. Disabled in evaluation mode.
        """
        super().__init__()
        if variant not in ('small', 'medium', 'large'):
            raise ValueError(f'Unknown SNRAware variant: {variant}')
        if image_size < 16 or image_size % 16:
            raise ValueError('image_size must be a positive multiple of sixteen.')
        if not 0 <= p_dropout < 1:
            raise ValueError('p_dropout must be in [0, 1).')
        self.image_size = image_size
        self.n_features = n_features
        depth = {'small': 1, 'medium': 2, 'large': 4}[variant]

        def stage(channels: int, size: int) -> Sequential:
            return Sequential(
                *(
                    ImagingTransformerBlock(channels, n_heads, size, mode, p_dropout)
                    for _ in range(depth)
                    for mode in ('temporal', 'local', 'global')
                )
            )

        self.pre = _conv(3, n_features)
        self.initial = stage(n_features, image_size)
        self.high = stage(n_features, image_size)
        self.low = stage(2 * n_features, image_size // 2)
        self.output_high = stage(n_features, image_size)
        self.output_low = stage(2 * n_features, image_size // 2)

        def down() -> Module:
            return PermutedBlock(
                (-2, -1),
                Sequential(
                    PixelUnshuffle(2),
                    convND(2)(4 * n_features, 2 * n_features, 1, bias=False),
                ),
            )

        self.down_initial = down()
        self.down_fusion = down()
        self.up_fusion = Sequential(Upsample((-2, -1)), _conv(2 * n_features, n_features, bias=False))
        self.up_output = Sequential(Upsample((-2, -1)), _conv(2 * n_features, 2 * n_features, bias=False))
        self.post = _conv(3 * n_features, 2)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise image patches in SNR units.

        Parameters
        ----------
        x
            Real image, imaginary image and g-factor map stacked as
            ``(batch, 3, frames, image_size, image_size)``.

        Returns
        -------
            Denoised real and imaginary channels, with shape
            ``(batch, 2, frames, image_size, image_size)``.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise a real tensor with shape ``(batch, 3, frames, image_size, image_size)``."""
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f'Expected (batch, 3, frames, {self.image_size}, {self.image_size}).')
        stem = self.pre(x)
        initial = self.initial(stem)
        high = self.high(initial)
        low = self.low(self.down_initial(initial))
        high_out = self.output_high(high + self.up_fusion(low))
        low_out = self.output_low(low + self.down_fusion(high))
        return self.post(torch.cat((high_out + stem, self.up_output(low_out)), dim=1))

    def convert_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert an official SNRAware state dictionary to this model's layout.

        Use ``model.load_state_dict(model.convert_state_dict(official.state_dict()))``
        where ``official = torch.jit.load(path, map_location='cpu')`` for a trusted
        official .pts artifact. Save the result with ordinary ``torch.save`` and
        subsequently load with ``torch.load(..., weights_only=True)``.

        Parameters
        ----------
        state_dict
            Unmodified official model state dictionary. Variant, feature width,
            heads and patch size must match this instance.

        Returns
        -------
            Native state dictionary. Unrecognized keys and incompatible shapes raise ValueError.
        """
        names = {
            'pre.conv': 'pre.module',
            'post.conv': 'post.module',
            'bk.down_00_11.block.D2_0.conv.conv': 'down_initial.module.1',
            'bk.down_0_1.block.D2_0.conv.conv': 'down_fusion.module.1',
            'bk.up_1_0.block.U2_0.conv.conv': 'up_fusion.1.module',
            'bk.up_1.block.U2_0.conv.conv': 'up_output.1.module',
        }
        ignored = set()
        stages = {'B00': 'initial', 'B01': 'high', 'B11': 'low', 'output_B0': 'output_high', 'output_B1': 'output_low'}
        for official, native in stages.items():
            for index in range(len(self.initial)):
                source = f'bk.{official}.block.cell_{index}'
                target = f'{native}.{index}'
                names[f'{source}.n1.ln'] = f'{target}.0.module.0.module'
                names[f'{source}.n2.ln'] = f'{target}.1.module.0.module'
                for old, new in [('query', 'to_q'), ('key', 'to_k'), ('value', 'to_v'), ('output_proj', 'to_out')]:
                    names[f'{source}.attn.{old}.conv'] = f'{target}.0.module.1.{new}.module'
                if index % 3:
                    names[f'{source}.attn'] = f'{target}.0.module.1'
                names[f'{source}.mlp.0.conv'] = f'{target}.1.module.1.module'
                names[f'{source}.mlp.1'] = f'{target}.1.module.2'
                names[f'{source}.mlp.2.conv'] = f'{target}.1.module.3.module'
                ignored.add(f'{source}.attn.mask' if index % 3 == 0 else f'{source}.attn.relative_position_index')

        expected = self.state_dict()
        converted = {}
        for key, tensor in state_dict.items():
            if key in ignored:
                continue
            prefix, _, parameter = key.rpartition('.')
            if prefix not in names:
                raise ValueError(f'Unrecognized official key: {key}')
            target = f'{names[prefix]}.{parameter}'
            if target not in expected or tensor.shape != expected[target].shape:
                raise ValueError(f'Incompatible official parameter: {key} -> {target}')
            if target.startswith(('down_initial.', 'down_fusion.')):
                tensor = tensor.reshape(2 * self.n_features, 2, 2, self.n_features).permute(0, 3, 2, 1)
                tensor = tensor.reshape_as(expected[target])
            converted[target] = tensor
        if missing := expected.keys() - converted.keys():
            raise ValueError(f'Missing official parameters: {sorted(missing)}')
        return converted
