SNRAware denoising networks
===========================

``mr2.nn.nets.SNRAware`` implements the small, medium and large imaging
transformers released with `SNRAware <https://arxiv.org/abs/2503.18162>`_. Each
uses a two-resolution HRNet and temporal, local and global convolutional attention.
The variants repeat that attention sequence once, twice and four times per stage.

Loading official weights
------------------------

Download the matching ``.pts`` file from
`Microsoft's model repository <https://huggingface.co/microsoft/SNRAware>`_. No
upstream package, YAML parser, or automatic download is needed::

    import torch
    from mr2.nn.nets import SNRAware

    model = SNRAware('small').eval()
    official = torch.jit.load('snraware_small_model.pts', map_location='cpu')
    converted = model.convert_state_dict(official.state_dict())
    model.load_state_dict(converted)
    torch.save(converted, 'snraware_small_mr2.pt')

Only load trusted TorchScript files. Subsequently the converted weights can be
loaded with ``torch.load('snraware_small_mr2.pt', weights_only=True)``. The converter
requires matching channels, heads, image size and variant. It rejects unknown or
missing parameters and only discards the official unused causal masks and
regenerated relative-position indices.

Input and output
----------------

The model accepts real tensors shaped ``(batch, 3, frames, 64, 64)``. Channels
are real image, imaginary image and g-factor map, in that order. It returns the
two image channels. Before inference, establish the noise scale from a compatible
noise measurement or reconstruction noise propagation. Let ``noise_std`` be the
positive standard deviation of one real noise component **before g-factor
amplification**, in the image's current units. The local real/imaginary noise
standard deviation must then be ``noise_std * gmap``. If the noise is spatially
uniform, use an all-ones map, ``gmap = torch.ones_like(image.real)``. For an already calibrated SNR-unit image, use
``noise_std = 1``. This assumes circular complex noise with equal real/imaginary
component variances, not a standard deviation estimated from magnitude pixels.

For complex image patches and real maps of matching shape::

    # image and gmap: (batch, frames, 64, 64)
    # noise_std: measured baseline component SD, broadcastable to image
    noise_std = torch.as_tensor(noise_std, device=image.device, dtype=image.real.dtype)
    image_snr = image / noise_std
    x = torch.stack((image_snr.real, image_snr.imag, gmap), dim=1)
    x = x.to(next(model.parameters()))  # match model device and real dtype
    with torch.inference_mode():
        prediction = model(x)
    denoised = torch.complex(prediction[:, 0], prediction[:, 1]) * noise_std.to(prediction)

The output is restored to the image's original intensity units; the g-map is
unchanged. When measured local component SD varies spatially, its decomposition
into baseline scale and g-map must come from the reconstruction/calibration;
do not divide by that local map and simultaneously supply it as g-factor noise
amplification. The model does not estimate or calibrate noise, and a g-map does
not represent arbitrary image intensity. Dividing by an image maximum changes
the meaning of its input.

Published weights use learned spatial normalization on 64 x 64 patches. Other
``image_size``, ``n_features`` and ``n_heads`` values support training new models,
not resizing the official weights. There is no fixed temporal extent in the
native operations; official checkpoint parity is checked at one and sixteen frames.
Call ``eval()`` for inference to disable stochastic depth.

For larger images, use existing ``mr2.operators.PatchOp`` and its overlap-add
adjoint with explicit padding and nonzero blend weights. This port adds no tiled
inference API.

Validation
----------

``ci/validate_snraware.py`` compares converted models against local official
checkpoints at one and sixteen frames::

    PYTHONPATH=src python ci/validate_snraware.py \
        --variant small --checkpoint /path/to/snraware_small_model.pts

The script prints checkpoint SHA256, parameter count and maximum/RMS output errors
and asserts float32 agreement with absolute and relative tolerances of ``5e-5``.
No checkpoint downloads are required by ordinary tests.

The official code was inspected at commit
``0cd04986fb3bdeb137c2f059a1b5e1c29a35e150``. Checkpoints were retrieved from model
repository revision ``2e42c3d3581fabb9b7472f65b04417ae9112a368``. The upstream
code and weights are provided for research and technical development; see their
model card and ``LICENSE-SNRAware`` for attribution.
