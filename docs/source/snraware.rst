SNRAware denoising networks
==========================

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
two image channels. For complex image patches and real maps of matching shape::

    # image and gmap: (batch, frames, 64, 64)
    x = torch.stack((image.real, image.imag, gmap), dim=1)
    with torch.inference_mode():
        prediction = model(x)
    denoised = torch.complex(prediction[:, 0], prediction[:, 1])

Images must use the upstream SNR-unit convention; a g-map describes spatial noise
amplification, not arbitrary image intensity. The model performs no calibration
or intensity normalization. Dividing by an image maximum changes the meaning of
its input. The upstream inference code optionally multiplies the image channels
by a positive factor and divides the predicted image by the same factor; smaller
factors increase denoising strength. The g-map is unchanged by that adjustment.

Published weights use learned spatial normalization on 64 x 64 patches. Other
``image_size``, ``n_features`` and ``n_heads`` values support training new models,
not resizing the official weights. There is no fixed temporal extent in the
native operations; official checkpoint parity is checked at one and sixteen frames.
Call ``eval()`` for inference to disable stochastic depth.

For larger images, existing ``mr2.operators.PatchOp`` extracts patches and its
adjoint overlap-adds predictions. Weighted blending divides assembled weighted
predictions by assembled weights. Ensure complete grid coverage, compatible
channel shapes and nonzero weights at image boundaries; pad and crop explicitly.
This port does not add a separate tiled inference API.

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

CPU float32 checks at that revision produced the following maximum absolute
errors (one frame / sixteen frames):

* small: 4.8e-06 / 2.9e-06; checkpoint SHA256
  ``08a78f0e61e7657356edfa6f37f1e3e97b6fa380fe41a357584e05123d7caed5``.
* medium: 6.2e-06 / 1.5e-06; checkpoint SHA256
  ``cd504782790b55b4e15422e18c34cac548182ee6aecf0aec4c4fb770de1c3b39``.
* large: 4.5e-06 / 1.1e-06; checkpoint SHA256
  ``505e0614b4a86de9c6546c22eed3c546b99fc07966e1fed0401a3364cbeffdee``.
