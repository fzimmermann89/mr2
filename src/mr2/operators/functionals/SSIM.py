"""Structural Similarity Index (SSIM) functional."""

from typing import Literal, cast

import torch

from mr2.operators.Operator import Operator


def ssim3d(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    data_range: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
    window_size: int = 7,
    reduction: Literal['full', 'volume', 'none'] = 'full',
) -> torch.Tensor:
    """Compute SSIM between two 3D volumes.

    For complex inputs, the SSIM is calculated separately for the real and imaginary parts and the results are averaged.
    For more details, see `SSIM`.

    Parameters
    ----------
    target
        Ground truth tensor, shape `(... z, y, x)` or broadcastable with prediction.
    prediction
        Predicted tensor, same shape as target.
    data_range
        Value range if the data. If `None`, the max-to-min per volume of the target will be used.
    weight
        Weight (or mask) tensor, same shape as target.
        Only windows with all weight values > 0 (or `True`) are considered.
        Windows will be weighted by the average value of the weight in the window.
        If `None`, all windows are considered without weighting.
    k1
        Constant for SSIM computation. Commonly 0.01.
    k2
        Constant for SSIM computation. Commonly 0.03.
    window_size
        Size of the rectangular sliding window.
        If any of the last 3 dimensions of the target is of size 1, the window size in this dimension will
        also be set to 1.

    reduction
        If `full`, return the weighted mean SSIM over all windows, i.e. a scalar value.
        If `volume`, return one value for each volume, i.e, an ``target.ndim - 3`` dimensional tensor.
        If `none`, return the unpadded SSIM map of shape
        ``(... z - window_size + 1 ,  y - window_size + 1 , x - window_size + 1)``.

    Returns
    -------
        mean SSIM or SSIM map (see `reduction`)
    """
    if target.is_complex() or prediction.is_complex():
        real_ssim = ssim3d(
            target.real,
            prediction.real,
            weight=weight,
            k1=k1,
            k2=k2,
            window_size=window_size,
            data_range=data_range,
            reduction=reduction,
        )
        imag_ssim = ssim3d(
            target.imag if target.is_complex() else torch.zeros_like(target),
            prediction.imag if prediction.is_complex() else torch.zeros_like(prediction),
            weight=weight,
            k1=k1,
            k2=k2,
            window_size=window_size,
            data_range=data_range,
            reduction=reduction,
        )
        return (real_ssim + imag_ssim) / 2
    if target.ndim < 3:
        raise ValueError('Input must be at least 3D (z, y, x)')
    window = tuple(window_size if s > 1 else 1 for s in target.shape[-3:])  # To support 1D and 2D uses
    if weight is not None:
        if (weight < 0).any():
            raise ValueError('Mask contains negative values')
        target, prediction, weight = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.broadcast_tensors(target, prediction, weight)
        )
    else:
        target, prediction = cast(tuple[torch.Tensor, torch.Tensor], torch.broadcast_tensors(target, prediction))
    leading_shape = target.shape[:-3]
    window_volume = window[0] * window[1] * window[2]

    def local_mean(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.reshape(-1, 1, *tensor.shape[-3:])
        mean = torch.nn.functional.avg_pool3d(tensor, kernel_size=window, stride=1)
        return mean.reshape(*leading_shape, *mean.shape[-3:])

    def local_max(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.reshape(-1, 1, *tensor.shape[-3:])
        maximum = torch.nn.functional.max_pool3d(tensor, kernel_size=window, stride=1)
        return maximum.reshape(*leading_shape, *maximum.shape[-3:])

    if weight is not None:
        weight = weight.to(dtype=torch.float32)
        # Set weights to 0 for windows that are not fully inside the mask.
        valid = local_mean((weight > 0).to(dtype=weight.dtype)) == 1
        weight = local_mean(weight) * valid
        weight /= weight.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1)

    if data_range is None:
        if weight is None:
            target_max = target.amax((-3, -2, -1), keepdim=True)
            target_min = target.amin((-3, -2, -1), keepdim=True)
        else:
            target_max = torch.where(weight > 0, local_max(target), -torch.inf).amax((-3, -2, -1), keepdim=True)
            target_min = torch.where(weight > 0, -local_max(-target), torch.inf).amin((-3, -2, -1), keepdim=True)
        data_range_ = target_max - target_min
    else:
        data_range_ = data_range
    data_range_ = data_range_.clamp_min(torch.finfo(target.dtype).eps)

    mean_tgt = local_mean(target)
    mean_img = local_mean(prediction)

    mean_tgt_tgt = local_mean(target.square())
    mean_img_img = local_mean(prediction.square())
    mean_tgt_img = local_mean(target * prediction)

    cov_norm = window_volume / max(window_volume - 1, 1)
    cov_tgt = cov_norm * (mean_tgt_tgt - mean_tgt * mean_tgt)
    cov_img = cov_norm * (mean_img_img - mean_img * mean_img)
    cov_tgt_img = cov_norm * (mean_tgt_img - mean_tgt * mean_img)

    c1 = (k1 * data_range_) ** 2
    c2 = (k2 * data_range_) ** 2
    a1 = 2 * mean_tgt * mean_img + c1
    a2 = 2 * cov_tgt_img + c2
    b1 = mean_tgt**2 + mean_img**2 + c1
    b2 = cov_tgt + cov_img + c2

    ssim_map = (a1 * a2) / (b1 * b2)
    if weight is not None:
        ssim_map = torch.where(weight > 0, ssim_map, torch.zeros_like(ssim_map))

    if reduction == 'full':
        if weight is not None:
            return (ssim_map * weight).sum((-3, -2, -1)).mean()
        return ssim_map.mean()
    elif reduction == 'volume':
        if weight is not None:
            return (ssim_map * weight).sum(dim=(-3, -2, -1))
        return ssim_map.mean(dim=(-3, -2, -1))
    elif reduction == 'none':
        return ssim_map


class SSIM(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """(masked) SSIM functional."""

    def __init__(
        self,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        *,
        data_range: torch.Tensor | None = None,
        window_size: int = 7,
        k1: float = 0.01,
        k2: float = 0.03,
        reduction: Literal['full', 'volume', 'none'] = 'full',
    ) -> None:
        """Initialize Volume SSIM.

        The Structural Similarity Index Measure [SSIM]_ is used to measure the similarity between two images.
        It considers luminance, contrast and structure differences between the images.
        SSIM values range from -1 to 1, where 1 indicates perfect structural similarity.
        Two random images have an SSIM of 0.

        Calculates the SSIM using a rectangular sliding window. If a boolean mask is used as weight, only the windows
        that are fully inside the mask are considered. This can be used to ignore the background of the volumes.
        Calculates the SSIM for a volume, i.e, 3D patches of the last three dimensions of the input are used.
        To apply it to 2D data, add a singleton dimension.
        For complex inputs, the SSIM is calculated separately for the real and imaginary parts and the results
        are averaged.

        For stability, it is advised to provide the data range. Otherwise it is estimates per volume from the target.

        References
        ----------
        .. [SSIM] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
               Image quality assessment: from error visibility to structural similarity.
               IEEE TMI, 13(4), 600-612. https://doi.org/10.1109/TIP.2003.819861


        Parameters
        ----------
        target
            Target volume. At least three dimensional.
        weight
            Either a boolean mask. Only windows with all values `True` will be considered.
            Or a weight tensor of the same shape as the target. Each window will be weighted by
            the average value of the weight in the window. Only windows with all weight values > 0 will be considered.
            Or `None`, meaning all windows are used.
        data_range
            Value range if the data. If None, the max-to-min per volume of the target will be used.
        window_size
            Size of the windows used in SSIM. Usually ``7`` or ``11``.
            If any of the last 3 dimensions of the target is of size 1, the window size in this dimension will
            also be set to 1.
        k1
            Constant. Usually ``0.01`` and rarely changed.
        k2
            Constant. Usually ``0.03`` and rarely changed.
        reduction
            If ``full``, return the weighted mean SSIM over all windows, i.e. a scalar value.
            If ``volume``, return one value for each volume, i.e, an ``target.ndim - 3`` dimensional tensor.
            If ``none``, return the unpadded SSIM map.
        """
        super().__init__()
        self.target = target
        self.weight = weight
        self.k1 = k1
        self.k2 = k2
        self.window_size = window_size
        self.data_range = data_range
        self.reduction = reduction

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Calculate the Structural Similarity Index (SSIM) between the input and a target.

        This method computes the SSIM between the provided input tensor `x` and
        the `target` tensor defined during initialization.

        Parameters
        ----------
        x
            Input tensor, expected to be comparable to `target`.

        Returns
        -------
            The SSIM value(s), the shape of which depends on the `reduction` parameter.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward of SSIM.

        .. note::
            Prefer calling the instance of the SSIM as ``operator(x)`` over directly calling this method.
            See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        ssim = ssim3d(
            self.target,
            x,
            weight=self.weight,
            k1=self.k1,
            k2=self.k2,
            window_size=self.window_size,
            data_range=self.data_range,
            reduction=self.reduction,
        )

        return (ssim,)
