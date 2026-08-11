"""Colour correction for the SeedVR2 output.

``wavelet_reconstruction`` / ``adain_color_fix`` are the transfers used by the
SeedVR research code (they originate from StableSR): the upscaled result keeps
its high frequency detail while the low frequency colour comes from the input.

``lab_color_transfer`` additionally matches CIELAB chrominance histograms and
is the pipeline default.

All functions take ``(B, C, H, W)`` tensors; the range only matters for
``lab_color_transfer`` which expects ``[-1, 1]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "adain_color_fix",
    "wavelet_reconstruction",
    "lab_color_transfer",
    "apply_color_correction",
]


# --- AdaIN -----------------------------------------------------------------

def adain_color_fix(content_feat: Tensor, style_feat: Tensor) -> Tensor:
    """Match the per-channel mean/std of ``content_feat`` to ``style_feat``."""
    c_mean = content_feat.mean(dim=(2, 3), keepdim=True)
    c_std = content_feat.std(dim=(2, 3), keepdim=True) + 1e-5
    s_mean = style_feat.mean(dim=(2, 3), keepdim=True)
    s_std = style_feat.std(dim=(2, 3), keepdim=True) + 1e-5
    return (content_feat - c_mean) / c_std * s_std + s_mean


# --- wavelet ---------------------------------------------------------------

def wavelet_blur(image: Tensor, radius: int) -> Tensor:
    """3x3 binomial kernel applied with dilation ``radius``."""
    radius = max(1, min(radius, max(1, min(image.shape[-2:]) // 8)))
    kernel = torch.tensor(
        [[0.0625, 0.125, 0.0625],
         [0.1250, 0.250, 0.1250],
         [0.0625, 0.125, 0.0625]],
        dtype=image.dtype, device=image.device,
    )
    channels = image.shape[1]
    kernel = kernel[None, None].repeat(channels, 1, 1, 1)
    image = F.pad(image, (radius,) * 4, mode="replicate")
    return F.conv2d(image, kernel, groups=channels, dilation=radius)


def wavelet_decomposition(image: Tensor, levels: int = 5) -> tuple[Tensor, Tensor]:
    """Split into (high frequency detail, low frequency colour)."""
    high_freq = torch.zeros_like(image)
    low_freq = image
    for i in range(levels):
        low_freq = wavelet_blur(image, 2 ** i)
        high_freq = high_freq + (image - low_freq)
        image = low_freq
    return high_freq, low_freq


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor) -> Tensor:
    """Content detail + style colour."""
    content_high, _ = wavelet_decomposition(content_feat)
    _, style_low = wavelet_decomposition(style_feat)
    return content_high + style_low


# --- CIELAB ----------------------------------------------------------------

_RGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_XYZ_TO_RGB = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)
_D65 = (0.95047, 1.0, 1.08883)
_EPS = 6.0 / 29.0


def _matmul_channels(x: Tensor, matrix) -> Tensor:
    m = torch.tensor(matrix, dtype=x.dtype, device=x.device)
    b, c, h, w = x.shape
    out = torch.matmul(x.permute(0, 2, 3, 1).reshape(-1, 3), m.T)
    return out.reshape(b, h, w, 3).permute(0, 3, 1, 2)


def rgb_to_lab(rgb: Tensor) -> Tensor:
    """sRGB in ``[0, 1]`` -> CIELAB (D65)."""
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = _matmul_channels(linear, _RGB_TO_XYZ)
    white = torch.tensor(_D65, dtype=xyz.dtype, device=xyz.device)[None, :, None, None]
    xyz = xyz / white
    f = torch.where(xyz > _EPS ** 3, xyz.clamp_min(1e-12) ** (1.0 / 3.0),
                    xyz / (3 * _EPS ** 2) + 4.0 / 29.0)
    lightness = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([lightness, a, b], dim=1)


def lab_to_rgb(lab: Tensor) -> Tensor:
    """CIELAB (D65) -> sRGB in ``[0, 1]``."""
    fy = (lab[:, 0:1] + 16.0) / 116.0
    fx = fy + lab[:, 1:2] / 500.0
    fz = fy - lab[:, 2:3] / 200.0
    f = torch.cat([fx, fy, fz], dim=1)
    xyz = torch.where(f > _EPS, f ** 3, 3 * _EPS ** 2 * (f - 4.0 / 29.0))
    white = torch.tensor(_D65, dtype=xyz.dtype, device=xyz.device)[None, :, None, None]
    linear = _matmul_channels(xyz * white, _XYZ_TO_RGB)
    linear = linear.clamp(0.0, 1.0)
    return torch.where(linear > 0.0031308,
                       1.055 * linear.clamp_min(1e-12) ** (1 / 2.4) - 0.055,
                       12.92 * linear).clamp(0.0, 1.0)


def match_histogram(source: Tensor, reference: Tensor) -> Tensor:
    """Quantile (CDF) matching of ``source`` onto ``reference``."""
    shape = source.shape
    src = source.flatten()
    ref = reference.flatten()
    src_sorted, src_index = torch.sort(src)
    ref_sorted, _ = torch.sort(ref)
    if src_sorted.numel() == ref_sorted.numel():
        matched_sorted = ref_sorted
    else:
        quantiles = torch.linspace(0, 1, src_sorted.numel(), device=source.device)
        idx = (quantiles * (ref_sorted.numel() - 1)).long().clamp_(0, ref_sorted.numel() - 1)
        matched_sorted = ref_sorted[idx]
    inverse = torch.argsort(src_index)
    return matched_sorted[inverse].reshape(shape)


def lab_color_transfer(content_feat: Tensor, style_feat: Tensor,
                       luminance_weight: float = 0.8) -> Tensor:
    """Wavelet reconstruction followed by CIELAB histogram matching.

    Inputs and outputs are in ``[-1, 1]``.  ``luminance_weight`` is how much of
    the content's own L* channel is kept.
    """
    dtype = content_feat.dtype
    style = style_feat.float()
    content_feat = content_feat.float()
    if content_feat.shape[-2:] != style.shape[-2:]:
        style = F.interpolate(style, size=content_feat.shape[-2:], mode="bilinear",
                              align_corners=False)
    content = wavelet_reconstruction(content_feat, style)

    content_lab = rgb_to_lab(content.add(1.0).mul(0.5).clamp_(0.0, 1.0))
    style_lab = rgb_to_lab(style.add(1.0).mul(0.5).clamp_(0.0, 1.0))

    matched_a = match_histogram(content_lab[:, 1], style_lab[:, 1])
    matched_b = match_histogram(content_lab[:, 2], style_lab[:, 2])
    if luminance_weight < 1.0:
        matched_l = match_histogram(content_lab[:, 0], style_lab[:, 0])
        lightness = content_lab[:, 0] * luminance_weight + matched_l * (1.0 - luminance_weight)
    else:
        lightness = content_lab[:, 0]

    result = lab_to_rgb(torch.stack([lightness, matched_a, matched_b], dim=1))
    return result.mul_(2.0).sub_(1.0).to(dtype)


def apply_color_correction(method: str, content_feat: Tensor, style_feat: Tensor) -> Tensor:
    """Dispatch on the ``color_correction`` name used by the workflow."""
    if method in (None, "none", "disabled"):
        return content_feat
    if method == "wavelet":
        return wavelet_reconstruction(content_feat, style_feat)
    if method == "adain":
        return adain_color_fix(content_feat, style_feat)
    if method == "lab":
        return lab_color_transfer(content_feat, style_feat)
    raise ValueError(f"unknown color correction method: {method!r}")
