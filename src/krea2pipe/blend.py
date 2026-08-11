"""Optional model-upscale, resize, and blend stage from the workflow."""

from __future__ import annotations

import torch
from torch import Tensor

from .imageutil import common_upscale, get_image_size, image_scale
from .upscale import image_upscale_with_model

MODES = ("normal", "multiply", "screen", "overlay", "soft_light", "difference")


def _g(x: Tensor) -> Tensor:
    return torch.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, torch.sqrt(x))


def _blend_mode(img1: Tensor, img2: Tensor, mode: str) -> Tensor:
    if mode == "normal":
        return img2
    if mode == "multiply":
        return img1 * img2
    if mode == "screen":
        return 1 - (1 - img1) * (1 - img2)
    if mode == "overlay":
        return torch.where(
            img1 <= 0.5,
            2 * img1 * img2,
            1 - 2 * (1 - img1) * (1 - img2),
        )
    if mode == "soft_light":
        return torch.where(
            img2 <= 0.5,
            img1 - (1 - 2 * img2) * img1 * (1 - img1),
            img1 + (2 * img2 - 1) * (_g(img1) - img1),
        )
    if mode == "difference":
        return img1 - img2
    choices = ", ".join(MODES)
    raise ValueError(f"unsupported blend mode {mode!r}; choose one of: {choices}")


def image_blend(image1: Tensor, image2: Tensor, blend_factor: float,
                blend_mode: str = "normal") -> Tensor:
    """Blend two BHWC image batches, matching ComfyUI's ImageBlend node."""
    image2 = image2.to(image1.device)
    if image1.shape != image2.shape:
        image2 = image2.permute(0, 3, 1, 2)
        image2 = common_upscale(
            image2, image1.shape[2], image1.shape[1], "bicubic", "center"
        )
        image2 = image2.permute(0, 2, 3, 1)
    blended = _blend_mode(image1, image2, blend_mode)
    blended = image1 * (1 - blend_factor) + blended * blend_factor
    return torch.clamp(blended, 0, 1)


def upscale_and_blend(upscale_model, source: Tensor, primary: Tensor,
                      blend_factor: float = 0.4,
                      blend_mode: str = "normal") -> Tensor:
    """Upscale ``source`` with a model, resize it, then blend into ``primary``."""
    target_w, target_h, _ = get_image_size(primary)
    hires = image_upscale_with_model(upscale_model, source)
    scaled = image_scale(hires, "lanczos", target_w, target_h, "disabled")
    del hires
    result = image_blend(primary, scaled, blend_factor, blend_mode)
    del scaled
    return result
