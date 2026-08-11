"""Image tensor helpers.

Faithful ports of the ComfyUI helpers used by the workflow:
``comfy/utils.py`` (``lanczos``, ``common_upscale``, ``tiled_scale``) and the
BHWC float32 [0, 1] "IMAGE" convention used by every node.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor


# --- conversions -------------------------------------------------------------------

def pil_to_tensor(img: Image.Image) -> Tensor:
    """PIL RGB -> (1, H, W, C) float32 in [0, 1]."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_pil(image: Tensor, index: int = 0) -> Image.Image:
    """(B, H, W, C) float32 in [0, 1] -> PIL RGB."""
    arr = 255.0 * image[index].detach().cpu().float().numpy()
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# --- resizing ----------------------------------------------------------------------

def lanczos(samples: Tensor, width: int, height: int) -> Tensor:
    """comfy.utils.lanczos - PIL based, operates on BCHW."""
    images = [
        Image.fromarray(np.clip(255.0 * image.movedim(0, -1).cpu().float().numpy(), 0, 255).astype(np.uint8))
        for image in samples
    ]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)


def common_upscale(samples: Tensor, width: int, height: int, upscale_method: str, crop: str) -> Tensor:
    """comfy.utils.common_upscale for 4D BCHW tensors."""
    if crop == "center":
        old_width = samples.shape[-1]
        old_height = samples.shape[-2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    else:
        s = samples

    if upscale_method == "lanczos":
        return lanczos(s, width, height)
    return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)


def image_scale(image: Tensor, upscale_method: str, width: int, height: int, crop: str = "disabled") -> Tensor:
    """``ImageScale`` node (nodes.py) - BHWC in, BHWC out."""
    if width == 0 and height == 0:
        return image
    samples = image.movedim(-1, 1)
    if width == 0:
        width = max(1, round(samples.shape[3] * height / samples.shape[2]))
    elif height == 0:
        height = max(1, round(samples.shape[2] * width / samples.shape[3]))
    return common_upscale(samples, width, height, upscale_method, crop).movedim(1, -1)


def get_image_size(image: Tensor) -> tuple[int, int, int]:
    """``GetImageSize`` node -> (width, height, batch_size)."""
    return int(image.shape[2]), int(image.shape[1]), int(image.shape[0])


# --- tiled upscale-model inference --------------------------------------------------

@torch.no_grad()
def tiled_scale(
    samples: Tensor,
    function: Callable[[Tensor], Tensor],
    tile_x: int = 64,
    tile_y: int = 64,
    overlap: int = 8,
    upscale_amount: float = 4,
    out_channels: int = 3,
    output_device: str | torch.device = "cpu",
    pbar=None,
) -> Tensor:
    """ComfyUI-style tiled scale, batching matching tiles across IMAGE entries."""
    out = torch.zeros(
        (samples.shape[0], out_channels)
        + tuple(round(a * upscale_amount) for a in samples.shape[2:]),
        device=output_device,
    )
    out_div = torch.zeros_like(out)

    for y in range(0, samples.shape[2], tile_y - overlap):
        for x in range(0, samples.shape[3], tile_x - overlap):
            x = max(0, min(samples.shape[-1] - overlap, x))
            y = max(0, min(samples.shape[-2] - overlap, y))
            s_in = samples[:, :, y : y + tile_y, x : x + tile_x]

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)
            feather = round(overlap * upscale_amount)
            for t in range(feather):
                mask[:, :, t : 1 + t, :] *= (1.0 / feather) * (t + 1)
                mask[:, :, mask.shape[2] - 1 - t : mask.shape[2] - t, :] *= (1.0 / feather) * (t + 1)
                mask[:, :, :, t : 1 + t] *= (1.0 / feather) * (t + 1)
                mask[:, :, :, mask.shape[3] - 1 - t : mask.shape[3] - t] *= (1.0 / feather) * (t + 1)

            o_y = round(y * upscale_amount)
            o_x = round(x * upscale_amount)
            out[:, :, o_y : o_y + ps.shape[2], o_x : o_x + ps.shape[3]] += ps * mask
            out_div[:, :, o_y : o_y + ps.shape[2], o_x : o_x + ps.shape[3]] += mask
            if pbar is not None:
                for _ in range(samples.shape[0]):
                    pbar()
    return out / out_div


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap) -> int:
    rows = math.ceil((height - overlap) / (tile_y - overlap))
    cols = math.ceil((width - overlap) / (tile_x - overlap))
    return rows * cols
