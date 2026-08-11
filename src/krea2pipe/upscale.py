"""``ImageUpscaleWithModel`` (comfy_extras/nodes_upscale_model.py) - spandrel + tiled_scale."""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from tqdm.auto import tqdm

from .imageutil import get_tiled_scale_steps, tiled_scale

logger = logging.getLogger(__name__)


@torch.no_grad()
def image_upscale_with_model(upscale_model, image: Tensor, tile: int = 512,
                             overlap: int = 32, device: str = "cuda") -> Tensor:
    """BHWC [0,1] -> BHWC [0,1] upscaled by ``upscale_model.scale``."""
    upscale_model.to(device)
    in_img = image.movedim(-1, -3).to(device)

    oom = True
    s = None
    while oom:
        try:
            steps = in_img.shape[0] * get_tiled_scale_steps(
                in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
            bar = tqdm(total=steps, desc="upscale", leave=False)
            s = tiled_scale(in_img, lambda a: upscale_model(
                                a.float().contiguous(memory_format=torch.channels_last)),
                            tile_x=tile, tile_y=tile,
                            overlap=overlap, upscale_amount=upscale_model.scale,
                            pbar=bar.update, output_device="cpu")
            bar.close()
            oom = False
        except torch.cuda.OutOfMemoryError:
            tile //= 2
            logger.warning("upscale model OOM, retrying with tile=%d", tile)
            if tile < 128:
                raise
    return torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
