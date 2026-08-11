"""Ultimate SD upscale with whole-image model scaling and tiled diffusion.

The default path redraws tiles in Chess order. Batched images retain independent
single-frame VAE dimensions so every image is refined.

Algorithm sources:

* ``repositories/ultimate_sd_upscale/scripts/ultimate-upscale.py`` - ``USDUpscaler``,
  ``USDURedraw.chess_process``, ``calc_rectangle``, ``get_factor(s)``.
* ``usdu_patch.py``   - ``round_length`` (multiple of 8) canvas / tile rounding.
* ``modules/processing.py`` - ``process_images`` (crop region, mask blur, encode,
  sample, decode, alpha-composite).
* ``usdu_utils.py``  - ``get_crop_region``, ``fix_crop_region``, ``expand_crop``.

The seam-fix pass is a no-op for this workflow (``seam_fix_mode="None"``) and is not
implemented; ``mode_type`` supports the "Chess", "Linear" and "None" redraw modes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
from PIL import Image, ImageDraw, ImageFilter
from torch import Tensor
from tqdm.auto import tqdm

from .imageutil import pil_to_tensor, tensor_to_pil
from .upscale import image_upscale_with_model

logger = logging.getLogger(__name__)


def round_length(length: float, multiple: int = 8) -> int:
    """usdu_patch.round_length - round to the nearest multiple (default 8)."""
    return round(length / multiple) * multiple


def get_crop_region(mask: Image.Image, pad: int = 0) -> tuple[int, int, int, int]:
    coordinates = mask.getbbox()
    if coordinates is not None:
        x1, y1, x2, y2 = coordinates
    else:
        x1, y1, x2, y2 = mask.width, mask.height, 0, 0
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, mask.width)
    y2 = min(y2 + pad, mask.height)
    return fix_crop_region((x1, y1, x2, y2), (mask.width, mask.height))


def fix_crop_region(region, image_size):
    image_width, image_height = image_size
    x1, y1, x2, y2 = region
    if x2 < image_width:
        x2 -= 1
    if y2 < image_height:
        y2 -= 1
    return x1, y1, x2, y2


def expand_crop(region, width, height, target_width, target_height):
    x1, y1, x2, y2 = region
    actual_width = x2 - x1
    actual_height = y2 - y1

    width_diff = target_width - actual_width
    x2 = min(x2 + width_diff // 2, width)
    width_diff = target_width - (x2 - x1)
    x1 = max(x1 - width_diff, 0)
    width_diff = target_width - (x2 - x1)
    x2 = min(x2 + width_diff, width)

    height_diff = target_height - actual_height
    y2 = min(y2 + height_diff // 2, height)
    height_diff = target_height - (y2 - y1)
    y1 = max(y1 - height_diff, 0)
    height_diff = target_height - (y2 - y1)
    y2 = min(y2 + height_diff, height)

    return (x1, y1, x2, y2), (target_width, target_height)


def get_factor(num: int) -> int:
    if num == 1:
        return 2
    if num % 4 == 0:
        return 4
    if num % 3 == 0:
        return 3
    if num % 2 == 0:
        return 2
    return 0


def get_factors(scale_factor: int) -> list[int]:
    scales: list[int] = []
    current_scale = 1
    current_scale_factor = get_factor(scale_factor)
    while current_scale_factor == 0:
        scale_factor += 1
        current_scale_factor = get_factor(scale_factor)
    while current_scale < scale_factor:
        current_scale_factor = get_factor(scale_factor // current_scale)
        scales.append(current_scale_factor)
        current_scale = current_scale * current_scale_factor
        if current_scale_factor == 0:
            break
    return scales


@dataclass
class USDUParams:
    upscale_by: float = 2.0
    seed: int = 0
    steps: int = 2
    cfg: float = 1.0
    sampler_name: str = "euler"
    scheduler: str = "simple"
    denoise: float = 0.1
    mode_type: str = "Chess"
    tile_width: int = 512
    tile_height: int = 512
    mask_blur: int = 8
    tile_padding: int = 32
    force_uniform_tiles: bool = True


class UltimateSDUpscale:
    def __init__(self, pipeline, cond: Tensor, upscale_model, params: USDUParams):
        self.pipe = pipeline
        self.cond = cond
        self.upscale_model = upscale_model
        self.p = params

    # --- step 1: whole-image upscale ------------------------------------------
    def _upscale(self, images: list[Image.Image],
                 target: tuple[int, int]) -> list[Image.Image]:
        if self.upscale_model is None:
            return [image.resize(target, resample=Image.Resampling.LANCZOS)
                    for image in images]
        scale_factor = math.ceil(max(target) / max(images[0].width, images[0].height))
        tensor = torch.cat([pil_to_tensor(image) for image in images], dim=0)
        for _ in get_factors(scale_factor):
            tensor = image_upscale_with_model(self.upscale_model, tensor,
                                              device=self.pipe.device)
        return [
            tensor_to_pil(tensor, i).resize(target, resample=Image.Resampling.LANCZOS)
            for i in range(tensor.shape[0])
        ]

    # --- step 2: per tile img2img ---------------------------------------------
    def _process_tile(self, images: list[Image.Image],
                      mask: Image.Image) -> list[Image.Image]:
        p = self.p
        proc_w = round_length(p.tile_width + p.tile_padding)
        proc_h = round_length(p.tile_height + p.tile_padding)

        crop_region = get_crop_region(mask, p.tile_padding)
        if p.force_uniform_tiles:
            x1, y1, x2, y2 = crop_region
            crop_width, crop_height = x2 - x1, y2 - y1
            crop_ratio = crop_width / crop_height
            p_ratio = proc_w / proc_h
            if crop_ratio > p_ratio:
                target_width = crop_width
                target_height = round(crop_width / p_ratio)
            else:
                target_width = round(crop_height * p_ratio)
                target_height = crop_height
            crop_region, _ = expand_crop(crop_region, mask.width, mask.height,
                                         target_width, target_height)
        tile_size = (proc_w, proc_h)

        blurred_mask = mask
        if p.mask_blur > 0:
            blurred_mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

        tiles = [image.crop(crop_region) for image in images]
        initial_tile_size = tiles[0].size
        tiles = [
            tile if tile.size == tile_size
            else tile.resize(tile_size, Image.Resampling.LANCZOS)
            for tile in tiles
        ]

        latent = self.pipe.vae_encode(
            torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
        )
        samples = self.pipe.sample(
            self.cond, latent, seed=p.seed, steps=p.steps, cfg=p.cfg,
            sampler_name=p.sampler_name, scheduler=p.scheduler, denoise=p.denoise,
            force_full_denoise=False, disable_pbar=True,
        )
        decoded = self.pipe.vae_decode(samples)
        results = []
        for i, image in enumerate(images):
            tile_sampled = tensor_to_pil(decoded, i)
            if tile_sampled.size != initial_tile_size:
                tile_sampled = tile_sampled.resize(
                    initial_tile_size, Image.Resampling.LANCZOS
                )

            image_tile_only = Image.new("RGBA", image.size)
            image_tile_only.paste(tile_sampled, crop_region[:2])
            temp = image_tile_only.copy()
            temp.putalpha(blurred_mask)
            image_tile_only.paste(temp, image_tile_only)

            result = image.convert("RGBA")
            result.alpha_composite(image_tile_only)
            results.append(result.convert("RGB"))
        return results

    def _calc_rectangle(self, xi: int, yi: int):
        x1 = xi * self.p.tile_width
        y1 = yi * self.p.tile_height
        return x1, y1, x1 + self.p.tile_width, y1 + self.p.tile_height

    def _redraw(self, images: list[Image.Image],
                rows: int, cols: int) -> list[Image.Image]:
        mode = self.p.mode_type
        if mode == "None":
            return images

        mask = Image.new("L", (images[0].width, images[0].height), "black")
        draw = ImageDraw.Draw(mask)

        if mode == "Linear":
            order = [(yi, xi) for yi in range(rows) for xi in range(cols)]
        elif mode == "Chess":
            tiles = [[(xi % 2 == 0) != (yi > 0 and yi % 2 != 0) for xi in range(cols)]
                     for yi in range(rows)]
            order = [(yi, xi) for yi in range(rows) for xi in range(cols) if tiles[yi][xi]]
            order += [(yi, xi) for yi in range(rows) for xi in range(cols) if not tiles[yi][xi]]
        else:
            raise ValueError(f"unsupported mode_type {mode!r}")

        for yi, xi in tqdm(order, desc="USDU", unit="tile"):
            draw.rectangle(self._calc_rectangle(xi, yi), fill="white")
            images = self._process_tile(images, mask)
            draw.rectangle(self._calc_rectangle(xi, yi), fill="black")
        return images

    # --- entry point -----------------------------------------------------------
    @torch.no_grad()
    def run(self, image: Tensor) -> Tensor:
        init_imgs = [tensor_to_pil(image, i) for i in range(image.shape[0])]
        target = (round_length(init_imgs[0].width * self.p.upscale_by),
                  round_length(init_imgs[0].height * self.p.upscale_by))
        logger.info("USDU: %d images, %dx%d -> %dx%d", len(init_imgs),
                    init_imgs[0].width, init_imgs[0].height, *target)

        upscaled = self._upscale(init_imgs, target)

        tile_width = self.p.tile_width if self.p.tile_width > 0 else self.p.tile_height
        tile_height = self.p.tile_height if self.p.tile_height > 0 else self.p.tile_width
        self.p.tile_width, self.p.tile_height = tile_width, tile_height
        rows = math.ceil(upscaled[0].height / tile_height)
        cols = math.ceil(upscaled[0].width / tile_width)
        logger.info("USDU: %d rows x %d cols of %dx%d tiles", rows, cols, tile_width, tile_height)

        result = self._redraw(upscaled, rows, cols)
        return torch.cat([pil_to_tensor(item) for item in result], dim=0)


def ultimate_sd_upscale(pipeline, image: Tensor, cond: Tensor, upscale_model,
                        params: USDUParams) -> Tensor:
    return UltimateSDUpscale(pipeline, cond, upscale_model, params).run(image)
