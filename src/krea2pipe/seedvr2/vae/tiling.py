"""Spatially tiled VAE encode / decode for the SeedVR2 causal video VAE.

The official SeedVR repository always runs the VAE on the whole frame.  The
``ComfyUI-SeedVR2_VideoUpscaler`` node adds spatial tiling, and ``krea2.json``
enables it (``encode_tiled``/``decode_tiled`` with a 1024 px tile and a 128 px
overlap).  Tiling is what keeps the 4096x4096 stage both fast and inside a
sane VRAM budget, so we reproduce it here.

Tiles are blended with a separable raised-cosine ramp over the overlap region.
Fades are only applied on *interior* tile edges so the outer image border keeps
full weight.  Tile size / overlap are expressed in **pixel** space and are
converted to latent space for the encoder (whose input is pixels but whose
output is latents) and used directly in latent space for the decoder.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = ["tiled_encode", "tiled_decode"]


def _ramp(steps: int, device, dtype) -> torch.Tensor:
    t = torch.linspace(0, 1, steps=steps, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(t * math.pi)


def _edge_weights(length: int, overlap: int, fade_start: bool, fade_end: bool,
                  ramp: torch.Tensor | None, device, dtype) -> torch.Tensor:
    w = torch.ones((length,), device=device, dtype=dtype)
    ov = max(0, min(overlap, length - 1))
    if ov > 0 and ramp is not None:
        if fade_start:
            w[:ov] = ramp[:ov]
        if fade_end:
            w[-ov:] = 1 - ramp[:ov]
    return w


def tiled_encode(vae, x: torch.Tensor, tile_size: Tuple[int, int] = (512, 512),
                 tile_overlap: Tuple[int, int] = (64, 64)) -> torch.Tensor:
    """Encode ``x`` (B, C, F, H, W pixels) tile by tile, blending in latent space."""
    if x.ndim != 5:
        x = x.unsqueeze(2)

    _, _, _, height, width = x.shape
    tile_h, tile_w = tile_size
    if height <= tile_h and width <= tile_w:
        return vae.slicing_encode(x)

    scale = vae.spatial_downsample_factor
    overlap_h, overlap_w = tile_overlap

    lat_tile_h = max(1, tile_h // scale)
    lat_tile_w = max(1, tile_w // scale)
    lat_ov_h = max(0, min(overlap_h // scale, lat_tile_h - 1))
    lat_ov_w = max(0, min(overlap_w // scale, lat_tile_w - 1))
    stride_h = max(1, lat_tile_h - lat_ov_h)
    stride_w = max(1, lat_tile_w - lat_ov_w)

    lat_h_total = (height + scale - 1) // scale
    lat_w_total = (width + scale - 1) // scale

    logger.info("tiled VAE encode %dx%d (tile=%s overlap=%s)", height, width,
                tile_size, tile_overlap)

    ramp_h = _ramp(lat_ov_h, x.device, x.dtype) if lat_ov_h > 0 else None
    ramp_w = _ramp(lat_ov_w, x.device, x.dtype) if lat_ov_w > 0 else None

    result = None
    count = None

    for y_lat in range(0, lat_h_total, stride_h):
        y_lat_end = min(y_lat + lat_tile_h, lat_h_total)
        if y_lat > 0 and (y_lat_end - y_lat) <= lat_ov_h:
            continue
        for x_lat in range(0, lat_w_total, stride_w):
            x_lat_end = min(x_lat + lat_tile_w, lat_w_total)
            if x_lat > 0 and (x_lat_end - x_lat) <= lat_ov_w:
                continue

            y0, x0 = y_lat * scale, x_lat * scale
            y1 = min(y_lat_end * scale, height)
            x1 = min(x_lat_end * scale, width)

            tile = vae.slicing_encode(x[:, :, :, y0:y1, x0:x1])

            if result is None:
                b_out, c_out, f_lat = tile.shape[:3]
                result = torch.zeros((b_out, c_out, f_lat, lat_h_total, lat_w_total),
                                     device=tile.device, dtype=tile.dtype)
                count = torch.zeros((1, 1, 1, lat_h_total, lat_w_total),
                                    device=tile.device, dtype=tile.dtype)

            eff_h = min(y_lat_end - y_lat, tile.shape[3], result.shape[3] - y_lat)
            eff_w = min(x_lat_end - x_lat, tile.shape[4], result.shape[4] - x_lat)
            tile = tile[:, :, : result.shape[2], :eff_h, :eff_w]

            w_h = _edge_weights(eff_h, lat_ov_h, y_lat > 0, y_lat_end < lat_h_total,
                                ramp_h, tile.device, tile.dtype).view(1, 1, 1, eff_h, 1)
            w_w = _edge_weights(eff_w, lat_ov_w, x_lat > 0, x_lat_end < lat_w_total,
                                ramp_w, tile.device, tile.dtype).view(1, 1, 1, 1, eff_w)
            tile = tile * w_h * w_w

            result[:, :, : tile.shape[2], y_lat:y_lat + eff_h, x_lat:x_lat + eff_w] += tile
            count[:, :, :, y_lat:y_lat + eff_h, x_lat:x_lat + eff_w].addcmul_(w_h, w_w)

    result.div_(count.clamp(min=1e-6))
    if x.shape[2] == 1:
        result = result.squeeze(2)
    return result


def tiled_decode(vae, z: torch.Tensor, tile_size: Tuple[int, int] = (512, 512),
                 tile_overlap: Tuple[int, int] = (64, 64)) -> torch.Tensor:
    """Decode ``z`` (B, C, F, H, W latents) tile by tile, blending in pixel space."""
    if z.ndim != 5:
        z = z.unsqueeze(2)

    _, _, _, height, width = z.shape
    scale = vae.spatial_downsample_factor
    tile_h, tile_w = tile_size
    overlap_h, overlap_w = tile_overlap

    lat_tile_h = max(1, tile_h // scale)
    lat_tile_w = max(1, tile_w // scale)
    if height <= lat_tile_h and width <= lat_tile_w:
        return vae.slicing_decode(z)

    lat_ov_h = max(0, min(overlap_h // scale, lat_tile_h - 1))
    lat_ov_w = max(0, min(overlap_w // scale, lat_tile_w - 1))
    stride_h = max(1, lat_tile_h - lat_ov_h)
    stride_w = max(1, lat_tile_w - lat_ov_w)

    logger.info("tiled VAE decode %dx%d latent (tile=%s overlap=%s)", height, width,
                tile_size, tile_overlap)

    ramp_h = _ramp(overlap_h, z.device, z.dtype) if overlap_h > 0 else None
    ramp_w = _ramp(overlap_w, z.device, z.dtype) if overlap_w > 0 else None

    result = None
    count = None

    for y_lat in range(0, height, stride_h):
        y_lat_end = min(y_lat + lat_tile_h, height)
        if y_lat > 0 and (y_lat_end - y_lat) <= lat_ov_h:
            continue
        for x_lat in range(0, width, stride_w):
            x_lat_end = min(x_lat + lat_tile_w, width)
            if x_lat > 0 and (x_lat_end - x_lat) <= lat_ov_w:
                continue

            tile = vae.slicing_decode(z[:, :, :, y_lat:y_lat_end, x_lat:x_lat_end])

            if result is None:
                b_out, c_out, f_out = tile.shape[:3]
                result = torch.zeros((b_out, c_out, f_out, height * scale, width * scale),
                                     device=tile.device, dtype=tile.dtype)
                count = torch.zeros((1, 1, 1, height * scale, width * scale),
                                    device=tile.device, dtype=tile.dtype)

            y0, y1 = y_lat * scale, y_lat_end * scale
            x0, x1 = x_lat * scale, x_lat_end * scale
            h_out, w_out = y1 - y0, x1 - x0

            w_h = _edge_weights(h_out, overlap_h, y_lat > 0, y_lat_end < height,
                                ramp_h, tile.device, tile.dtype).view(1, 1, 1, h_out, 1)
            w_w = _edge_weights(w_out, overlap_w, x_lat > 0, x_lat_end < width,
                                ramp_w, tile.device, tile.dtype).view(1, 1, 1, 1, w_out)
            tile = tile * w_h * w_w

            result[:, :, : tile.shape[2], y0:y1, x0:x1] += tile
            count[:, :, :, y0:y1, x0:x1].addcmul_(w_h, w_w)

    result.div_(count.clamp(min=1e-6))
    if z.shape[2] == 1:
        result = result.squeeze(2)
    return result
