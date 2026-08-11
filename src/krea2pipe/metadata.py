"""Portable, versioned generation metadata for saved images."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import torch
from PIL import Image

from . import __version__
from .prompting import EXPANSION_SOURCE

if TYPE_CHECKING:
    from .workflow import WorkflowConfig

SCHEMA = "krea2pipe.generation"
SCHEMA_VERSION = 1
PNG_KEY = "krea2pipe"


def _model_id(name: str) -> str:
    """Keep relative model identifiers but do not expose machine-local absolute paths."""
    expanded = Path(name).expanduser()
    return expanded.name if expanded.is_absolute() else name


def build_generation_manifest(
    cfg: WorkflowConfig,
    prompt: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Capture all settings that affect generated pixels in a JSON-safe mapping."""
    seedvr2 = cfg.seedvr2
    prompt_metadata: dict[str, Any] = {
        "positive": prompt,
        "negative": cfg.negative_prompt,
    }
    if cfg.prompt_theme is not None:
        prompt_metadata["expansion"] = {
            "theme": cfg.prompt_theme,
            "index": cfg.prompt_index,
            "seed": cfg.prompt_seed,
            "system_prompt": EXPANSION_SOURCE,
        }
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "software": {
            "krea2pipe": __version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": cfg.device,
        },
        "prompt": prompt_metadata,
        "models": {
            "diffusion": _model_id(cfg.unet_name),
            "text_encoder": _model_id(cfg.clip_name),
            "vae": _model_id(cfg.vae_name),
            "loras": [
                {"name": _model_id(name), "strength": strength}
                for name, strength in cfg.resolve_loras()
            ],
            "usdu_upscaler": _model_id(cfg.upscale_model_name),
            "blend_upscaler": _model_id(cfg.blend_upscale_model_name),
            "seedvr2_dit": _model_id(seedvr2.dit_model),
            "seedvr2_vae": _model_id(seedvr2.vae_model),
        },
        "base": {
            "aspect_ratio": cfg.aspect_ratio,
            "megapixels": cfg.megapixels,
            "multiple_of": cfg.multiple_of,
            "width": width,
            "height": height,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
            "steps": cfg.steps,
            "cfg": cfg.cfg,
            "sampler": cfg.sampler_name,
            "scheduler": cfg.scheduler,
            "dtype": str(cfg.dtype).removeprefix("torch."),
        },
        "stages": {
            "usdu": {
                "enabled": cfg.run_usdu,
                "seed": cfg.usdu_seed,
                "steps": cfg.usdu_steps,
                "cfg": cfg.usdu_cfg,
                "sampler": cfg.usdu_sampler_name,
                "scheduler": cfg.usdu_scheduler,
                "denoise": cfg.usdu_denoise,
                "upscale_by": cfg.usdu_upscale_by,
                "mode": cfg.usdu_mode,
                "mask_blur": cfg.usdu_mask_blur,
                "tile_padding": cfg.usdu_tile_padding,
                "force_uniform_tiles": cfg.usdu_force_uniform_tiles,
            },
            "color_match": {
                "enabled": cfg.run_color_match,
                "method": cfg.color_match_method,
                "strength": cfg.color_match_strength,
            },
            "seedvr2": {
                "enabled": cfg.run_seedvr2,
                "variant": seedvr2.variant,
                "seed": seedvr2.seed,
                "resolution": seedvr2.resolution,
                "max_resolution": seedvr2.max_resolution,
                "sample_steps": seedvr2.sample_steps,
                "cfg_scale": seedvr2.cfg_scale,
                "cfg_rescale": seedvr2.cfg_rescale,
                "color_correction": seedvr2.color_correction,
                "cond_noise_scale": seedvr2.cond_noise_scale,
                "dtype": seedvr2.dtype,
                "keep_dit_resident": seedvr2.keep_dit_resident,
                "vae_tile": seedvr2.vae_tile,
                "vae_tile_overlap": seedvr2.vae_tile_overlap,
            },
            "blend": {
                "enabled": cfg.run_blend and cfg.run_seedvr2,
                "factor": cfg.blend_factor,
                "mode": cfg.blend_mode,
            },
        },
        "output": {
            "format": cfg.extension,
            "quality": cfg.quality,
        },
    }


def manifest_for_image(
    manifest: Mapping[str, Any],
    *,
    stage: str,
    batch_index: int,
    width: int,
    height: int,
    image_format: str,
) -> dict[str, Any]:
    """Add file-specific details without mutating the shared run manifest."""
    return {
        **manifest,
        "image": {
            "stage": stage,
            "batch_index": batch_index,
            "width": width,
            "height": height,
            "format": image_format,
        },
    }


def encode_manifest(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def read_generation_manifest(path: str | os.PathLike) -> dict[str, Any] | None:
    """Read a krea2pipe manifest from PNG, JPEG, or WebP."""
    file = Path(path)
    with Image.open(file) as image:
        if image.format == "PNG":
            raw = image.info.get(PNG_KEY)
        else:
            import piexif

            exif = piexif.load(str(file))
            raw = exif.get("0th", {}).get(piexif.ImageIFD.ImageDescription)
            if isinstance(raw, bytes):
                raw = raw.rstrip(b"\0").decode("utf-8")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"invalid {PNG_KEY} metadata in {file}")
    manifest = json.loads(raw)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported {PNG_KEY} metadata in {file}")
    return manifest
