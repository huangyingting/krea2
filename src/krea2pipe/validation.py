"""Fail-fast validation for standalone pipeline configuration and resources."""

from __future__ import annotations

import math
import os
import tempfile
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from . import blend, color_match, loaders, nodes, sampling

if TYPE_CHECKING:
    from .workflow import WorkflowConfig


class DeviceConfigurationError(RuntimeError):
    """Raised when the requested compute device cannot run the pipeline."""


def _positive(value: int | float, option: str) -> None:
    _number(value, option)
    if value <= 0:
        raise ValueError(f"{option} must be greater than zero")


def _number(value: object, option: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{option} must be a finite number")


def _integer(value: object, option: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{option} must be an integer")


def _string(value: object, option: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value):
        suffix = "" if allow_empty else " non-empty"
        raise ValueError(f"{option} must be a{suffix} string")


def validate_settings(cfg: WorkflowConfig) -> None:
    """Validate values without touching the filesystem or loading models."""
    for option, value in (
        ("model-root", cfg.model_root),
        ("unet", cfg.unet_name),
        ("clip", cfg.clip_name),
        ("vae", cfg.vae_name),
        ("upscale-model", cfg.upscale_model_name),
        ("blend-upscale-model", cfg.blend_upscale_model_name),
        ("aspect-ratio", cfg.aspect_ratio),
        ("sampler", cfg.sampler_name),
        ("scheduler", cfg.scheduler),
        ("usdu-sampler", cfg.usdu_sampler_name),
        ("usdu-scheduler", cfg.usdu_scheduler),
        ("usdu-mode", cfg.usdu_mode),
        ("color-match-method", cfg.color_match_method),
        ("blend-mode", cfg.blend_mode),
        ("state-dir", cfg.state_dir),
        ("output-dir", cfg.output_dir),
        ("filename", cfg.filename),
        ("extension", cfg.extension),
        ("time-format", cfg.time_format),
        ("device", cfg.device),
        ("SeedVR2 checkpoint", cfg.seedvr2.dit_model),
        ("SeedVR2 VAE", cfg.seedvr2.vae_model),
    ):
        _string(value, option)
    _string(cfg.subdir, "subdir", allow_empty=True)
    if cfg.prompt is not None:
        _string(cfg.prompt, "prompt", allow_empty=True)
    _string(cfg.negative_prompt, "negative-prompt", allow_empty=True)
    expansion_values = (cfg.prompt_theme, cfg.prompt_index, cfg.prompt_seed)
    if any(value is not None for value in expansion_values):
        if any(value is None for value in expansion_values):
            raise ValueError("prompt theme, index, and seed must be configured together")
        _string(cfg.prompt_theme, "prompt theme")
        _string(cfg.theme_system_prompt, "theme system prompt")
        _integer(cfg.prompt_index, "prompt index")
        _integer(cfg.prompt_seed, "prompt seed")
        if cfg.prompt_index < 0:
            raise ValueError("prompt index must be non-negative")
        if not 0 <= cfg.prompt_seed <= (1 << 64) - 1:
            raise ValueError("prompt seed must be an unsigned 64-bit integer")
    if "/" in cfg.time_format or "\\" in cfg.time_format:
        raise ValueError("time-format must not contain path separators")
    if cfg.loras is not None:
        if not isinstance(cfg.loras, list):
            raise ValueError("loras must be a list of (name, strength) pairs")
        for index, item in enumerate(cfg.loras, start=1):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"LoRA {index} must be a (name, strength) pair")
            _string(item[0], f"LoRA {index} name")
            _number(item[1], f"LoRA {index} strength")
    else:
        _string(cfg.lora_name, "lora-name")
        _number(cfg.lora_strength, "lora-strength")
    for option, value in (
        ("batch-size", cfg.batch_size),
        ("steps", cfg.steps),
        ("multiple-of", cfg.multiple_of),
        ("quality", cfg.quality),
        ("seed", cfg.seed),
    ):
        _integer(value, option)
    for option, value in (
        ("megapixels", cfg.megapixels),
        ("cfg", cfg.cfg),
        ("color-match-strength", cfg.color_match_strength),
        ("blend-factor", cfg.blend_factor),
    ):
        _number(value, option)
    for option, value in (
        ("run-usdu", cfg.run_usdu),
        ("run-color-match", cfg.run_color_match),
        ("run-seedvr2", cfg.run_seedvr2),
        ("run-blend", cfg.run_blend),
        ("save", cfg.save),
        ("save-intermediates", cfg.save_intermediates),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{option} must be true or false")
    loaders.normalize_model_root(cfg.model_root)
    _positive(cfg.batch_size, "batch-size")
    _positive(cfg.steps, "steps")
    _positive(cfg.megapixels, "megapixels")
    _positive(cfg.multiple_of, "multiple-of")
    if cfg.multiple_of % 8:
        raise ValueError("multiple-of must be divisible by 8")
    _positive(cfg.quality, "quality")
    if cfg.quality > 100:
        raise ValueError("quality must be between 1 and 100")
    if cfg.cfg != 1.0:
        raise ValueError("cfg must be 1.0; this Krea 2 pipeline has no negative pass")
    if cfg.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
        raise ValueError(f"unsupported model dtype {cfg.dtype}")
    if cfg.extension not in {"png", "jpg", "jpeg", "webp"}:
        raise ValueError("extension must be png, jpg, jpeg, or webp")
    if cfg.sampler_name not in sampling.SAMPLERS:
        raise ValueError(f"unsupported sampler {cfg.sampler_name!r}")
    if cfg.scheduler not in sampling.SCHEDULERS:
        raise ValueError(f"unsupported scheduler {cfg.scheduler!r}")
    if not 0 <= cfg.seed <= (1 << 64) - 1:
        raise ValueError("seed must be an unsigned 64-bit integer")
    for index, (name, strength) in enumerate(cfg.resolve_loras(), start=1):
        _string(name, f"LoRA {index} name")
        _number(strength, f"LoRA {index} strength")
    if (cfg.width is None) != (cfg.height is None):
        raise ValueError("width and height must be configured together")
    if cfg.width is not None:
        _integer(cfg.width, "width")
        _integer(cfg.height, "height")
        _positive(cfg.width, "width")
        _positive(cfg.height, "height")
        if cfg.width % 8 or cfg.height % 8:
            raise ValueError("width and height must be multiples of 8")
    else:
        cfg.resolve_size()
    if cfg.run_usdu:
        _integer(cfg.usdu_steps, "usdu-steps")
        _integer(cfg.usdu_seed, "usdu-seed")
        _integer(cfg.usdu_mask_blur, "USDU mask blur")
        _integer(cfg.usdu_tile_padding, "USDU tile padding")
        _number(cfg.usdu_upscale_by, "usdu-upscale-by")
        _number(cfg.usdu_denoise, "usdu-denoise")
        _number(cfg.usdu_cfg, "USDU cfg")
        _positive(cfg.usdu_steps, "usdu-steps")
        _positive(cfg.usdu_upscale_by, "usdu-upscale-by")
        if cfg.usdu_sampler_name not in sampling.SAMPLERS:
            raise ValueError(f"unsupported USDU sampler {cfg.usdu_sampler_name!r}")
        if cfg.usdu_scheduler not in sampling.SCHEDULERS:
            raise ValueError(f"unsupported USDU scheduler {cfg.usdu_scheduler!r}")
        if cfg.usdu_mode not in {"Linear", "Chess", "None"}:
            raise ValueError(f"unsupported USDU mode {cfg.usdu_mode!r}")
        if not 0 <= cfg.usdu_denoise <= 1:
            raise ValueError("usdu-denoise must be between 0 and 1")
        if cfg.usdu_cfg != 1.0:
            raise ValueError("USDU cfg must be 1.0; negative guidance is not implemented")
        if not 0 <= cfg.usdu_seed <= (1 << 64) - 1:
            raise ValueError("usdu-seed must be an unsigned 64-bit integer")
        if cfg.usdu_mask_blur < 0:
            raise ValueError("USDU mask blur must be non-negative")
        if cfg.usdu_tile_padding < 0:
            raise ValueError("USDU tile padding must be non-negative")
    if cfg.run_seedvr2:
        _integer(cfg.seedvr2.resolution, "seedvr2-resolution")
        _integer(cfg.seedvr2.max_resolution, "seedvr2-max-resolution")
        _integer(cfg.seedvr2.sample_steps, "SeedVR2 sample steps")
        _integer(cfg.seedvr2.seed, "seedvr2-seed")
        _integer(cfg.seedvr2.vae_tile, "SeedVR2 VAE tile")
        _integer(cfg.seedvr2.vae_tile_overlap, "SeedVR2 VAE tile overlap")
        if cfg.seedvr2.model_dir is not None:
            _string(cfg.seedvr2.model_dir, "SeedVR2 model_dir")
        if cfg.seedvr2.embeds_dir is not None:
            _string(cfg.seedvr2.embeds_dir, "SeedVR2 embeds_dir")
        _positive(cfg.seedvr2.resolution, "seedvr2-resolution")
        _positive(cfg.seedvr2.max_resolution, "seedvr2-max-resolution")
        _positive(cfg.seedvr2.sample_steps, "SeedVR2 sample steps")
        if not 0 <= cfg.seedvr2.seed <= (1 << 32) - 1:
            raise ValueError("seedvr2-seed must be an unsigned 32-bit integer")
        if cfg.seedvr2.color_correction not in {"none", "wavelet", "adain", "lab"}:
            raise ValueError(
                f"unsupported SeedVR2 color correction {cfg.seedvr2.color_correction!r}"
            )
        if cfg.seedvr2.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"unsupported SeedVR2 dtype {cfg.seedvr2.dtype!r}")
        if cfg.seedvr2.vae_tile:
            _positive(cfg.seedvr2.vae_tile, "seedvr2 vae_tile")
            if not 0 <= cfg.seedvr2.vae_tile_overlap < cfg.seedvr2.vae_tile:
                raise ValueError(
                    "SeedVR2 VAE tile overlap must be non-negative and smaller than the tile"
                )
    if not 0 <= cfg.color_match_strength <= 1:
        raise ValueError("color-match-strength must be between 0 and 1")
    if cfg.color_match_method not in color_match.METHODS:
        raise ValueError(f"unsupported color-match method {cfg.color_match_method!r}")
    if not 0 <= cfg.blend_factor <= 1:
        raise ValueError("blend-factor must be between 0 and 1")
    if cfg.blend_mode not in blend.MODES:
        raise ValueError(f"unsupported blend mode {cfg.blend_mode!r}")


def _required_model(cfg: WorkflowConfig, kind: str, name: str, label: str) -> Path:
    return Path(loaders.require_model(kind, name, cfg.model_root, label))


def _required_seedvr2_model(cfg: WorkflowConfig, name: str, label: str) -> Path:
    path = Path(name).expanduser()
    if not path.is_absolute():
        model_dir = cfg.seedvr2.model_dir or os.path.join(cfg.model_root, "SEEDVR2")
        model_dir_path = Path(model_dir).expanduser()
        if not model_dir_path.is_absolute():
            raise ValueError("SeedVR2 model_dir must be an absolute path")
        path = model_dir_path / path
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{label} is not readable: {path}")
    return path


def validate_models(cfg: WorkflowConfig) -> None:
    """Ensure every checkpoint needed by enabled stages exists and is readable."""
    relative_models = [
        cfg.unet_name,
        cfg.clip_name,
        cfg.vae_name,
        *(name for name, strength in cfg.resolve_loras() if strength),
    ]
    if cfg.run_usdu:
        relative_models.append(cfg.upscale_model_name)
    if cfg.run_blend and cfg.run_seedvr2:
        relative_models.append(cfg.blend_upscale_model_name)
    if any(not Path(name).expanduser().is_absolute() for name in relative_models):
        root = Path(loaders.normalize_model_root(cfg.model_root))
        if not root.is_dir():
            raise FileNotFoundError(f"model-root directory not found: {root}")

    _required_model(cfg, "diffusion_models", cfg.unet_name, "Krea 2 checkpoint")
    _required_model(cfg, "text_encoders", cfg.clip_name, "text encoder")
    _required_model(cfg, "vae", cfg.vae_name, "image VAE")
    for name, strength in cfg.resolve_loras():
        if strength:
            _required_model(cfg, "loras", name, f"LoRA {name!r}")
    if cfg.run_usdu:
        _required_model(
            cfg, "upscale_models", cfg.upscale_model_name, "USDU upscale model"
        )
    if cfg.run_seedvr2:
        _required_seedvr2_model(cfg, cfg.seedvr2.dit_model, "SeedVR2 checkpoint")
        _required_seedvr2_model(cfg, cfg.seedvr2.vae_model, "SeedVR2 VAE")
        if not cfg.seedvr2.config_path().is_file():
            raise FileNotFoundError(
                f"SeedVR2 configuration not found: {cfg.seedvr2.config_path()}"
            )
        from .seedvr2.runner import resolve_embeddings

        resolve_embeddings(cfg.seedvr2)
    if cfg.run_blend and cfg.run_seedvr2:
        _required_model(
            cfg,
            "upscale_models",
            cfg.blend_upscale_model_name,
            "blend upscale model",
        )


def validate_device(device_name: str) -> None:
    """Validate the configured torch device before loading any checkpoints."""
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device {device_name!r}: {exc}") from exc
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise DeviceConfigurationError(
            f"device {device_name!r} requires CUDA, but CUDA is unavailable"
        )
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise DeviceConfigurationError(
            f"device {device_name!r} does not exist; found "
            f"{torch.cuda.device_count()} CUDA device(s)"
        )


def _prepare_writable_directory(value: str, option: str) -> Path:
    path = Path(value).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"cannot create {option} {path}: {exc}") from exc
    if not path.is_dir():
        raise NotADirectoryError(f"{option} is not a directory: {path}")
    if not os.access(path, os.W_OK):
        raise PermissionError(f"{option} is not writable: {path}")
    try:
        with tempfile.NamedTemporaryFile(dir=path):
            pass
    except OSError as exc:
        raise OSError(f"cannot write to {option} {path}: {exc}") from exc
    return path


def prepare_output(cfg: WorkflowConfig) -> None:
    """Create state and image destinations before expensive inference begins."""
    if not cfg.save:
        return
    _prepare_writable_directory(cfg.state_dir, "state-dir")
    path = _prepare_writable_directory(cfg.output_dir, "output-dir")
    width, height = cfg.resolve_size()
    target, _basename = nodes.resolve_output_target(
        str(path),
        cfg.filename,
        cfg.subdir,
        cfg.time_format,
        0,
        width,
        height,
    )
    try:
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target):
            pass
    except OSError as exc:
        raise OSError(f"cannot write to output destination {target}: {exc}") from exc


def preflight(cfg: WorkflowConfig) -> None:
    validate_settings(cfg)
    validate_device(cfg.device)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        with torch.cuda.device(device):
            if (
                cfg.dtype == torch.bfloat16
                or (cfg.run_seedvr2 and cfg.seedvr2.dtype == "bfloat16")
            ) and not torch.cuda.is_bf16_supported():
                raise DeviceConfigurationError(
                    f"device {cfg.device!r} does not support bfloat16; use float16"
                )
    if cfg.run_seedvr2 and cfg.seedvr2.device != cfg.device:
        raise DeviceConfigurationError(
            "SeedVR2 device must match the pipeline device "
            f"({cfg.seedvr2.device!r} != {cfg.device!r})"
        )
    validate_models(cfg)
    prepare_output(cfg)
