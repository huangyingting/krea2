"""Command-line front end for the standalone Krea 2 pipeline."""

from __future__ import annotations

import argparse
import logging
import math
import os
import secrets
import sqlite3
import sys
import time
from collections.abc import Collection
from contextlib import nullcontext
from dataclasses import replace
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch

from . import batch, blend, color_match, loaders, sampling
from .config import config_options, load_config, write_config_template
from .prompting import EXPANSION_SYSTEM_PROMPT
from .seedvr2 import SeedVR2Config
from .validation import DeviceConfigurationError, validate_settings
from .workflow import (
    PipelineOutOfMemoryError,
    WorkflowConfig,
    expand_theme,
    run_workflow,
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
MAX_SEED = (1 << 64) - 1
MAX_SEEDVR2 = (1 << 32) - 1
DEFAULT_SOURCE_RECONCILE_INTERVAL = 300.0
DEFAULT_THEME_PROMPT_COUNT = 0
logger = logging.getLogger(__name__)


def _seed_value(value: object) -> int | str:
    if isinstance(value, str) and value.strip().lower() == "random":
        return "random"
    if isinstance(value, bool):
        raise ValueError("must be an integer or 'random'")
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("must be an integer or 'random'") from exc
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"must be between 0 and {MAX_SEED}")
    return seed


def _seed_argument(value: str) -> int | str:
    try:
        return _seed_value(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _resolve_seed(value: object, option: str, max_seed: int = MAX_SEED) -> int:
    try:
        parsed = _seed_value(value)
    except ValueError as exc:
        raise SystemExit(f"{option}: {exc}") from exc
    if parsed == "random":
        return secrets.randbits(max_seed.bit_length())
    if parsed > max_seed:
        raise SystemExit(f"{option}: must be between 0 and {max_seed}")
    return parsed


def _parse_loras(value: object) -> list[tuple[str, float]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise SystemExit("loras: expected an array of { name, strength } entries")
    result: list[tuple[str, float]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, dict):
            extra = set(item) - {"name", "strength"}
            if extra or "name" not in item or "strength" not in item:
                raise SystemExit(
                    f"loras[{index}]: expected only 'name' and 'strength'"
                )
            name, strength = item["name"], item["strength"]
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            name, strength = item
        else:
            raise SystemExit(f"loras[{index}]: expected {{ name, strength }}")
        if not isinstance(name, str) or not name.strip():
            raise SystemExit(f"loras[{index}].name: expected a non-empty string")
        if isinstance(strength, bool):
            raise SystemExit(f"loras[{index}].strength: expected a number")
        try:
            numeric_strength = float(strength)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"loras[{index}].strength: expected a number") from exc
        if not math.isfinite(numeric_strength):
            raise SystemExit(f"loras[{index}].strength: expected a finite number")
        result.append((name, numeric_strength))
    return result


class _AppendLoRA(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not getattr(namespace, "_cli_loras", False):
            setattr(namespace, self.dest, [])
            namespace.lora_name = None
            namespace._cli_loras = True
        getattr(namespace, self.dest).append(values)


def _require_choice(value: str, choices: Collection[str], option: str) -> str:
    if not isinstance(value, str) or value not in choices:
        allowed = ", ".join(sorted(choices))
        raise SystemExit(f"{option}: unsupported value {value!r}; choose one of: {allowed}")
    return value


def _unit_float(value: object, option: str) -> float:
    if isinstance(value, bool):
        raise SystemExit(f"{option}: expected a number from 0 to 1")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{option}: expected a number from 0 to 1") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise SystemExit(f"{option}: expected a number from 0 to 1")
    return number


def build_config_parser() -> argparse.ArgumentParser:
    """Build the internal parser that defines TOML settings and defaults."""
    p = argparse.ArgumentParser(
        prog="krea2pipe",
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = WorkflowConfig()

    g = p.add_argument_group("input mode")
    g.add_argument("--prompt-mode", default="source",
                   help="Select source or theme mode; ignored by CLI --prompt")
    g.add_argument("--sources", default=None, metavar="PATHS",
                   help="SOURCE MODE: Git-style include and !exclude glob list")
    g.add_argument("--reconcile-interval", default=None, metavar="SECONDS",
                   help="SOURCE MODE ONLY: full-scan safety interval; omitted uses 300 "
                        "seconds, while 0 processes the current queue and exits")
    g.add_argument("--theme", default=None,
                   help="THEME MODE: use resident Qwen to expand this theme into prompts")
    g.add_argument("--theme-system-prompt", default=EXPANSION_SYSTEM_PROMPT,
                   help="THEME MODE ONLY: Qwen system instructions used to expand themes")
    g.add_argument("--prompt-count", default=None,
                   help="THEME MODE ONLY: prompt limit; omitted or 0 runs continuously")

    g = p.add_argument_group("resolution")
    g.add_argument("--aspect-ratio", default=d.aspect_ratio,
                   help="1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, or 21:9")
    g.add_argument("--megapixels", type=float, default=d.megapixels,
                   help="approximate base-image area before USDU and SeedVR2 upscaling")
    g.add_argument("--multiple-of", type=int, default=d.multiple_of,
                   help="round computed width and height to this model-safe multiple")
    g.add_argument("--width", type=int, default=None,
                   help="exact base width override; set height as well")
    g.add_argument("--height", type=int, default=None,
                   help="exact base height override; set width as well")
    g.add_argument("--batch-size", type=int, default=d.batch_size,
                   help="number of independent images generated for each prompt")

    g = p.add_argument_group("sampling")
    g.add_argument("--seed", type=_seed_argument, default="random",
                   help="integer seed or 'random' (chosen once at startup)")
    g.add_argument("--steps", type=int, default=d.steps,
                   help="base Krea 2 diffusion steps")
    g.add_argument("--cfg", type=float, default=d.cfg,
                   help="base guidance scale; 1 matches Krea 2 and skips negative guidance")
    g.add_argument("--sampler", dest="sampler_name", default=d.sampler_name,
                   choices=sorted(sampling.SAMPLERS),
                   help="base sampler: euler or euler_ancestral")
    g.add_argument("--scheduler", default=d.scheduler, choices=sorted(sampling.SCHEDULERS),
                   help="base scheduler: normal, sgm_uniform, or simple")

    g = p.add_argument_group("models")
    g.add_argument("--model-root", default=d.model_root,
                   help="absolute root containing diffusion_models, text_encoders, "
                        "vae, loras, upscale_models, and SEEDVR2")
    g.add_argument("--unet", dest="unet_name", default=d.unet_name,
                   help="Krea 2 checkpoint under model-root/diffusion_models")
    g.add_argument("--clip", dest="clip_name", default=d.clip_name,
                   help="Qwen text encoder under model-root/text_encoders")
    g.add_argument("--vae", dest="vae_name", default=d.vae_name,
                   help="Qwen Image VAE under model-root/vae")
    g.add_argument("--lora", dest="lora_name", default=None,
                   help="legacy single LoRA; prefer --add-lora or TOML 'loras'")
    g.add_argument("--lora-strength", type=float, default=d.lora_strength,
                   help="0 disables the LoRA entirely")
    g.add_argument(
        "--add-lora", dest="loras", action=_AppendLoRA, nargs=2,
        metavar=("FILE", "STRENGTH"), default=None,
        help="LoRAs under model-root/loras; add entries with individual strengths",
    )
    g.add_argument("--upscale-model", dest="upscale_model_name",
                   default=d.upscale_model_name,
                   help="model under model-root/upscale_models used by UltimateSDUpscale")

    g = p.add_argument_group("UltimateSDUpscale")
    g.add_argument("--usdu-upscale-by", type=float, default=d.usdu_upscale_by,
                   help="UltimateSDUpscale output scale relative to the base image")
    g.add_argument("--usdu-seed", type=_seed_argument, default="random",
                   help="integer seed or 'random' (chosen once at startup)")
    g.add_argument("--usdu-steps", type=int, default=d.usdu_steps,
                   help="diffusion steps per USDU tile")
    g.add_argument("--usdu-sampler", dest="usdu_sampler_name",
                   default=d.usdu_sampler_name, choices=sorted(sampling.SAMPLERS),
                   help="upscale sampler: euler or euler_ancestral")
    g.add_argument("--usdu-scheduler", default=d.usdu_scheduler,
                   choices=sorted(sampling.SCHEDULERS),
                   help="upscale scheduler: normal, sgm_uniform, or simple")
    g.add_argument("--usdu-denoise", type=float, default=d.usdu_denoise,
                   help="tile denoise strength from 0 to 1; higher values alter more detail")
    g.add_argument("--usdu-mode", default=d.usdu_mode, choices=["Linear", "Chess", "None"],
                   help="tile traversal/seam-fixing mode: Linear, Chess, or None")

    g = p.add_argument_group("ColorMatch")
    g.add_argument("--no-color-match", dest="run_color_match", action="store_false",
                   help="skip the standalone ColorMatch stage")
    g.add_argument("--color-match-method", default=d.color_match_method,
                   choices=color_match.METHODS,
                   help="color transfer: default, hm, reinhard, mvgd, mkl, "
                        "hm-mvgd-hm, or hm-mkl-hm")
    g.add_argument("--color-match-strength", type=float, default=d.color_match_strength,
                   help="transfer strength from 0 (off) to 1 (full)")

    g = p.add_argument_group("SeedVR2")
    g.add_argument("--seedvr2-resolution", type=int, default=d.seedvr2.resolution,
                   help="target short-edge pixels; aspect ratio is preserved")
    g.add_argument("--seedvr2-max-resolution", type=int, default=d.seedvr2.max_resolution,
                   help="long-edge pixel cap used to limit VRAM")
    g.add_argument("--seedvr2-seed", type=_seed_argument, default="random",
                   help="32-bit integer seed or 'random' (chosen once at startup)")
    g.add_argument("--seedvr2-model", default=d.seedvr2.dit_model,
                   help="SeedVR2 DiT checkpoint under model-root/SEEDVR2")
    g.add_argument("--seedvr2-color-correction", default=d.seedvr2.color_correction,
                   choices=["none", "wavelet", "adain", "lab"],
                   help="SeedVR2-internal correction: none, wavelet, adain, or lab")

    g = p.add_argument_group("Upscale blend")
    g.add_argument("--no-blend", dest="run_blend", action="store_false",
                   help="skip the separate model-upscale/Lanczos/blend stage")
    g.add_argument("--blend-upscale-model", dest="blend_upscale_model_name",
                   default=d.blend_upscale_model_name,
                   help="model under model-root/upscale_models used only by final blend")
    g.add_argument("--blend-factor", type=float, default=d.blend_factor,
                   help="blend strength from 0 to 1")
    g.add_argument("--blend-mode", default=d.blend_mode, choices=blend.MODES,
                   help="normal, multiply, screen, overlay, soft_light, or difference")

    g = p.add_argument_group("output")
    g.add_argument("--output-dir", "-o", default=d.output_dir,
                   help="persistent output root; also stores the batch resume ledger")
    g.add_argument("--filename", default=d.filename,
                   help="filename template; supports %time, %date, %width, %height, %counter")
    g.add_argument("--subdir", default=d.subdir,
                   help="subdirectory created below output-dir; empty writes to the root")
    g.add_argument("--extension", default=d.extension,
                   choices=["png", "jpg", "jpeg", "webp"],
                   help="output image format: png, jpg, jpeg, or webp")
    g.add_argument("--quality", type=int, default=d.quality,
                   help="JPEG/WebP quality from 1 to 100")
    g.add_argument("--save-intermediates", action="store_true",
                   help="also write the base / USDU / ColorMatch / SeedVR2 stages")
    g.add_argument("--no-save", dest="save", action="store_false",
                   help="do not write files; incompatible with resumable batch/service mode")

    g = p.add_argument_group("stages / runtime")
    g.add_argument("--no-usdu", dest="run_usdu", action="store_false",
                   help="skip UltimateSDUpscale")
    g.add_argument("--no-seedvr2", dest="run_seedvr2", action="store_false",
                   help="skip the SeedVR2 upscaler (and the blend)")
    g.add_argument("--device", default=d.device,
                   help="PyTorch device; use cuda for the supported production path")
    g.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES),
                   help="model compute precision; keep bfloat16 unless debugging "
                        "or using hardware without bfloat16 support")
    g.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="minimum severity written to the console and log file")
    g.add_argument("--log-file", default=None,
                   help="optional rotating log file (10 MiB, five backups)")
    g.add_argument("--verbose", "-v", action="store_true",
                   help="enable DEBUG logging, overriding log-level")
    return p


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line interface."""
    p = argparse.ArgumentParser(
        prog="krea2pipe",
        description="Standalone Krea 2 image-generation and upscaling pipeline.",
    )
    p.add_argument(
        "--config",
        "-c",
        default=None,
        metavar="FILE",
        help="TOML generation configuration",
    )
    action = p.add_mutually_exclusive_group()
    action.add_argument(
        "--generate-config",
        nargs="?",
        const="krea2pipe.toml",
        default=None,
        metavar="FILE",
        help="write a complete default TOML config and exit "
             "(default file: krea2pipe.toml)",
    )
    action.add_argument(
        "--prompt",
        "-p",
        default=None,
        help="generate this prompt once, ignoring the configured source or theme",
    )
    action.add_argument(
        "--reset-status",
        action="store_true",
        help="clear completion state for the configured source or theme and exit",
    )
    return p


def config_from_args(args: argparse.Namespace) -> WorkflowConfig:
    defaults = WorkflowConfig()
    try:
        model_root = loaders.normalize_model_root(args.model_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    loras = _parse_loras(args.loras)
    if args.lora_name is not None:
        loras = [(args.lora_name, args.lora_strength)]
    elif loras is None:
        loras = [(defaults.lora_name, args.lora_strength)]
    sampler_name = _require_choice(args.sampler_name, sampling.SAMPLERS, "sampler")
    scheduler = _require_choice(args.scheduler, sampling.SCHEDULERS, "scheduler")
    usdu_sampler = _require_choice(
        args.usdu_sampler_name, sampling.SAMPLERS, "usdu-sampler"
    )
    usdu_scheduler = _require_choice(
        args.usdu_scheduler, sampling.SCHEDULERS, "usdu-scheduler"
    )
    color_method = _require_choice(
        args.color_match_method, color_match.METHODS, "color-match-method"
    )
    color_strength = _unit_float(args.color_match_strength, "color-match-strength")
    blend_mode = _require_choice(args.blend_mode, blend.MODES, "blend-mode")
    blend_factor = _unit_float(args.blend_factor, "blend-factor")
    dtype_name = _require_choice(args.dtype, DTYPES, "dtype")
    seedvr2 = SeedVR2Config(
        dit_model=args.seedvr2_model,
        model_dir=os.path.join(model_root, "SEEDVR2"),
        seed=_resolve_seed(args.seedvr2_seed, "seedvr2-seed", MAX_SEEDVR2),
        resolution=args.seedvr2_resolution,
        max_resolution=args.seedvr2_max_resolution,
        color_correction=args.seedvr2_color_correction,
        device=args.device,
        dtype=dtype_name,
    )
    cfg = WorkflowConfig(
        prompt=args.prompt,
        theme_system_prompt=args.theme_system_prompt,
        model_root=model_root,
        unet_name=args.unet_name,
        clip_name=args.clip_name,
        vae_name=args.vae_name,
        lora_name=args.lora_name or defaults.lora_name,
        lora_strength=args.lora_strength,
        loras=loras,
        upscale_model_name=args.upscale_model_name,
        blend_upscale_model_name=args.blend_upscale_model_name,
        aspect_ratio=args.aspect_ratio,
        megapixels=args.megapixels,
        multiple_of=args.multiple_of,
        batch_size=args.batch_size,
        width=args.width,
        height=args.height,
        seed=_resolve_seed(args.seed, "seed"),
        steps=args.steps,
        cfg=args.cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        usdu_seed=_resolve_seed(args.usdu_seed, "usdu-seed"),
        usdu_steps=args.usdu_steps,
        usdu_sampler_name=usdu_sampler,
        usdu_scheduler=usdu_scheduler,
        usdu_denoise=args.usdu_denoise,
        usdu_upscale_by=args.usdu_upscale_by,
        usdu_mode=args.usdu_mode,
        color_match_method=color_method,
        color_match_strength=color_strength,
        run_color_match=args.run_color_match,
        seedvr2=seedvr2,
        blend_factor=blend_factor,
        blend_mode=blend_mode,
        output_dir=args.output_dir,
        filename=args.filename,
        subdir=args.subdir,
        extension=args.extension,
        quality=args.quality,
        save_intermediates=args.save_intermediates,
        run_usdu=args.run_usdu,
        run_seedvr2=args.run_seedvr2,
        run_blend=args.run_blend,
        save=args.save,
        device=args.device,
        dtype=DTYPES[dtype_name],
    )
    try:
        validate_settings(cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the small public CLI and merge it over the TOML-backed defaults."""
    cli_args = build_parser().parse_args(argv)
    settings_parser = build_config_parser()
    if cli_args.config and not cli_args.generate_config:
        configured = load_config(
            cli_args.config,
            config_options(settings_parser),
        )
        settings_parser.set_defaults(**configured)
    args = settings_parser.parse_args([])
    vars(args).update(vars(cli_args))
    return args


def _render(cfg: WorkflowConfig):
    result = run_workflow(cfg, progress=lambda m: print(m, flush=True))
    for path in result.paths:
        print(path)
    if not result.paths:
        print(f"{result.width}x{result.height} image produced (saving disabled)")
    return result


def _render_pending(
    cfg: WorkflowConfig,
    queue: batch.SourceQueue,
    announce_empty: bool = False,
    max_prompts: int | None = None,
) -> int:
    """Render pending prompts directly from the durable source index."""
    pending: int | None = None
    if announce_empty:
        total, done, pending = queue.counts()
        print(
            f"{total} prompts, {done} already rendered, {pending} to go",
            flush=True,
        )
    rendered = 0
    while (prompt := queue.next_pending()) is not None:
        rendered += 1
        queue_status = (
            f"{pending - rendered + 1} pending"
            if pending is not None
            else "source queue"
        )
        print(
            f"=== [{queue_status}] "
            f"{prompt.file.name}:{prompt.line} {prompt.text[:70]}",
            flush=True,
        )
        # Stable per-line seeds make a resumed run reproduce the same image.
        # Models and compiled graphs are cached process-wide, so everything
        # after the first image skips loading and compilation.
        offset = prompt.seed_offset
        result = _render(replace(
            cfg,
            prompt=prompt.text,
            seed=(cfg.seed + offset) & MAX_SEED,
            usdu_seed=(cfg.usdu_seed + offset) & MAX_SEED,
            seedvr2=replace(
                cfg.seedvr2,
                seed=(cfg.seedvr2.seed + offset) & MAX_SEEDVR2,
            ),
        ))
        if not result.paths:
            raise RuntimeError(f"no output was saved for {prompt.file}:{prompt.line}")
        queue.mark(prompt)
        if max_prompts is not None and rendered >= max_prompts:
            break
    return rendered


def _run_source_queue(
    cfg: WorkflowConfig,
    source_spec: batch.SourceSpec,
    reconcile_interval: float,
) -> int:
    """Run an event-driven source queue with periodic full reconciliation."""
    with batch.SourceQueue(source_spec, cfg.output_dir) as queue:
        if reconcile_interval == 0:
            queue.reconcile()
            _render_pending(cfg, queue, announce_empty=True)
            return 0

        try:
            with batch.SourceWatcher(source_spec) as watcher:
                queue.reconcile()
                last_reconcile = time.monotonic()
                announce = True
                while True:
                    rendered = _render_pending(
                        cfg,
                        queue,
                        announce_empty=announce,
                        max_prompts=1,
                    )
                    announce = False
                    remaining = max(
                        0.1,
                        reconcile_interval - (time.monotonic() - last_reconcile),
                    )
                    changed_paths = watcher.wait(0 if rendered else remaining)
                    if changed_paths:
                        queue.update_paths(changed_paths)
                    if time.monotonic() - last_reconcile >= reconcile_interval:
                        queue.reconcile()
                        last_reconcile = time.monotonic()
        except batch.SourceWatchError as exc:
            logger.warning(
                "filesystem events unavailable (%s); using %.0fs reconciliation fallback",
                exc,
                reconcile_interval,
            )
            _run_reconciliation_fallback(cfg, queue, reconcile_interval)


def _run_reconciliation_fallback(
    cfg: WorkflowConfig,
    queue: batch.SourceQueue,
    reconcile_interval: float,
) -> None:
    """Keep reconciliation periodic when filesystem events are unavailable."""
    queue.reconcile()
    last_reconcile = time.monotonic()
    announce = True
    while True:
        rendered = _render_pending(
            cfg,
            queue,
            announce_empty=announce,
            max_prompts=1,
        )
        announce = False
        now = time.monotonic()
        if now - last_reconcile >= reconcile_interval:
            queue.reconcile()
            last_reconcile = time.monotonic()
        elif not rendered:
            time.sleep(reconcile_interval - (now - last_reconcile))
            queue.reconcile()
            last_reconcile = time.monotonic()


def _render_theme(cfg: WorkflowConfig, theme: str, prompt_count: int) -> int:
    progress = batch.ThemeProgress(
        cfg.output_dir,
        theme,
        {"base": cfg.seed, "usdu": cfg.usdu_seed, "seedvr2": cfg.seedvr2.seed},
        cfg.theme_system_prompt,
    )
    seeds = progress.seeds
    cfg = replace(
        cfg,
        seed=seeds["base"],
        usdu_seed=seeds["usdu"],
        seedvr2=replace(cfg.seedvr2, seed=seeds["seedvr2"]),
    )
    index = progress.next_index
    target = "unbounded" if prompt_count == 0 else str(prompt_count)
    print(f"theme queue: {index} completed, target {target}", flush=True)
    rendered = 0
    while prompt_count == 0 or index < prompt_count:
        prompt_seed = (cfg.seed + index) & MAX_SEED
        print(f"=== [theme {index}] expanding prompt (seed {prompt_seed})", flush=True)
        prompt = expand_theme(cfg, theme, prompt_seed)
        print(f"expanded prompt: {prompt}", flush=True)
        result = _render(replace(
            cfg,
            prompt=prompt,
            prompt_theme=theme,
            prompt_index=index,
            prompt_seed=prompt_seed,
            seed=(cfg.seed + index) & MAX_SEED,
            usdu_seed=(cfg.usdu_seed + index) & MAX_SEED,
            seedvr2=replace(
                cfg.seedvr2,
                seed=(cfg.seedvr2.seed + index) & MAX_SEEDVR2,
            ),
        ))
        if not result.paths:
            raise RuntimeError(f"no output was saved for theme prompt {index}")
        progress.mark_completed(index)
        index += 1
        rendered += 1
    return rendered


def _configure_logging(level_name: str, verbose: bool, log_file: str | None) -> None:
    if not isinstance(level_name, str) or level_name not in {
        "DEBUG", "INFO", "WARNING", "ERROR"
    }:
        raise SystemExit("log-level must be DEBUG, INFO, WARNING, or ERROR")
    if not isinstance(verbose, bool):
        raise SystemExit("verbose must be true or false")
    if log_file is not None and (not isinstance(log_file, str) or not log_file):
        raise SystemExit("log-file must be a non-empty path")
    level = logging.DEBUG if verbose else getattr(logging, level_name)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(
                RotatingFileHandler(
                    path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
                )
            )
        except OSError as exc:
            raise SystemExit(f"cannot initialize log-file {path}: {exc}") from exc
    for handler in handlers:
        handler.setFormatter(formatter)
    root = logging.getLogger()
    for handler in root.handlers:
        handler.close()
    root.handlers.clear()
    root.setLevel(level)
    for handler in handlers:
        root.addHandler(handler)
    logging.captureWarnings(True)


def _resolve_input_mode(args: argparse.Namespace) -> str:
    """Validate input-only settings and return prompt, source, or theme."""
    if args.prompt is not None:
        if not isinstance(args.prompt, str) or not args.prompt.strip():
            raise SystemExit("--prompt must be a non-empty string")
        args.prompt = args.prompt.strip()
        return "prompt"

    mode = _require_choice(args.prompt_mode, {"source", "theme"}, "prompt-mode")
    if mode == "source":
        if isinstance(args.sources, str):
            args.sources = [args.sources]
        if (
            not isinstance(args.sources, (list, tuple))
            or not args.sources
            or any(not isinstance(item, str) or not item.strip() for item in args.sources)
        ):
            raise SystemExit(
                "prompt-mode is 'source', but sources is not a non-empty path array"
            )
        args.sources = [item.strip() for item in args.sources]
    else:
        value = args.theme
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(
                "prompt-mode is 'theme', but theme is not a non-empty string"
            )
        args.theme = value.strip()

    if mode == "source":
        if args.reconcile_interval is None:
            args.reconcile_interval = DEFAULT_SOURCE_RECONCILE_INTERVAL
        if (
            isinstance(args.reconcile_interval, bool)
            or not isinstance(args.reconcile_interval, (int, float))
            or not math.isfinite(args.reconcile_interval)
        ):
            raise SystemExit("reconcile-interval must be a finite number of seconds")
        if args.reconcile_interval < 0:
            raise SystemExit("reconcile-interval must be zero or greater")
    else:
        if args.prompt_count is None:
            args.prompt_count = DEFAULT_THEME_PROMPT_COUNT
        if isinstance(args.prompt_count, bool) or not isinstance(args.prompt_count, int):
            raise SystemExit("prompt-count must be an integer")
        if args.prompt_count < 0:
            raise SystemExit("prompt-count must be zero or greater")
        if (
            not isinstance(args.theme_system_prompt, str)
            or not args.theme_system_prompt.strip()
        ):
            raise SystemExit("theme-system-prompt must be a non-empty string")
        args.theme_system_prompt = args.theme_system_prompt.strip()
    return mode


def _source_spec(args: argparse.Namespace) -> batch.SourceSpec:
    try:
        return batch.SourceSpec(args.sources)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _reset_status(
    cfg: WorkflowConfig,
    mode: str,
    args: argparse.Namespace,
    source_spec: batch.SourceSpec | None,
) -> int:
    if mode == "source":
        if source_spec is None:
            raise RuntimeError("source mode requires a resolved source specification")
        with batch.SourceQueue(source_spec, cfg.output_dir) as queue:
            queue.reconcile()
            reset = queue.reset()
    else:
        progress = batch.ThemeProgress(
            cfg.output_dir,
            args.theme,
            {
                "base": cfg.seed,
                "usdu": cfg.usdu_seed,
                "seedvr2": cfg.seedvr2.seed,
            },
            cfg.theme_system_prompt,
        )
        reset = progress.reset()
    print(f"reset {reset} completed {mode} prompts")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.generate_config:
        if args.config:
            raise SystemExit("--config cannot be combined with --generate-config")
        path = write_config_template(args.generate_config, build_config_parser())
        print(f"wrote {path}")
        return 0
    if not args.config:
        raise SystemExit(
            "generation requires --config FILE; create one with --generate-config"
        )
    _configure_logging(args.log_level, args.verbose, args.log_file)
    mode = _resolve_input_mode(args)
    source_spec = _source_spec(args) if mode == "source" else None
    cfg = config_from_args(args)
    if not args.reset_status and mode == "source" and not cfg.save:
        raise SystemExit("batch mode requires saving so completed lines can be resumed safely")
    if not args.reset_status and mode == "theme" and not cfg.save:
        raise SystemExit("theme mode requires saving so generation can resume safely")

    lock = (
        batch.OutputLock(cfg.output_dir)
        if cfg.save or args.reset_status
        else nullcontext()
    )
    try:
        with lock:
            logger.info(
                "starting device=%s batch-size=%d model-root=%s output-dir=%s",
                cfg.device,
                cfg.batch_size,
                cfg.model_root,
                cfg.output_dir if cfg.save else "(saving disabled)",
            )
            if args.reset_status:
                return _reset_status(cfg, mode, args, source_spec)
            if mode == "prompt":
                _render(cfg)
                return 0
            if mode == "theme":
                _render_theme(cfg, args.theme, args.prompt_count)
                return 0

            if source_spec is None:
                raise RuntimeError("source mode requires a resolved source specification")
            return _run_source_queue(
                cfg,
                source_spec,
                args.reconcile_interval,
            )
    except KeyboardInterrupt:
        logger.warning("interrupted; the active prompt remains pending for the next run")
        return 130
    except PipelineOutOfMemoryError as exc:
        logger.error("%s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        return 1
    except torch.cuda.OutOfMemoryError:
        logger.error(
            "CUDA out of memory outside a tracked stage; reduce batch-size or resolution",
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return 1
    except (
        batch.AlreadyRunningError,
        batch.SourceWatchError,
        batch.ThemeProgressError,
        DeviceConfigurationError,
        OSError,
        sqlite3.DatabaseError,
        ValueError,
    ) as exc:
        logger.error("%s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
