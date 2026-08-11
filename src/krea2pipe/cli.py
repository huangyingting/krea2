"""Command-line front end for the standalone Krea 2 pipeline."""

from __future__ import annotations

import argparse
import logging
import math
import os
import secrets
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
from .seedvr2 import SeedVR2Config
from .validation import DeviceConfigurationError, validate_settings
from .workflow import (
    DEFAULT_PROMPT,
    PipelineOutOfMemoryError,
    WorkflowConfig,
    run_workflow,
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
MAX_SEED = (1 << 64) - 1
MAX_SEEDVR2 = (1 << 32) - 1
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="krea2pipe",
        description="Standalone Krea 2 image-generation and upscaling pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = WorkflowConfig()

    g = p.add_argument_group("prompt")
    g.add_argument("--config", "-c", default=None, metavar="FILE",
                   help="TOML (or YAML) file of the options below; flags override it")
    g.add_argument("--generate-config", nargs="?", const="krea2pipe.toml", default=None,
                   metavar="FILE",
                   help="write a complete default TOML config and exit "
                        "(default file: krea2pipe.toml)")
    p.add_argument("source", nargs="?", default=None, metavar="FILE_OR_DIR",
                   help="prompt file, or a folder of prompt files; one image is rendered "
                        "per non-empty line and completed lines are skipped on restart")
    g.add_argument("--prompt", "-p", default=None,
                   help=f"a single prompt (default: built-in demo prompt {DEFAULT_PROMPT[:40]!r}...)")
    g.add_argument("--watch", type=float, default=0, metavar="SECONDS",
                   help="after finishing, keep rescanning FILE_OR_DIR every SECONDS for new "
                        "lines instead of exiting (for running as a service)")

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
    g.add_argument("--seed", type=_seed_argument, default=d.seed,
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
    g.add_argument("--usdu-seed", type=_seed_argument, default=d.usdu_seed,
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
    g.add_argument("--seedvr2-seed", type=_seed_argument, default=d.seedvr2.seed,
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
    )
    cfg = WorkflowConfig(
        prompt=args.prompt,
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
    """Parse ``argv``, seeding the defaults from ``--config`` when one is given."""
    parser = build_parser()
    config_path = parser.parse_known_args(argv)[0].config
    if config_path:
        parser.set_defaults(**load_config(config_path, config_options(parser)))
    return parser.parse_args(argv)


def _render(cfg: WorkflowConfig):
    result = run_workflow(cfg, progress=lambda m: print(m, flush=True))
    for path in result.paths:
        print(path)
    if not result.paths:
        print(f"{result.width}x{result.height} image produced (saving disabled)")
    return result


def _render_pending(cfg: WorkflowConfig, source: str, announce_empty: bool = False) -> int:
    """Render every prompt under ``source`` that is not recorded as done yet."""
    prompts = list(batch.iter_prompts(source))
    done = batch.Progress(cfg.output_dir)
    todo = [p for p in prompts if p not in done]
    if todo or announce_empty:
        print(f"{len(prompts)} prompts, {len(prompts) - len(todo)} already rendered, "
              f"{len(todo)} to go", flush=True)
    for n, prompt in enumerate(todo, start=1):
        print(f"=== [{n}/{len(todo)}] {prompt.file.name}:{prompt.line} {prompt.text[:70]}",
              flush=True)
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
        done.mark(prompt)
    return len(todo)


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.generate_config:
        path = write_config_template(args.generate_config, build_parser())
        print(f"wrote {path}")
        return 0
    _configure_logging(args.log_level, args.verbose, args.log_file)
    cfg = config_from_args(args)
    if args.source is not None and not isinstance(args.source, str):
        raise SystemExit("source must be a file or directory path")
    if (
        isinstance(args.watch, bool)
        or not isinstance(args.watch, (int, float))
        or not math.isfinite(args.watch)
    ):
        raise SystemExit("--watch must be a finite number of seconds")
    if args.source and args.prompt:
        raise SystemExit("use either FILE_OR_DIR or --prompt, not both")
    if args.source and not cfg.save:
        raise SystemExit("batch mode requires saving so completed lines can be resumed safely")
    if args.watch > 0 and not args.source:
        raise SystemExit("--watch requires FILE_OR_DIR")
    if args.watch < 0:
        raise SystemExit("--watch must be zero or greater")

    lock = batch.OutputLock(cfg.output_dir) if cfg.save else nullcontext()
    try:
        with lock:
            logger.info(
                "starting device=%s batch-size=%d model-root=%s output-dir=%s",
                cfg.device,
                cfg.batch_size,
                cfg.model_root,
                cfg.output_dir if cfg.save else "(saving disabled)",
            )
            if not args.source:
                _render(cfg)
                return 0

            _render_pending(cfg, args.source, announce_empty=True)
            while args.watch > 0:
                time.sleep(args.watch)
                _render_pending(cfg, args.source)
            return 0
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
    except (batch.AlreadyRunningError, DeviceConfigurationError, OSError, ValueError) as exc:
        logger.error("%s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
