"""Standalone Krea 2 generation and modular upscaling workflow."""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import torch
from torch import Tensor

from . import accel, blend, color_match, loaders, metadata, nodes, usdu
from .imageutil import get_image_size
from .pipeline import Krea2Models, Krea2Pipeline
from .prompting import EXPANSION_SYSTEM_PROMPT
from .seedvr2 import SeedVR2Config
from .validation import preflight

logger = logging.getLogger(__name__)

#: Prompt used when no direct prompt or batch source is given.
DEFAULT_PROMPT = (
    "A moody cinematic portrait of a red fox in a misty pine forest at dawn, "
    "volumetric god rays, shallow depth of field, 85mm lens, rich film grain"
)


@dataclass
class WorkflowConfig:
    """Configuration for every standalone pipeline stage."""

    # --- prompt ---
    prompt: Optional[str] = None
    negative_prompt: str = ""
    prompt_theme: Optional[str] = None
    prompt_index: Optional[int] = None
    prompt_seed: Optional[int] = None
    theme_system_prompt: str = EXPANSION_SYSTEM_PROMPT

    # --- models ---
    model_root: str = loaders.DEFAULT_MODEL_ROOT
    unet_name: str = "moodyKrea2Mix_v50BF16.safetensors"
    clip_name: str = "qwen3vl_4b_bf16.safetensors"
    vae_name: str = "qwen_image_vae.safetensors"
    lora_name: str = "atmospheric photography.safetensors"
    lora_strength: float = 0.6
    loras: list[tuple[str, float]] | None = None
    upscale_model_name: str = "4xNomosWebPhoto_RealPLKSR.pth"
    blend_upscale_model_name: str = "4xNomosWebPhoto_RealPLKSR.pth"

    # --- resolution ---
    aspect_ratio: str = "1:1"
    megapixels: float = 1.5
    multiple_of: int = 32
    batch_size: int = 1
    width: Optional[int] = None     # explicit override of ResolutionSelector
    height: Optional[int] = None

    # --- base sampling ---
    seed: int = 1099257494857840
    steps: int = 8
    cfg: float = 1.0
    sampler_name: str = "euler_ancestral"
    scheduler: str = "sgm_uniform"

    # --- tiled diffusion upscale ---
    usdu_seed: int = 82616517812345
    usdu_steps: int = 2
    usdu_cfg: float = 1.0
    usdu_sampler_name: str = "euler"
    usdu_scheduler: str = "simple"
    usdu_denoise: float = 0.1
    usdu_upscale_by: float = 2.0
    usdu_mode: str = "Chess"
    usdu_mask_blur: int = 64
    usdu_tile_padding: int = 96
    usdu_force_uniform_tiles: bool = True

    # --- color match ---
    color_match_method: str = "hm-mkl-hm"
    color_match_strength: float = 0.22
    run_color_match: bool = True

    # --- SeedVR2 ---
    seedvr2: SeedVR2Config = field(default_factory=SeedVR2Config)

    # --- final blend ---
    blend_factor: float = 0.4
    blend_mode: str = "normal"

    # --- image output ---
    state_dir: str = "state"
    output_dir: str = "output"
    filename: str = "%time"
    subdir: str = "AIKC"
    extension: str = "jpg"
    quality: int = 100
    time_format: str = "%Y-%m-%d-%H%M%S"
    save_intermediates: bool = False

    # --- stage toggles / runtime ---
    run_usdu: bool = True
    run_seedvr2: bool = True
    run_blend: bool = True
    save: bool = True
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    def resolve_prompt(self) -> str:
        if self.prompt is not None:
            return self.prompt
        return DEFAULT_PROMPT

    def resolve_size(self) -> tuple[int, int]:
        if self.width and self.height:
            return self.width, self.height
        return nodes.resolution_selector(self.aspect_ratio, self.megapixels, self.multiple_of)

    def resolve_loras(self) -> list[tuple[str, float]]:
        if self.loras is not None:
            return [(name, strength) for name, strength in self.loras if strength]
        return [(self.lora_name, self.lora_strength)] if self.lora_strength else []

    def resolve_seedvr2(self) -> SeedVR2Config:
        model_dir = self.seedvr2.model_dir or os.path.join(
            self.model_root, "SEEDVR2"
        )
        return replace(
            self.seedvr2,
            model_dir=model_dir,
            device=self.device,
            dtype=str(self.dtype).removeprefix("torch."),
        )


@dataclass
class WorkflowResult:
    image: Tensor
    base_image: Tensor
    prompt: str
    width: int
    height: int
    paths: list[str] = field(default_factory=list)
    stages: dict[str, Tensor] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


class PipelineOutOfMemoryError(RuntimeError):
    """A CUDA allocation failure annotated with the active pipeline stage."""


class _Timer:
    def __init__(self, sink: dict[str, float], name: str):
        self.sink, self.name = sink, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sink[self.name] = time.perf_counter() - self.t0
        if exc_type is None:
            logger.info("%s: %.1fs", self.name, self.sink[self.name])
        else:
            logger.warning(
                "%s failed after %.1fs: %s",
                self.name,
                self.sink[self.name],
                exc_value,
            )
        return False


def _oom_message(stage: str, cfg: WorkflowConfig, width: int, height: int) -> str:
    current = f"batch-size={cfg.batch_size}, base-resolution={width}x{height}"
    if stage == "seedvr2":
        advice = (
            "reduce batch-size, seedvr2-resolution, or seedvr2-max-resolution; "
            "a smaller SeedVR2 VAE tile can also reduce peak memory"
        )
    elif stage in {"usdu", "blend"}:
        advice = (
            "reduce batch-size, the base resolution/megapixels, or the upscale factor; "
            f"you can also disable the {stage} stage"
        )
    else:
        advice = "reduce batch-size or the base resolution/megapixels"
    return f"CUDA out of memory during {stage} ({current}); {advice}"


@contextmanager
def _stage(timings: dict[str, float], name: str, cfg: WorkflowConfig,
           width: int, height: int):
    device = torch.device(cfg.device)
    track_cuda = device.type == "cuda" and torch.cuda.is_available()
    if track_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with _Timer(timings, name):
            yield
        if track_cuda:
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            logger.info("%s peak CUDA memory: %.2f GiB", name, peak_gib)
    except torch.cuda.OutOfMemoryError as exc:
        memory = ""
        if track_cuda:
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
            memory = f" (allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB)"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise PipelineOutOfMemoryError(
            _oom_message(name, cfg, width, height) + memory
        ) from exc


def _prefetch(fn: Callable[[], object]) -> Callable[[], None]:
    """Run ``fn`` on a worker and return a waiter that re-raises its failure."""
    failure: list[BaseException] = []

    def run() -> None:
        try:
            fn()
        except BaseException as exc:
            failure.append(exc)

    thread = threading.Thread(target=run, name="krea2-prefetch", daemon=True)
    thread.start()

    def wait() -> None:
        thread.join()
        if failure:
            raise failure[0]

    return wait


#: Loaded models are kept for the lifetime of the process, so a second
#: ``run_workflow`` call in the same process skips all checkpoint loading.
_PIPELINES: dict[tuple, Krea2Pipeline] = {}
_UPSCALERS: dict[tuple, object] = {}


def _cached_pipeline(cfg: WorkflowConfig) -> Krea2Pipeline:
    models = Krea2Models(
        model_root=cfg.model_root,
        unet_name=cfg.unet_name,
        clip_name=cfg.clip_name,
        vae_name=cfg.vae_name,
        loras=cfg.resolve_loras(),
        device=cfg.device,
        dtype=cfg.dtype,
    )
    key = (models.model_root, models.unet_name, models.clip_name, models.vae_name,
           tuple(models.loras), models.device, str(models.dtype))
    pipe = _PIPELINES.get(key)
    if pipe is None:
        _PIPELINES.clear()          # only ever keep one set of weights resident
        pipe = _PIPELINES.setdefault(key, Krea2Pipeline(models))
    return pipe


def _cached_upscale_model(model_name: str, device: str, model_root: str):
    key = (model_root, model_name, device)
    model = _UPSCALERS.get(key)
    if model is None:
        _UPSCALERS.clear()
        model = _UPSCALERS.setdefault(
            key, loaders.load_upscale_model(model_name, device, model_root))
    return model


def release_models() -> None:
    """Drop every cached model and free the VRAM they hold."""
    _PIPELINES.clear()
    _UPSCALERS.clear()
    from .seedvr2 import release_upscaler

    release_upscaler()
    gc.collect()
    torch.cuda.empty_cache()


def expand_theme(config: WorkflowConfig, theme: str, seed: int) -> str:
    """Expand one theme using the cached Qwen model."""
    preflight(config)
    return _cached_pipeline(config).expand_theme(
        theme,
        seed,
        config.theme_system_prompt,
    )


def run_workflow(config: WorkflowConfig | None = None,
                 progress: Callable[[str], None] | None = None) -> WorkflowResult:
    """Execute the pipeline and return the final image plus intermediates."""
    cfg = config or WorkflowConfig()
    preflight(cfg)
    accel.tune_backends()
    say = progress or (lambda msg: logger.info("%s", msg))
    timings: dict[str, float] = {}
    stages: dict[str, Tensor] = {}

    prompt = cfg.resolve_prompt()
    width, height = cfg.resolve_size()
    say(f"prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    say(f"base resolution: {width}x{height}")

    pipe = _cached_pipeline(cfg)

    # Loading the DiT is dominated by file IO and host->device copies, which
    # release the GIL, so it overlaps almost perfectly with the (CPU bound)
    # ``import transformers`` that the text encoder triggers.
    prefetch = _prefetch(lambda: pipe.dit)

    say("[1/6] encoding prompt")
    with _stage(timings, "text_encode", cfg, width, height):
        cond = pipe.encode_prompt(prompt)
    prefetch()

    say(f"[2/6] sampling {cfg.steps} steps @ {width}x{height} (seed {cfg.seed})")
    with _stage(timings, "base_sample", cfg, width, height):
        latent = pipe.empty_latent(width, height, cfg.batch_size)
        samples = pipe.sample(
            cond, latent, cfg.seed, cfg.steps, cfg.cfg,
            cfg.sampler_name, cfg.scheduler,
        )
        base = pipe.vae_decode(samples)
    stages["base"] = base

    if cfg.run_usdu:
        bw, bh, _ = get_image_size(base)
        tile_w = nodes.simple_math("(a*b + (96 * 2))/2", a=bw, b=cfg.usdu_upscale_by)[0]
        tile_h = nodes.simple_math("(a*b + (96 * 2))/2", a=bh, b=cfg.usdu_upscale_by)[0]
        say(
            f"[3/6] UltimateSDUpscale x{cfg.usdu_upscale_by} "
            f"(tiles {tile_w}x{tile_h}, {cfg.usdu_mode}, seed {cfg.usdu_seed})"
        )
        with _stage(timings, "usdu", cfg, width, height):
            upscale_model = _cached_upscale_model(
                cfg.upscale_model_name, cfg.device, cfg.model_root
            )
            params = usdu.USDUParams(
                upscale_by=cfg.usdu_upscale_by,
                seed=cfg.usdu_seed,
                steps=cfg.usdu_steps,
                cfg=cfg.usdu_cfg,
                sampler_name=cfg.usdu_sampler_name,
                scheduler=cfg.usdu_scheduler,
                denoise=cfg.usdu_denoise,
                mode_type=cfg.usdu_mode,
                tile_width=tile_w,
                tile_height=tile_h,
                mask_blur=cfg.usdu_mask_blur,
                tile_padding=cfg.usdu_tile_padding,
                force_uniform_tiles=cfg.usdu_force_uniform_tiles,
            )
            upscaled = usdu.ultimate_sd_upscale(pipe, base, cond, upscale_model, params)
            gc.collect()
            torch.cuda.empty_cache()
    else:
        say("[3/6] UltimateSDUpscale skipped")
        upscaled = base
    stages["usdu"] = upscaled

    if cfg.run_color_match:
        say(f"[4/6] ColorMatch {cfg.color_match_method} @ {cfg.color_match_strength}")
        with _stage(timings, "color_match", cfg, width, height):
            matched = color_match.color_match(
                base, upscaled, cfg.color_match_method, cfg.color_match_strength
            )
    else:
        say("[4/6] ColorMatch skipped")
        matched = upscaled
    stages["color_match"] = matched

    # The diffusion models are no longer needed - free the VRAM for SeedVR2.
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    if cfg.run_seedvr2:
        say(f"[5/6] SeedVR2 -> {cfg.seedvr2.resolution}px")
        with _stage(timings, "seedvr2", cfg, width, height):
            from .seedvr2 import seedvr2_upscale

            seed_out = seedvr2_upscale(matched, cfg.resolve_seedvr2())
        stages["seedvr2"] = seed_out
    else:
        say("[5/6] SeedVR2 skipped")
        seed_out = matched

    if cfg.run_blend and cfg.run_seedvr2:
        target_w, target_h, _ = get_image_size(seed_out)
        say(f"[6/6] 4x model upscale + lanczos to {target_w}x{target_h} + blend {cfg.blend_factor}")
        with _stage(timings, "blend", cfg, width, height):
            final = blend.upscale_and_blend(
                _cached_upscale_model(
                    cfg.blend_upscale_model_name, cfg.device, cfg.model_root
                ),
                matched,
                seed_out,
                cfg.blend_factor,
                cfg.blend_mode,
            )
        stages["blend"] = final
    else:
        say("[6/6] blend skipped")
        final = seed_out

    out_w, out_h, _ = get_image_size(final)
    paths: list[str] = []
    if cfg.save:
        with _stage(timings, "save", cfg, width, height):
            meta = nodes.a1111_metadata(
                prompt, cfg.negative_prompt, cfg.steps, cfg.sampler_name, cfg.cfg,
                cfg.seed, width, height, cfg.unet_name,
            )
            manifest = metadata.build_generation_manifest(cfg, prompt, width, height)
            paths = nodes.save_image(
                final, cfg.output_dir, cfg.filename, cfg.subdir, cfg.extension,
                cfg.quality, cfg.time_format, metadata=meta,
                generation_manifest=manifest,
            )
            if cfg.save_intermediates:
                for name, img in stages.items():
                    if img is final:
                        continue
                    nodes.save_image(img, cfg.output_dir, f"{cfg.filename}_{name}",
                                     cfg.subdir, "png", cfg.quality, cfg.time_format,
                                     metadata=meta, generation_manifest=manifest,
                                     image_stage=name)

    say(f"done: {out_w}x{out_h} in {sum(timings.values()):.1f}s")
    return WorkflowResult(
        image=final, base_image=base, prompt=prompt, width=out_w, height=out_h,
        paths=paths, stages=stages, timings=timings,
    )
