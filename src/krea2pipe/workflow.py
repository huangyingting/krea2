"""The complete ``krea2.json`` graph, node for node.

Execution order (matching the ComfyUI topological order)::

    Text Load Line From File ─┐
    ResolutionSelector ───────┤
                              ├─> CLIPTextEncode ─> KSamplerAdvanced ─> VAEDecode
    UNETLoader -> Power Lora ─┘                          (node 31, "base")
                                                              │
              ┌───────────────────────────────────────────────┤
              │                                               ▼
              │                                    UltimateSDUpscale (node 73)
              │                                               │
              └──────────────> ColorMatch(hm-mkl-hm, 0.22) <──┘
                                        │
                    ┌───────────────────┴──────────────────┐
                    ▼                                      ▼
        SeedVR2VideoUpscaler (node 66)        ImageUpscaleWithModel (4xNomos)
                    │                                      │
                    │                            ImageScale(lanczos, W/H of SeedVR2)
                    │                                      │
                    └──────────> ImageBlend(0.4, normal) <─┘
                                        │
                                   Image Saver
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from . import accel, blend, color_match, loaders, nodes, usdu
from .imageutil import get_image_size
from .pipeline import Krea2Models, Krea2Pipeline
from .seedvr2 import SeedVR2Config

logger = logging.getLogger(__name__)

#: Prompt used when no direct prompt or batch source is given.
DEFAULT_PROMPT = (
    "A moody cinematic portrait of a red fox in a misty pine forest at dawn, "
    "volumetric god rays, shallow depth of field, 85mm lens, rich film grain"
)


@dataclass
class WorkflowConfig:
    """Every widget value of ``krea2.json``, overridable."""

    # --- prompt (node 6 CLIPTextEncode) ---
    prompt: Optional[str] = None
    negative_prompt: str = ""

    # --- models (nodes 12 / 38 / 10 / 51 Power Lora Loader) ---
    unet_name: str = "moodyKrea2Mix_v50BF16.safetensors"
    clip_name: str = "qwen3vl_4b_bf16.safetensors"
    vae_name: str = "qwen_image_vae.safetensors"
    lora_name: str = "atmospheric photography.safetensors"
    lora_strength: float = 0.6
    loras: list[tuple[str, float]] | None = None
    upscale_model_name: str = "4xNomosWebPhoto_RealPLKSR.pth"
    blend_upscale_model_name: str = "4xNomosWebPhoto_RealPLKSR.pth"

    # --- resolution (node 43 ResolutionSelector -> EmptyLatentImage) ---
    aspect_ratio: str = "1:1"
    megapixels: float = 1.5
    multiple_of: int = 32
    batch_size: int = 1
    width: Optional[int] = None     # explicit override of ResolutionSelector
    height: Optional[int] = None

    # --- KSamplerAdvanced (node 3) ---
    seed: int = 1099257494857840
    steps: int = 8
    cfg: float = 1.0
    sampler_name: str = "euler_ancestral"
    scheduler: str = "sgm_uniform"

    # --- UltimateSDUpscale (node 73) + SimpleMath+ tile size (node 62) ---
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

    # --- ColorMatch (node 61) ---
    color_match_method: str = "hm-mkl-hm"
    color_match_strength: float = 0.22
    run_color_match: bool = True

    # --- SeedVR2 (nodes 64 / 65 / 66) ---
    seedvr2: SeedVR2Config = field(default_factory=SeedVR2Config)

    # --- ImageBlend (node 71) ---
    blend_factor: float = 0.4
    blend_mode: str = "normal"

    # --- Image Saver (node 74) ---
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


class _Timer:
    def __init__(self, sink: dict[str, float], name: str):
        self.sink, self.name = sink, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.sink[self.name] = time.perf_counter() - self.t0
        logger.info("%s: %.1fs", self.name, self.sink[self.name])
        return False


def _prefetch(fn: Callable[[], object]) -> Callable[[], None]:
    """Start ``fn`` on a worker thread and return a callable that waits for it.

    Failures are swallowed: the caller always touches the same lazy property
    afterwards, so a broken prefetch just falls back to loading inline (and
    raises there, with a meaningful traceback).
    """
    def run() -> None:
        try:
            fn()
        except BaseException:  # pragma: no cover - retried synchronously
            logger.debug("prefetch failed, falling back to inline loading",
                         exc_info=True)

    thread = threading.Thread(target=run, name="krea2-prefetch", daemon=True)
    thread.start()
    return thread.join


#: Loaded models are kept for the lifetime of the process, so a second
#: ``run_workflow`` call in the same process skips all checkpoint loading.
_PIPELINES: dict[tuple, Krea2Pipeline] = {}
_UPSCALERS: dict[tuple, object] = {}


def _cached_pipeline(cfg: WorkflowConfig) -> Krea2Pipeline:
    models = Krea2Models(
        unet_name=cfg.unet_name,
        clip_name=cfg.clip_name,
        vae_name=cfg.vae_name,
        loras=cfg.resolve_loras(),
        device=cfg.device,
        dtype=cfg.dtype,
    )
    key = (models.unet_name, models.clip_name, models.vae_name, tuple(models.loras),
           models.device, str(models.dtype))
    pipe = _PIPELINES.get(key)
    if pipe is None:
        _PIPELINES.clear()          # only ever keep one set of weights resident
        pipe = _PIPELINES.setdefault(key, Krea2Pipeline(models))
    return pipe


def _cached_upscale_model(model_name: str, device: str):
    key = (model_name, device)
    model = _UPSCALERS.get(key)
    if model is None:
        _UPSCALERS.clear()
        model = _UPSCALERS.setdefault(
            key, loaders.load_upscale_model(model_name, device))
    return model


def release_models() -> None:
    """Drop every cached model and free the VRAM they hold."""
    _PIPELINES.clear()
    _UPSCALERS.clear()
    from .seedvr2 import release_upscaler

    release_upscaler()
    gc.collect()
    torch.cuda.empty_cache()


def run_workflow(config: WorkflowConfig | None = None,
                 progress: Callable[[str], None] | None = None) -> WorkflowResult:
    """Execute the whole graph and return the final image (plus intermediates)."""
    cfg = config or WorkflowConfig()
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

    # --- CLIPLoader + CLIPTextEncode ---------------------------------------
    say("[1/6] encoding prompt")
    with _Timer(timings, "text_encode"):
        cond = pipe.encode_prompt(prompt)
        pipe.free_text_encoder()
    prefetch()

    # --- KSamplerAdvanced + VAEDecode --------------------------------------
    say(f"[2/6] sampling {cfg.steps} steps @ {width}x{height} (seed {cfg.seed})")
    with _Timer(timings, "base_sample"):
        latent = pipe.empty_latent(width, height, cfg.batch_size)
        samples = pipe.sample(
            cond, latent, cfg.seed, cfg.steps, cfg.cfg,
            cfg.sampler_name, cfg.scheduler,
        )
        base = pipe.vae_decode(samples)
    stages["base"] = base

    # --- UltimateSDUpscale --------------------------------------------------
    if cfg.run_usdu:
        bw, bh, _ = get_image_size(base)
        # SimpleMath+ nodes 62 / 63: "(a*b + (96 * 2))/2"
        tile_w = nodes.simple_math("(a*b + (96 * 2))/2", a=bw, b=cfg.usdu_upscale_by)[0]
        tile_h = nodes.simple_math("(a*b + (96 * 2))/2", a=bh, b=cfg.usdu_upscale_by)[0]
        say(
            f"[3/6] UltimateSDUpscale x{cfg.usdu_upscale_by} "
            f"(tiles {tile_w}x{tile_h}, {cfg.usdu_mode}, seed {cfg.usdu_seed})"
        )
        with _Timer(timings, "usdu"):
            upscale_model = _cached_upscale_model(cfg.upscale_model_name, cfg.device)
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

    # --- ColorMatch ---------------------------------------------------------
    if cfg.run_color_match:
        say(f"[4/6] ColorMatch {cfg.color_match_method} @ {cfg.color_match_strength}")
        with _Timer(timings, "color_match"):
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

    # --- SeedVR2VideoUpscaler ----------------------------------------------
    if cfg.run_seedvr2:
        say(f"[5/6] SeedVR2 -> {cfg.seedvr2.resolution}px")
        with _Timer(timings, "seedvr2"):
            from .seedvr2 import seedvr2_upscale

            seed_out = seedvr2_upscale(matched, cfg.seedvr2)
        stages["seedvr2"] = seed_out
    else:
        say("[5/6] SeedVR2 skipped")
        seed_out = matched

    # --- ImageUpscaleWithModel -> ImageScale -> ImageBlend ------------------
    if cfg.run_blend and cfg.run_seedvr2:
        target_w, target_h, _ = get_image_size(seed_out)
        say(f"[6/6] 4x model upscale + lanczos to {target_w}x{target_h} + blend {cfg.blend_factor}")
        with _Timer(timings, "blend"):
            final = blend.upscale_and_blend(
                _cached_upscale_model(cfg.blend_upscale_model_name, cfg.device),
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
        meta = nodes.a1111_metadata(
            prompt, cfg.negative_prompt, cfg.steps, cfg.sampler_name, cfg.cfg,
            cfg.seed, width, height, cfg.unet_name,
        )
        paths = nodes.save_image(
            final, cfg.output_dir, cfg.filename, cfg.subdir, cfg.extension,
            cfg.quality, cfg.time_format, metadata=meta,
        )
        if cfg.save_intermediates:
            for name, img in stages.items():
                if img is final:
                    continue
                nodes.save_image(img, cfg.output_dir, f"{cfg.filename}_{name}",
                                 cfg.subdir, "png", cfg.quality, cfg.time_format)

    say(f"done: {out_w}x{out_h} in {sum(timings.values()):.1f}s")
    return WorkflowResult(
        image=final, base_image=base, prompt=prompt, width=out_w, height=out_h,
        paths=paths, stages=stages, timings=timings,
    )
