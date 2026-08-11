"""SeedVR2 one-step diffusion upscaler - single image / short clip runner.

This mirrors ``projects/inference_seedvr2_7b.py`` from the official SeedVR
repository (``generation_step`` + ``generation_loop``), reduced to
single-process inference and extended with resolution limits and color correction.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch import Tensor
from torchvision.transforms import Compose, Lambda, Normalize

from .color_fix import apply_color_correction
from .config import load_config
from .infer import VideoDiffusionInfer
from .parallel import get_device
from .seed import set_seed
from .transforms.divisible_crop import DivisibleCrop
from .transforms.na_resize import NaResize
from .transforms.rearrange import Rearrange

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "configs"
#: Text embeddings of the fixed positive / negative prompt, released with the
#: SeedVR2 weights (``pos_emb.pt`` / ``neg_emb.pt`` in the official repo).
EMBED_FILES = ("pos_emb.pt", "neg_emb.pt")


@dataclass
class SeedVR2Config:
    """SeedVR2 inference and output settings."""

    dit_model: str = "seedvr2_ema_7b_fp16.safetensors"
    vae_model: str = "ema_vae_fp16.safetensors"
    variant: str = "7b"
    model_dir: Optional[str] = None
    embeds_dir: Optional[str] = None

    seed: int = 1234567892
    resolution: int = 4096
    max_resolution: int = 4096
    sample_steps: int = 1
    cfg_scale: float = 1.0
    cfg_rescale: float = 0.0
    color_correction: str = "lab"
    cond_noise_scale: float = 0.0

    device: str = "cuda"
    dtype: str = "bfloat16"
    #: keep the DiT on the GPU between the encode / decode phases (needs ~17 GB
    #: extra VRAM but avoids two model transfers)
    keep_dit_resident: bool = True

    #: Spatially tiled VAE encode/decode settings.
    #: Tiling is always the fast path at 4K (~10 s and ~35 GB cheaper than the
    #: official whole-frame VAE); it is skipped automatically when the image
    #: already fits in a single tile.
    vae_tile: int = 1024
    vae_tile_overlap: int = 128

    def config_path(self) -> Path:
        return CONFIG_DIR / f"main_{self.variant}.yaml"


def default_model_dir() -> str:
    from ..loaders import DEFAULT_MODEL_ROOT

    return os.path.join(DEFAULT_MODEL_ROOT, "SEEDVR2")


def _resolve(name: str, model_dir: str) -> str:
    name = os.path.expanduser(name)
    path = name if os.path.isabs(name) else os.path.join(model_dir, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing SeedVR2 weight at {path!r}. Download it from "
            "https://huggingface.co/ByteDance-Seed/SeedVR2-7B"
        )
    return path


def resolve_embeddings(cfg: SeedVR2Config) -> tuple[Path, Path]:
    candidates = []
    if cfg.embeds_dir:
        embeds_dir = Path(cfg.embeds_dir).expanduser()
        if not embeds_dir.is_absolute() and cfg.model_dir:
            embeds_dir = Path(cfg.model_dir) / embeds_dir
        candidates.append(embeds_dir)
    candidates.append(Path(__file__).parent / "embeddings")
    if cfg.model_dir:
        candidates.append(Path(cfg.model_dir))
    candidates.append(Path(default_model_dir()))
    for base in candidates:
        if all((base / name).is_file() for name in EMBED_FILES):
            return base / EMBED_FILES[0], base / EMBED_FILES[1]
    raise FileNotFoundError(
        "pos_emb.pt / neg_emb.pt not found (searched: "
        + ", ".join(str(c) for c in candidates)
        + "). They ship with the SeedVR repository and the SeedVR2 weights."
    )


class SeedVR2Upscaler:
    """Loads the SeedVR2 DiT + VAE once and upscales images with it."""

    def __init__(self, cfg: SeedVR2Config | None = None):
        self.cfg = cfg or SeedVR2Config()
        self.model_dir = self.cfg.model_dir or default_model_dir()
        self.runner: VideoDiffusionInfer | None = None

    # --- setup -------------------------------------------------------------
    def load(self) -> VideoDiffusionInfer:
        """``configure_runner`` of the official inference script."""
        if self.runner is not None:
            return self.runner
        cfg = self.cfg
        config = load_config(str(cfg.config_path()))
        OmegaConf.set_readonly(config, False)
        config.dit.dtype = cfg.dtype
        config.vae.dtype = cfg.dtype
        config.vae.checkpoint = _resolve(cfg.vae_model, self.model_dir)
        config.diffusion.cfg.scale = cfg.cfg_scale
        config.diffusion.cfg.rescale = cfg.cfg_rescale
        config.diffusion.timesteps.sampling.steps = cfg.sample_steps

        runner = VideoDiffusionInfer(config)
        runner.configure_dit_model(
            device=cfg.device, checkpoint=_resolve(cfg.dit_model, self.model_dir)
        )
        runner.configure_vae_model()
        if hasattr(runner.vae, "set_memory_limit"):
            runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
        runner.configure_diffusion()
        if cfg.vae_tile:
            runner.vae_tiling = ((cfg.vae_tile, cfg.vae_tile),
                                 (cfg.vae_tile_overlap, cfg.vae_tile_overlap))
        self.runner = runner
        return runner

    def unload(self) -> None:
        self.runner = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- inference ---------------------------------------------------------
    def _text_embeds(self) -> dict[str, list[Tensor]]:
        pos_path, neg_path = resolve_embeddings(self.cfg)
        device = get_device()
        dtype = getattr(torch, self.cfg.dtype)
        pos = torch.load(pos_path, weights_only=True, map_location="cpu").to(device, dtype)
        neg = torch.load(neg_path, weights_only=True, map_location="cpu").to(device, dtype)
        return {"texts_pos": [pos], "texts_neg": [neg]}

    @staticmethod
    def _cut_videos(video: Tensor) -> Tensor:
        """Pad the temporal axis to ``4n + 1`` frames (from the official script)."""
        t = video.size(1)
        if t == 1 or (t - 1) % 4 == 0:
            return video
        padding = video[:, -1:].repeat(1, 4 - ((t - 1) % 4), 1, 1)
        return torch.cat([video, padding], dim=1)

    def _generation_step(self, cond_latents: list[Tensor],
                         text_embeds: dict[str, list[Tensor]],
                         independent: bool = False) -> list[Tensor]:
        runner = self.runner
        assert runner is not None
        device = get_device()
        noises = []
        aug_noises = []
        for latent in cond_latents:
            if independent:
                # Independent images deliberately reset to the configured seed.
                set_seed(self.cfg.seed, same_across_ranks=True)
            noises.append(torch.randn_like(latent))
            aug_noises.append(torch.randn_like(latent))

        def _add_noise(x: Tensor, aug_noise: Tensor) -> Tensor:
            t = torch.tensor([1000.0], device=device) * self.cfg.cond_noise_scale
            shape = torch.tensor(x.shape[1:], device=device)[None]
            t = runner.timestep_transform(t, shape)
            return runner.schedule.forward(x, aug_noise, t)

        conditions = [
            runner.get_condition(noise, task="sr",
                                 latent_blur=_add_noise(latent_blur, aug_noise))
            for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
        ]
        text_embeds = {
            key: value * len(cond_latents) if len(value) == 1 else value
            for key, value in text_embeds.items()
        }

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
            if independent and len(cond_latents) > 1:
                # Preserve batch_size=1 DiT numerics, then group same-shaped VAE
                # decodes. Variable-length attention across multiple images is
                # faster but measurably changes pixels.
                output_latents = []
                for i, (noise, condition) in enumerate(zip(noises, conditions)):
                    one_text = {
                        key: [value[i]] for key, value in text_embeds.items()
                    }
                    output_latents.extend(runner.inference_latents(
                        noises=[noise],
                        conditions=[condition],
                        dit_offload=not self.cfg.keep_dit_resident,
                        **one_text,
                    ))
                runner.vae.to(device)
                video_tensors = runner.vae_decode(output_latents)
            else:
                video_tensors = runner.inference(
                    noises=noises,
                    conditions=conditions,
                    dit_offload=not self.cfg.keep_dit_resident,
                    **text_embeds,
                )

        return [
            rearrange(v[:, None] if v.ndim == 3 else v, "c t h w -> t c h w")
            for v in video_tensors
        ]

    @torch.no_grad()
    def _upscale_videos(self, images: list[Tensor], independent: bool) -> list[Tensor]:
        cfg = self.cfg
        runner = self.load()
        device = get_device()
        set_seed(cfg.seed, same_across_ranks=True)

        video_transform = Compose([
            NaResize(resolution=cfg.resolution, mode="side", downsample_only=False,
                     max_resolution=cfg.max_resolution),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            DivisibleCrop((16, 16)),
            Normalize(0.5, 0.5),
            Rearrange("t c h w -> c t h w"),
        ])

        cond_latents = []
        input_videos = []
        ori_lengths = []
        for image in images:
            video = image[..., :3].permute(0, 3, 1, 2).to(device).float()
            cond = video_transform(video)
            ori_lengths.append(cond.size(1))
            input_videos.append(cond)
            cond_latents.append(self._cut_videos(cond))
            logger.info("SeedVR2 input %s -> %s", tuple(video.shape[-2:]),
                        tuple(cond.shape[-2:]))

        if not cfg.keep_dit_resident:
            runner.dit.to("cpu")
        runner.vae.to(device)
        cond_latents = runner.vae_encode(cond_latents)
        if not cfg.keep_dit_resident:
            runner.vae.to("cpu")
            runner.dit.to(device)

        samples = self._generation_step(
            cond_latents, self._text_embeds(), independent=independent
        )
        del cond_latents

        outputs = []
        for sample, ori_length, input_video in zip(samples, ori_lengths, input_videos):
            if ori_length < sample.shape[0]:
                sample = sample[:ori_length]
            reference = rearrange(
                input_video[:, None] if input_video.ndim == 3 else input_video,
                "c t h w -> t c h w",
            )[: sample.shape[0]]
            sample = apply_color_correction(
                cfg.color_correction, sample.float(),
                reference.float().to(sample.device),
            )
            sample = sample.clamp(-1, 1).mul_(0.5).add_(0.5)
            outputs.append(
                sample.permute(0, 2, 3, 1).contiguous().cpu().float()
            )
        return outputs

    @torch.no_grad()
    def upscale(self, image: Tensor) -> Tensor:
        """Upscale one video represented as ``(T, H, W, C)``."""
        return self._upscale_videos([image], independent=False)[0]

    @torch.no_grad()
    def upscale_images(self, image: Tensor) -> Tensor:
        """Upscale independent ``(B, H, W, C)`` images in one model call."""
        outputs = self._upscale_videos(
            [image[i:i + 1] for i in range(image.shape[0])],
            independent=True,
        )
        return torch.cat(outputs, dim=0)


#: The 7B DiT takes ~7 s to load, so keep one upscaler for the whole process.
_CACHED: dict[tuple, SeedVR2Upscaler] = {}


def release_upscaler() -> None:
    """Drop the cached upscaler and free its VRAM."""
    for upscaler in _CACHED.values():
        upscaler.unload()
    _CACHED.clear()


def seedvr2_upscale(image: Tensor, cfg: SeedVR2Config | None = None, **overrides) -> Tensor:
    """Upscale an IMAGE batch, reusing the models loaded by a previous call.

    Images are split into independent one-frame jobs;
    ``SeedVR2Upscaler.upscale`` remains the lower-level video API.
    """
    cfg = cfg or SeedVR2Config()
    if overrides:
        cfg = SeedVR2Config(**{**cfg.__dict__, **overrides})
    key = (cfg.dit_model, cfg.vae_model, cfg.variant, cfg.model_dir, cfg.device,
           cfg.dtype, cfg.cfg_scale, cfg.cfg_rescale, cfg.sample_steps,
           cfg.vae_tile, cfg.vae_tile_overlap)
    upscaler = _CACHED.get(key)
    if upscaler is None:
        release_upscaler()
        upscaler = _CACHED.setdefault(key, SeedVR2Upscaler(cfg))
    upscaler.cfg = cfg          # seed / resolution / colour correction may change
    return upscaler.upscale_images(image)
