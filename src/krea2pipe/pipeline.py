"""Krea 2 diffusion pipeline: LoRAs, text conditioning, sampling, and VAE."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field

import torch
from torch import Tensor

from . import accel, loaders, sampling
from .models import vae as vae_module
from .models.dit import SingleStreamDiT
from .models.vae import WanVAE
from .prompting import EXPANSION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class Krea2Models:
    model_root: str = loaders.DEFAULT_MODEL_ROOT
    unet_name: str = "moodyKrea2Mix_v50BF16.safetensors"
    clip_name: str = "qwen3vl_4b_bf16.safetensors"
    vae_name: str = "qwen_image_vae.safetensors"
    loras: list[tuple[str, float]] = field(default_factory=list)
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


class Krea2Pipeline:
    """Holds the DiT / VAE / text-encoder and exposes node-level operations."""

    def __init__(self, config: Krea2Models | None = None):
        self.config = config or Krea2Models()
        self.device = self.config.device
        self.dtype = self.config.dtype
        self.model_sampling = sampling.ModelSamplingFlux(shift=1.15)
        self._dit: SingleStreamDiT | None = None
        self._vae: WanVAE | None = None
        self._text_encoder = None

    # --- lazy loaders ----------------------------------------------------------
    @property
    def dit(self) -> SingleStreamDiT:
        if self._dit is None:
            self._dit = loaders.load_dit(
                self.config.unet_name, self.device, self.dtype, self.config.model_root
            )
            for lora_name, strength in self.config.loras:
                from .lora import apply_lora, load_lora_file

                path = loaders.require_model(
                    "loras", lora_name, self.config.model_root, f"LoRA {lora_name!r}"
                )
                apply_lora(self._dit, load_lora_file(path), strength)
            # After the LoRA is merged the weights are static, so the blocks can
            # be compiled once and reused by every layer, step and tile size.
            accel.compile_repeated_blocks(self._dit.blocks)
        return self._dit

    @property
    def vae(self) -> WanVAE:
        if self._vae is None:
            self._vae = loaders.load_vae(
                self.config.vae_name, self.device, self.dtype, self.config.model_root
            )
        return self._vae

    @property
    def text_encoder(self):
        if self._text_encoder is None:
            self._text_encoder = loaders.load_text_encoder(
                self.config.clip_name, self.device, self.dtype, self.config.model_root
            )
        return self._text_encoder

    def free_text_encoder(self) -> None:
        """Drop the Qwen3-VL weights once conditioning has been computed."""
        if self._text_encoder is not None:
            self._text_encoder = None
            gc.collect()
            torch.cuda.empty_cache()

    # --- nodes -----------------------------------------------------------------
    def encode_prompt(self, prompt: str) -> Tensor:
        """``CLIPTextEncode`` -> (1, seq, txtlayers*txtdim) conditioning."""
        return self.text_encoder.encode(prompt)

    def expand_theme(
        self,
        theme: str,
        seed: int,
        system_prompt: str = EXPANSION_SYSTEM_PROMPT,
    ) -> str:
        """Generate one image prompt with the same resident Qwen model used for encoding."""
        return self.text_encoder.generate_prompt(theme, seed, system_prompt=system_prompt)

    @staticmethod
    def empty_latent(width: int, height: int, batch_size: int = 1) -> Tensor:
        """``EmptyLatentImage`` (+ ``fix_empty_latent_channels`` -> 16 channels)."""
        return torch.zeros([batch_size, 16, height // 8, width // 8], dtype=torch.float32)

    @torch.no_grad()
    def vae_encode(self, image: Tensor) -> Tensor:
        """``VAEEncode``: BHWC [0,1] -> latent (B, 16, h, w)."""
        pixels = image[:, :, :, :3]
        h = (pixels.shape[1] // 8) * 8
        w = (pixels.shape[2] // 8) * 8
        if h != pixels.shape[1] or w != pixels.shape[2]:
            y_off = (pixels.shape[1] % 8) // 2
            x_off = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, y_off:y_off + h, x_off:x_off + w]
        x = pixels.movedim(-1, 1).unsqueeze(2).to(self.device, self.dtype)  # B,C,1,H,W
        x = x * 2.0 - 1.0
        latent = self.vae.encode(x)
        return latent.squeeze(2).float()

    @torch.no_grad()
    def vae_decode(self, latent: Tensor) -> Tensor:
        """``VAEDecode``: latent (B, 16, h, w) -> BHWC [0,1]."""
        z = latent.to(self.device, self.dtype)
        if z.ndim == 4:
            z = z.unsqueeze(2)
        pixels = self.vae.decode(z)
        pixels = torch.clamp((pixels.float() + 1.0) / 2.0, min=0.0, max=1.0)
        return pixels.squeeze(2).movedim(1, -1).cpu()

    def _make_denoiser(self, cond: Tensor):
        context = cond.to(self.device, self.dtype)
        dit = self.dit

        def denoise(x: Tensor, sigma: Tensor) -> Tensor:
            if context.shape[0] not in (1, x.shape[0]):
                raise ValueError(
                    f"conditioning batch {context.shape[0]} cannot drive latent batch {x.shape[0]}"
                )
            out = dit(x.to(self.dtype), sigma.to(self.device).float(), context)
            return out.float()

        return denoise

    @torch.no_grad()
    def sample(
        self,
        cond: Tensor,
        latent: Tensor,
        seed: int,
        steps: int,
        cfg: float = 1.0,
        sampler_name: str = "euler_ancestral",
        scheduler: str = "sgm_uniform",
        denoise: float = 1.0,
        start_step: int | None = None,
        last_step: int | None = None,
        force_full_denoise: bool = True,
        add_noise: bool = True,
        disable_pbar: bool = False,
    ) -> Tensor:
        """Sample a latent batch (cfg == 1 skips the unconditional pass)."""
        if cfg != 1.0:
            raise NotImplementedError(
                "the default pipeline runs at cfg=1.0 and skips negative guidance"
            )
        sigmas = sampling.calculate_sigmas(self.model_sampling, scheduler, steps, denoise)
        sigmas = sampling.slice_sigmas(sigmas, start_step, last_step, force_full_denoise)
        if len(sigmas) <= 1:
            return latent

        noise = (
            sampling.prepare_noise(latent, seed)
            if add_noise
            else torch.zeros(latent.size(), dtype=latent.dtype)
        )
        noise = noise.to(self.device)
        latent_image = latent.to(self.device)
        is_empty = bool(torch.count_nonzero(latent_image) == 0)
        if not is_empty:
            latent_image = vae_module.process_latent_in(latent_image)

        samples = sampling.sample(
            self._make_denoiser(cond),
            noise,
            latent_image,
            sigmas,
            sampler_name,
            self.model_sampling,
            seed=seed,
            disable_pbar=disable_pbar,
        )
        return vae_module.process_latent_out(samples.float())

    @torch.no_grad()
    def txt2img(self, cond: Tensor, width: int, height: int, seed: int, steps: int,
                sampler_name: str = "euler_ancestral", scheduler: str = "sgm_uniform",
                cfg: float = 1.0) -> Tensor:
        latent = self.empty_latent(width, height)
        out = self.sample(cond, latent, seed, steps, cfg, sampler_name, scheduler)
        return self.vae_decode(out)
