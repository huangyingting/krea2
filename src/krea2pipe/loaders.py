"""Model loading helpers.

Loads the ComfyUI-format safetensors checkpoints straight off disk:

* ``models/diffusion_models/*.safetensors`` -> :class:`SingleStreamDiT`
  (keys are prefixed with ``model.diffusion_model.``)
* ``models/vae/qwen_image_vae.safetensors`` -> :class:`WanVAE`
* ``models/text_encoders/qwen3vl_4b_bf16.safetensors`` -> :class:`Krea2TextEncoder`
* ``models/upscale_models/*.pth`` -> a spandrel image-to-image model
"""

from __future__ import annotations

import logging
import os

import torch
from safetensors.torch import load_file

from . import accel
from .models.dit import Krea2Config, SingleStreamDiT
from .models.vae import WanVAE

logger = logging.getLogger(__name__)

DEFAULT_COMFY_ROOT = os.environ.get("COMFYUI_ROOT", "/data/ComfyUI")


def comfy_path(*parts: str) -> str:
    return os.path.join(DEFAULT_COMFY_ROOT, *parts)


def resolve_model(kind: str, name: str) -> str:
    """Resolve ``name`` inside ``<comfy>/models/<kind>`` unless it is already a path."""
    if os.path.isabs(name) or os.path.exists(name):
        return name
    return comfy_path("models", kind, name)


def resolve_prompt_file(name: str) -> str:
    """Resolve a prompt text file.

    ``Text Load Line From File`` stores paths relative to the ComfyUI root (the
    workflow uses ``t2i/prompts/...``), so try that first and fall back to the
    ``input`` directory before giving up.
    """
    if os.path.isabs(name) or os.path.exists(name):
        return name
    for candidate in (comfy_path(name), comfy_path("input", name)):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"prompt file {name!r} not found under {DEFAULT_COMFY_ROOT} "
        "(set COMFYUI_ROOT or pass an absolute path)"
    )


def load_state_dict(path: str, prefix: str | None = None) -> dict[str, torch.Tensor]:
    sd = load_file(path, device="cpu")
    if prefix:
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    return sd


def _load_into(model: torch.nn.Module, sd: dict[str, torch.Tensor], dtype: torch.dtype,
               device: str | torch.device) -> torch.nn.Module:
    converted = {}
    model_sd = model.state_dict()
    for k, v in sd.items():
        if k not in model_sd:
            raise KeyError(f"unexpected checkpoint key {k!r}")
        target = model_sd[k]
        converted[k] = v.to(device=device, dtype=dtype if v.is_floating_point() else target.dtype)
    missing, unexpected = model.load_state_dict(converted, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()
    model.requires_grad_(False)
    return model


def load_dit(name: str, device="cuda", dtype=torch.bfloat16) -> SingleStreamDiT:
    path = resolve_model("diffusion_models", name)
    logger.info("loading diffusion model %s", path)
    sd = load_state_dict(path, prefix="model.diffusion_model.")
    config = Krea2Config.from_state_dict(sd)
    logger.info("krea2 config: %s", config)
    with torch.device("meta"):
        model = SingleStreamDiT(config)
    return _load_into(model, sd, dtype, device)


def load_vae(name: str = "qwen_image_vae.safetensors", device="cuda", dtype=torch.bfloat16) -> WanVAE:
    path = resolve_model("vae", name)
    logger.info("loading vae %s", path)
    sd = load_state_dict(path)
    dim = sd["decoder.head.0.gamma"].shape[0]
    image_channels = sd["encoder.conv1.weight"].shape[1]
    conv_out_channels = sd["decoder.head.2.weight"].shape[0]
    with torch.device("meta"):
        model = WanVAE(dim=dim, z_dim=16, dim_mult=(1, 2, 4, 4), num_res_blocks=2,
                       attn_scales=(), temperal_downsample=(False, True, True),
                       image_channels=image_channels, conv_out_channels=conv_out_channels)
    return _load_into(model, sd, dtype, device)


def load_text_encoder(name: str = "qwen3vl_4b_bf16.safetensors", device="cuda",
                      dtype=torch.bfloat16):
    from .models.text_encoder import Krea2TextEncoder

    path = resolve_model("text_encoders", name)
    logger.info("loading text encoder %s", path)
    return Krea2TextEncoder(path, device=device, dtype=dtype)


def load_upscale_model(name: str, device="cuda"):
    """``UpscaleModelLoader`` - spandrel, same as comfy_extras/nodes_upscale_model.py."""
    import spandrel

    path = resolve_model("upscale_models", name)
    logger.info("loading upscale model %s", path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    out = spandrel.ModelLoader().load_from_state_dict(sd).eval()
    if not isinstance(out, spandrel.ImageModelDescriptor):
        raise RuntimeError("Upscale model must be a single-image model.")
    out = out.to(device)
    if device != "cpu":
        # ``out.model`` is the bare nn.Module; the descriptor wrapper only
        # forwards to it, so NHWC + compilation can be applied underneath it.
        accel.tune_backends()
        out._model = accel.compile_module(accel.channels_last(out.model))
    return out
