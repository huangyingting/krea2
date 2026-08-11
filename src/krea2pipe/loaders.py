"""Model loading helpers.

Loads supported checkpoints directly from a configurable model directory:

* ``diffusion_models/*.safetensors`` -> :class:`SingleStreamDiT`
  (keys are prefixed with ``model.diffusion_model.``)
* ``vae/qwen_image_vae.safetensors`` -> :class:`WanVAE`
* ``text_encoders/qwen3vl_4b_bf16.safetensors`` -> :class:`Krea2TextEncoder`
* ``upscale_models/*.pth`` -> a spandrel image-to-image model
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

DEFAULT_MODEL_ROOT = os.environ.get("KREA2_MODEL_ROOT", "/data/models")


def normalize_model_root(model_root: str) -> str:
    if not isinstance(model_root, str) or not model_root:
        raise ValueError("model-root must be a non-empty absolute path")
    root = os.path.expanduser(model_root)
    if not os.path.isabs(root):
        raise ValueError("model-root must be an absolute path")
    return os.path.normpath(root)


def resolve_model(kind: str, name: str, model_root: str = DEFAULT_MODEL_ROOT) -> str:
    """Resolve absolute paths directly and relative names below ``model_root/kind``."""
    if not isinstance(name, str) or not name:
        raise ValueError("model name must be a non-empty path")
    name = os.path.expanduser(name)
    if os.path.isabs(name):
        return name
    return os.path.join(normalize_model_root(model_root), kind, name)


def require_model(kind: str, name: str, model_root: str = DEFAULT_MODEL_ROOT,
                  label: str = "model") -> str:
    """Resolve a model and fail with configuration context when it is unavailable."""
    path = resolve_model(kind, name, model_root)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{label} not found: {path} (configured as {name!r}; "
            f"model-root={normalize_model_root(model_root)!r})"
        )
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{label} is not readable: {path}")
    return path


def load_state_dict(path: str, prefix: str | None = None) -> dict[str, torch.Tensor]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
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


def load_dit(name: str, device="cuda", dtype=torch.bfloat16,
             model_root: str = DEFAULT_MODEL_ROOT) -> SingleStreamDiT:
    path = require_model("diffusion_models", name, model_root, "Krea 2 checkpoint")
    logger.info("loading diffusion model %s", path)
    sd = load_state_dict(path, prefix="model.diffusion_model.")
    config = Krea2Config.from_state_dict(sd)
    logger.info("krea2 config: %s", config)
    with torch.device("meta"):
        model = SingleStreamDiT(config)
    return _load_into(model, sd, dtype, device)


def load_vae(name: str = "qwen_image_vae.safetensors", device="cuda",
             dtype=torch.bfloat16, model_root: str = DEFAULT_MODEL_ROOT) -> WanVAE:
    path = require_model("vae", name, model_root, "image VAE")
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
                      dtype=torch.bfloat16, model_root: str = DEFAULT_MODEL_ROOT):
    from .models.text_encoder import Krea2TextEncoder

    path = require_model("text_encoders", name, model_root, "text encoder")
    logger.info("loading text encoder %s", path)
    return Krea2TextEncoder(path, device=device, dtype=dtype)


def load_upscale_model(name: str, device="cuda",
                       model_root: str = DEFAULT_MODEL_ROOT):
    """Load a spandrel-compatible image upscaler."""
    import spandrel

    path = require_model("upscale_models", name, model_root, "upscale model")
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
