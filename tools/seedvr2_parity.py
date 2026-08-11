"""Numerical parity check: krea2pipe's SeedVR2 port vs the ComfyUI custom node.

The port in ``krea2pipe.seedvr2`` is a first-party port of the *official*
ByteDance SeedVR repository (https://github.com/ByteDance-Seed/SeedVR).  The
ComfyUI node ``ComfyUI-SeedVR2_VideoUpscaler`` is an independent adaptation of
the same code, so it makes a good cross-check.

Both implementations are fed **identical** latents and noise so that the
comparison measures the model/attention/normalisation port rather than the
(deliberately different) RNG draw order::

    uv run python tools/seedvr2_parity.py --input /tmp/stage1.png --resolution 1024

Reported metrics on an A100 80GB with ``seedvr2_ema_7b_fp16.safetensors``:

    VAE latent (deterministic mode) : mean |d| 0.0000   (bit-exact)
    DiT output latent               : mean |d| 0.0464, corr 0.9970
    decoded pixels                  : mean |d| 0.45/255

The remaining DiT residual comes from bf16 accumulation order: the port batches
variable-length attention with ``scaled_dot_product_attention`` differently
from the node, and the two use different RMSNorm implementations.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

WRAPPER_ROOT = os.environ.get(
    "SEEDVR2_NODE_ROOT",
    "/data/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler",
)


def load_image(path: str, size: int | None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    return torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)[None]


def run_port(image: torch.Tensor, args) -> dict:
    from torchvision.transforms import Compose, Lambda, Normalize

    from krea2pipe.seedvr2 import SeedVR2Config, SeedVR2Upscaler
    from krea2pipe.seedvr2.transforms.divisible_crop import DivisibleCrop
    from krea2pipe.seedvr2.transforms.na_resize import NaResize
    from krea2pipe.seedvr2.transforms.rearrange import Rearrange

    upscaler = SeedVR2Upscaler(SeedVR2Config(
        resolution=args.resolution, max_resolution=args.resolution, dit_model=args.model))
    runner = upscaler.load()

    transform = Compose([
        NaResize(resolution=args.resolution, mode="side", downsample_only=False,
                 max_resolution=args.resolution),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])
    video = transform(image[..., :3].permute(0, 3, 1, 2).to(args.device).float())

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    latent = runner.vae_encode([video])[0]

    runner.config.vae.use_sample = False
    latent_mode = runner.vae_encode([video])[0].float().cpu()
    runner.config.vae.use_sample = True

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(latent.shape, generator=generator).to(latent.device, latent.dtype)

    captured: dict = {}
    original_decode = runner.vae_decode

    def capturing_decode(latents):
        captured["latent"] = [x.detach().cpu().float() for x in latents]
        return original_decode(latents)

    runner.vae_decode = capturing_decode
    conditions = [runner.get_condition(noise, task="sr", latent_blur=latent)]
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        runner.inference(noises=[noise], conditions=conditions, **upscaler._text_embeds())

    return {
        "video": video.cpu(),
        "latent": latent.float().cpu(),
        "latent_mode": latent_mode,
        "noise": noise.float().cpu(),
        "out_latent": captured["latent"][0],
    }


def run_node(reference: dict, args) -> dict:
    sys.path.insert(0, WRAPPER_ROOT)
    from src.core.generation_utils import (load_text_embeddings, prepare_runner,  # noqa: E402
                                           script_directory, setup_generation_context)
    from src.core.model_loader import materialize_model  # noqa: E402
    from src.utils.debug import Debug  # noqa: E402

    debug = Debug(enabled=False)
    ctx = setup_generation_context(dit_device=args.device, vae_device=args.device,
                                   dit_offload_device=None, vae_offload_device=None,
                                   tensor_offload_device=None, debug=debug)
    runner, _ = prepare_runner(
        dit_model=args.model, vae_model="ema_vae_fp16.safetensors",
        model_dir=args.model_dir, debug=debug, ctx=ctx, dit_cache=False, vae_cache=False,
        dit_id=None, vae_id=None, block_swap_config=None, encode_tiled=False,
        decode_tiled=False, tile_debug="false", attention_mode="sdpa")
    embeds = load_text_embeddings(script_directory, ctx["dit_device"], ctx["compute_dtype"], debug)
    for name in ("vae", "dit"):
        model = getattr(runner, name, None)
        if model is not None and next(model.parameters()).device.type == "meta":
            materialize_model(runner, name, ctx["dit_device"], runner.config, debug)

    video = reference["video"].to(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    latent = runner.vae_encode([video])[0]

    runner.config.vae.use_sample = False
    latent_mode = runner.vae_encode([video])[0].float().cpu()
    runner.config.vae.use_sample = True

    # The ComfyUI node forces one step / cfg 1 for the distilled model.
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion(device=ctx["dit_device"], dtype=ctx["compute_dtype"])

    noise = reference["noise"].to(latent.device, latent.dtype)
    blur = reference["latent"].to(latent.device, latent.dtype)
    conditions = [runner.get_condition(noise, task="sr", latent_blur=blur)]
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        out = runner.inference(noises=[noise], conditions=conditions,
                               texts_pos=embeds["texts_pos"], texts_neg=embeds["texts_neg"])

    decoded_port = runner.vae_decode([reference["out_latent"].to(args.device, torch.bfloat16)])[0]
    decoded_node = runner.vae_decode([out[0].to(args.device, torch.bfloat16)])[0]
    return {
        "latent": latent.float().cpu(),
        "latent_mode": latent_mode,
        "out_latent": out[0].float().cpu(),
        "decoded_port": decoded_port.float().cpu(),
        "decoded_node": decoded_node.float().cpu(),
    }


def report(port: dict, node: dict) -> dict:
    def diff(a, b):
        d = (a - b).abs()
        return float(d.mean()), float(d.max())

    mean_mode, max_mode = diff(port["latent_mode"], node["latent_mode"])
    mean_out, max_out = diff(port["out_latent"], node["out_latent"])
    corr = float(torch.corrcoef(torch.stack([
        port["out_latent"].flatten(), node["out_latent"].flatten()]))[0, 1])
    mean_px, max_px = diff(node["decoded_port"], node["decoded_node"])
    metrics = {
        "vae_latent_mean_abs": mean_mode,
        "vae_latent_max_abs": max_mode,
        "vae_latent_std": float(port["latent_mode"].std()),
        "dit_latent_mean_abs": mean_out,
        "dit_latent_max_abs": max_out,
        "dit_latent_corr": corr,
        "decoded_mean_abs_255": mean_px * 127.5,   # decoded images live in [-1, 1]
        "decoded_max_abs_255": max_px * 127.5,
    }
    width = max(len(k) for k in metrics)
    for key, value in metrics.items():
        print(f"{key:<{width}} : {value:.6f}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="input image")
    ap.add_argument("--input-size", type=int, default=512, help="resize the input to NxN first")
    ap.add_argument("--resolution", type=int, default=1024, help="SeedVR2 target resolution")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--model", default="seedvr2_ema_7b_fp16.safetensors")
    ap.add_argument("--model-dir", default="/data/ComfyUI/models/SEEDVR2")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if not os.path.isdir(WRAPPER_ROOT):
        print(f"ComfyUI SeedVR2 node not found at {WRAPPER_ROOT}", file=sys.stderr)
        return 2

    image = load_image(args.input, args.input_size)
    port = run_port(image, args)
    del_cuda_cache()
    node = run_node(port, args)
    report(port, node)
    return 0


def del_cuda_cache() -> None:
    import gc

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
