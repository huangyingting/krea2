"""Headless development reference used for numerical validation.

Run with the reference runtime's interpreter::

    $KREA2_REFERENCE_ROOT/venv/bin/python tools/reference_runtime.py --stage base

It executes an equivalent development graph and writes images/latents for
cross-implementation checks. It is not imported by the application.
"""

import argparse
import asyncio
import os
import sys
import time

REFERENCE_ROOT = os.environ.get("KREA2_REFERENCE_ROOT", "/data/reference-runtime")
sys.path.insert(0, REFERENCE_ROOT)
os.chdir(REFERENCE_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


class Timer:
    """Per stage wall clock timing (CUDA synchronised)."""

    def __init__(self, sink, name):
        self.sink, self.name = sink, name

    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.sink[self.name] = time.perf_counter() - self.t0
        print(f"[reference] {self.name}: {self.sink[self.name]:.1f}s", flush=True)
        return False


def save(image, path):
    arr = 255.0 * image[0].detach().cpu().float().numpy()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)
    print("wrote", path, tuple(image.shape))
    if image.shape[0] > 1:
        deltas = [
            round(float((image[0] - image[i]).abs().mean()), 6)
            for i in range(1, image.shape[0])
        ]
        print("[reference] mean batch deltas from image 0:", deltas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a fox walking in the snow, cinematic photography")
    ap.add_argument("--seed", type=int, default=1099257494857840)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--width", type=int, default=1248)
    ap.add_argument("--height", type=int, default=1248)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out", default="/tmp/ref_base.png")
    ap.add_argument("--out-latent", default="/tmp/ref_base_latent.pt")
    ap.add_argument("--stage", default="base", choices=["base", "usdu", "full"])
    ap.add_argument("--usdu-out", default="/tmp/ref_usdu.png")
    ap.add_argument("--usdu-seed", type=int, default=82616517812345)
    ap.add_argument("--lora-strength", type=float, default=0.6)
    ap.add_argument("--final-out", default="/tmp/ref_final.png")
    ap.add_argument("--seedvr2-model",
                    default="seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors")
    ap.add_argument("--seedvr2-attention", default="sageattn_3",
                    help="the workflow asks for sageattn_3; the node falls back to sdpa")
    ap.add_argument("--seedvr2-blocks-to-swap", type=int, default=36,
                    help="reference default is 36; pass 0 to keep every block resident")
    ap.add_argument("--input-image", default=None,
                    help="skip the base sampling stage and load this image instead")
    ap.add_argument("--trace-shapes", action="store_true",
                    help="print the first transformer block input shape for each model call")
    ap.add_argument("--usdu-independent-images", action="store_true",
                    help="work around video-VAE batch collapse and refine every image")
    args = ap.parse_args()

    torch.set_grad_enabled(False)

    import nodes
    from nodes import (CLIPLoader, CLIPTextEncode, EmptyLatentImage, KSamplerAdvanced,
                       LoraLoaderModelOnly, UNETLoader, VAEDecode, VAELoader)

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=args.stage in ("usdu", "full"),
                                       init_api_nodes=False))

    timings = {}
    (model,) = UNETLoader().load_unet("moodyKrea2Mix_v50BF16.safetensors", "default")
    if args.lora_strength:
        (model,) = LoraLoaderModelOnly().load_lora_model_only(
            model, "atmospheric photography.safetensors", args.lora_strength)
    if args.trace_shapes:
        model.model.diffusion_model.blocks[0].register_forward_pre_hook(
            lambda _module, inputs: print(
                "[reference] block input", tuple(inputs[0].shape), flush=True
            )
        )
    (clip,) = CLIPLoader().load_clip("qwen3vl_4b_bf16.safetensors", "krea2")
    (vae,) = VAELoader().load_vae("qwen_image_vae.safetensors")

    with Timer(timings, "text_encode"):
        (positive,) = CLIPTextEncode().encode(clip, args.prompt)
        (negative,) = CLIPTextEncode().encode(clip, "")

    (latent,) = EmptyLatentImage().generate(args.width, args.height, args.batch_size)
    if args.input_image:
        image = torch.from_numpy(
            np.array(Image.open(args.input_image).convert("RGB")).astype(np.float32) / 255.0
        ).unsqueeze(0)
        if args.batch_size > 1:
            image = image.repeat(args.batch_size, 1, 1, 1)
    else:
        with Timer(timings, "base_sample"):
            (sampled,) = KSamplerAdvanced().sample(
                model, "enable", args.seed, args.steps, 1.0, "euler_ancestral", "sgm_uniform",
                positive, negative, latent, 0, 9999, "disable")
            torch.save(sampled["samples"].cpu(), args.out_latent)
            (image,) = VAEDecode().decode(vae, sampled)
        save(image, args.out)

    if args.stage in ("usdu", "full"):
        from comfy_extras.nodes_upscale_model import UpscaleModelLoader

        usdu_cls = nodes.NODE_CLASS_MAPPINGS["UltimateSDUpscale"]
        if args.trace_shapes:
            processing_globals = (
                usdu_cls.upscale.__globals__["StableDiffusionProcessing"]
                .__init__.__globals__
            )
            original_sample = processing_globals["sample"]
            vae_decode = processing_globals["VAEDecode"]
            original_decode = vae_decode.decode

            def traced_sample(*call_args, **call_kwargs):
                latent = call_kwargs.get(
                    "latent", call_args[8] if len(call_args) > 8 else None
                )
                print(
                    "[reference] USDU latent",
                    tuple(latent["samples"].shape),
                    "shared images",
                    len(processing_globals["shared"].batch),
                    flush=True,
                )
                samples = original_sample(*call_args, **call_kwargs)
                print(
                    "[reference] USDU samples",
                    tuple(samples["samples"].shape),
                    flush=True,
                )
                return samples

            processing_globals["sample"] = traced_sample

            def traced_decode(self, vae, samples):
                decoded = original_decode(self, vae, samples)
                print(
                    "[reference] USDU decoded",
                    tuple(decoded[0].shape),
                    flush=True,
                )
                return decoded

            vae_decode.decode = traced_decode
        (upscale_model,) = UpscaleModelLoader().load_model("4xNomosWebPhoto_RealPLKSR.pth")
        tile_w = round((image.shape[2] * 2.0 + 96 * 2) / 2)
        tile_h = round((image.shape[1] * 2.0 + 96 * 2) / 2)
        print("tile size", tile_w, tile_h)
        if args.usdu_independent_images:
            if not hasattr(vae, "not_video"):
                raise RuntimeError("loaded VAE does not support independent image batches")
            vae.not_video = True
        with Timer(timings, "usdu"):
            (usdu_image,) = usdu_cls().upscale(
                image, model, positive, negative, vae, 2.0, args.usdu_seed, 2,
                1.0, "euler", "simple", 0.1, upscale_model, "Chess", tile_w,
                tile_h, 64, 96, "None", 1.0, 8, 64, 16, True, False, 1,
            )
        save(usdu_image, args.usdu_out)

    if args.stage == "full":
        run_final_chain(nodes, image, usdu_image, upscale_model, args, timings)

    print("[reference] TIMINGS", {k: round(v, 1) for k, v in timings.items()})
    print(f"[reference] TOTAL {sum(timings.values()):.1f}s  "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")


def run_final_chain(nodes, base_image, usdu_image, upscale_model, args, timings):
    """Run ColorMatch, SeedVR2, model upscale, Lanczos resize, and blending."""
    color_match_cls = nodes.NODE_CLASS_MAPPINGS["ColorMatch"]
    blend_cls = nodes.NODE_CLASS_MAPPINGS["ImageBlend"]
    scale_cls = nodes.NODE_CLASS_MAPPINGS["ImageScale"]
    upscale_with_model_cls = nodes.NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]

    with Timer(timings, "color_match"):
        (matched,) = color_match_cls().colormatch(base_image, usdu_image, "hm-mkl-hm",
                                                  strength=0.22)

    with Timer(timings, "seedvr2"):
        seed_image = run_seedvr2_node(matched, args)

    with Timer(timings, "blend"):
        (hires,) = upscale_with_model_cls().upscale(upscale_model, matched)
        (scaled,) = scale_cls().upscale(hires, "lanczos", seed_image.shape[2],
                                        seed_image.shape[1], "disabled")
        blend_fn = getattr(blend_cls, "execute", None) or blend_cls().blend_images
        (final,) = blend_fn(seed_image, scaled, 0.4, "normal")
    save(final, args.final_out)


def run_seedvr2_node(image, args):
    """Drive the development reference SeedVR2 implementation."""
    sys.path.insert(
        0,
        os.environ.get(
            "SEEDVR2_REFERENCE_ROOT", os.path.join(REFERENCE_ROOT, "seedvr2")
        ),
    )
    from src.core.generation_phases import (decode_all_batches, encode_all_batches,
                                            postprocess_all_batches, upscale_all_batches)
    from src.core.generation_utils import (compute_generation_info, load_text_embeddings,
                                           prepare_runner, script_directory,
                                           setup_generation_context)
    from src.utils.debug import Debug

    debug = Debug(enabled=False)
    block_swap = None
    if args.seedvr2_blocks_to_swap:
        block_swap = {"blocks_to_swap": args.seedvr2_blocks_to_swap,
                      "offload_io_components": False, "enable_debug": False}
    ctx = setup_generation_context(dit_device="cuda:0", vae_device="cuda:0",
                                   dit_offload_device="cpu", vae_offload_device="cpu",
                                   tensor_offload_device=None, debug=debug)
    runner, cache_context = prepare_runner(
        dit_model=args.seedvr2_model, vae_model="ema_vae_fp16.safetensors",
        model_dir=os.path.join(REFERENCE_ROOT, "models", "SEEDVR2"), debug=debug, ctx=ctx,
        dit_cache=False, vae_cache=False, dit_id=None, vae_id=None,
        block_swap_config=block_swap,
        encode_tiled=True, encode_tile_size=(1024, 1024), encode_tile_overlap=(128, 128),
        decode_tiled=True, decode_tile_size=(1024, 1024), decode_tile_overlap=(128, 128),
        tile_debug="false", attention_mode=args.seedvr2_attention)
    ctx["cache_context"] = cache_context
    ctx["text_embeds"] = load_text_embeddings(script_directory, ctx["dit_device"],
                                              ctx["compute_dtype"], debug)
    seed = 1234567892
    common = dict(batch_size=1, uniform_batch_size=False, seed=seed, temporal_overlap=0)
    image, _ = compute_generation_info(ctx=ctx, images=image, resolution=4096,
                                       max_resolution=4096, prepend_frames=0, debug=debug,
                                       **common)
    ctx = encode_all_batches(runner, ctx=ctx, images=image, debug=debug,
                             progress_callback=None, resolution=4096, max_resolution=4096,
                             input_noise_scale=0.0, color_correction="lab", **common)
    ctx = upscale_all_batches(runner, ctx=ctx, debug=debug, progress_callback=None,
                              seed=seed, latent_noise_scale=0.0, cache_model=False)
    ctx = decode_all_batches(runner, ctx=ctx, debug=debug, progress_callback=None,
                             cache_model=False)
    ctx = postprocess_all_batches(ctx=ctx, debug=debug, progress_callback=None,
                                  color_correction="lab", prepend_frames=0,
                                  temporal_overlap=0, batch_size=1)
    return ctx["final_video"].cpu().float()


if __name__ == "__main__":
    main()
