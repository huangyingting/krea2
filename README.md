# krea2pipe

A **pure-Python port of the "Moody Krea2 4KHD" ComfyUI workflow** (`krea2.json`).
No ComfyUI server, no node graph, no custom-node loader — just a package you
import (or a CLI you run) that reproduces the workflow end to end:

```
prompt ─▶ Krea 2 txt2img (1248²) ─▶ UltimateSDUpscale ×2 (2496²) ─▶ ColorMatch
                                                                       │
                              ┌────────────────────────────────────────┤
                              ▼                                        ▼
                    SeedVR2 upscaler (4096²)              4×NomosWebPhoto + Lanczos
                              └──────────────▶ ImageBlend 0.4 ◀────────┘
                                                    │
                                              4096×4096 JPEG
```

The model weights, LoRAs and prompt files are read straight out of an existing
ComfyUI installation (`/data/ComfyUI` by default) in their original
`safetensors` / `.pth` format — nothing has to be converted.

---

## Quick start

```bash
uv sync                     # creates .venv and installs everything
uv run krea2pipe --help

# one image
uv run krea2pipe --prompt "a red fox in a misty pine forest at dawn" -o output

# batch: one image per non-empty line; a directory is scanned recursively
uv run krea2pipe /data/krea2/prompts

# one prompt, four independent images
uv run krea2pipe -p "a red fox in a misty forest" --batch-size 4

# generate, edit, and use a complete TOML configuration
uv run krea2pipe --generate-config
# $EDITOR krea2pipe.toml
uv run krea2pipe --config krea2pipe.toml
```

Python API:

```python
from krea2pipe.workflow import WorkflowConfig, run_workflow

result = run_workflow(WorkflowConfig(prompt="a red fox in a misty pine forest"))
print(result.width, result.height, result.paths, result.timings)
image = result.image            # torch.Tensor, (B, H, W, C) float32 in [0, 1]
base = result.stages["base"]    # every intermediate stage is kept
```

### Requirements

* Python ≥ 3.12, CUDA GPU. Developed and tested on an **NVIDIA A100 80 GB**
  (the full 4096² run peaks at **55 GB**; use `--seedvr2-resolution 2048` or
  `--no-seedvr2` on smaller cards).
* An existing ComfyUI model tree. Override the location with
  `export COMFYUI_ROOT=/path/to/ComfyUI`. Files used:

| Path (below `COMFYUI_ROOT`) | Purpose |
| --- | --- |
| `models/diffusion_models/moodyKrea2Mix_v50BF16.safetensors` | Krea 2 DiT |
| `models/text_encoders/qwen3vl_4b_bf16.safetensors` | Qwen3-VL-4B text encoder |
| `models/vae/qwen_image_vae.safetensors` | VAE |
| `models/loras/atmospheric photography.safetensors` | LoRA (strength 0.6) |
| `models/upscale_models/4xNomosWebPhoto_RealPLKSR.pth` | 4× ESRGAN-style upscaler |
| `models/SEEDVR2/seedvr2_ema_7b_fp16.safetensors` | SeedVR2 7B DiT |
| `models/SEEDVR2/ema_vae_fp16.safetensors` | SeedVR2 video VAE |
| `t2i/prompts/...` | optional prompt line files |

---

## What was ported

Every node of `krea2.json` (38 nodes, 21 functional) is re-implemented from the
corresponding ComfyUI / custom-node source:

| Workflow node | Implementation |
| --- | --- |
| `UNETLoader`, `CLIPLoader`, `VAELoader`, `UpscaleModelLoader` | `loaders.py` |
| `Power Lora Loader (rgthree)` | `lora.py` (weights merged in place: `W += s·B·A`) |
| `CLIPTextEncode` (Krea 2 / Qwen3-VL) | `models/text_encoder.py` |
| `EmptyLatentImage`, `KSamplerAdvanced`, `VAEDecode` | `pipeline.py`, `sampling.py`, `models/{dit,vae}.py` |
| `ResolutionSelector`, `SimpleMath+`, `Text Load Line From File`, `Image Saver` | `nodes.py` |
| `ColorMatch` | `color_match.py` (independent, optional color-transfer stage) |
| `ImageBlend` plus its separate model upscaler | `blend.py` (independent, optional final stage) |
| `ImageScale`, `GetImageSize`, `ImageUpscaleWithModel` | `imageutil.py`, `upscale.py` |
| `UltimateSDUpscale` | `usdu.py` (Chess/Linear tiling, seam fix, mask blur, padding) |
| `SeedVR2LoadDiTModel` / `SeedVR2LoadVAEModel` / `SeedVR2VideoUpscaler` | `seedvr2/` (see below) |
| `Seed`, `PrimitiveInt`, `FloatConstant`, `GetNode`/`SetNode`, notes, bypassers | folded into `WorkflowConfig` |
| `ConditioningZeroOut` | not needed — `cfg == 1` skips the unconditional pass |

The graph itself lives in `workflow.py`; `WorkflowConfig` carries **every widget
value of the original JSON** as a default, so `run_workflow()` with no arguments
is the original workflow (only the prompt differs, see *Prompts* below).

### SeedVR2

`src/krea2pipe/seedvr2/` is a **first-party port of the official
[ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR) code**
(Apache-2.0, `seedvr2/LICENSE`), not a wrapper around the ComfyUI custom node.
`runner.py` mirrors `projects/inference_seedvr2_7b.py`. Deviations from
upstream, all documented in `seedvr2/__init__.py`:

* `common.distributed` → `parallel.py`: single process, every sequence-parallel
  collective is an identity op.
* `flash_attn_varlen_func` → `dit/attention.py::sdpa_varlen_func`, built on
  `torch.scaled_dot_product_attention` (equal-length fast path + grouped
  general path).
* apex `FusedLayerNorm`/`FusedRMSNorm` → torch equivalents (`FusedRMSNormCompat`
  keeps the checkpoint key names).
* `.safetensors` checkpoints (including the fp8 releases) are loadable.
* `color_fix.py` and the `max_resolution` long-edge cap reproduce the ComfyUI
  node's `color_correction='lab'` and resolution semantics, which the official
  repo does not have.
* `vae/tiling.py` adds spatial VAE tiling (`encode_tiled`/`decode_tiled`, on by
  default with the 1024/128 tile of `krea2.json`). The official repo always runs
  the VAE on the whole frame, which costs ~10 s and 35 GB more at 4096².

It is ~8× faster than the ComfyUI node here (9.3 s vs 73 s for 512²→1024²)
because the node block-swaps 36 layers to the CPU to fit small GPUs.

---

## Validation

The port was checked numerically against the **real ComfyUI 0.30.0** running the
same graph (`tools/comfy_reference.py`) and against the SeedVR2 custom node
(`tools/seedvr2_parity.py`):

| Stage | Metric | Result |
| --- | --- | --- |
| `CLIPTextEncode` | cosine similarity of the conditioning | **0.999985** |
| `KSamplerAdvanced` (1 step) | latent cosine similarity | **0.99995** (1248²) |
| base image (8 steps) | mean abs pixel difference | **3.97 / 255** |
| `UltimateSDUpscale` | mean abs pixel difference | **0.35 / 255** |
| scheduler sigmas | `sgm_uniform`, `simple`, `normal`, … | exact match |
| SeedVR2 DiT (identical latents + noise) | latent correlation | **0.9970** |
| SeedVR2 decoded pixels | mean abs pixel difference | **0.45 / 255** |
| SeedVR2 VAE encode | all 83 module outputs vs the node | **bit-exact** |
| SeedVR2 VAE decode (including tiled) | mean abs difference vs the node | **0.0000 / 255** |

The residuals are bf16 accumulation-order noise (different attention batching,
different RMSNorm kernels); an 8-step sampling chain amplifies the per-step
difference, which is why the base image sits at ~4/255 while a single step is at
1e-5 cosine distance.

### Running the tests

```bash
uv run pytest                       # CPU + real-model GPU tests
uv run pytest -m gpu                # just the 13 end-to-end GPU tests, ~60 s
KREA2_PARITY=1 uv run pytest -m gpu # + the two ComfyUI cross-checks, ~40 s
```

The GPU tests are skipped automatically when there is no CUDA device or no
`COMFYUI_ROOT` model tree.

CPU tests cover the node maths (ResolutionSelector → 1248², `SimpleMath+` tile
sizes, prompt-line wraparound, blend modes, colour matching, EXIF metadata),
scheduler/sampler/LoRA maths, the USDU tiling helpers, and the rewritten SeedVR2
pieces (`sdpa_varlen_func` vs a naive attention loop, LAB round-trip, wavelet
reconstruction, `NaResize` semantics). GPU tests run the real models and assert
determinism, shapes, ranges and the tolerances above.

---

## Performance (A100 80 GB, defaults)

| Stage | Output | ComfyUI warm | krea2pipe warm |
| --- | --- | --- | --- |
| text encode | — | 1.0 s | 1.3 s |
| Krea 2 sampling + VAE decode | 1248² | 9.4 s | **6.9 s** |
| UltimateSDUpscale ×2 (Chess, 1344² tiles) | 2496² | 15.1 s | **14.1 s** |
| ColorMatch (hm-mkl-hm, 0.22) | 2496² | 2.1 s | 2.2 s |
| SeedVR2 (7B) | 4096² | 21.1 s | **19.3 s** |
| 4× model upscale + Lanczos + blend 0.4 | 4096² | 11.0 s | **7.4 s** |
| **total** | **4096²** | **59.6 s**, 41.9 GB | **51.6 s**, 47.5 GB |

The ComfyUI column comes from `tools/comfy_reference.py --stage full` on the
same machine and the same graph. It **excludes model loading** (ComfyUI keeps
the UNet/CLIP/VAE/upscaler resident between runs). The fair comparison is
therefore warm-to-warm: krea2pipe is about **8 seconds faster**.

The first image after service startup is about **78 s** because checkpoints and
compiled kernels must be loaded. Every later image is about **51.6 s**. The
one-time compile from source takes longer on the very first run, but Inductor's
disk cache is persistent in the supplied service. A real three-prompt batch
completed in **3m04s**; its final image took **50.7 s**.

`batch-size=N` follows the graph's `EmptyLatentImage` semantics: one prompt
produces N different deterministic images. Krea2 samples the latent batch
together, USDU batches corresponding tiles from every image, and SeedVR2
batches its parity-safe VAE work while preserving the node's independent
batch-1 DiT numerics. The saver writes `_00`, `_01`, … suffixes.

| Batch | Warm total | Per image | Peak VRAM |
| ---: | ---: | ---: | ---: |
| 1 | 51.6 s | 51.6 s | 47.5 GB |
| 2 | 99.7 s | 49.9 s | 47.6 GB |
| 4 | **191.8 s** | **48.0 s** | 53.5 GB |

The batch-4 result was verified as `(4, 4096, 4096, 3)`. Unpatched ComfyUI
reports 183.0 s for the same nominal batch, but its Krea image VAE interprets
the IMAGE batch as one video. Wan VAE rounds four frames down to the valid
`4n+1` prefix of one, so tracing shows four images in `shared.batch` but USDU
latent/sample shapes of `(1, 16, 1, 180, 180)` and a decoded batch of one.
Consequently only image 0 receives tile diffusion. This is
[ComfyUI issue #14039](https://github.com/Comfy-Org/ComfyUI/issues/14039);
[PR #14269](https://github.com/Comfy-Org/ComfyUI/pull/14269) proposes the same
independent-image layout used here.

Applying that fix to the reference (`vae.not_video = True`) produces real
latent/sample batches of `(4, 16, 1, 180, 180)` and decodes four images for
every tile. Corrected batch-4 USDU takes **59.3 s** in ComfyUI versus **54.6 s**
in krea2pipe, so there is no equivalent-work performance deficit. The
uncorrected 23.3 s ComfyUI USDU timing is faster only because it diffuses one
image. The USDU node's own `batch_size=1` controls how many tile coordinates
are grouped and is not the cause.

### Where the time went

A naive port of the *official* SeedVR repository took **88 s / 62.8 GB**. Five
things closed the gap:

1. **cuDNN `Conv3d` workaround** (`conv3d_compat.py`, ~9 s). PyTorch 2.9/2.10
   built against cuDNN ≥ 9.10.2 dispatch fp16/bf16 `Conv3d` through a path that
   uses ~3× the memory and is much slower. Calling `torch.cudnn_convolution`
   directly avoids it. `ComfyUI-SeedVR2_VideoUpscaler` ships the same
   workaround (`src/optimization/compatibility.py`); both VAEs in this project
   are `Conv3d`-based, so it helps the Krea 2 VAE too.
   SeedVR2 VAE encode 5.9 → 3.6 s, decode 12.8 → 7.8 s.
2. **Spatial VAE tiling** (`seedvr2/vae/tiling.py`, ~10 s and 35 GB). The
   official repo only has whole-frame VAE; `krea2.json` asks for 1024 px tiles
   with a 128 px overlap. Our output is bit-identical to the node's tiled decode
   (mean |Δ| 0.0000/255, corr 1.0).
3. **numpy pin** (~1.5 s). `color-matcher`'s `hm-mkl-hm` transfer is ~1.7×
   slower on numpy ≥ 2.4 (5.9 s vs 3.5 s on a 2496² image), so `pyproject.toml`
   caps it at `<2.4`.
4. **Regional `torch.compile`** (~7 s warm). The 28 repeated Krea2 blocks share
   one dynamic compiled graph across base and USDU resolutions. The RealPLKSR
   4× model runs channels-last and compiled, reducing each 512 px tile from
   240 ms to 99 ms. Its mean output drift is only 0.02/255.
5. **Cross-image batching**. RealPLKSR and USDU submit corresponding tiles
   together. Independent `ColorMatch` transfers run on four CPU workers; at
   batch 4 this reduced that stage from 8.0 s to 2.6 s.

SageAttention 2.2 was built locally and measured at the real Krea2 shape
(48 heads, head dimension 128, about 6.6K tokens): **5.81 ms SDPA vs 5.31 ms
Sage**, only 1.09× at the attention kernel and under one second end-to-end.
It also changes attention output (correlation 0.99993), so the port deliberately
keeps exact PyTorch SDPA. FlashAttention 2 and xFormers use effectively the same
kernel path as modern PyTorch SDPA on this A100.

---

## Batch queue and service

Pass one text file or a directory. Each non-empty, non-comment line is one
prompt; directories are scanned recursively in stable filename order. When
`batch-size` is greater than one, every line produces that many images:

```bash
uv run krea2pipe /data/krea2/prompts
```

After an image is saved, its absolute source filename, line number and content
digest are fsynced to `OUTPUT_DIR/.krea2pipe-progress.tsv`. If the process or
machine stops mid-image, that line is not marked; the next run safely retries
it. Completed lines are never rendered twice, while an edited line is treated
as new work. New files and appended lines are picked up by service/watch mode.

The recommended service configuration is TOML:

```bash
uv run krea2pipe --generate-config
# edit source and output-dir, then:
sudo cp deploy/krea2pipe.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now krea2pipe
journalctl -u krea2pipe -f
```

`--generate-config` writes `krea2pipe.toml` in the current directory. Pass a
path to write elsewhere, for example
`krea2pipe --generate-config /etc/krea2pipe.toml`. The generated file contains
every supported default with comments; optional `source`, `prompt`, `width`,
and `height` entries are left commented. Existing files are never overwritten.
Command-line flags take precedence over TOML values.

Common service settings use concise values and support independent sampling
controls:

```toml
aspect-ratio = "16:9"
seed = "random"
sampler = "euler_ancestral"
scheduler = "sgm_uniform"

loras = [
  { name = "atmospheric photography.safetensors", strength = 0.6 },
  { name = "another-style.safetensors", strength = 0.35 },
]

usdu-seed = "random"
usdu-sampler = "euler"
usdu-scheduler = "simple"

run-color-match = true
color-match-method = "hm-mkl-hm"
color-match-strength = 0.22

run-blend = true
blend-upscale-model = "4xNomosWebPhoto_RealPLKSR.pth"
blend-mode = "normal"
blend-factor = 0.4
```

Supported aspect ratios are `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`,
`16:9`, and `21:9`. A random seed is chosen when the service starts; prompt
lines still receive stable offsets during that process, and the resolved seed
is written to image metadata. LoRAs are merged in listed order and may each
use a different strength. `dtype` controls model compute precision and VRAM;
`bfloat16` is recommended for the target A100, while `float16` is available
for hardware compatibility and `float32` mainly for debugging.

The standalone ColorMatch stage supports `default`, `hm`, `reinhard`, `mvgd`,
`mkl`, `hm-mvgd-hm`, and `hm-mkl-hm`; it is distinct from SeedVR2's internal
`seedvr2-color-correction`. The final blend stage independently loads its
`blend-upscale-model`, applies Lanczos sizing, and blends it with SeedVR2. Set
either `run-color-match = false` or `run-blend = false` to bypass that module.

The supplied unit keeps the `torch.compile` cache under
`/var/cache/krea2pipe`, so it survives a reboot, and restarts after a failure.
Adjust its `User`, paths and config before installing it on another account.
Future AI prompt generation can simply append lines or write new text files to
the watched source directory; the renderer and resume mechanism need no change.

---

## Layout

```
src/krea2pipe/
  workflow.py      the krea2.json graph (WorkflowConfig / run_workflow)
  color_match.py   optional standalone color-transfer stage
  blend.py         optional model-upscale / Lanczos / blend stage
  cli.py           `krea2pipe` command line entry point
  batch.py         prompt file/folder queue and persistent resume ledger
  config.py        TOML/YAML service configuration
  accel.py         transparent backend tuning and regional torch.compile
  pipeline.py      Krea2Pipeline: encode / sample / decode
  sampling.py      ModelSamplingFlux, schedulers, euler & euler_ancestral
  models/          dit.py (Krea 2 single-stream DiT), vae.py (Wan VAE), text_encoder.py
  loaders.py       ComfyUI-format checkpoint loading
  lora.py          rgthree Power Lora Loader
  nodes.py         small utility nodes (resolution, math, colour, blend, saver)
  imageutil.py     tensor/PIL helpers, common_upscale, tiled_scale
  upscale.py       ImageUpscaleWithModel (spandrel + tiling)
  usdu.py          UltimateSDUpscale
  conv3d_compat.py cuDNN Conv3d fast path (see "Where the time went")
  seedvr2/         port of the official SeedVR2 (Apache-2.0)
    vae/tiling.py  spatial tiled VAE encode/decode (from the ComfyUI node)
tools/
  comfy_reference.py   headless real-ComfyUI reference renders
  seedvr2_parity.py    SeedVR2 port vs ComfyUI node comparison
tests/               pytest suite (CPU + opt-in GPU)
```

## Licence / attribution

* `src/krea2pipe/seedvr2/` — derived from ByteDance-Seed/SeedVR, Apache-2.0
  (licence retained in that directory).
* The remaining node ports follow the algorithms of ComfyUI (GPL-3.0) and its
  custom nodes (ComfyUI_UltimateSDUpscale, KJNodes, rgthree-comfy,
  ComfyUI_essentials, WAS Node Suite, ComfyUI-Image-Saver).
* Krea 2 model reference: <https://github.com/krea-ai/krea-2>.
