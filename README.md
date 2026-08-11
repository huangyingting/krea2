# krea2pipe

A **standalone Python Krea 2 image-generation and 4K upscaling application**.
It runs directly as a library, CLI, or persistent system service:

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

Model weights are loaded from the configurable `model-root` directory in their
original `safetensors` / `.pth` format. Absolute checkpoint paths are also
accepted, and no conversion or external workflow runtime is required.

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
* A model library containing these category directories. Set its absolute path
  with `model-root` in TOML or `--model-root` on the CLI:

| Path (below `model-root`) | Purpose |
| --- | --- |
| `diffusion_models/moodyKrea2Mix_v50BF16.safetensors` | Krea 2 DiT |
| `text_encoders/qwen3vl_4b_bf16.safetensors` | Qwen3-VL-4B text encoder |
| `vae/qwen_image_vae.safetensors` | VAE |
| `loras/atmospheric photography.safetensors` | LoRA (strength 0.6) |
| `upscale_models/4xNomosWebPhoto_RealPLKSR.pth` | 4× ESRGAN-style upscaler |
| `SEEDVR2/seedvr2_ema_7b_fp16.safetensors` | SeedVR2 7B DiT |
| `SEEDVR2/ema_vae_fp16.safetensors` | SeedVR2 video VAE |

---

## Pipeline architecture

| Operation | Implementation |
| --- | --- |
| Model and LoRA loading | `loaders.py`, `lora.py` |
| Qwen3-VL text conditioning | `models/text_encoder.py` |
| Krea 2 sampling and VAE | `pipeline.py`, `sampling.py`, `models/{dit,vae}.py` |
| Resolution, metadata, and image saving | `nodes.py` |
| Optional standalone color transfer | `color_match.py` |
| Optional model upscale and final blend | `blend.py` |
| Tiled image upscaling | `imageutil.py`, `upscale.py`, `usdu.py` |
| SeedVR2 diffusion upscaling | `seedvr2/` |

`workflow.py` orchestrates these independent stages. `WorkflowConfig` contains
their defaults and exposes every setting through Python, CLI flags, and TOML.

### SeedVR2

`src/krea2pipe/seedvr2/` embeds the official
[ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR) implementation
(Apache-2.0, `seedvr2/LICENSE`) directly.
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
* `color_fix.py` provides selectable correction, and `max_resolution` caps the
  long edge.
* `vae/tiling.py` adds spatial VAE tiling (`encode_tiled`/`decode_tiled`, on by
  default with a 1024/128 tile). The official repo always runs
  the VAE on the whole frame, which costs ~10 s and 35 GB more at 4096².

---

## Validation

The pipeline was checked numerically against independent reference execution
and the official SeedVR2 implementation:

| Stage | Metric | Result |
| --- | --- | --- |
| `CLIPTextEncode` | cosine similarity of the conditioning | **0.999985** |
| Diffusion sampler (1 step) | latent cosine similarity | **0.99995** (1248²) |
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
KREA2_PARITY=1 uv run pytest -m gpu # include development reference checks
```

GPU tests are skipped automatically when CUDA or `KREA2_MODEL_ROOT` is absent.

CPU tests cover the node maths (ResolutionSelector → 1248², `SimpleMath+` tile
sizes, prompt-line wraparound, blend modes, colour matching, EXIF metadata),
scheduler/sampler/LoRA maths, the USDU tiling helpers, and the rewritten SeedVR2
pieces (`sdpa_varlen_func` vs a naive attention loop, LAB round-trip, wavelet
reconstruction, `NaResize` semantics). GPU tests run the real models and assert
determinism, shapes, ranges and the tolerances above.

---

## Performance (A100 80 GB, defaults)

| Stage | Output | Warm time |
| --- | --- | ---: |
| text encode | — | 1.3 s |
| Krea 2 sampling + VAE decode | 1248² | 6.9 s |
| Ultimate SD upscale ×2 (Chess, 1344² tiles) | 2496² | 14.1 s |
| ColorMatch (hm-mkl-hm, 0.22) | 2496² | 2.2 s |
| SeedVR2 (7B) | 4096² | 19.3 s |
| 4× model upscale + Lanczos + blend 0.4 | 4096² | 7.4 s |
| **total** | **4096²** | **51.6 s**, 47.5 GB |

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

The batch-4 result was verified as `(4, 4096, 4096, 3)`. Every image is encoded
as an independent one-frame VAE input and receives tile diffusion.

### Where the time went

Baseline official SeedVR inference took **88 s / 62.8 GB**. Five optimizations
reduced that cost:

1. **cuDNN `Conv3d` workaround** (`conv3d_compat.py`, ~9 s). PyTorch 2.9/2.10
   built against cuDNN ≥ 9.10.2 dispatch fp16/bf16 `Conv3d` through a path that
   uses ~3× the memory and is much slower. Calling `torch.cudnn_convolution`
   directly avoids it. Both VAEs are `Conv3d`-based, so the optimization also
   helps the Krea 2 VAE.
   SeedVR2 VAE encode 5.9 → 3.6 s, decode 12.8 → 7.8 s.
2. **Spatial VAE tiling** (`seedvr2/vae/tiling.py`, ~10 s and 35 GB). The
   official repo only has whole-frame VAE. The default uses 1024 px tiles with
   128 px overlap and is numerically stable.
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
It also changes attention output (correlation 0.99993), so the pipeline deliberately
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
it. Completed ledger entries are not rendered again, while an edited line is
treated as new work. New files and appended lines are picked up by
service/watch mode.

The recommended service configuration is TOML:

```bash
uv run krea2pipe --generate-config
# edit source, output-dir, and model-root, then:
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
model-root = "/data/models"
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
`/var/cache/krea2pipe`, so it survives a reboot. It restarts after transient
failures but stops after three failures in five minutes instead of looping on
a bad configuration. Adjust its `User`, paths and config before installing it
on another account. Future AI prompt generation can simply append lines or
write new text files to the watched source directory; the renderer and resume
mechanism need no change.

### Production failure handling and logging

Before loading weights, the application validates numeric settings, CUDA
availability, every checkpoint needed by enabled stages, and output-directory
access. Saved images are written to a temporary file, fsynced, and atomically
renamed. An exclusive output-directory lock prevents a service and a manual run
from writing the same resume ledger concurrently.

Each stage logs elapsed time and peak allocated CUDA memory. A CUDA allocation
failure identifies the failed stage, current batch and base resolution, and
the relevant settings to reduce. The application deliberately does not
silently split a batch because doing so can change numerical results; the
active prompt remains pending and can be retried after adjusting the config.
Color-transfer failures and output errors are also fatal rather than producing
a success-shaped fallback.

Console logging defaults to `INFO` and is available through `journalctl` under
systemd. Set `log-level = "DEBUG"` for diagnostics or configure
`log-file = "/data/krea2/logs/krea2pipe.log"` for an additional rotating log
(10 MiB per file, five backups). A clean interruption exits with status 130
without marking the active prompt complete.

---

## Layout

```
src/krea2pipe/
  workflow.py      stage orchestration (WorkflowConfig / run_workflow)
  color_match.py   optional standalone color-transfer stage
  blend.py         optional model-upscale / Lanczos / blend stage
  cli.py           `krea2pipe` command line entry point
  batch.py         prompt file/folder queue and persistent resume ledger
  config.py        TOML/YAML service configuration
  accel.py         transparent backend tuning and regional torch.compile
  pipeline.py      Krea2Pipeline: encode / sample / decode
  sampling.py      ModelSamplingFlux, schedulers, euler & euler_ancestral
  models/          dit.py (Krea 2 single-stream DiT), vae.py (Wan VAE), text_encoder.py
  loaders.py       model-root checkpoint loading
  lora.py          LoRA weight merging
  nodes.py         resolution, arithmetic, metadata, and saving utilities
  imageutil.py     tensor/PIL helpers, common_upscale, tiled_scale
  upscale.py       ImageUpscaleWithModel (spandrel + tiling)
  usdu.py          UltimateSDUpscale
  conv3d_compat.py cuDNN Conv3d fast path (see "Where the time went")
  seedvr2/         embedded official SeedVR2 implementation (Apache-2.0)
    vae/tiling.py  spatial tiled VAE encode/decode
tests/               pytest suite (CPU + opt-in GPU)
```

## Licence / attribution

* `src/krea2pipe/seedvr2/` — derived from ByteDance-Seed/SeedVR, Apache-2.0
  (licence retained in that directory).
* Krea 2 model reference: <https://github.com/krea-ai/krea-2>.
* Additional provenance and development-only reference tooling are documented
  in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
