"""End-to-end GPU tests.

They need a CUDA device and model library (``KREA2_MODEL_ROOT``) and are
skipped automatically otherwise::

    uv run pytest -m gpu                      # ~4 minutes on an A100 80GB
    KREA2_PARITY=1 uv run pytest -m gpu       # + development reference checks

The tolerances below were measured against an independent reference execution.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

PROMPT = "a red fox in a snowy pine forest, cinematic photography, golden hour"
PARITY = os.environ.get("KREA2_PARITY") == "1"
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pipeline(model_root):
    """The same process-wide pipeline ``run_workflow`` uses, so only one copy is resident."""
    from krea2pipe.workflow import WorkflowConfig, _cached_pipeline

    return _cached_pipeline(WorkflowConfig())


# --- individual stages ------------------------------------------------------------

def test_text_encoder_produces_conditioning(pipeline):
    cond = pipeline.encode_prompt(PROMPT)
    assert cond.ndim == 3
    assert cond.shape[0] == 1
    # 12 tapped Qwen3-VL hidden states of width 2560 are concatenated
    assert cond.shape[-1] == 12 * 2560
    assert torch.isfinite(cond).all()


def test_prompt_encoding_is_deterministic(pipeline):
    a = pipeline.encode_prompt(PROMPT)
    b = pipeline.encode_prompt(PROMPT)
    assert torch.equal(a, b)


def test_resident_qwen_expands_a_theme(pipeline):
    encoder_id = id(pipeline.text_encoder)
    prompt = pipeline.expand_theme("a red fox in a quiet winter forest", seed=2026)
    assert len(prompt) >= 40
    assert "\n" not in prompt
    assert id(pipeline.text_encoder) == encoder_id


def test_sampling_is_seed_deterministic(pipeline):
    cond = pipeline.encode_prompt(PROMPT)
    latent = pipeline.empty_latent(256, 256)
    a = pipeline.sample(cond, latent, seed=1234, steps=2)
    b = pipeline.sample(cond, latent, seed=1234, steps=2)
    c = pipeline.sample(cond, latent, seed=4321, steps=2)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.shape == (1, 16, 32, 32)      # /8 VAE stride


def test_vae_round_trip(pipeline):
    torch.manual_seed(0)
    image = torch.rand(1, 128, 128, 3)
    latent = pipeline.vae_encode(image)
    assert latent.shape == (1, 16, 16, 16)
    decoded = pipeline.vae_decode(latent)
    assert decoded.shape == image.shape
    assert 0.0 <= float(decoded.min()) and float(decoded.max()) <= 1.0


def test_txt2img_small(pipeline):
    cond = pipeline.encode_prompt(PROMPT)
    image = pipeline.txt2img(cond, 256, 256, seed=7, steps=4)
    assert image.shape == (1, 256, 256, 3)
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert float(image.std()) > 0.02       # not a flat/black image


def test_upscale_model_is_4x(model_root):
    from krea2pipe import loaders, upscale

    model = loaders.load_upscale_model("4xNomosWebPhoto_RealPLKSR.pth")
    out = upscale.image_upscale_with_model(model, torch.rand(1, 64, 64, 3))
    assert out.shape == (1, 256, 256, 3)


# --- SeedVR2 ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def seedvr2_input():
    from krea2pipe import imageutil

    path = REPO / "tests" / "data" / "seedvr2_input.png"
    if path.exists():
        return imageutil.pil_to_tensor(__import__("PIL.Image", fromlist=["Image"]).open(path))
    torch.manual_seed(0)
    base = torch.rand(1, 3, 16, 16)
    return torch.nn.functional.interpolate(
        base, size=(256, 256), mode="bicubic", align_corners=False
    ).clamp(0, 1).permute(0, 2, 3, 1).contiguous()


def test_seedvr2_upscales_and_is_deterministic(seedvr2_input, model_root):
    from krea2pipe.seedvr2 import SeedVR2Config, SeedVR2Upscaler

    cfg = SeedVR2Config(resolution=512, max_resolution=512)
    upscaler = SeedVR2Upscaler(cfg)
    try:
        first = upscaler.upscale(seedvr2_input)
        second = upscaler.upscale(seedvr2_input)
    finally:
        upscaler.unload()
        torch.cuda.empty_cache()

    assert first.shape == (1, 512, 512, 3)
    assert 0.0 <= float(first.min()) and float(first.max()) <= 1.0
    assert torch.equal(first, second), "same seed must give the same output"


def test_seedvr2_respects_max_resolution(model_root):
    from krea2pipe.seedvr2 import SeedVR2Config, SeedVR2Upscaler

    torch.manual_seed(0)
    wide = torch.rand(1, 128, 256, 3)      # 2:1 aspect ratio
    upscaler = SeedVR2Upscaler(SeedVR2Config(resolution=512, max_resolution=512))
    try:
        out = upscaler.upscale(wide)
    finally:
        upscaler.unload()
        torch.cuda.empty_cache()
    assert max(out.shape[1], out.shape[2]) <= 512
    assert out.shape[2] > out.shape[1]     # still landscape


def test_seedvr2_tiled_vae_round_trip(model_root):
    """Tiled encode/decode must reconstruct the image as well as the whole-frame path.

    The default uses 1024 px tiles with 128 px overlap; tiling keeps the 4K
    stage fast, so check that seams remain invisible.
    """
    from krea2pipe.seedvr2 import SeedVR2Config, SeedVR2Upscaler

    upscaler = SeedVR2Upscaler(SeedVR2Config(resolution=1024, max_resolution=1024))
    try:
        vae = upscaler.load().vae
        torch.manual_seed(0)
        small = torch.rand(1, 3, 24, 24, device="cuda")
        image = torch.nn.functional.interpolate(
            small, size=(768, 768), mode="bicubic", align_corners=False
        ).clamp(0, 1).to(torch.bfloat16).unsqueeze(2) * 2 - 1

        with torch.no_grad():
            tiled = vae.encode(image, tiled=True, tile_size=(256, 256),
                               tile_overlap=(64, 64)).latent
            whole = vae.encode(image).posterior.mode().squeeze(2)
            assert tiled.shape == whole.shape
            out_tiled = vae.decode(whole, tiled=True, tile_size=(256, 256),
                                   tile_overlap=(64, 64)).sample
            out_whole = vae.decode(whole).sample
    finally:
        upscaler.unload()
        torch.cuda.empty_cache()

    assert out_tiled.shape == out_whole.shape
    diff = (out_tiled.float() - out_whole.float()).abs() / 2 * 255
    assert float(diff.mean()) < 3.0, f"tiling changed the image too much: {diff.mean()}"


def test_seedvr2_tiling_is_enabled_by_default():
    """SeedVR2 defaults to a 1024 px VAE tile and 128 px overlap."""
    from krea2pipe.seedvr2 import SeedVR2Config

    cfg = SeedVR2Config()
    assert cfg.vae_tile == 1024
    assert cfg.vae_tile_overlap == 128


def test_conv3d_workaround_matches_the_reference_kernel():
    """The cuDNN fast path must be numerically identical to ``F.conv3d``."""
    from krea2pipe.conv3d_compat import CONV3D_CUDNN_WORKAROUND, conv3d_forward

    conv = torch.nn.Conv3d(8, 8, 3, padding=1).cuda().to(torch.bfloat16)
    x = torch.randn(1, 8, 1, 32, 32, device="cuda", dtype=torch.bfloat16)
    fast = conv3d_forward(conv, x, conv.weight, conv.bias)
    if not CONV3D_CUDNN_WORKAROUND:
        assert fast is None
        pytest.skip("this torch/cuDNN build is not affected by the Conv3d bug")
    reference = torch.nn.functional.conv3d(x, conv.weight, conv.bias, conv.stride,
                                           conv.padding, conv.dilation, conv.groups)
    assert torch.allclose(fast.float(), reference.float(), atol=2e-2, rtol=2e-2)


# --- whole workflow ----------------------------------------------------------------

def test_workflow_without_seedvr2(tmp_path, model_root):
    from krea2pipe.workflow import WorkflowConfig, run_workflow

    result = run_workflow(WorkflowConfig(
        prompt=PROMPT, width=512, height=512, steps=4,
        usdu_upscale_by=1.5, run_color_match=False,
        run_seedvr2=False, run_blend=False,
        output_dir=str(tmp_path), extension="png",
    ))
    assert (result.width, result.height) == (768, 768)
    assert result.image.shape == (1, 768, 768, 3)
    assert result.base_image.shape == (1, 512, 512, 3)
    assert set(result.stages) >= {"base", "usdu", "color_match"}
    assert len(result.paths) == 1 and os.path.exists(result.paths[0])
    assert {"text_encode", "base_sample", "usdu"} <= set(result.timings)
    assert "color_match" not in result.timings


@pytest.mark.slow
def test_workflow_all_stages(tmp_path, model_root):
    from krea2pipe.seedvr2 import SeedVR2Config
    from krea2pipe.workflow import WorkflowConfig, run_workflow

    result = run_workflow(WorkflowConfig(
        prompt=PROMPT, width=512, height=512, steps=4, usdu_upscale_by=1.5,
        batch_size=2,
        seedvr2=SeedVR2Config(resolution=1024, max_resolution=1024),
        output_dir=str(tmp_path),
    ))
    assert (result.width, result.height) == (1024, 1024)
    assert result.image.shape == (2, 1024, 1024, 3)
    assert set(result.stages) == {"base", "usdu", "color_match", "seedvr2", "blend"}
    assert 0.0 <= float(result.image.min()) and float(result.image.max()) <= 1.0
    assert len(result.paths) == 2 and all(os.path.exists(path) for path in result.paths)


# --- opt-in development reference checks -------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not PARITY, reason="set KREA2_PARITY=1 for reference checks")
def test_base_stage_matches_reference(tmp_path):
    """The same prompt, seed, and steps produce a matching reference latent."""
    reference_python = (
        Path(os.environ.get("KREA2_REFERENCE_ROOT", "/data/reference-runtime"))
        / "venv" / "bin" / "python"
    )
    if not reference_python.exists():
        pytest.skip(f"{reference_python} not found")

    reference_latent = tmp_path / "ref.pt"
    subprocess.run(
        [str(reference_python), str(REPO / "tools" / "reference_runtime.py"),
         "--stage", "base", "--prompt", PROMPT, "--steps", "1",
         "--width", "512", "--height", "512",
         "--out", str(tmp_path / "ref.png"), "--out-latent", str(reference_latent)],
        check=True, cwd=str(REPO),
    )

    from krea2pipe.pipeline import Krea2Models, Krea2Pipeline

    pipe = Krea2Pipeline(Krea2Models(loras=[("atmospheric photography.safetensors", 0.6)]))
    cond = pipe.encode_prompt(PROMPT)
    ours = pipe.sample(cond, pipe.empty_latent(512, 512), seed=1099257494857840, steps=1)
    theirs = torch.load(reference_latent).to(ours.device, ours.dtype)

    cosine = torch.nn.functional.cosine_similarity(
        ours.flatten().float(), theirs.flatten().float(), dim=0)
    # measured: 0.99995 at 1248x1248, 0.99987 at 512x512 (bf16 accumulation order)
    assert float(cosine) > 0.9995, f"latent cosine similarity {float(cosine)}"


@pytest.mark.slow
@pytest.mark.skipif(not PARITY, reason="set KREA2_PARITY=1 for reference checks")
def test_seedvr2_matches_reference(tmp_path):
    """SeedVR2 agrees with independent execution for identical latents and noise."""
    node_root = Path(os.environ.get(
        "SEEDVR2_REFERENCE_ROOT", "/data/reference-runtime/seedvr2"
    ))
    if not node_root.is_dir():
        pytest.skip(f"{node_root} not found")

    from PIL import Image

    source = tmp_path / "input.png"
    torch.manual_seed(0)
    small = torch.rand(1, 3, 16, 16)
    image = torch.nn.functional.interpolate(small, size=(512, 512), mode="bicubic",
                                            align_corners=False).clamp(0, 1)
    Image.fromarray((image[0].permute(1, 2, 0).numpy() * 255).astype("uint8")).save(source)

    out = subprocess.run(
        [sys.executable, str(REPO / "tools" / "seedvr2_reference.py"),
         "--input", str(source), "--input-size", "512", "--resolution", "1024"],
        check=True, capture_output=True, text=True, cwd=str(REPO),
    )
    metrics = {}
    for line in out.stdout.splitlines():
        if " : " in line:
            key, value = line.split(" : ")
            metrics[key.strip()] = float(value)

    assert metrics["dit_latent_corr"] > 0.99
    assert metrics["decoded_mean_abs_255"] < 2.0
