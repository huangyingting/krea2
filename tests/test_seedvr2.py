"""Tests for the embedded SeedVR2 implementation.

These cover the local attention implementation, colour correction, and
resolution-limit behavior.
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from krea2pipe.seedvr2 import SeedVR2Config, color_fix
from krea2pipe.seedvr2.dit.attention import sdpa_varlen_func
from krea2pipe.seedvr2.seed import set_seed
from krea2pipe.seedvr2.transforms.na_resize import NaResize
from krea2pipe.seedvr2.vae.tiling import tiled_decode, tiled_encode


def test_seedvr2_seed_rejects_values_numpy_cannot_represent():
    with pytest.raises(ValueError, match=r"between 0 and 2\*\*32 - 1"):
        set_seed(1 << 32, same_across_ranks=True)


# --- variable length attention (flash-attn replacement) -------------------------------

def _naive_varlen(q, k, v, cu_q, cu_k):
    outs = []
    for i in range(len(cu_q) - 1):
        qi = q[cu_q[i]:cu_q[i + 1]].transpose(0, 1)      # (heads, len, dim)
        ki = k[cu_k[i]:cu_k[i + 1]].transpose(0, 1)
        vi = v[cu_k[i]:cu_k[i + 1]].transpose(0, 1)
        attn = torch.softmax(qi @ ki.transpose(-1, -2) / math.sqrt(qi.shape[-1]), dim=-1)
        outs.append((attn @ vi).transpose(0, 1))
    return torch.cat(outs, dim=0)


@pytest.mark.parametrize("lengths", [[7], [4, 4, 4], [3, 5, 2], [1, 9]])
def test_sdpa_varlen_matches_a_naive_loop(lengths):
    torch.manual_seed(0)
    heads, dim = 3, 8
    total = sum(lengths)
    q = torch.randn(total, heads, dim, dtype=torch.float64)
    k = torch.randn(total, heads, dim, dtype=torch.float64)
    v = torch.randn(total, heads, dim, dtype=torch.float64)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32)
    out = sdpa_varlen_func(q, k, v, cu, cu)
    assert out.shape == q.shape
    assert torch.allclose(out, _naive_varlen(q, k, v, cu, cu), atol=1e-9)


def test_sdpa_varlen_cross_attention_lengths():
    """Query and key/value can have different segment lengths (cross attention)."""
    torch.manual_seed(0)
    heads, dim = 2, 4
    lens_q, lens_k = [3, 5], [7, 2]
    q = torch.randn(sum(lens_q), heads, dim, dtype=torch.float64)
    k = torch.randn(sum(lens_k), heads, dim, dtype=torch.float64)
    v = torch.randn(sum(lens_k), heads, dim, dtype=torch.float64)
    cu_q = torch.tensor([0, 3, 8], dtype=torch.int32)
    cu_k = torch.tensor([0, 7, 9], dtype=torch.int32)
    out = sdpa_varlen_func(q, k, v, cu_q, cu_k)
    assert out.shape == q.shape
    assert torch.allclose(out, _naive_varlen(q, k, v, cu_q, cu_k), atol=1e-9)


def test_sdpa_varlen_fast_path_equals_general_path():
    """Equal lengths take the reshape fast path - it must agree with the grouped one."""
    torch.manual_seed(0)
    q = torch.randn(8, 2, 4, dtype=torch.float64)
    k, v = torch.randn_like(q), torch.randn_like(q)
    equal = torch.tensor([0, 4, 8], dtype=torch.int32)
    assert torch.allclose(sdpa_varlen_func(q, k, v, equal, equal),
                          _naive_varlen(q, k, v, equal, equal), atol=1e-9)


def test_sdpa_varlen_chunks_large_unequal_groups(monkeypatch):
    import krea2pipe.seedvr2.dit.attention as attention

    torch.manual_seed(0)
    lens_q = [3, 3, 3, 3, 4]
    lens_k = [5, 5, 5, 5, 2]
    q = torch.randn(sum(lens_q), 2, 4, dtype=torch.float64)
    k = torch.randn(sum(lens_k), 2, 4, dtype=torch.float64)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0] + list(torch.tensor(lens_q).cumsum(0)), dtype=torch.int32)
    cu_k = torch.tensor([0] + list(torch.tensor(lens_k).cumsum(0)), dtype=torch.int32)
    calls = []
    original_sdpa = attention._sdpa

    def recording_sdpa(q_batch, *args, **kwargs):
        calls.append(q_batch.shape[0])
        return original_sdpa(q_batch, *args, **kwargs)

    monkeypatch.setattr(attention, "_sdpa", recording_sdpa)
    monkeypatch.setattr(attention, "SDPA_VARLEN_MAX_GROUP_BYTES", 2048)
    out = attention.sdpa_varlen_func(q, k, v, cu_q, cu_k)

    assert max(calls) == 2
    assert torch.allclose(out, _naive_varlen(q, k, v, cu_q, cu_k), atol=1e-9)


# --- colour correction ----------------------------------------------------------------

def test_rgb_lab_round_trip():
    torch.manual_seed(0)
    rgb = torch.rand(2, 3, 16, 16)
    back = color_fix.lab_to_rgb(color_fix.rgb_to_lab(rgb))
    assert torch.allclose(back, rgb, atol=2e-3)


def test_rgb_to_lab_reference_values():
    white = torch.ones(1, 3, 1, 1)
    black = torch.zeros(1, 3, 1, 1)
    lab_white = color_fix.rgb_to_lab(white)
    lab_black = color_fix.rgb_to_lab(black)
    assert float(lab_white[0, 0]) == pytest.approx(100.0, abs=0.5)   # L* = 100
    assert float(lab_white[0, 1]) == pytest.approx(0.0, abs=0.5)     # neutral a*
    assert float(lab_white[0, 2]) == pytest.approx(0.0, abs=0.5)     # neutral b*
    assert float(lab_black[0, 0]) == pytest.approx(0.0, abs=0.5)


def test_match_histogram_transfers_the_distribution():
    torch.manual_seed(0)
    source = torch.rand(1, 32, 32) * 10
    reference = torch.rand(1, 32, 32) * 2 + 50
    matched = color_fix.match_histogram(source, reference)
    assert matched.shape == source.shape
    assert float(matched.mean()) == pytest.approx(float(reference.mean()), abs=0.5)
    # monotonic mapping: the ordering of the source is preserved
    assert torch.equal(matched.flatten().argsort(), source.flatten().argsort())


def test_wavelet_reconstruction_keeps_content_detail():
    torch.manual_seed(0)
    content = torch.randn(1, 3, 32, 32) * 0.2
    style = torch.full((1, 3, 32, 32), 0.5)
    out = color_fix.wavelet_reconstruction(content, style)
    assert out.shape == content.shape
    # low frequencies come from the style, high frequencies from the content
    assert float(out.mean()) == pytest.approx(0.5, abs=0.05)
    content_high, content_low = color_fix.wavelet_decomposition(content)
    _, style_low = color_fix.wavelet_decomposition(style)
    assert torch.allclose(content_high + content_low, content, atol=1e-5)
    assert torch.allclose(out, content_high + style_low, atol=1e-6)


def test_adain_matches_mean_and_std():
    torch.manual_seed(0)
    content = torch.randn(1, 3, 16, 16)
    style = torch.randn(1, 3, 16, 16) * 3 + 1
    out = color_fix.adain_color_fix(content, style)
    for c in range(3):
        assert float(out[0, c].mean()) == pytest.approx(float(style[0, c].mean()), abs=1e-3)
        assert float(out[0, c].std()) == pytest.approx(float(style[0, c].std()), abs=1e-2)


def test_lab_color_transfer_shape_and_range():
    torch.manual_seed(0)
    content = torch.rand(1, 3, 32, 32) * 2 - 1
    style = torch.rand(1, 3, 16, 16) * 2 - 1     # different size on purpose
    out = color_fix.lab_color_transfer(content, style)
    assert out.shape == content.shape
    assert float(out.min()) >= -1.001 and float(out.max()) <= 1.001


@pytest.mark.parametrize("method", ["none", "wavelet", "adain", "lab"])
def test_apply_color_correction_dispatch(method):
    content = torch.rand(1, 3, 16, 16) * 2 - 1
    style = torch.rand(1, 3, 16, 16) * 2 - 1
    out = color_fix.apply_color_correction(method, content, style)
    assert out.shape == content.shape
    if method == "none":
        assert out is content


def test_apply_color_correction_rejects_unknown():
    with pytest.raises(ValueError):
        color_fix.apply_color_correction("nope", torch.rand(1, 3, 4, 4), torch.rand(1, 3, 4, 4))


# --- NaResize resolution limits ---------------------------------------------------------

def _resize(image: torch.Tensor, resolution: int, max_resolution: int) -> tuple[int, int]:
    out = NaResize(resolution=resolution, mode="side", downsample_only=False,
                   max_resolution=max_resolution)(image)
    return out.shape[-1], out.shape[-2]


def test_na_resize_scales_the_short_edge():
    w, h = _resize(torch.rand(1, 3, 512, 256), resolution=1024, max_resolution=0)
    assert (w, h) == (1024, 2048)      # short edge (256) -> 1024


def test_na_resize_caps_the_long_edge():
    """``max_resolution`` caps the long edge."""
    w, h = _resize(torch.rand(1, 3, 512, 256), resolution=1024, max_resolution=1024)
    assert max(w, h) <= 1024
    assert (w, h) == (512, 1024)       # aspect ratio preserved


def test_side_resize_applies_the_final_capped_size_once(monkeypatch):
    import krea2pipe.seedvr2.transforms.side_resize as side_resize

    calls = []
    original_resize = side_resize.TVF.resize

    def recording_resize(image, size, interpolation):
        calls.append(tuple(size))
        return original_resize(image, size, interpolation)

    monkeypatch.setattr(side_resize.TVF, "resize", recording_resize)
    out = side_resize.SideResize(size=1024, max_size=1024)(
        torch.rand(1, 3, 512, 256)
    )

    assert out.shape[-2:] == (1024, 512)
    assert calls == [(1024, 512)]


def test_na_resize_square_input_hits_the_target():
    assert _resize(torch.rand(1, 3, 2496, 2496), 4096, 4096) == (4096, 4096)


def test_na_resize_downsample_only():
    small = torch.rand(1, 3, 256, 256)
    out = NaResize(resolution=1024, mode="side", downsample_only=True)(small)
    assert out.shape[-2:] == (256, 256)


def test_na_resize_area_mode():
    out = NaResize(resolution=1024, mode="area", downsample_only=False)(torch.rand(1, 3, 512, 2048))
    area = out.shape[-1] * out.shape[-2]
    assert area == pytest.approx(1024 ** 2, rel=0.02)


# --- config -----------------------------------------------------------------------------

def test_seedvr2_config_defaults_match_the_workflow():
    cfg = SeedVR2Config()
    assert cfg.seed == 1234567892
    assert cfg.resolution == 4096 and cfg.max_resolution == 4096
    assert cfg.sample_steps == 1 and cfg.cfg_scale == 1.0
    assert cfg.color_correction == "lab"
    assert cfg.config_path().exists()


def test_seedvr2_config_path_for_3b():
    assert SeedVR2Config(variant="3b").config_path().name == "main_3b.yaml"


def test_seedvr2_3b_config_instantiates_on_meta():
    from krea2pipe.seedvr2.config import create_object, load_config

    cfg = load_config(str(SeedVR2Config(variant="3b").config_path()))
    with torch.device("meta"):
        model = create_object(cfg.dit.model)

    assert cfg.dit.model.qk_rope is True
    assert len(model.blocks) == 32


def test_workflow_device_and_dtype_override_seedvr2_defaults():
    from krea2pipe.workflow import WorkflowConfig

    cfg = WorkflowConfig(
        model_root="/models",
        device="cpu",
        dtype=torch.float32,
        seedvr2=SeedVR2Config(device="cuda:7", dtype="float16"),
    )

    resolved = cfg.resolve_seedvr2()
    assert resolved.device == "cpu"
    assert resolved.dtype == "float32"
    assert resolved.model_dir == "/models/SEEDVR2"


def test_cli_propagates_device_and_dtype_into_seedvr2():
    from krea2pipe import cli

    args = cli.build_config_parser().parse_args([
        "--device", "cpu", "--dtype", "float32",
    ])
    args.prompt = "test"
    cfg = cli.config_from_args(args)

    assert cfg.seedvr2.device == "cpu"
    assert cfg.seedvr2.dtype == "float32"


def test_seedvr2_device_resolver_preserves_cuda_index():
    from krea2pipe.seedvr2.parallel import get_device

    assert get_device("cuda:3") == torch.device("cuda:3")


def test_dit_assign_loading_finishes_on_the_configured_device_and_dtype(monkeypatch):
    import krea2pipe.seedvr2.infer as infer

    class TinyDit(nn.Linear):
        def set_gradient_checkpointing(self, enabled):
            self.gradient_checkpointing = enabled

    config = OmegaConf.create({
        "dit": {
            "dtype": "float32",
            "gradient_checkpoint": False,
            "model": {"unused": True},
        },
    })
    requested = []

    monkeypatch.setattr(infer, "create_object", lambda config: TinyDit(2, 2))

    def load_state(path, dtype=None, device="cpu"):
        requested.append((dtype, device))
        return {
            "weight": torch.ones(2, 2, dtype=torch.float64),
            "bias": torch.ones(2, dtype=torch.float64),
        }

    monkeypatch.setattr(infer, "load_checkpoint_state", load_state)
    runner = infer.VideoDiffusionInfer(config, device="cpu")
    runner.configure_dit_model(checkpoint="dit.safetensors")

    assert requested == [(torch.float32, "cpu")]
    assert next(runner.dit.parameters()).device.type == "cpu"
    assert next(runner.dit.parameters()).dtype == torch.float32


@pytest.mark.parametrize("has_unexpected_key", [False, True])
def test_vae_checkpoint_loading_rejects_key_mismatches(
    monkeypatch, has_unexpected_key
):
    import krea2pipe.seedvr2.infer as infer

    config = OmegaConf.create({
        "vae": {
            "dtype": "float32",
            "checkpoint": "vae.safetensors",
            "model": {"unused": True},
        },
    })
    monkeypatch.setattr(infer, "create_object", lambda config: nn.Linear(2, 2))
    state = {
        "weight": torch.ones(2, 2),
        "bias": torch.ones(2),
    }
    if has_unexpected_key:
        state["extra"] = torch.ones(1)
    else:
        state.pop("bias")
    monkeypatch.setattr(
        infer,
        "load_checkpoint_state",
        lambda *args, **kwargs: state,
    )

    runner = infer.VideoDiffusionInfer(config, device="cpu")
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        runner.configure_vae_model()


def test_rope_cache_is_cleared_when_the_module_moves():
    from krea2pipe.seedvr2.dit.rope import RotaryEmbeddingBase

    rope = RotaryEmbeddingBase(dim=12, rope_dim=3)
    rope.get_axial_freqs.cache_clear()
    rope.get_axial_freqs(1, 2, 2)
    assert rope.get_axial_freqs.cache_info().currsize == 1

    rope.to(dtype=torch.float64)
    assert rope.get_axial_freqs.cache_info().currsize == 0


def test_dit_forward_reuses_and_discards_request_cache():
    from krea2pipe.seedvr2.dit.nadit import NaDiT

    counter = 0
    values = []
    caches = []

    class PatchIn(nn.Module):
        def forward(self, hidden, shape):
            return hidden, shape

    class PatchOut(nn.Module):
        def forward(self, hidden, shape, cache):
            caches.append(cache.cache)
            return hidden, shape

    class Embedding(nn.Module):
        def forward(self, timestep, device, dtype):
            return torch.zeros(1, 1, device=device, dtype=dtype)

    class Block(nn.Module):
        def forward(self, vid, txt, vid_shape, txt_shape, emb, cache):
            nonlocal counter

            def build():
                nonlocal counter
                counter += 1
                return counter

            values.append(cache("indices", build))
            return vid, txt, vid_shape, txt_shape

    model = NaDiT.__new__(NaDiT)
    nn.Module.__init__(model)
    model.need_txt_repeat = False
    model.gradient_checkpointing = False
    model.txt_in = nn.Identity()
    model.vid_in = PatchIn()
    model.emb_in = Embedding()
    model.blocks = nn.ModuleList([Block(), Block()])
    model.vid_out = PatchOut()

    args = {
        "vid": torch.zeros(1, 2),
        "txt": torch.zeros(1, 2),
        "vid_shape": torch.tensor([[1, 1, 1]]),
        "txt_shape": torch.tensor([[1]]),
        "timestep": torch.tensor([1.0]),
    }
    model(**args)
    model(**args)

    assert values == [1, 1, 2, 2]
    assert caches[0] is not caches[1]
    assert caches == [{}, {}]


def test_dit_offload_does_not_reload_until_the_next_sampling_request():
    import krea2pipe.seedvr2.infer as infer

    class FakeDit(nn.Module):
        def __init__(self):
            super().__init__()
            self.moves = []

        def to(self, device, *args, **kwargs):
            self.moves.append(str(torch.device(device)))
            return self

    class FakeSampler:
        timesteps = [1]

        def sample(self, x, f):
            return x

    config = OmegaConf.create({
        "diffusion": {"cfg": {"scale": 1.0, "rescale": 0.0, "partial": 1.0}},
    })
    runner = infer.VideoDiffusionInfer(config, device="cuda:3")
    runner.dit = FakeDit()
    runner.sampler = FakeSampler()
    latent = torch.zeros(1, 1, 1, 2)
    text = torch.zeros(1, 2)

    runner.inference_latents(
        noises=[latent],
        conditions=[latent],
        texts_pos=[text],
        texts_neg=[text],
        dit_offload=True,
    )

    assert runner.dit.moves == ["cuda:3", "cpu"]


@pytest.mark.parametrize("fails", [False, True])
def test_upscale_request_always_clears_vae_temporal_caches(monkeypatch, fails):
    import krea2pipe.seedvr2.runner as runner

    class FakeRunner:
        def __init__(self):
            self.clears = 0

        def clear_vae_temporal_caches(self):
            self.clears += 1

    fake = FakeRunner()
    upscaler = runner.SeedVR2Upscaler(
        SeedVR2Config(device="cpu", dtype="float32")
    )
    monkeypatch.setattr(upscaler, "load", lambda: fake)

    if fails:
        def run(*args):
            raise RuntimeError("failed request")
    else:
        def run(*args):
            return ["done"]

    monkeypatch.setattr(upscaler, "_upscale_videos_unlocked", run)
    if fails:
        with pytest.raises(RuntimeError, match="failed request"):
            upscaler._upscale_videos([], independent=False)
    else:
        assert upscaler._upscale_videos([], independent=False) == ["done"]
    assert fake.clears == 1


def test_cached_upscaler_serializes_shared_requests(monkeypatch):
    import krea2pipe.seedvr2.runner as runner

    state_lock = threading.Lock()
    active = 0
    max_active = 0

    class FakeUpscaler:
        def __init__(self, cfg):
            self.cfg = cfg

        def upscale_images(self, image):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            seed = self.cfg.seed
            time.sleep(0.02)
            with state_lock:
                active -= 1
            return image + seed

        def unload(self):
            pass

    runner.release_upscaler()
    monkeypatch.setattr(runner, "SeedVR2Upscaler", FakeUpscaler)
    image = torch.zeros(1, 1, 1, 1)
    configs = [
        SeedVR2Config(device="cpu", dtype="float32", seed=1),
        SeedVR2Config(device="cpu", dtype="float32", seed=2),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda cfg: runner.seedvr2_upscale(image, cfg), configs))
    runner.release_upscaler()

    assert [out.item() for out in outputs] == [1.0, 2.0]
    assert max_active == 1


def test_clear_vae_temporal_caches_drops_retained_tensors():
    from krea2pipe.seedvr2.infer import VideoDiffusionInfer
    from krea2pipe.seedvr2.vae.causal_inflation_lib import InflatedCausalConv3d

    conv = InflatedCausalConv3d(
        1, 1, kernel_size=1, inflation_mode="none"
    )
    conv.memory = torch.ones(1)
    runner = VideoDiffusionInfer(OmegaConf.create({}), device="cpu")
    runner.vae = nn.Sequential(conv)

    runner.clear_vae_temporal_caches()
    assert conv.memory is None


# --- spatial VAE tiling -----------------------------------------------------------------

class _FakeVAE:
    """Identity-ish stand-in that lets us check the tiling geometry without weights."""

    spatial_downsample_factor = 8

    def __init__(self):
        self.tiles = []

    def slicing_encode(self, x):
        self.tiles.append(tuple(x.shape[-2:]))
        b, _, f, h, w = x.shape
        return torch.full((b, 32, f, h // 8, w // 8), 1.0)

    def slicing_decode(self, z):
        self.tiles.append(tuple(z.shape[-2:]))
        b, _, f, h, w = z.shape
        return torch.full((b, 3, f, h * 8, w * 8), 1.0)


def test_tiled_encode_shape_and_blend_weights():
    vae = _FakeVAE()
    out = tiled_encode(vae, torch.zeros(1, 3, 1, 1024, 1024), tile_size=(512, 512),
                       tile_overlap=(128, 128))
    assert out.shape == (1, 32, 128, 128)
    assert len(vae.tiles) > 1, "a 1024px image must be split into several 512px tiles"
    # Constant tiles must reconstruct exactly: the blend weights have to sum to 1.
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)


def test_tiled_decode_shape_and_blend_weights():
    vae = _FakeVAE()
    out = tiled_decode(vae, torch.zeros(1, 32, 1, 128, 128), tile_size=(512, 512),
                       tile_overlap=(128, 128))
    assert out.shape == (1, 3, 1024, 1024)
    assert len(vae.tiles) > 1
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)


def test_tiling_is_skipped_when_the_image_fits_in_one_tile():
    vae = _FakeVAE()
    tiled_encode(vae, torch.zeros(1, 3, 1, 256, 256), tile_size=(512, 512),
                 tile_overlap=(128, 128))
    assert vae.tiles == [(256, 256)]

    vae = _FakeVAE()
    tiled_decode(vae, torch.zeros(1, 32, 1, 32, 32), tile_size=(512, 512),
                 tile_overlap=(128, 128))
    assert vae.tiles == [(32, 32)]


def test_workflow_helper_processes_images_as_independent_seedvr_jobs(monkeypatch):
    import krea2pipe.seedvr2.runner as runner

    calls = []

    class FakeUpscaler:
        def __init__(self, cfg):
            self.cfg = cfg

        def upscale_images(self, image):
            for item in image.split(1):
                calls.append(item.clone())
            return image + torch.arange(
                1, image.shape[0] + 1, dtype=image.dtype
            ).view(-1, 1, 1, 1)

        def unload(self):
            pass

    runner._CACHED.clear()
    monkeypatch.setattr(runner, "SeedVR2Upscaler", FakeUpscaler)
    image = torch.zeros(3, 4, 4, 3)
    out = runner.seedvr2_upscale(image, SeedVR2Config(device="cpu"))
    runner._CACHED.clear()

    assert [tuple(item.shape) for item in calls] == [(1, 4, 4, 3)] * 3
    assert torch.equal(out[:, 0, 0, 0], torch.tensor([1.0, 2.0, 3.0]))
