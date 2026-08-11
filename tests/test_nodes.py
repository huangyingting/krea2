"""Unit tests for standalone workflow operations."""

from __future__ import annotations

import pytest
import torch

from krea2pipe import blend, color_match, metadata, nodes


# --- ResolutionSelector (node 43) ---------------------------------------------------

def test_resolution_selector_matches_workflow():
    """The workflow's widgets (1:1, 1.5 MP, /32) must give 1248x1248."""
    assert nodes.resolution_selector("1:1 (Square)", 1.5, 32) == (1248, 1248)


@pytest.mark.parametrize("ratio", sorted(nodes.ASPECT_RATIOS))
def test_resolution_selector_is_multiple_of(ratio):
    w, h = nodes.resolution_selector(ratio, 1.5, 32)
    assert w % 32 == 0 and h % 32 == 0
    # ~1.5 megapixels, allowing for the rounding to a multiple of 32
    assert 1.3 < (w * h) / (1024 * 1024) < 1.7


def test_resolution_selector_orientation():
    w, h = nodes.resolution_selector("16:9 (Widescreen)", 1.5, 32)
    assert w > h
    pw, ph = nodes.resolution_selector("9:16 (Portrait Widescreen)", 1.5, 32)
    assert (pw, ph) == (h, w)


# --- SimpleMath+ (nodes 62 / 63) ----------------------------------------------------

def test_simple_math_tile_size_expression():
    """``(a*b + (96 * 2))/2`` with the workflow's values -> 1344 tile size."""
    assert nodes.simple_math("(a*b + (96 * 2))/2", a=1248, b=2.0)[0] == 1344


def test_simple_math_returns_int_and_float():
    as_int, as_float = nodes.simple_math("a/b", a=7, b=2)
    assert (as_int, as_float) == (4, 3.5)  # round() is banker's rounding, like the node


def test_simple_math_supports_functions_and_comparisons():
    assert nodes.simple_math("max(a, b)", a=3, b=9)[0] == 9
    assert nodes.simple_math("a > b", a=3, b=9)[0] == 0
    assert nodes.simple_math("a > b", a=9, b=3)[0] == 1


def test_simple_math_nan_is_zero():
    assert nodes.simple_math("a", a=float("nan"))[1] == 0.0


# --- Text Load Line From File (node 79) ---------------------------------------------

def test_text_load_line_from_file_wraps(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("first\n  second  \nthird\n", encoding="utf-8")
    assert nodes.text_load_line_from_file(str(path), 0) == "first"
    assert nodes.text_load_line_from_file(str(path), 1) == "second"  # stripped
    # the workflow asks for line 430 of a 430-line file -> wraps back to line 0
    assert nodes.text_load_line_from_file(str(path), 3) == "first"
    assert nodes.text_load_line_from_file(str(path), 7) == "second"


def test_text_load_line_from_file_negative_index(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    assert nodes.text_load_line_from_file(str(path), -1) == ""


# --- ImageBlend (node 71) -----------------------------------------------------------

def _img(value: float, size: int = 8) -> torch.Tensor:
    return torch.full((1, size, size, 3), value)


def test_image_blend_normal_is_a_lerp():
    out = blend.image_blend(_img(0.0), _img(1.0), 0.4, "normal")
    assert torch.allclose(out, _img(0.4))


def test_image_blend_factor_extremes():
    a, b = _img(0.2), _img(0.9)
    assert torch.allclose(blend.image_blend(a, b, 0.0), a)
    assert torch.allclose(blend.image_blend(a, b, 1.0), b)


def test_image_blend_multiply_and_screen():
    a, b = _img(0.5), _img(0.5)
    assert torch.allclose(blend.image_blend(a, b, 1.0, "multiply"), _img(0.25))
    assert torch.allclose(blend.image_blend(a, b, 1.0, "screen"), _img(0.75))


def test_image_blend_resizes_mismatched_input():
    out = blend.image_blend(_img(0.0, 16), _img(1.0, 8), 0.5)
    assert out.shape == (1, 16, 16, 3)


def test_image_blend_rejects_unknown_mode():
    with pytest.raises(ValueError):
        blend.image_blend(_img(0.0), _img(1.0), 0.5, "does_not_exist")


def test_upscale_and_blend_is_a_standalone_stage(monkeypatch):
    monkeypatch.setattr(
        blend,
        "image_upscale_with_model",
        lambda _model, image: image.repeat_interleave(2, 1).repeat_interleave(2, 2),
    )
    source = _img(0.8, 8)
    primary = _img(0.2, 16)
    out = blend.upscale_and_blend(object(), source, primary, 0.5)
    assert torch.allclose(out, _img(0.5, 16))


# --- ColorMatch (node 61) -----------------------------------------------------------

def test_color_match_strength_zero_is_identity():
    target = torch.rand(1, 16, 16, 3)
    out = color_match.color_match(
        torch.rand(1, 16, 16, 3), target, "hm-mkl-hm", 0.0
    )
    assert out is target


def test_color_match_shifts_towards_reference():
    torch.manual_seed(0)
    target = torch.rand(1, 32, 32, 3) * 0.3            # dark image
    reference = torch.rand(1, 32, 32, 3) * 0.3 + 0.6   # bright image
    out = color_match.color_match(reference, target, "mkl", 1.0)
    assert out.shape == target.shape
    assert out.dtype == torch.float32
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
    assert float(out.mean()) > float(target.mean())


def test_color_match_strength_interpolates():
    torch.manual_seed(0)
    target = torch.rand(1, 32, 32, 3) * 0.3
    reference = torch.rand(1, 32, 32, 3) * 0.3 + 0.6
    full = color_match.color_match(reference, target, "mkl", 1.0)
    partial = color_match.color_match(reference, target, "mkl", 0.22)
    expected = (target + 0.22 * (full - target)).clamp(0, 1)
    assert torch.allclose(partial, expected, atol=2e-3)


def test_color_match_batch_matches_independent_images():
    torch.manual_seed(1)
    target = torch.rand(4, 24, 24, 3) * 0.3
    reference = torch.rand(4, 24, 24, 3) * 0.3 + 0.6
    batched = color_match.color_match(reference, target, "mkl", 0.22)
    independent = torch.cat([
        color_match.color_match(reference[i:i + 1], target[i:i + 1], "mkl", 0.22)
        for i in range(4)
    ])
    assert torch.equal(batched, independent)


def test_color_match_does_not_hide_transfer_failures(monkeypatch):
    from color_matcher import ColorMatcher

    def fail(*_args, **_kwargs):
        raise RuntimeError("transfer failed")

    monkeypatch.setattr(ColorMatcher, "transfer", fail)
    with pytest.raises(RuntimeError, match="transfer failed"):
        color_match.color_match(_img(0.8), _img(0.2), "mkl")


@pytest.mark.parametrize("method", color_match.METHODS)
def test_color_match_supports_all_documented_methods(method):
    torch.manual_seed(2)
    target = torch.rand(1, 16, 16, 3)
    reference = torch.rand(1, 16, 16, 3)
    out = color_match.color_match(reference, target, method, 0.5)
    assert out.shape == target.shape
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


# --- Image Saver (node 74) ----------------------------------------------------------

def test_save_image_tokens_and_batch(tmp_path):
    image = torch.rand(2, 8, 16, 3)
    paths = nodes.save_image(image, str(tmp_path), "%width-%height-%counter",
                             subdir="AIKC", extension="png")
    assert len(paths) == 2
    assert paths[0].endswith("16-8-0_00.png")   # batch index suffix
    assert paths[1].endswith("16-8-0_01.png")
    for path in paths:
        assert "AIKC" in path


def test_save_image_jpeg_with_metadata(tmp_path):
    from PIL import Image

    meta = nodes.a1111_metadata("a prompt", "", 8, "euler_ancestral", 1.0, 42, 64, 64, "model.safetensors")
    (path,) = nodes.save_image(torch.rand(1, 64, 64, 3), str(tmp_path), "meta",
                               extension="jpg", quality=100, metadata=meta)
    assert path.endswith(".jpg")
    with Image.open(path) as img:
        assert img.size == (64, 64)
        # piexif stores the A1111 parameter string as a UTF-16 UserComment
        assert "a prompt".encode("utf-16-be") in img.info.get("exif", b"")


def test_save_image_is_atomic_and_leaves_no_temporary_file(tmp_path):
    (path,) = nodes.save_image(
        torch.rand(1, 8, 8, 3), str(tmp_path), "atomic", extension="png"
    )
    assert path.endswith("atomic.png")
    assert [item.name for item in tmp_path.iterdir()] == ["atomic.png"]


def test_save_image_rejects_a_subdirectory_outside_output_root(tmp_path):
    with pytest.raises(ValueError, match="escapes output-dir"):
        nodes.save_image(
            torch.rand(1, 8, 8, 3), str(tmp_path), "escape",
            subdir="../outside", extension="png",
        )


@pytest.mark.parametrize("extension", ["png", "jpg", "webp"])
def test_generation_manifest_round_trips_in_every_output_format(tmp_path, extension):
    from krea2pipe.workflow import WorkflowConfig

    cfg = WorkflowConfig(
        prompt="portable prompt",
        prompt_theme="future city",
        prompt_index=7,
        prompt_seed=108,
        unet_name="/private/models/krea2.safetensors",
        loras=[("style-a.safetensors", 0.4), ("style-b.safetensors", 0.7)],
        batch_size=2,
        seed=101,
        scheduler="simple",
        usdu_seed=202,
        color_match_method="mkl",
        blend_factor=0.3,
        extension=extension,
    )
    manifest = metadata.build_generation_manifest(cfg, cfg.prompt, 640, 384)
    assert "/private/models" not in metadata.encode_manifest(manifest)
    assert "model_root" not in metadata.encode_manifest(manifest)
    parameters = nodes.a1111_metadata(
        cfg.prompt, "", cfg.steps, cfg.sampler_name, cfg.cfg, cfg.seed,
        640, 384, cfg.unet_name,
    )
    paths = nodes.save_image(
        torch.rand(2, 12, 20, 3),
        str(tmp_path),
        f"manifest-{extension}",
        extension=extension,
        metadata=parameters,
        generation_manifest=manifest,
        image_stage="seedvr2",
    )

    for index, path in enumerate(paths):
        restored = metadata.read_generation_manifest(path)
        assert restored is not None
        assert restored["schema_version"] == 1
        assert restored["prompt"]["positive"] == "portable prompt"
        assert restored["prompt"]["expansion"]["theme"] == "future city"
        assert restored["prompt"]["expansion"]["index"] == 7
        assert restored["prompt"]["expansion"]["seed"] == 108
        assert restored["models"]["diffusion"] == "krea2.safetensors"
        assert restored["models"]["loras"][1] == {
            "name": "style-b.safetensors",
            "strength": 0.7,
        }
        assert restored["base"]["seed"] == 101
        assert restored["base"]["scheduler"] == "simple"
        assert restored["stages"]["usdu"]["seed"] == 202
        assert restored["stages"]["color_match"]["method"] == "mkl"
        assert restored["stages"]["blend"]["factor"] == 0.3
        assert restored["image"] == {
            "stage": "seedvr2",
            "batch_index": index,
            "width": 20,
            "height": 12,
            "format": extension,
        }
        if extension != "png":
            import piexif
            import piexif.helper

            exif = piexif.load(path)
            user_comment = exif["Exif"][piexif.ExifIFD.UserComment]
            assert piexif.helper.UserComment.load(user_comment) == parameters


def test_png_keeps_a1111_parameters_alongside_generation_manifest(tmp_path):
    from PIL import Image
    from krea2pipe.workflow import WorkflowConfig

    cfg = WorkflowConfig(prompt="a prompt")
    (path,) = nodes.save_image(
        torch.rand(1, 8, 8, 3),
        str(tmp_path),
        "metadata",
        extension="png",
        metadata="A1111 parameters",
        generation_manifest=metadata.build_generation_manifest(cfg, cfg.prompt, 8, 8),
    )
    with Image.open(path) as image:
        assert image.info["parameters"] == "A1111 parameters"
        assert metadata.PNG_KEY in image.info


def test_manifest_reader_returns_none_for_an_unrelated_image(tmp_path):
    (path,) = nodes.save_image(
        torch.rand(1, 8, 8, 3), str(tmp_path), "plain", extension="png"
    )
    assert metadata.read_generation_manifest(path) is None


def test_a1111_metadata_format():
    meta = nodes.a1111_metadata("pos", "neg", 8, "euler_ancestral", 1.0, 7, 1248, 1248, "m.safetensors")
    assert meta.startswith("pos")
    assert "Negative prompt: neg" in meta
    assert "Steps: 8" in meta
    assert "Seed: 7" in meta
    assert "Size: 1248x1248" in meta


# --- workflow-level wiring ----------------------------------------------------------

def test_workflow_defaults_match_the_json():
    from krea2pipe.workflow import WorkflowConfig

    cfg = WorkflowConfig()
    assert cfg.resolve_size() == (1248, 1248)
    assert cfg.seed == 1099257494857840
    assert cfg.steps == 8 and cfg.cfg == 1.0
    assert cfg.sampler_name == "euler_ancestral" and cfg.scheduler == "sgm_uniform"
    assert cfg.lora_strength == pytest.approx(0.6)
    assert cfg.usdu_upscale_by == 2.0 and cfg.usdu_seed == 82616517812345
    assert cfg.usdu_denoise == pytest.approx(0.1) and cfg.usdu_mode == "Chess"
    assert cfg.color_match_method == "hm-mkl-hm"
    assert cfg.color_match_strength == pytest.approx(0.22)
    assert cfg.run_color_match
    assert cfg.blend_factor == pytest.approx(0.4)
    assert cfg.blend_upscale_model_name == "4xNomosWebPhoto_RealPLKSR.pth"
    assert cfg.seedvr2.resolution == 4096 and cfg.seedvr2.max_resolution == 4096
    assert cfg.seedvr2.seed == 1234567892
    assert cfg.seedvr2.color_correction == "lab"


def test_workflow_usdu_tile_size_from_simple_math():
    """Tile size follows the SimpleMath+ nodes: (base * upscale_by + 192) / 2."""
    width, _ = nodes.resolution_selector("1:1 (Square)", 1.5, 32)
    assert nodes.simple_math("(a*b + (96 * 2))/2", a=width, b=2.0)[0] == 1344


def test_resolution_selector_accepts_short_aspect_ratio():
    assert nodes.resolution_selector("16:9", 1.5, 32) == nodes.resolution_selector(
        "16:9 (Widescreen)", 1.5, 32
    )


def test_config_parses_workflow_defaults(tmp_path):
    from krea2pipe.cli import config_from_args, parse_args

    cfg = config_from_args(parse_args([]))
    assert cfg.resolve_size() == (1248, 1248)
    assert cfg.run_usdu and cfg.run_seedvr2 and cfg.run_blend and cfg.save

    file = tmp_path / "settings.toml"
    file.write_text(
        "width = 512\n"
        "height = 512\n"
        "run-usdu = false\n"
        "run-seedvr2 = false\n"
        "save = false\n"
        "seedvr2-resolution = 1024\n"
    )
    cfg = config_from_args(parse_args(["--config", str(file)]))
    assert cfg.resolve_size() == (512, 512)
    assert not cfg.run_usdu and not cfg.run_seedvr2 and not cfg.save
    assert cfg.seedvr2.resolution == 1024


def test_config_dtype_choice(tmp_path):
    from krea2pipe.cli import config_from_args, parse_args

    file = tmp_path / "settings.toml"
    file.write_text('dtype = "float32"\n')
    cfg = config_from_args(parse_args(["--config", str(file)]))
    assert cfg.dtype is torch.float32
