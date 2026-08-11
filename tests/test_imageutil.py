"""Tests for the image helpers and the UltimateSDUpscale tiling maths."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from krea2pipe import imageutil, usdu


# --- tensor <-> PIL ------------------------------------------------------------------

def test_pil_tensor_round_trip():
    img = Image.new("RGB", (7, 5), (10, 200, 30))
    tensor = imageutil.pil_to_tensor(img)
    assert tensor.shape == (1, 5, 7, 3)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0
    back = imageutil.tensor_to_pil(tensor)
    assert back.size == (7, 5)
    assert back.getpixel((0, 0)) == (10, 200, 30)


def test_tensor_to_pil_selects_batch_index():
    batch = torch.cat([torch.zeros(1, 4, 4, 3), torch.ones(1, 4, 4, 3)])
    assert imageutil.tensor_to_pil(batch, 0).getpixel((0, 0)) == (0, 0, 0)
    assert imageutil.tensor_to_pil(batch, 1).getpixel((0, 0)) == (255, 255, 255)


def test_get_image_size_returns_width_height_batch():
    assert imageutil.get_image_size(torch.zeros(3, 5, 7, 3)) == (7, 5, 3)


# --- resizing --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"])
def test_image_scale_methods(method):
    out = imageutil.image_scale(torch.rand(1, 16, 16, 3), method, 32, 24)
    assert out.shape == (1, 24, 32, 3)
    # like ComfyUI, only the (PIL based) lanczos path clamps; bicubic may overshoot
    tol = 0.0 if method != "bicubic" else 0.2
    assert -tol <= float(out.min()) and float(out.max()) <= 1.0 + tol


def test_lanczos_preserves_flat_colour():
    flat = torch.full((1, 3, 16, 16), 0.5)
    out = imageutil.lanczos(flat, 32, 32)
    assert out.shape == (1, 3, 32, 32)
    # comfy.utils.lanczos round-trips through 8 bit PIL images
    assert torch.allclose(out, torch.full_like(out, 0.5), atol=1 / 255)


def test_common_upscale_center_crop_keeps_aspect():
    image = torch.rand(1, 3, 16, 32)                       # wide
    out = imageutil.common_upscale(image, 16, 16, "bilinear", "center")
    assert out.shape == (1, 3, 16, 16)


def test_get_tiled_scale_steps():
    assert imageutil.get_tiled_scale_steps(1024, 1024, 512, 512, 32) > 0
    assert imageutil.get_tiled_scale_steps(512, 512, 512, 512, 0) == 1


def test_tiled_scale_matches_direct_upscale_for_identity():
    image = torch.rand(1, 3, 64, 64)
    out = imageutil.tiled_scale(image, lambda x: x, tile_x=32, tile_y=32, overlap=8,
                                upscale_amount=1.0)
    assert out.shape == image.shape
    assert torch.allclose(out, image, atol=2e-2)


def test_tiled_scale_batches_matching_tiles_across_images():
    seen = []

    def identity(tile):
        seen.append(tuple(tile.shape))
        return tile

    image = torch.rand(3, 3, 64, 64)
    out = imageutil.tiled_scale(
        image, identity, tile_x=32, tile_y=32, overlap=8, upscale_amount=1.0
    )
    assert torch.allclose(out, image, atol=2e-2)
    assert seen and all(shape[0] == 3 for shape in seen)


# --- UltimateSDUpscale helpers -----------------------------------------------------------

def test_round_length_snaps_to_multiple_of_8():
    assert usdu.round_length(96) == 96
    assert usdu.round_length(100) == 96      # round(12.5) is banker's rounding, like upstream
    assert usdu.round_length(101) == 104
    assert usdu.round_length(1248 * 1.5) == 1872


def test_get_crop_region_of_a_mask():
    mask = Image.new("L", (64, 64), 0)
    for x in range(10, 20):
        for y in range(30, 40):
            mask.putpixel((x, y), 255)
    # fix_crop_region shaves one pixel off edges that are inside the image
    assert usdu.get_crop_region(mask, pad=0) == (10, 30, 19, 39)
    assert usdu.get_crop_region(mask, pad=5) == (5, 25, 24, 44)


def test_get_crop_region_clamps_padding_to_the_image():
    mask = Image.new("L", (16, 16), 0)
    mask.putpixel((0, 0), 255)
    x1, y1, x2, y2 = usdu.get_crop_region(mask, pad=8)
    assert (x1, y1) == (0, 0)
    assert x2 <= 16 and y2 <= 16


def test_fix_crop_region_shrinks_interior_edges():
    """Verbatim upstream behaviour: interior edges lose one pixel, borders do not."""
    assert usdu.fix_crop_region((0, 0, 32, 32), (64, 64)) == (0, 0, 31, 31)
    assert usdu.fix_crop_region((0, 0, 64, 64), (64, 64)) == (0, 0, 64, 64)


def test_expand_crop_reaches_the_target_size():
    x1, y1, x2, y2 = 10, 10, 20, 20
    (nx1, ny1, nx2, ny2), (w, h) = usdu.expand_crop((x1, y1, x2, y2), 64, 64, 32, 32)
    assert (w, h) == (32, 32)
    assert nx2 - nx1 == 32 and ny2 - ny1 == 32
    assert 0 <= nx1 and nx2 <= 64 and 0 <= ny1 and ny2 <= 64


def test_expand_crop_is_clamped_by_the_image():
    (nx1, ny1, nx2, ny2), size = usdu.expand_crop((0, 0, 8, 8), 16, 16, 64, 64)
    assert (nx1, ny1, nx2, ny2) == (0, 0, 16, 16)   # cannot grow past the image
    assert size == (64, 64)                        # upstream returns the requested tile size


def test_get_factors_multiply_to_the_scale():
    for scale in (2, 3, 4, 6, 8):
        factors = usdu.get_factors(scale)
        product = 1
        for f in factors:
            product *= f
        assert product == scale
        assert all(f <= 4 for f in factors)


def test_usdu_params_defaults_match_the_workflow():
    params = usdu.USDUParams()
    assert params.mode_type in {"Linear", "Chess", "None"}
    assert params.tile_padding >= 0


def test_usdu_passes_the_outer_image_batch_to_one_worker(monkeypatch):
    seen = []

    def fake_run(self, image):
        seen.append(tuple(image.shape))
        return image + 1

    monkeypatch.setattr(usdu.UltimateSDUpscale, "run", fake_run)
    image = torch.zeros(3, 4, 4, 3)
    out = usdu.ultimate_sd_upscale(None, image, None, None, usdu.USDUParams())

    assert seen == [(3, 4, 4, 3)]
    assert torch.equal(out[:, 0, 0, 0], torch.ones(3))
