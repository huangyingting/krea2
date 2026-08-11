"""Small standalone workflow operations."""

from __future__ import annotations

import ast
import logging
import math
import operator as op
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from . import metadata as image_metadata
from .blend import image_blend
from .color_match import color_match
from .imageutil import tensor_to_pil

logger = logging.getLogger(__name__)


# --- Resolution selection ------------------------------------------------------------

ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}
_SHORT_ASPECT_RATIOS = {
    label.split(" ", 1)[0]: dimensions for label, dimensions in ASPECT_RATIOS.items()
}


def resolution_selector(aspect_ratio: str, megapixels: float, multiple: int) -> tuple[int, int]:
    dimensions = ASPECT_RATIOS.get(aspect_ratio) or _SHORT_ASPECT_RATIOS.get(
        aspect_ratio.strip()
    )
    if dimensions is None:
        choices = ", ".join(_SHORT_ASPECT_RATIOS)
        raise ValueError(f"unsupported aspect ratio {aspect_ratio!r}; choose one of: {choices}")
    w_ratio, h_ratio = dimensions
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
    width = round(w_ratio * scale / multiple) * multiple
    height = round(h_ratio * scale / multiple) * multiple
    return width, height


# --- Safe arithmetic -----------------------------------------------------------------

_MATH_OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod,
    ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le, ast.Gt: op.gt,
    ast.GtE: op.ge, ast.And: lambda x, y: x and y, ast.Or: lambda x, y: x or y,
    ast.Not: op.not_,
}
_MATH_FUNCS = {"min": min, "max": max, "round": round, "sum": sum, "len": len}


def simple_math(value: str, a=0.0, b=0.0, c=0.0, d=0.0) -> tuple[int, float]:
    """Evaluate restricted arithmetic and return integer and floating results."""

    def eval_(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return {"a": a, "b": b, "c": c, "d": d}.get(node.id, 0)
        if isinstance(node, ast.BinOp):
            return _MATH_OPERATORS[type(node.op)](eval_(node.left), eval_(node.right))
        if isinstance(node, ast.UnaryOp):
            return _MATH_OPERATORS[type(node.op)](eval_(node.operand))
        if isinstance(node, ast.Compare):
            left = eval_(node.left)
            for operator, comparator in zip(node.ops, node.comparators):
                if not _MATH_OPERATORS[type(operator)](left, eval_(comparator)):
                    return 0
            return 1
        if isinstance(node, ast.BoolOp):
            values = [eval_(v) for v in node.values]
            return _MATH_OPERATORS[type(node.op)](*values)
        if isinstance(node, ast.Call) and node.func.id in _MATH_FUNCS:
            return _MATH_FUNCS[node.func.id](*[eval_(arg) for arg in node.args])
        if isinstance(node, ast.Subscript):
            container = eval_(node.value)
            if isinstance(node.slice, ast.Constant):
                return container[node.slice.value]
            return 0
        return 0

    result = eval_(ast.parse(value, mode="eval").body)
    if isinstance(result, float) and math.isnan(result):
        result = 0.0
    return round(result), result


# --- Text Load Line From File (WAS node suite) ---------------------------------------

def text_load_line_from_file(file_path: str, index: int, mode: str = "index") -> str:
    """``Text Load Line From File`` (mode="index"): strip lines, wrap the index modulo."""
    with open(file_path, "r", encoding="utf-8", newline="\n") as fh:
        lines = [line.strip() for line in fh]
    if not lines:
        return ""
    if mode != "index":
        raise ValueError("only mode='index' is supported")
    if index >= len(lines):
        index = index % len(lines)
    if index < 0:
        return ""
    return lines[index]


# --- Image saving ---------------------------------------------------------------------

def _sanitize(name: str) -> str:
    return "".join(ch for ch in name if ch not in '\\/:*?"<>|').strip()


def save_image(
    image: Tensor,
    output_dir: str,
    filename: str = "%time",
    subdir: str = "",
    extension: str = "jpg",
    quality: int = 100,
    time_format: str = "%Y-%m-%d-%H%M%S",
    metadata: str | None = None,
    counter: int = 0,
    *,
    generation_manifest: Mapping[str, Any] | None = None,
    image_stage: str = "final",
) -> list[str]:
    """Save a BHWC image batch with portable parameters and generation metadata."""
    name = filename
    name = name.replace("%date", datetime.now().strftime("%Y-%m-%d"))
    name = name.replace("%time", datetime.now().strftime(time_format))
    name = name.replace("%counter", str(counter))
    name = name.replace("%width", str(image.shape[2]))
    name = name.replace("%height", str(image.shape[1]))
    directory, basename = os.path.split(name)
    basename = _sanitize(basename) or "image"

    output_root = Path(output_dir).expanduser().resolve()
    target_dir = (output_root / subdir / directory).resolve()
    try:
        target_dir.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"output subdirectory escapes output-dir: {target_dir}"
        ) from exc
    target_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    batch = image.shape[0]
    for i in range(batch):
        suffix = "" if batch == 1 else f"_{i:02d}"
        path = str(target_dir / f"{basename}{suffix}.{extension}")
        n = 1
        while os.path.exists(path):
            path = str(target_dir / f"{basename}{suffix}_{n:02d}.{extension}")
            n += 1
        img = tensor_to_pil(image, i)
        manifest_text = None
        if generation_manifest is not None:
            manifest_text = image_metadata.encode_manifest(
                image_metadata.manifest_for_image(
                    generation_manifest,
                    stage=image_stage,
                    batch_index=i,
                    width=image.shape[2],
                    height=image.shape[1],
                    image_format=extension,
                )
            )
        fd, temporary = tempfile.mkstemp(
            prefix=f".{basename}-", suffix=f".{extension}", dir=target_dir
        )
        os.close(fd)
        try:
            if extension == "png":
                from PIL.PngImagePlugin import PngInfo

                info = PngInfo()
                if metadata:
                    info.add_text("parameters", metadata)
                if manifest_text:
                    info.add_itxt(image_metadata.PNG_KEY, manifest_text, zip=True)
                img.save(temporary, format="PNG", pnginfo=info, optimize=True)
            else:
                image_format = "JPEG" if extension in {"jpg", "jpeg"} else extension.upper()
                img.save(temporary, format=image_format, optimize=True, quality=quality)
                if metadata or manifest_text:
                    import piexif
                    import piexif.helper

                    exif: dict[str, dict[int, bytes]] = {}
                    if manifest_text:
                        exif["0th"] = {
                            piexif.ImageIFD.ImageDescription: manifest_text.encode("utf-8")
                        }
                    if metadata:
                        exif["Exif"] = {
                            piexif.ExifIFD.UserComment:
                                piexif.helper.UserComment.dump(metadata, encoding="unicode")
                        }
                    piexif.insert(piexif.dump(exif), temporary)
            with open(temporary, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(target_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        paths.append(path)
        logger.info("saved %s", path)
    return paths


def a1111_metadata(positive: str, negative: str, steps: int, sampler: str, cfg: float,
                   seed: int, width: int, height: int, model_name: str = "") -> str:
    """Build an A1111-compatible image metadata string."""
    return (
        f"{positive}\n"
        f"Negative prompt: {negative}\n"
        f"Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg}, Seed: {seed}, "
        f"Size: {width}x{height}, Model: {model_name}, Version: krea2pipe"
    )
