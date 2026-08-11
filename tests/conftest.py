"""Shared pytest fixtures / markers.

Tests are split in two groups:

* CPU tests - pure maths, run everywhere, no model weights needed.
* GPU tests (``@pytest.mark.gpu``) - need a CUDA device and the ComfyUI model
  tree.  They are skipped automatically when either is missing; run them with
  ``uv run pytest -m gpu``.
"""

from __future__ import annotations

import os

import pytest
import torch

COMFY_ROOT = os.environ.get("COMFYUI_ROOT", "/data/ComfyUI")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires a CUDA device and the ComfyUI model tree")
    config.addinivalue_line("markers", "slow: end-to-end run, several minutes")


@pytest.fixture(scope="session")
def comfy_root() -> str:
    if not os.path.isdir(COMFY_ROOT):
        pytest.skip(f"ComfyUI tree not found at {COMFY_ROOT}")
    return COMFY_ROOT


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.is_available() and os.path.isdir(COMFY_ROOT):
        return
    reason = "needs CUDA" if not torch.cuda.is_available() else f"needs {COMFY_ROOT}"
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
