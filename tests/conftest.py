"""Shared pytest fixtures / markers.

Tests are split in two groups:

* CPU tests - pure maths, run everywhere, no model weights needed.
* GPU tests (``@pytest.mark.gpu``) - need a CUDA device and the standalone model
  library. They are skipped automatically when either is missing; run them with
  ``uv run pytest -m gpu``.
"""

from __future__ import annotations

import os

import pytest
import torch

MODEL_ROOT = os.environ.get("KREA2_MODEL_ROOT", "/data/models")
os.environ.setdefault("KREA2_MODEL_ROOT", MODEL_ROOT)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires CUDA and the model library")
    config.addinivalue_line("markers", "slow: end-to-end run, several minutes")


@pytest.fixture(scope="session")
def model_root() -> str:
    if not os.path.isdir(MODEL_ROOT):
        pytest.skip(f"model library not found at {MODEL_ROOT}")
    return MODEL_ROOT


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.is_available() and os.path.isdir(MODEL_ROOT):
        return
    reason = "needs CUDA" if not torch.cuda.is_available() else f"needs {MODEL_ROOT}"
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
