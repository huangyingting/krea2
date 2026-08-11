"""Backend tuning and ``torch.compile`` helpers.

The pipeline always takes the fastest path that is numerically safe and falls
back silently when a backend is missing or a graph refuses to compile, so none
of this is exposed as user-facing configuration.

``torch.compile`` results are cached by Inductor in ``/tmp/torchinductor_$USER``
(or ``$TORCHINDUCTOR_CACHE_DIR``), so the ~20 s first compile becomes ~3 s in
every later process on the same machine.
"""

from __future__ import annotations

import logging
import os
from typing import TypeVar

import torch

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=torch.nn.Module)

__all__ = ["tune_backends", "compile_module", "compile_repeated_blocks", "channels_last"]

_TUNED = False


def tune_backends() -> None:
    """Enable the backend flags that cost nothing and change no results."""
    global _TUNED
    if _TUNED or not torch.cuda.is_available():
        return
    _TUNED = True
    # Picks the best convolution algorithm per input shape.  Every stage here
    # runs many convolutions at a handful of repeated shapes, so the one-off
    # search pays for itself immediately.
    torch.backends.cudnn.benchmark = True


def _compile_disabled() -> bool:
    return os.environ.get("TORCHDYNAMO_DISABLE") == "1"


def compile_module(module: M, **kwargs) -> M:
    """``torch.compile`` ``module``, returning it unchanged if that is not possible."""
    if _compile_disabled() or not torch.cuda.is_available():
        return module
    try:
        return torch.compile(module, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on the local toolchain
        logger.debug("torch.compile unavailable (%s), running eager", exc)
        return module


def compile_repeated_blocks(blocks: torch.nn.ModuleList) -> None:
    """Compile each repeated transformer block in place.

    Compiling the blocks rather than the whole model keeps the compile to a
    single graph that is reused by every layer, which is what makes the cold
    cost bearable, and ``dynamic=True`` keeps one graph across the several
    sequence lengths the workflow uses (base resolution, upscale tiles).
    """
    if _compile_disabled() or not torch.cuda.is_available():
        return
    for i, block in enumerate(blocks):
        blocks[i] = compile_module(block, dynamic=True)


def channels_last(module: M) -> M:
    """Move a convolutional model to NHWC, which cuDNN prefers."""
    try:
        return module.to(memory_format=torch.channels_last)
    except (TypeError, RuntimeError):  # pragma: no cover - some wrappers reject kwargs
        return module
