"""Work around the PyTorch 2.9/2.10 + cuDNN >= 9.10.2 ``Conv3d`` dispatch bug.

On those builds a half/bfloat16 ``Conv3d`` is routed through a path that
allocates roughly 3x the memory it needs and runs noticeably slower.  Calling
cuDNN directly sidesteps the buggy dispatch layer.  Both VAEs in this project
(Qwen-Image / Wan2.1 for Krea2 and the SeedVR2 causal video VAE) are built out
of ``Conv3d``, so this is worth several seconds per 4K image.

The workaround is probed once and falls back to stock ``Conv3d`` whenever it
does not apply.
"""

from __future__ import annotations

import torch

__all__ = ["CONV3D_CUDNN_WORKAROUND", "conv3d_forward"]


def _probe() -> bool:
    try:
        if getattr(torch.version, "hip", None) is not None:
            return False
        if not torch.cuda.is_available() or not torch.backends.cudnn.is_available():
            return False
        major, minor = (int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
        if not ((2, 9) <= (major, minor) <= (2, 10)):
            return False
        cudnn_version = torch.backends.cudnn.version()
        return cudnn_version is not None and cudnn_version >= 91002
    except Exception:  # pragma: no cover - defensive probe
        return False


CONV3D_CUDNN_WORKAROUND = _probe()


def conv3d_forward(module: torch.nn.Conv3d, input: torch.Tensor, weight: torch.Tensor,
                   bias: torch.Tensor | None):
    """Return ``module``'s convolution, or ``None`` if the fast path is unavailable."""
    if not (CONV3D_CUDNN_WORKAROUND and weight.dtype in (torch.float16, torch.bfloat16)
            and input.is_cuda and torch.backends.cudnn.enabled):
        return None
    try:
        out = torch.cudnn_convolution(
            input, weight, module.padding, module.stride, module.dilation, module.groups,
            benchmark=False, deterministic=False, allow_tf32=True,
        )
    except RuntimeError:
        return None
    if bias is not None:
        out += bias.reshape((1, -1) + (1,) * (out.ndim - 2))
    return out
