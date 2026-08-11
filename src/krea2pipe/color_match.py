"""Standalone color transfer stage corresponding to the KJNodes ColorMatch node."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

METHODS = ("default", "hm", "reinhard", "mvgd", "mkl", "hm-mvgd-hm", "hm-mkl-hm")


def color_match(image_ref: Tensor, image_target: Tensor, method: str = "mkl",
                strength: float = 1.0) -> Tensor:
    """Transfer reference colors to a BHWC image batch."""
    if method not in METHODS:
        choices = ", ".join(METHODS)
        raise ValueError(f"unsupported color-match method {method!r}; choose one of: {choices}")
    if strength == 0:
        return image_target
    from color_matcher import ColorMatcher

    image_ref = image_ref.cpu()
    image_target = image_target.cpu()
    batch_size = image_target.size(0)
    images_target = image_target.squeeze()
    images_ref = image_ref.squeeze()
    image_ref_np = images_ref.numpy()
    images_target_np = images_target.numpy()

    def match_one(i: int) -> Tensor:
        cm = ColorMatcher()
        target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
        ref_np = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
        try:
            result = cm.transfer(src=target_np, ref=ref_np, method=method)
            if strength != 1:
                result = target_np + strength * (result - target_np)
            return torch.from_numpy(result)
        except Exception as exc:  # pragma: no cover - matches the node's behaviour
            logger.warning("color match failed (%s), passing target through", exc)
            return torch.from_numpy(target_np)

    if batch_size == 1:
        out = [match_one(0)]
    else:
        # ColorMatcher releases the GIL; cap workers to avoid oversubscribing NumPy.
        with ThreadPoolExecutor(max_workers=min(batch_size, 4)) as pool:
            out = list(pool.map(match_one, range(batch_size)))

    result = torch.stack(out, dim=0)
    if result.is_complex():
        result = result.real
    return result.to(torch.float32).clamp_(0, 1)
