"""LoRA loading and in-place inference-weight merging.

Krea 2 LoRAs use diffusers-style keys
(``diffusion_model.<module>.lora_A/lora_B.weight``). Without an ``alpha``
tensor, the update is::

    W += strength * (B @ A)

The update is computed in float32 and cast back to the weight dtype. Because
the model is inference-only, patches are merged directly into its weights.
"""

from __future__ import annotations

import logging

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def load_lora_file(path: str) -> dict[str, torch.Tensor]:
    return load_file(path, device="cpu")


def apply_lora(
    model: torch.nn.Module,
    lora: dict[str, torch.Tensor],
    strength: float,
    key_prefix: str = "diffusion_model.",
) -> int:
    """Merge a LoRA into ``model`` in-place.  Returns the number of patched weights."""
    if strength == 0:
        return 0

    params = dict(model.named_parameters())
    patched = 0
    unmatched: list[str] = []

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in lora.items():
        for suffix, slot in ((".lora_A.weight", "A"), (".lora_B.weight", "B"),
                             (".lora_down.weight", "A"), (".lora_up.weight", "B"),
                             (".alpha", "alpha")):
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                if base.startswith(key_prefix):
                    base = base[len(key_prefix):]
                pairs.setdefault(base, {})[slot] = value
                break

    for base, tensors in pairs.items():
        name = base + ".weight"
        if "A" not in tensors or "B" not in tensors:
            continue
        if name not in params:
            unmatched.append(name)
            continue
        weight = params[name]
        mat_a = tensors["A"].to(device=weight.device, dtype=torch.float32)
        mat_b = tensors["B"].to(device=weight.device, dtype=torch.float32)
        alpha = 1.0
        if "alpha" in tensors:
            alpha = float(tensors["alpha"].item()) / mat_a.shape[0]
        delta = torch.mm(mat_b.flatten(1), mat_a.flatten(1)).reshape(weight.shape)
        with torch.no_grad():
            weight.add_((strength * alpha * delta).to(weight.dtype))
        patched += 1

    if unmatched:
        logger.warning("LoRA: %d keys did not match the model (e.g. %s)", len(unmatched), unmatched[:3])
    logger.info("LoRA: merged %d weights at strength %.3f", patched, strength)
    return patched
