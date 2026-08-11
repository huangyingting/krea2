"""SeedVR2 diffusion upscaler.

Derived from the official ByteDance-Seed/SeedVR repository (Apache-2.0; see
``LICENSE``). Runtime adaptations:

* ``common.distributed`` is replaced by :mod:`krea2pipe.seedvr2.parallel`
  (single process - every sequence-parallel collective is an identity);
* ``flash_attn_varlen_func`` is replaced by a ``scaled_dot_product_attention``
  based implementation (:mod:`krea2pipe.seedvr2.dit.attention`);
* apex ``FusedLayerNorm`` / ``FusedRMSNorm`` are replaced by the equivalent
  torch layers;
* checkpoints may be ``.safetensors`` (including the fp8 releases);
* :mod:`krea2pipe.seedvr2.color_fix` provides selectable color correction;
* ``max_resolution`` caps the long edge for predictable memory use.
"""

from .runner import (SeedVR2Config, SeedVR2Upscaler, release_upscaler,
                     seedvr2_upscale)

__all__ = ["SeedVR2Config", "SeedVR2Upscaler", "release_upscaler", "seedvr2_upscale"]
