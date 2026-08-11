# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
"""Single-process replacements for SeedVR's ``common.distributed`` package.

The official repository shards long sequences across GPUs with a sequence
parallel process group.  ``krea2pipe`` always runs on a single device, and every
one of those collectives is documented to be a no-op when no process group
exists (see ``common/distributed/ops.py``), so they are simply identities here.
"""

from typing import Any, List, Optional

import torch
from torch import Tensor, nn

__all__ = [
    "get_device",
    "get_global_rank",
    "get_world_size",
    "get_sequence_parallel_group",
    "get_sequence_parallel_rank",
    "get_sequence_parallel_world_size",
    "get_next_sequence_parallel_rank",
    "get_prev_sequence_parallel_rank",
    "Gather",
    "get_data_parallel_rank",
    "get_data_parallel_world_size",
    "slice_inputs",
    "gather_outputs",
    "gather_heads_scatter_seq",
    "gather_seq_scatter_heads",
    "gather_seq_scatter_heads_qkv",
    "scatter_heads",
    "gather_heads",
    "remove_seqeunce_parallel_padding",
    "sync_data",
    "meta_non_persistent_buffer_init_fn",
    "partition_by_groups",
    "partition_by_size",
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_global_rank() -> int:
    return 0


def get_world_size() -> int:
    return 1


def get_sequence_parallel_group() -> None:
    return None


def get_sequence_parallel_rank() -> int:
    return 0


def get_sequence_parallel_world_size() -> int:
    return 1


def get_next_sequence_parallel_rank() -> int:
    return 0


def get_prev_sequence_parallel_rank() -> int:
    return 0


class Gather(torch.autograd.Function):
    """Single-process stand-in for the sequence-parallel all-gather."""

    @staticmethod
    def forward(ctx, group, local_input: Tensor, dim: int = 0, grad_scale: bool = True) -> Tensor:
        return local_input

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return None, grad_output, None, None


get_data_parallel_rank = get_global_rank
get_data_parallel_world_size = get_world_size


def slice_inputs(x: Tensor, dim: int, padding: bool = True) -> Tensor:
    return x


def gather_outputs(x: Tensor, **kwargs: Any) -> Tensor:
    return x


def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int) -> Tensor:
    return x


def gather_seq_scatter_heads(x: Tensor, seq_dim: int, head_dim: int) -> Tensor:
    return x


def gather_seq_scatter_heads_qkv(qkv_tensor: Tensor, **kwargs: Any) -> Tensor:
    return qkv_tensor


def scatter_heads(x: Tensor, dim: int) -> Tensor:
    return x


def gather_heads(x: Tensor, dim: int, grad_scale: Optional[bool] = False) -> Tensor:
    return x


def remove_seqeunce_parallel_padding(x: Tensor, dim: int, unpad_dim_size: int) -> Tensor:
    return x


def sync_data(data: Any, sp_idx: int = 0, name: str = "tmp") -> Any:
    return data


def meta_non_persistent_buffer_init_fn(module: nn.Module) -> nn.Module:
    """Materialise non-persistent buffers left on the meta device.

    ``RotaryEmbedding`` keeps ``dummy`` (device probe) and, since
    rotary-embedding-torch 0.6, ``cached_freqs`` / ``cached_scales`` out of the
    state dict.  All of them are zero-initialised upstream, so materialising
    them as zeros reproduces a freshly constructed module.
    """
    from rotary_embedding_torch import RotaryEmbedding

    with torch.no_grad():
        for submodule in module.modules():
            if not isinstance(submodule, RotaryEmbedding):
                continue
            for buffer_name, buffer in submodule.named_buffers(recurse=False):
                if buffer.is_meta:
                    setattr(submodule, buffer_name, torch.zeros_like(buffer, device="cpu"))
    assert not any(b.is_meta for _, b in module.named_buffers())
    return module


def partition_by_groups(data: List[Any], groups: int) -> List[List[Any]]:
    assert groups > 0
    return [data[i::groups] for i in range(groups)]


def partition_by_size(data: List[Any], size: int) -> List[List[Any]]:
    assert size > 0
    return [data[i:i + size] for i in range(0, len(data), size)]
