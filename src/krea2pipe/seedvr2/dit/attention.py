# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.
"""Attention layers.

Upstream calls ``flash_attn_varlen_func``; this port uses PyTorch's
``scaled_dot_product_attention`` instead so no compiled extension is required.
Sequences that share a length are batched together (identical lengths are the
norm here, since SeedVR attends over fixed-size windows), which keeps a single
fused kernel launch per group instead of one call per window.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _sdpa(q: Tensor, k: Tensor, v: Tensor, softmax_scale=None, causal=False) -> Tensor:
    """(b, s, h, d) -> (b, s, h, d)."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, scale=softmax_scale,
    )
    return out.transpose(1, 2)


def sdpa_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None,
                     max_seqlen_k=None, dropout_p=0.0, softmax_scale=None,
                     causal=False, **kwargs) -> Tensor:
    """Drop-in replacement for ``flash_attn_varlen_func``.

    ``q``/``k``/``v`` are ``(total_tokens, heads, head_dim)`` and ``cu_seqlens_*``
    are the exclusive prefix sums of the per-sequence lengths.
    """
    lens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int64)
    lens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int64)
    n = lens_q.numel()

    if n == 1:
        return _sdpa(q[None], k[None], v[None], softmax_scale, causal)[0]

    # Fast path: every sequence has the same length -> a plain reshape (no copy).
    same_q = bool((lens_q == lens_q[0]).all())
    same_k = bool((lens_k == lens_k[0]).all())
    if same_q and same_k and bool(lens_q[0] == lens_k[0]):
        length = int(lens_q[0])
        h, d = q.shape[1], q.shape[2]
        out = _sdpa(q.view(n, length, h, d), k.view(n, length, h, d),
                    v.view(n, length, h, d), softmax_scale, causal)
        return out.reshape(-1, h, d)

    # General path: group the sequences by (len_q, len_k) and batch each group.
    lens_q_cpu = lens_q.cpu()
    lens_k_cpu = lens_k.cpu()
    q_splits = torch.split(q, lens_q_cpu.tolist(), dim=0)
    k_splits = torch.split(k, lens_k_cpu.tolist(), dim=0)
    v_splits = torch.split(v, lens_k_cpu.tolist(), dim=0)

    groups: dict[tuple[int, int], list[int]] = {}
    for i, (lq, lk) in enumerate(zip(lens_q_cpu.tolist(), lens_k_cpu.tolist())):
        groups.setdefault((lq, lk), []).append(i)

    outputs: list = [None] * n
    for idx in groups.values():
        qb = torch.stack([q_splits[i] for i in idx])
        kb = torch.stack([k_splits[i] for i in idx])
        vb = torch.stack([v_splits[i] for i in idx])
        ob = _sdpa(qb, kb, vb, softmax_scale, causal)
        for j, i in enumerate(idx):
            outputs[i] = ob[j]
    return torch.cat(outputs, dim=0)


class TorchAttention(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        assert len(args) == 0 or len(args) > 2, "query, key should both provided by args / kwargs"
        q = kwargs.get("query") or args[0]
        k = kwargs.get("key") or args[1]
        b, h, sq, d = q.shape
        b, h, sk, d = k.shape
        return b * h * (4 * d * (sq / 1e6) * (sk / 1e6))

    def forward(self, *args, **kwargs):
        return F.scaled_dot_product_attention(*args, **kwargs)


class FlashAttentionVarlen(nn.Module):
    """Variable-length attention (SDPA backed, flash-attn compatible signature)."""

    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        _, h, d = output.shape
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
        return h * (4 * d * (seqlens_q * seqlens_k).sum())

    def forward(self, *args, **kwargs):
        kwargs.pop("deterministic", None)
        return sdpa_varlen_func(*args, **kwargs)
