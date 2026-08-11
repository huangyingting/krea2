"""Krea 2 (K2) single-stream MMDiT.

The architecture follows the official implementation
(https://github.com/krea-ai/krea-2, ``mmdit.py``): no attention mask, no
sequence padding, 3-axis RoPE, and a 12-layer Qwen3-VL text-fusion adapter.

Checkpoint layout (``model.diffusion_model.*`` keys of e.g.
``moodyKrea2Mix_v50BF16.safetensors``) is loaded verbatim - the module names here match
the checkpoint names exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


@dataclass
class Krea2Config:
    features: int = 6144
    tdim: int = 256
    txtdim: int = 2560
    heads: int = 48
    kvheads: int = 12
    multiplier: int = 4
    layers: int = 28
    patch: int = 2
    channels: int = 16
    bias: bool = False
    theta: float = 1e3
    txtlayers: int = 12
    txtheads: int = 20
    txtkvheads: int = 20

    @classmethod
    def from_state_dict(cls, sd: dict[str, Tensor]) -> "Krea2Config":
        """Infer architecture dimensions from checkpoint tensors."""
        head_dim = 128
        first_w = sd["first.weight"]
        patch = 2
        layers = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
        return cls(
            features=first_w.shape[0],
            channels=first_w.shape[1] // (patch * patch),
            patch=patch,
            layers=layers,
            heads=sd["blocks.0.attn.wq.weight"].shape[0] // head_dim,
            kvheads=sd["blocks.0.attn.wk.weight"].shape[0] // head_dim,
            txtlayers=sd["txtfusion.projector.weight"].shape[1],
            txtdim=sd["txtfusion.layerwise_blocks.0.prenorm.scale"].shape[0],
        )


def rope(pos: Tensor, dim: int, theta: float) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.float(), omega.float())
    out = torch.stack([out.cos(), -out.sin(), out.sin(), out.cos()], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def apply_rope(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_out = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    def __init__(self, theta: float, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(ids.shape[-1])],
            dim=-3,
        )
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim: int, max_period: float = 1e4, time_factor: float = 1e3) -> Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb.to(t.dtype) if torch.is_floating_point(t) else emb


class RMSNorm(nn.Module):
    """RMSNorm with the reference ``(1 + scale)`` weight convention."""

    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        weight = self.scale.to(device=x.device, dtype=torch.float32) + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=self.eps).to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k)


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)).mul_(self.up(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int | None = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: Tensor, freqs: Tensor | None = None) -> Tensor:
        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "B H L D -> B L (H D)")
        return self.wo(out * F.sigmoid(gate))


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))

    def forward(self, vec: Tensor) -> tuple[Tensor, Tensor]:
        out = vec + self.lin.to(dtype=vec.dtype, device=vec.device).unsqueeze(0)
        scale, shift = out.chunk(2, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: Tensor):
        return (vec + self.lin.to(dtype=vec.dtype, device=vec.device)).chunk(6, dim=-1)


class TextFusionBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.prenorm(x))
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):
    """Fuses the 12 tapped Qwen3-VL hidden-state layers into a single token stream."""

    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )

    def forward(self, x: Tensor) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous())
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, vec: Tensor, freqs: Tensor) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs)
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        return x


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)


class SingleStreamDiT(nn.Module):
    def __init__(self, config: Krea2Config):
        super().__init__()
        self.config = config
        c = config
        headdim = c.features // c.heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim, f"axes {axes} sum != headdim {headdim}"
        self.pe_embedder = EmbedND(theta=int(c.theta), axes_dim=axes)

        self.first = nn.Linear(c.channels * c.patch**2, c.features, bias=True)
        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(c.features, c.heads, c.multiplier, c.bias, c.kvheads)
                for _ in range(c.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(c.tdim, c.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(c.features, c.features),
        )
        self.txtfusion = TextFusionTransformer(
            c.txtlayers, c.txtdim, c.txtheads, c.multiplier, c.bias, c.txtkvheads
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(c.txtdim),
            nn.Linear(c.txtdim, c.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(c.features, c.features),
        )
        self.last = LastLayer(c.features, c.patch, c.channels)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"), nn.Linear(c.features, c.features * 6)
        )

    def process_img(self, x: Tensor) -> tuple[Tensor, Tensor, int, int]:
        patch = self.config.patch
        pad_h = (patch - x.shape[-2] % patch) % patch
        pad_w = (patch - x.shape[-1] % patch) % patch
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        h, w = x.shape[-2] // patch, x.shape[-1] // patch
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        ids = torch.zeros(h, w, 3, device=x.device, dtype=torch.float32)
        ids[..., 1] = torch.arange(h, device=x.device, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(w, device=x.device, dtype=torch.float32)[None, :]
        return img, ids.reshape(1, h * w, 3).repeat(x.shape[0], 1, 1), h, w

    def forward(self, x: Tensor, timesteps: Tensor, context: Tensor) -> Tensor:
        """x: (B, C, H, W) latent; context: (B, seq, txtlayers*txtdim); timesteps: (B,)."""
        bs, _, h_orig, w_orig = x.shape
        patch = self.config.patch

        context_batch, seq, fused = context.shape
        if context_batch not in (1, bs):
            raise ValueError(
                f"conditioning batch {context_batch} cannot drive latent batch {bs}"
            )
        expected = self.config.txtlayers * self.config.txtdim
        if fused != expected:
            raise ValueError(
                f"Krea2 expects conditioning with {self.config.txtlayers}x{self.config.txtdim}"
                f"={expected} features, got {fused}."
            )
        context = context.reshape(
            context_batch, seq, self.config.txtlayers, self.config.txtdim
        )

        img, imgpos, h_, w_ = self.process_img(x)
        img_tokens = img.shape[1]
        img = self.first(img)

        t = self.tmlp(timestep_embedding(timesteps, self.config.tdim).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)

        context = self.txtfusion(context)
        context = self.txtmlp(context)
        if context_batch == 1 and bs != 1:
            context = context.expand(bs, -1, -1)

        txtlen = context.shape[1]
        txtpos = torch.zeros(bs, txtlen, 3, device=context.device, dtype=torch.float32)

        combined = torch.cat((context, img), dim=1)
        del context, img
        pos = torch.cat((txtpos, imgpos), dim=1)
        freqs = self.pe_embedder(pos)
        del pos, txtpos, imgpos

        for block in self.blocks:
            combined = block(combined, tvec, freqs)

        final = self.last(combined, t)
        del combined
        out = final[:, txtlen : txtlen + img_tokens, :]
        out = rearrange(
            out,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_,
            w=w_,
            ph=patch,
            pw=patch,
            c=self.config.channels,
        )
        return out[:, :, :h_orig, :w_orig]
