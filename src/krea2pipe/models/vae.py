"""Qwen-Image VAE using the Wan2.1 architecture for still-image inference.

Still-image latents have one temporal frame, so temporal feature caching and
temporal up/downsampling are unnecessary. Checkpoint key names remain unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ..conv3d_compat import conv3d_forward


class CausalConv3d(nn.Conv3d):
    """Conv3d with causal zero padding along the temporal axis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = 2 * self.padding[0]
        self.padding = (0, self.padding[1], self.padding[2])

    def _conv_forward(self, input, weight, bias):
        out = conv3d_forward(self, input, weight, bias)
        return super()._conv_forward(input, weight, bias) if out is None else out

    def forward(self, x: Tensor) -> Tensor:
        if self._padding > 0:
            x = F.pad(x, (0, 0, 0, 0, self._padding, 0))
        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma.to(x)
        if self.bias is not None:
            out = out + self.bias.to(x)
        return out


class Resample(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        self.dim = dim
        self.mode = mode
        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # Temporal convolutions are only active for cached multi-frame inference.
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        return rearrange(x, "(b t) c h w -> b c t h w", t=t)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        b, c, h, w = q.shape
        q, k, v = (t_.view(b, 1, c, h * w).transpose(2, 3).contiguous() for t_ in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(2, 3).reshape(b, c, h, w)
        out = self.proj(out)
        out = rearrange(out, "(b t) c h w -> b c t h w", t=t)
        return out + identity


class Encoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, input_channels=3, dim_mult=(1, 2, 4, 4), num_res_blocks=2,
                 attn_scales=(), temperal_downsample=(True, True, False), dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [1] + list(dim_mult)]
        scale = 1.0
        self.conv1 = CausalConv3d(input_channels, dims[0], 3, padding=1)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.downsamples(x)
        x = self.middle(x)
        return self.head(x)


class Decoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, output_channels=3, dim_mult=(1, 2, 4, 4), num_res_blocks=2,
                 attn_scales=(), temperal_upsample=(False, True, True), dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + list(dim_mult)[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, output_channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.middle(x)
        x = self.upsamples(x)
        return self.head(x)


class WanVAE(nn.Module):
    def __init__(self, dim=96, z_dim=16, dim_mult=(1, 2, 4, 4), num_res_blocks=2, attn_scales=(),
                 temperal_downsample=(False, True, True), image_channels=3, conv_out_channels=3,
                 dropout=0.0):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder3d(dim, z_dim * 2, image_channels, dim_mult, num_res_blocks,
                                 attn_scales, temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, conv_out_channels, dim_mult, num_res_blocks,
                                 attn_scales, tuple(temperal_downsample)[::-1], dropout)

    def encode(self, x: Tensor) -> Tensor:
        """x: (B, 3, 1, H, W) in [-1, 1] -> mu (B, 16, 1, H/8, W/8)."""
        assert x.shape[2] == 1, "image-only VAE supports a single frame"
        out = self.encoder(x)
        mu, _log_var = self.conv1(out).chunk(2, dim=1)
        return mu

    def decode(self, z: Tensor) -> Tensor:
        """z: (B, 16, 1, h, w) -> (B, 3, 1, h*8, w*8) in [-1, 1]."""
        assert z.shape[2] == 1, "image-only VAE supports a single frame"
        return self.decoder(self.conv2(z))


# --- Wan2.1 latent format --------------------------------------------------------------

LATENTS_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
LATENTS_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]


def _stats(latent: Tensor) -> tuple[Tensor, Tensor]:
    shape = (1, len(LATENTS_MEAN)) + (1,) * (latent.ndim - 2)
    mean = torch.tensor(LATENTS_MEAN, device=latent.device, dtype=latent.dtype).view(shape)
    std = torch.tensor(LATENTS_STD, device=latent.device, dtype=latent.dtype).view(shape)
    return mean, std


def process_latent_in(latent: Tensor, scale_factor: float = 1.0) -> Tensor:
    mean, std = _stats(latent)
    return (latent - mean) * scale_factor / std


def process_latent_out(latent: Tensor, scale_factor: float = 1.0) -> Tensor:
    mean, std = _stats(latent)
    return latent * std / scale_factor + mean
