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

from typing import Callable, Optional
from diffusers.models.normalization import RMSNorm
import torch
from torch import nn


class FusedRMSNormCompat(nn.Module):
    """``apex.normalization.FusedRMSNorm`` in plain PyTorch.

    Apex accumulates the mean square in fp32 and applies the (optional) weight
    in the input dtype - reproduced here so bf16/fp16 inference matches.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) \
            else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        variance = x.float().pow(2).mean(dims, keepdim=True)
        out = (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype)
        if self.weight is not None:
            out = out * self.weight
        return out

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, " \
               f"elementwise_affine={self.elementwise_affine}"

# (dim: int, eps: float, elementwise_affine: bool)
norm_layer_type = Callable[[int, float, bool], nn.Module]


def get_norm_layer(norm_type: Optional[str]) -> norm_layer_type:

    def _norm_layer(dim: int, eps: float, elementwise_affine: bool):
        if norm_type is None:
            return nn.Identity()

        if norm_type == "layer":
            return nn.LayerNorm(
                normalized_shape=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if norm_type == "rms":
            return RMSNorm(
                dim=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        # ``fusedln`` / ``fusedrms`` are apex kernels upstream.  Apex is not a
        # dependency here, so the mathematically identical torch layers are used
        # (apex fuses the same formula; only the kernel differs).
        if norm_type == "fusedln":
            return nn.LayerNorm(
                normalized_shape=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if norm_type == "fusedrms":
            return FusedRMSNormCompat(
                normalized_shape=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        raise NotImplementedError(f"{norm_type} is not supported")

    return _norm_layer
