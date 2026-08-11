"""Krea 2 text conditioning: Qwen3-VL-4B with a 12-layer hidden-state tap.

Reproduces ComfyUI's ``CLIPLoader(type="krea2")`` + ``CLIPTextEncode`` path:

* prompt is wrapped in the Krea2 chat template (system + user, no ``<think>`` block),
* hidden states ``[2, 5, 8, ..., 35]`` are stacked (12 layers),
* the ``<|im_start|>system ... <|im_start|>user\\n`` prefix is stripped,
* the stack is flattened to ``(B, seq, 12*2560)`` which is what the DiT consumes.

The single-file ComfyUI checkpoint ``qwen3vl_4b_bf16.safetensors`` already uses HuggingFace
key names (``model.language_model.*`` / ``model.visual.*``), so it is loaded straight into
``transformers``' ``Qwen3VLModel``.
"""

from __future__ import annotations

import os

import torch
from safetensors.torch import load_file
from transformers import Qwen2Tokenizer

TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qwen_tokenizer")

# comfy/text_encoders/krea2.py :: KREA2_TAP_LAYERS
KREA2_TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

# comfy/text_encoders/krea2.py :: KREA2_TEMPLATE
KREA2_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)

IM_START = 151644
TOKEN_USER = 872
TOKEN_NEWLINE = 198

# Qwen/Qwen3-VL-4B-Instruct config.json
QWEN3VL_4B_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "use_cache": False,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}


class Krea2TextEncoder(torch.nn.Module):
    def __init__(self, checkpoint: str, device="cuda", dtype=torch.bfloat16,
                 tap_layers=KREA2_TAP_LAYERS, keep_loaded: bool = True):
        super().__init__()
        from transformers import Qwen3VLConfig, Qwen3VLTextModel

        self.tap_layers = tuple(tap_layers)
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = Qwen2Tokenizer.from_pretrained(TOKENIZER_DIR)

        config = Qwen3VLConfig(**QWEN3VL_4B_CONFIG)
        with torch.device("meta"):
            model = Qwen3VLTextModel(config.text_config)
        model = model.to(dtype)

        sd = load_file(checkpoint)
        text_sd = {
            k[len("model.language_model."):]: v
            for k, v in sd.items()
            if k.startswith("model.language_model.")
        }
        missing, unexpected = model.load_state_dict(text_sd, strict=False, assign=True)
        missing = [m for m in missing if "rotary_emb.inv_freq" not in m]
        if missing or unexpected:
            raise RuntimeError(f"text encoder weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        del sd, text_sd

        # Non-persistent buffers (rope inv_freq) stay on the meta device after `assign=True`.
        model.rotary_emb = type(model.rotary_emb)(config.text_config)
        meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
        if meta:
            raise RuntimeError(f"text encoder tensors left on meta device: {meta[:5]}")

        self.model = model.eval().requires_grad_(False)
        self.keep_loaded = keep_loaded
        self._on_device = False

    def to_device(self):
        if not self._on_device:
            self.model.to(self.device)
            self._on_device = True

    def offload(self):
        if self._on_device and not self.keep_loaded:
            self.model.to("cpu")
            torch.cuda.empty_cache()
            self._on_device = False

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer(KREA2_TEMPLATE.format(prompt), add_special_tokens=False)["input_ids"]

    @staticmethod
    def _template_end(tokens: list[int]) -> int:
        """comfy/text_encoders/krea2.py :: Krea2TEModel.encode_token_weights prefix strip."""
        template_end = -1
        count_im_start = 0
        for i, tok in enumerate(tokens):
            if tok == IM_START and count_im_start < 2:
                template_end = i
                count_im_start += 1
        if len(tokens) > (template_end + 3):
            if tokens[template_end + 1] == TOKEN_USER and tokens[template_end + 2] == TOKEN_NEWLINE:
                template_end += 3
        return template_end

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        """Returns conditioning of shape (1, seq, len(tap_layers) * hidden_size)."""
        self.to_device()
        tokens = self.tokenize(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        stacked = torch.stack([out.hidden_states[i] for i in self.tap_layers], dim=2)  # (B, seq, n, dim)
        del out

        template_end = self._template_end(tokens)
        stacked = stacked[:, template_end:]
        b, seq, n, h = stacked.shape
        cond = stacked.reshape(b, seq, n * h)
        self.offload()
        return cond


def conditioning_zero_out(cond: torch.Tensor) -> torch.Tensor:
    """ComfyUI ``ConditioningZeroOut``: zeroes the conditioning tensor (keeps the shape)."""
    return torch.zeros_like(cond)
