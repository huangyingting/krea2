"""Prompt expansion using the resident Qwen text model."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from krea2pipe.models.text_encoder import Krea2TextEncoder


class _Tokenizer:
    eos_token_id = 2

    def __init__(self):
        self.input = ""

    def __call__(self, text, **_kwargs):
        self.input = text
        return {"input_ids": torch.tensor([[0]])}

    @staticmethod
    def decode(tokens, **_kwargs):
        return "expanded image prompt" if tokens == [1] else ""


class _CausalTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(3, 2)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(torch.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]))
        self.calls = 0

    def forward(self, **_kwargs):
        hidden = (
            torch.tensor([[[100.0, 0.0]]])
            if self.calls == 0
            else torch.tensor([[[0.0, 100.0]]])
        )
        self.calls += 1
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=object())


def test_qwen_expands_theme_with_official_system_prompt():
    encoder = Krea2TextEncoder.__new__(Krea2TextEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.device = torch.device("cpu")
    encoder.model = _CausalTextModel()
    encoder.tokenizer = _Tokenizer()
    encoder._on_device = False

    prompt = encoder.generate_prompt("a quiet forest", seed=123, max_new_tokens=8)

    assert prompt == "expanded image prompt"
    assert "expert prompt engineer for text-to-image models" in encoder.tokenizer.input
    assert "<|im_start|>user\na quiet forest<|im_end|>" in encoder.tokenizer.input


def test_qwen_rejects_empty_theme():
    encoder = Krea2TextEncoder.__new__(Krea2TextEncoder)
    torch.nn.Module.__init__(encoder)
    with torch.no_grad():
        try:
            encoder.generate_prompt("  ", seed=1)
        except ValueError as exc:
            assert str(exc) == "theme must not be empty"
        else:
            raise AssertionError("empty theme was accepted")
