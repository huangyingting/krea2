import pytest
import torch

from krea2pipe import lora, pipeline as pipeline_module
from krea2pipe.pipeline import Krea2Models, Krea2Pipeline


class RecordingVAE:
    def __init__(self):
        self.encoded_shape = None
        self.decoded_shape = None

    def encode(self, image):
        self.encoded_shape = tuple(image.shape)
        batch, _, frames, height, width = image.shape
        return torch.zeros(batch, 16, frames, height // 8, width // 8)

    def decode(self, latent):
        self.decoded_shape = tuple(latent.shape)
        batch, _, frames, height, width = latent.shape
        return torch.zeros(batch, 3, frames, height * 8, width * 8)


class TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1, bias=False)
        self.layer.weight.data.zero_()
        self.blocks = torch.nn.ModuleList()


def lora_update(value):
    return {
        "diffusion_model.layer.lora_A.weight": torch.ones(1, 1),
        "diffusion_model.layer.lora_B.weight": torch.tensor([[value]]),
    }


def test_qwen_vae_preserves_independent_image_batch():
    pipeline = Krea2Pipeline(Krea2Models(device="cpu", dtype=torch.float32))
    vae = RecordingVAE()
    pipeline._vae = vae

    latent = pipeline.vae_encode(torch.rand(4, 16, 24, 3))
    decoded = pipeline.vae_decode(latent)

    assert vae.encoded_shape == (4, 3, 1, 16, 24)
    assert latent.shape == (4, 16, 2, 3)
    assert vae.decoded_shape == (4, 16, 1, 2, 3)
    assert decoded.shape == (4, 16, 24, 3)


def test_dit_publishes_only_after_all_loras_and_retries_cleanly(monkeypatch):
    models = []
    compiled = []
    second_loads = 0

    def load_dit(*_args):
        model = TinyDiT()
        models.append(model)
        return model

    def load_lora_file(path):
        nonlocal second_loads
        if path == "second.safetensors":
            second_loads += 1
            if second_loads == 1:
                raise RuntimeError("corrupt LoRA")
            return lora_update(2.0)
        return lora_update(1.0)

    monkeypatch.setattr(pipeline_module.loaders, "load_dit", load_dit)
    monkeypatch.setattr(
        pipeline_module.loaders,
        "require_model",
        lambda _kind, name, _root, _description: name,
    )
    monkeypatch.setattr(lora, "load_lora_file", load_lora_file)
    monkeypatch.setattr(
        pipeline_module.accel,
        "compile_repeated_blocks",
        lambda _blocks: compiled.append(True),
    )
    pipe = Krea2Pipeline(
        Krea2Models(
            loras=[
                ("first.safetensors", 1.0),
                ("second.safetensors", 1.0),
            ],
            device="cpu",
            dtype=torch.float32,
        )
    )

    with pytest.raises(RuntimeError, match="corrupt LoRA"):
        _ = pipe.dit

    assert pipe._dit is None
    assert models[0].layer.weight.item() == 1.0
    assert compiled == []

    dit = pipe.dit

    assert dit is models[1]
    assert pipe._dit is dit
    assert dit.layer.weight.item() == 3.0
    assert len(models) == 2
    assert compiled == [True]


def test_dit_rejects_zero_weight_lora_without_caching(monkeypatch):
    models = []

    def load_dit(*_args):
        model = TinyDiT()
        models.append(model)
        return model

    monkeypatch.setattr(pipeline_module.loaders, "load_dit", load_dit)
    monkeypatch.setattr(
        pipeline_module.loaders,
        "require_model",
        lambda _kind, name, _root, _description: name,
    )
    monkeypatch.setattr(
        lora,
        "load_lora_file",
        lambda _path: {
            "diffusion_model.unknown.lora_A.weight": torch.ones(1, 1),
            "diffusion_model.unknown.lora_B.weight": torch.ones(1, 1),
        },
    )
    pipe = Krea2Pipeline(
        Krea2Models(
            loras=[("incompatible.safetensors", 1.0)],
            device="cpu",
            dtype=torch.float32,
        )
    )

    for _ in range(2):
        with pytest.raises(
            ValueError,
            match=(
                "Enabled LoRA 'incompatible.safetensors' patched zero model "
                "weights"
            ),
        ):
            _ = pipe.dit
        assert pipe._dit is None

    assert len(models) == 2
