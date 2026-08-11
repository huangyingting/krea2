import torch

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
