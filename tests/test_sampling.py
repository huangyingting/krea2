"""Tests for the sampling / scheduler / LoRA maths (``krea2pipe.sampling``, ``lora``)."""

from __future__ import annotations

import math

import pytest
import torch

from krea2pipe import lora, sampling


# --- ModelSamplingFlux --------------------------------------------------------------

def test_model_sampling_flux_range():
    ms = sampling.ModelSamplingFlux(shift=1.15)
    assert float(ms.sigma_min) == pytest.approx(0.0, abs=1e-3)
    assert float(ms.sigma_max) == pytest.approx(1.0, abs=1e-6)
    assert ms.sigmas.shape == (10000,)
    assert torch.all(ms.sigmas[1:] >= ms.sigmas[:-1])  # monotonically increasing


def test_flux_time_shift_matches_reference_formula():
    mu, sigma, t = 1.15, 1.0, 0.5
    expected = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    assert sampling.flux_time_shift(mu, sigma, t) == pytest.approx(expected)


def test_const_noise_scaling_round_trip():
    ms = sampling.ModelSamplingFlux()
    latent = torch.randn(1, 16, 8, 8)
    noise = torch.randn_like(latent)
    sigma = torch.tensor([0.3])
    noised = ms.noise_scaling(sigma, noise, latent)
    assert torch.allclose(noised, 0.3 * noise + 0.7 * latent, atol=1e-6)


def test_calculate_denoised_is_a_flow_step():
    ms = sampling.ModelSamplingFlux()
    x = torch.randn(1, 16, 4, 4)
    v = torch.randn_like(x)
    out = ms.calculate_denoised(torch.tensor([0.25]), v, x)
    assert torch.allclose(out, x - 0.25 * v, atol=1e-6)


# --- schedulers ---------------------------------------------------------------------

@pytest.mark.parametrize("scheduler", sorted(sampling.SCHEDULERS))
def test_scheduler_shapes_and_endpoints(scheduler):
    ms = sampling.ModelSamplingFlux()
    sigmas = sampling.calculate_sigmas(ms, scheduler, 8)
    assert sigmas.shape == (9,)                       # steps + 1
    assert float(sigmas[0]) == pytest.approx(1.0, abs=1e-3)
    assert float(sigmas[-1]) == 0.0                   # sgm/simple both end at zero
    assert torch.all(sigmas[1:] <= sigmas[:-1])       # decreasing


def test_sgm_uniform_matches_comfy_values():
    """8-step sgm_uniform sigmas of ModelSamplingFlux(shift=1.15)."""
    sigmas = sampling.calculate_sigmas(sampling.ModelSamplingFlux(), "sgm_uniform", 8)
    # golden values captured from ComfyUI 0.30.0:
    #   comfy.samplers.calculate_sigmas(ModelSamplingFlux(shift=1.15), "sgm_uniform", 8)
    expected = torch.tensor([1.0000, 0.9567, 0.9046, 0.8404, 0.7596, 0.6548, 0.5132, 0.3114, 0.0])
    assert torch.allclose(sigmas, expected, atol=1e-3)


def test_denoise_shortens_the_schedule():
    ms = sampling.ModelSamplingFlux()
    full = sampling.calculate_sigmas(ms, "simple", 2, denoise=1.0)
    partial = sampling.calculate_sigmas(ms, "simple", 2, denoise=0.1)
    assert partial.shape == (3,)
    assert float(partial[0]) < float(full[0])         # starts much closer to the image
    # denoise=0.1 with 2 steps == the last 3 sigmas of a 20 step schedule (ComfyUI reference)
    reference = sampling.calculate_sigmas(ms, "simple", 20)[-3:]
    assert torch.allclose(partial, reference, atol=1e-4)
    assert float(partial[0]) == pytest.approx(0.2598, abs=1e-3)


def test_slice_sigmas_start_and_end():
    sigmas = sampling.calculate_sigmas(sampling.ModelSamplingFlux(), "sgm_uniform", 8)
    sliced = sampling.slice_sigmas(sigmas, 0, 9999, force_full_denoise=True)
    assert torch.equal(sliced, sigmas)
    sliced = sampling.slice_sigmas(sigmas, 2, 4, force_full_denoise=False)
    assert len(sliced) == 3
    assert torch.equal(sliced, sigmas[2:5])


# --- noise ---------------------------------------------------------------------------

def test_prepare_noise_is_seed_deterministic():
    latent = torch.zeros(1, 16, 8, 8)
    a = sampling.prepare_noise(latent, 1099257494857840)
    b = sampling.prepare_noise(latent, 1099257494857840)
    c = sampling.prepare_noise(latent, 42)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.shape == latent.shape


def test_prepare_noise_is_cpu_generated_and_batch_wise():
    """ComfyUI generates one noise tensor per batch item, on the CPU."""
    single = sampling.prepare_noise(torch.zeros(1, 16, 4, 4), 7)
    batch = sampling.prepare_noise(torch.zeros(3, 16, 4, 4), 7)
    assert batch.device.type == "cpu"
    assert torch.equal(batch[:1], single)


# --- samplers -------------------------------------------------------------------------

def test_euler_matches_an_analytic_denoiser():
    """With a constant-velocity model, euler integrates exactly."""
    velocity = torch.randn(1, 4, 2, 2)

    def model(x, sigma):
        return x - sigma.view(-1, 1, 1, 1) * velocity  # denoised = x - sigma * v

    sigmas = torch.tensor([1.0, 0.5, 0.0])
    x = torch.randn(1, 4, 2, 2)
    out = sampling.sample_euler(model, x.clone(), sigmas)
    assert torch.allclose(out, x - velocity, atol=1e-5)


def test_euler_ancestral_is_seed_deterministic():
    def model(x, sigma):
        return x * 0.5

    sigmas = torch.tensor([1.0, 0.6, 0.3, 0.0])
    x = torch.randn(1, 4, 4, 4)
    a = sampling.sample_euler_ancestral_rf(model, x.clone(), sigmas, seed=3)
    b = sampling.sample_euler_ancestral_rf(model, x.clone(), sigmas, seed=3)
    c = sampling.sample_euler_ancestral_rf(model, x.clone(), sigmas, seed=4)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_samplers_registry():
    assert {"euler", "euler_ancestral"} <= set(sampling.SAMPLERS)


# --- LoRA ------------------------------------------------------------------------------

def test_apply_lora_merges_b_times_a():
    linear = torch.nn.Linear(4, 6, bias=False)
    before = linear.weight.detach().clone()
    down = torch.randn(2, 4)   # lora_A / lora_down: (rank, in)
    up = torch.randn(6, 2)     # lora_B / lora_up:   (out, rank)
    patched = lora.apply_lora(
        torch.nn.Sequential(),  # placeholder replaced below
        {}, 0.0,
    )
    assert patched == 0        # strength 0 is a no-op

    model = torch.nn.Module()
    model.layer = linear
    patched = lora.apply_lora(
        model,
        {"diffusion_model.layer.lora_down.weight": down,
         "diffusion_model.layer.lora_up.weight": up},
        strength=0.6,
    )
    assert patched == 1
    assert torch.allclose(linear.weight, before + 0.6 * (up @ down), atol=1e-6)


def test_apply_lora_honours_alpha():
    linear = torch.nn.Linear(4, 4, bias=False)
    before = linear.weight.detach().clone()
    model = torch.nn.Module()
    model.layer = linear
    rank = 2
    down, up = torch.randn(rank, 4), torch.randn(4, rank)
    lora.apply_lora(model, {
        "diffusion_model.layer.lora_A.weight": down,
        "diffusion_model.layer.lora_B.weight": up,
        "diffusion_model.layer.alpha": torch.tensor(float(rank)),
    }, strength=1.0)
    assert torch.allclose(linear.weight, before + (up @ down), atol=1e-6)


def test_apply_lora_ignores_unknown_keys():
    model = torch.nn.Module()
    model.layer = torch.nn.Linear(4, 4, bias=False)
    patched = lora.apply_lora(model, {
        "diffusion_model.nope.lora_A.weight": torch.randn(2, 4),
        "diffusion_model.nope.lora_B.weight": torch.randn(4, 2),
    }, strength=1.0)
    assert patched == 0
