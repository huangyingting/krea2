"""Flow-model schedules, deterministic noise, and Euler samplers."""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor
from tqdm.auto import trange


def flux_time_shift(mu: float, sigma: float, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class ModelSamplingFlux:
    """Flux-style constant prediction with shifted timesteps."""

    def __init__(self, shift: float = 1.15, timesteps: int = 10000):
        self.shift = shift
        self.sigmas = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps))

    @property
    def sigma_min(self) -> Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> Tensor:
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return flux_time_shift(self.shift, 1.0, timestep)

    # --- CONST ---
    @staticmethod
    def calculate_input(sigma, noise):
        return noise

    @staticmethod
    def calculate_denoised(sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    @staticmethod
    def noise_scaling(sigma, noise, latent_image):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return sigma * noise + (1.0 - sigma) * latent_image

    @staticmethod
    def inverse_noise_scaling(sigma, latent):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (latent.ndim - 1))
        return latent / (1.0 - sigma)


# --- schedulers ---------------------------------------------------------------------

def normal_scheduler(model_sampling: ModelSamplingFlux, steps: int, sgm: bool = False) -> Tensor:
    s = model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)

    append_zero = True
    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        if math.isclose(float(s.sigma(end)), 0, abs_tol=0.00001):
            steps += 1
            append_zero = False
        timesteps = torch.linspace(start, end, steps)

    sigs = [float(s.sigma(timesteps[x])) for x in range(len(timesteps))]
    if append_zero:
        sigs += [0.0]
    return torch.FloatTensor(sigs)


def simple_scheduler(model_sampling: ModelSamplingFlux, steps: int) -> Tensor:
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


SCHEDULERS = {
    "sgm_uniform": lambda ms, steps: normal_scheduler(ms, steps, sgm=True),
    "normal": lambda ms, steps: normal_scheduler(ms, steps),
    "simple": simple_scheduler,
}


def calculate_sigmas(model_sampling: ModelSamplingFlux, scheduler: str, steps: int,
                     denoise: float | None = None) -> Tensor:
    """Calculate a schedule, including partial-denoise sigma slicing."""
    if scheduler not in SCHEDULERS:
        raise ValueError(f"unsupported scheduler {scheduler!r}")
    if denoise is None or denoise > 0.9999:
        return SCHEDULERS[scheduler](model_sampling, steps)
    if denoise <= 0.0:
        return torch.FloatTensor([])
    new_steps = int(steps / denoise)
    sigmas = SCHEDULERS[scheduler](model_sampling, new_steps)
    return sigmas[-(steps + 1):]


# --- noise --------------------------------------------------------------------------

def prepare_noise(latent_image: Tensor, seed: int) -> Tensor:
    """Create deterministic CPU float32 noise from a manual seed."""
    generator = torch.manual_seed(seed)
    return torch.randn(
        latent_image.size(), dtype=torch.float32, layout=latent_image.layout,
        generator=generator, device="cpu",
    ).to(dtype=latent_image.dtype)


def default_noise_sampler(x: Tensor, seed: int | None = None):
    if seed is not None:
        if x.device == torch.device("cpu"):
            seed += 1
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None
    return lambda sigma, sigma_next: torch.randn(
        x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator
    )


# --- samplers -----------------------------------------------------------------------

Denoiser = Callable[[Tensor, Tensor], Tensor]


@torch.no_grad()
def sample_euler_ancestral_rf(model: Denoiser, x: Tensor, sigmas: Tensor, seed: int | None = None,
                              eta: float = 1.0, s_noise: float = 1.0, disable: bool = False,
                              callback=None) -> Tensor:
    """Euler ancestral sampler for rectified-flow models."""
    noise_sampler = default_noise_sampler(x, seed=seed)
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in)
        if callback is not None:
            callback(i, sigmas[i], denoised)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
            sigma_down = sigmas[i + 1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i + 1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i + 1] ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
            if eta > 0:
                x = (alpha_ip1 / alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff
    return x


@torch.no_grad()
def sample_euler(model: Denoiser, x: Tensor, sigmas: Tensor, seed: int | None = None,
                 disable: bool = False, callback=None, **kwargs) -> Tensor:
    """Deterministic Euler sampler."""
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in)
        d = (x - denoised) / sigma_hat.view(*([1] * x.ndim))
        if callback is not None:
            callback(i, sigmas[i], denoised)
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x


SAMPLERS: dict[str, Callable] = {
    "euler_ancestral": sample_euler_ancestral_rf,
    "euler": sample_euler,
}


# --- driver -------------------------------------------------------------------------

@torch.no_grad()
def sample(
    denoise_fn: Callable[[Tensor, Tensor], Tensor],
    noise: Tensor,
    latent_image: Tensor,
    sigmas: Tensor,
    sampler_name: str,
    model_sampling: ModelSamplingFlux,
    seed: int | None = None,
    disable_pbar: bool = False,
    callback=None,
) -> Tensor:
    """Scale noise, execute the selected sampler, and unscale the result.

    ``denoise_fn(x, sigma)`` must return the *model output* (velocity); the CONST
    ``calculate_denoised`` conversion is applied here, exactly like
    ``BaseModel._apply_model``.
    """
    if sampler_name not in SAMPLERS:
        raise ValueError(f"unsupported sampler {sampler_name!r}")
    if len(sigmas) <= 1:
        return latent_image

    device = noise.device
    sigmas = sigmas.to(device)

    def model_k(x: Tensor, sigma: Tensor) -> Tensor:
        model_output = denoise_fn(x, sigma).float()
        return model_sampling.calculate_denoised(sigma, model_output, x)

    x = model_sampling.noise_scaling(sigmas[0].unsqueeze(0), noise, latent_image)
    samples = SAMPLERS[sampler_name](model_k, x, sigmas, seed=seed, disable=disable_pbar,
                                     callback=callback)
    return model_sampling.inverse_noise_scaling(sigmas[-1].unsqueeze(0), samples)


def slice_sigmas(sigmas: Tensor, start_step: int | None, last_step: int | None,
                 force_full_denoise: bool = False) -> Tensor:
    """Execute advanced sampling with optional start and final step bounds."""
    sigmas = sigmas.clone()
    if last_step is not None and last_step < (len(sigmas) - 1):
        sigmas = sigmas[: last_step + 1]
        if force_full_denoise:
            sigmas[-1] = 0
    if start_step is not None:
        if start_step < (len(sigmas) - 1):
            sigmas = sigmas[start_step:]
        else:
            return sigmas[:0]
    return sigmas
