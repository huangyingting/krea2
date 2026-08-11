"""Production preflight and resource-error tests."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from krea2pipe import validation, workflow
from krea2pipe.cli import _configure_logging
from krea2pipe.workflow import WorkflowConfig


def _minimal_model_root(tmp_path):
    files = (
        ("diffusion_models", "dit.safetensors"),
        ("text_encoders", "text.safetensors"),
        ("vae", "vae.safetensors"),
    )
    for directory, name in files:
        path = tmp_path / directory / name
        path.parent.mkdir(parents=True)
        path.touch()
    return WorkflowConfig(
        model_root=str(tmp_path),
        unet_name="dit.safetensors",
        clip_name="text.safetensors",
        vae_name="vae.safetensors",
        loras=[],
        run_usdu=False,
        run_color_match=False,
        run_seedvr2=False,
        run_blend=False,
        save=False,
        device="cpu",
    )


def test_preflight_accepts_a_complete_minimal_configuration(tmp_path):
    validation.preflight(_minimal_model_root(tmp_path))


def test_preflight_reports_the_missing_model_and_resolved_path(tmp_path):
    cfg = _minimal_model_root(tmp_path)
    (tmp_path / "vae" / "vae.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match=r"image VAE not found: .*vae/vae.safetensors"):
        validation.preflight(cfg)


def test_preflight_rejects_output_path_escape_before_inference(tmp_path):
    cfg = _minimal_model_root(tmp_path / "models")
    cfg.save = True
    cfg.state_dir = str(tmp_path / "state")
    cfg.output_dir = str(tmp_path / "output")
    cfg.subdir = "../outside"
    with pytest.raises(ValueError, match="escapes output-dir"):
        validation.preflight(cfg)


def test_preflight_creates_and_probes_the_resolved_output_destination(
    tmp_path, monkeypatch
):
    cfg = _minimal_model_root(tmp_path / "models")
    cfg.save = True
    cfg.state_dir = str(tmp_path / "state")
    cfg.output_dir = str(tmp_path / "output")
    cfg.subdir = "renders"
    cfg.filename = "daily/%width/image"
    probed = []
    original = validation.tempfile.NamedTemporaryFile

    def record_probe(*args, **kwargs):
        probed.append(Path(kwargs["dir"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(validation.tempfile, "NamedTemporaryFile", record_probe)

    validation.preflight(cfg)

    assert (tmp_path / "state").is_dir()
    width, _height = cfg.resolve_size()
    destination = tmp_path / "output" / "renders" / "daily" / str(width)
    assert destination.is_dir()
    assert probed[-1] == destination


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"batch_size": 0}, "batch-size must be greater than zero"),
        ({"width": 1024, "height": None}, "width and height must be configured together"),
        ({"width": 1025, "height": 1024}, "width and height must be multiples of 8"),
        ({"quality": 101}, "quality must be between 1 and 100"),
        ({"usdu_denoise": 1.1}, "usdu-denoise must be between 0 and 1"),
    ],
)
def test_settings_reject_invalid_values(changes, message):
    cfg = WorkflowConfig(**changes)
    with pytest.raises(ValueError, match=message):
        validation.validate_settings(cfg)


def test_cuda_device_fails_before_model_loading_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(validation.DeviceConfigurationError, match="CUDA is unavailable"):
        validation.validate_device("cuda")


def test_stage_oom_has_actionable_batch_and_resolution_context():
    cfg = WorkflowConfig(batch_size=4, run_seedvr2=False, device="cpu")
    with pytest.raises(
        workflow.PipelineOutOfMemoryError,
        match=r"base_sample.*batch-size=4.*base-resolution=1024x768.*reduce batch-size",
    ):
        with workflow._stage({}, "base_sample", cfg, 1024, 768):
            raise torch.cuda.OutOfMemoryError("allocation failed")


def test_file_logging_records_context_and_filters_by_level(tmp_path):
    path = tmp_path / "logs" / "app.log"
    root = logging.getLogger()
    try:
        _configure_logging("INFO", False, str(path))
        logger = logging.getLogger("krea2pipe.test")
        logger.debug("hidden detail")
        logger.warning("actionable warning")
        for handler in root.handlers:
            handler.flush()
        text = path.read_text()
        assert "WARNING [krea2pipe.test] actionable warning" in text
        assert "hidden detail" not in text
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers.clear()
