from pathlib import Path

import pytest

from krea2pipe import loaders


def test_model_names_resolve_under_configured_root():
    assert loaders.resolve_model("vae", "image.safetensors", "/srv/models") == (
        "/srv/models/vae/image.safetensors"
    )


def test_absolute_model_paths_are_preserved(tmp_path):
    model = tmp_path / "model.safetensors"
    assert loaders.resolve_model("vae", str(model), "/srv/models") == str(model)


def test_relative_model_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    local = tmp_path / "model.safetensors"
    local.touch()
    monkeypatch.chdir(tmp_path)
    assert loaders.resolve_model("vae", local.name, "/srv/models") == (
        "/srv/models/vae/model.safetensors"
    )


def test_model_root_expands_home(monkeypatch):
    monkeypatch.setenv("HOME", "/home/tester")
    assert Path(loaders.resolve_model("loras", "style.safetensors", "~/models")) == (
        Path("/home/tester/models/loras/style.safetensors")
    )


def test_model_root_must_be_absolute():
    with pytest.raises(ValueError, match="must be an absolute path"):
        loaders.resolve_model("vae", "image.safetensors", "relative/models")
