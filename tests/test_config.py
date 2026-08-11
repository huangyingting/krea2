"""TOML configuration and CLI precedence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

import krea2pipe.cli as cli
from krea2pipe.cli import config_from_args, main, parse_args


def test_toml_config_supplies_service_options(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'source = "/data/prompts"\n'
        'output-dir = "/data/output"\n'
        "watch = 30\n"
        "steps = 6\n"
    )
    args = parse_args(["--config", str(file)])
    assert args.source == "/data/prompts"
    assert args.output_dir == "/data/output"
    assert args.watch == 30
    assert args.steps == 6


def test_command_line_overrides_config(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text('source = "/config/prompts"\nwatch = 30\n')
    args = parse_args(["--config", str(file), "--watch", "5", "/cli/prompts"])
    assert args.source == "/cli/prompts"
    assert args.watch == 5


def test_config_rejects_unknown_options(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text("stpes = 8\n")
    with pytest.raises(SystemExit, match="unknown option 'stpes'"):
        parse_args(["--config", str(file)])


def test_minimal_batch_cli_is_just_a_path():
    args = parse_args(["/data/prompts"])
    assert args.source == "/data/prompts"


def test_generate_config_writes_all_defaults_and_round_trips(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    assert main(["--generate-config", str(file)]) == 0

    with file.open("rb") as fh:
        generated = tomllib.load(fh)
    assert generated["batch-size"] == 1
    assert generated["aspect-ratio"] == "1:1"
    assert generated["steps"] == 8
    assert generated["sampler"] == "euler_ancestral"
    assert generated["log-level"] == "INFO"
    assert "log-file = " in file.read_text()
    assert generated["prompt-count"] == 1
    assert "# theme = " in file.read_text()
    from krea2pipe.workflow import WorkflowConfig

    assert generated["model-root"] == WorkflowConfig().model_root
    assert generated["loras"] == [
        {"name": "atmospheric photography.safetensors", "strength": 0.6}
    ]
    assert "lora" not in generated
    assert generated["usdu-sampler"] == "euler"
    assert generated["usdu-scheduler"] == "simple"
    assert generated["run-color-match"] is True
    assert generated["color-match-method"] == "hm-mkl-hm"
    assert generated["color-match-strength"] == 0.22
    assert generated["run-blend"] is True
    assert generated["blend-upscale-model"] == "4xNomosWebPhoto_RealPLKSR.pth"
    assert generated["blend-mode"] == "normal"
    assert generated["run-seedvr2"] is True
    assert generated["output-dir"] == "output"

    args = parse_args(["--config", str(file)])
    assert args.batch_size == 1
    assert args.steps == 8
    assert args.sampler_name == "euler_ancestral"
    assert args.run_seedvr2 is True
    assert args.output_dir == "output"
    assert config_from_args(args) == config_from_args(parse_args([]))


def test_generate_config_refuses_to_overwrite(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text("steps = 4\n")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["--generate-config", str(file)])
    assert file.read_text() == "steps = 4\n"


def test_generate_config_uses_default_filename():
    args = parse_args(["--generate-config"])
    assert args.generate_config == "krea2pipe.toml"


def test_toml_accepts_cli_option_aliases(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'sampler = "euler"\n'
        'unet = "custom.safetensors"\n'
        "run-usdu = false\n"
    )
    args = parse_args(["--config", str(file)])
    assert args.sampler_name == "euler"
    assert args.unet_name == "custom.safetensors"
    assert args.run_usdu is False


def test_toml_supports_random_seeds(tmp_path, monkeypatch):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'seed = "random"\nusdu-seed = "random"\nseedvr2-seed = "random"\n'
    )
    values = iter([123, 456, 789])
    bits = []

    def random_bits(count):
        bits.append(count)
        return next(values)

    monkeypatch.setattr(cli.secrets, "randbits", random_bits)

    cfg = config_from_args(parse_args(["--config", str(file)]))

    assert cfg.seed == 456
    assert cfg.usdu_seed == 789
    assert cfg.seedvr2.seed == 123
    assert bits == [32, 64, 64]


def test_seedvr2_rejects_a_seed_above_the_numpy_limit():
    with pytest.raises(SystemExit, match=r"seedvr2-seed: must be between 0 and 4294967295"):
        config_from_args(parse_args(["--seedvr2-seed", str(1 << 32)]))


def test_batch_prompt_offsets_keep_seedvr2_seed_in_32_bit_range(tmp_path, monkeypatch):
    from krea2pipe import batch
    from krea2pipe.seedvr2 import SeedVR2Config
    from krea2pipe.workflow import WorkflowConfig

    source = tmp_path / "prompts.txt"
    source.write_text("one prompt\n")
    prompt = next(batch.iter_prompts(source))
    rendered = []
    monkeypatch.setattr(
        cli,
        "_render",
        lambda cfg: rendered.append(cfg) or SimpleNamespace(paths=["image.jpg"]),
    )
    cfg = WorkflowConfig(
        output_dir=str(tmp_path / "output"),
        seedvr2=SeedVR2Config(seed=cli.MAX_SEEDVR2),
    )

    assert cli._render_pending(cfg, str(source)) == 1
    assert rendered[0].seedvr2.seed == (
        cfg.seedvr2.seed + prompt.seed_offset
    ) & cli.MAX_SEEDVR2


def test_theme_mode_resumes_with_saved_seeds(tmp_path, monkeypatch):
    from krea2pipe.seedvr2 import SeedVR2Config
    from krea2pipe.workflow import WorkflowConfig

    expanded = []
    rendered = []
    monkeypatch.setattr(
        cli,
        "expand_theme",
        lambda _cfg, theme, seed: expanded.append((theme, seed)) or f"prompt {seed}",
    )
    monkeypatch.setattr(
        cli,
        "_render",
        lambda cfg: rendered.append(cfg) or SimpleNamespace(paths=["image.jpg"]),
    )
    cfg = WorkflowConfig(
        output_dir=str(tmp_path),
        seed=100,
        usdu_seed=200,
        seedvr2=SeedVR2Config(seed=300),
    )

    assert cli._render_theme(cfg, "quiet forest", 2) == 2
    assert [item[1] for item in expanded] == [100, 101]
    assert [item.prompt_index for item in rendered] == [0, 1]
    assert rendered[1].prompt_theme == "quiet forest"
    assert rendered[1].prompt_seed == 101
    assert rendered[1].seed == 101
    assert rendered[1].usdu_seed == 201
    assert rendered[1].seedvr2.seed == 301

    resumed = WorkflowConfig(
        output_dir=str(tmp_path),
        seed=999,
        usdu_seed=999,
        seedvr2=SeedVR2Config(seed=999),
    )
    assert cli._render_theme(resumed, "quiet forest", 3) == 1
    assert rendered[-1].prompt_index == 2
    assert rendered[-1].seed == 102
    assert rendered[-1].usdu_seed == 202
    assert rendered[-1].seedvr2.seed == 302


def test_theme_cli_rejects_conflicting_prompt_sources():
    with pytest.raises(SystemExit, match="do not combine"):
        main(["--theme", "forest", "--prompt", "fox"])


def test_prompt_count_requires_theme():
    with pytest.raises(SystemExit, match="only valid with theme"):
        main(["--prompt-count", "2"])


def test_toml_supports_multiple_loras_and_usdu_sampling(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'loras = [\n'
        '  { name = "first.safetensors", strength = 0.4 },\n'
        '  { name = "second.safetensors", strength = 0.75 },\n'
        ']\n'
        'usdu-sampler = "euler_ancestral"\n'
        'usdu-scheduler = "sgm_uniform"\n'
    )

    cfg = config_from_args(parse_args(["--config", str(file)]))

    assert cfg.resolve_loras() == [
        ("first.safetensors", 0.4),
        ("second.safetensors", 0.75),
    ]
    assert cfg.usdu_sampler_name == "euler_ancestral"
    assert cfg.usdu_scheduler == "sgm_uniform"


def test_repeated_add_lora_cli_replaces_default_lora():
    cfg = config_from_args(parse_args([
        "--add-lora", "first.safetensors", "0.4",
        "--add-lora", "second.safetensors", "0.75",
    ]))
    assert cfg.resolve_loras() == [
        ("first.safetensors", 0.4),
        ("second.safetensors", 0.75),
    ]


def test_toml_rejects_malformed_lora(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text('loras = [{ name = "missing-strength.safetensors" }]\n')
    with pytest.raises(SystemExit, match="expected only 'name' and 'strength'"):
        config_from_args(parse_args(["--config", str(file)]))


def test_toml_rejects_unsupported_usdu_sampler(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text('usdu-sampler = "not-a-sampler"\n')
    with pytest.raises(SystemExit, match="usdu-sampler: unsupported value"):
        config_from_args(parse_args(["--config", str(file)]))


def test_toml_controls_color_match_and_blend_modules(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        "run-color-match = false\n"
        'color-match-method = "reinhard"\n'
        "color-match-strength = 0.5\n"
        "run-blend = false\n"
        'blend-upscale-model = "other-upscaler.pth"\n'
        'blend-mode = "screen"\n'
        "blend-factor = 0.25\n"
    )
    cfg = config_from_args(parse_args(["--config", str(file)]))
    assert not cfg.run_color_match
    assert cfg.color_match_method == "reinhard"
    assert cfg.color_match_strength == 0.5
    assert not cfg.run_blend
    assert cfg.blend_upscale_model_name == "other-upscaler.pth"
    assert cfg.blend_mode == "screen"
    assert cfg.blend_factor == 0.25


def test_service_example_exposes_every_optional_stage():
    example = Path(__file__).parents[1] / "krea2pipe.example.toml"
    cfg = config_from_args(parse_args(["--config", str(example)]))
    assert cfg.run_usdu
    assert cfg.run_color_match
    assert cfg.run_seedvr2
    assert cfg.run_blend
    assert cfg.model_root == "/data/models"
    assert cfg.seedvr2.model_dir == "/data/models/SEEDVR2"
    assert cfg.color_match_method == "hm-mkl-hm"
    assert cfg.blend_upscale_model_name == "4xNomosWebPhoto_RealPLKSR.pth"


def test_config_rejects_relative_model_root(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text('model-root = "relative/models"\n')
    with pytest.raises(SystemExit, match="model-root must be an absolute path"):
        config_from_args(parse_args(["--config", str(file)]))


def test_config_rejects_wrong_scalar_types(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text('batch-size = "four"\n')
    with pytest.raises(SystemExit) as exc:
        config_from_args(parse_args(["--config", str(file)]))
    assert exc.value.code == 2


def test_config_rejects_invalid_log_level(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text('log-level = "TRACE"\n')
    with pytest.raises(SystemExit, match="log-level must be"):
        main(["--config", str(file), "--device", "cpu", "--no-save"])
