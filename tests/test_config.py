"""TOML configuration and the intentionally small public CLI."""

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
        'prompt-mode = "source"\n'
        'source = "/data/prompts"\n'
        'output-dir = "/data/output"\n'
        "reconcile-interval = 30\n"
        "steps = 6\n"
    )
    args = parse_args(["--config", str(file)])
    assert args.sources == "/data/prompts"
    assert args.output_dir == "/data/output"
    assert args.reconcile_interval == 30
    assert args.steps == 6


def test_one_shot_prompt_keeps_toml_generation_settings(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'prompt-mode = "source"\n'
        'source = "/config/prompts"\n'
        "reconcile-interval = 30\n"
        "steps = 6\n"
    )
    args = parse_args(["--config", str(file), "--prompt", "a fox"])
    assert args.sources == "/config/prompts"
    assert args.reconcile_interval == 30
    assert args.steps == 6
    assert args.prompt == "a fox"


def test_config_rejects_unknown_options(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text("stpes = 8\n")
    with pytest.raises(SystemExit, match="unknown option 'stpes'"):
        parse_args(["--config", str(file)])


def test_config_rejects_non_toml_files(tmp_path):
    file = tmp_path / "krea2pipe.yaml"
    file.write_text("source: /data/prompts\n")
    with pytest.raises(SystemExit, match=r"configuration must be a \.toml file"):
        parse_args(["--config", str(file)])


@pytest.mark.parametrize("option", [
    ["/data/prompts"],
    ["--theme", "forest"],
    ["--watch", "5"],
    ["--reconcile-interval", "5"],
    ["--batch-size", "2"],
    ["--device", "cpu"],
])
def test_public_cli_rejects_configuration_switches(option):
    with pytest.raises(SystemExit):
        parse_args(option)


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
    assert generated["seed"] == "random"
    assert generated["usdu-seed"] == "random"
    assert generated["seedvr2-seed"] == "random"
    assert generated["prompt-mode"] == "source"
    from krea2pipe.prompting import EXPANSION_SYSTEM_PROMPT

    assert generated["theme-system-prompt"] == EXPANSION_SYSTEM_PROMPT
    assert "prompt-count" not in generated
    assert "reconcile-interval" not in generated
    assert "# prompt-count = 0" in file.read_text()
    assert "# reconcile-interval = 300" in file.read_text()
    assert "# theme = " in file.read_text()
    assert "# sources = " in file.read_text()
    assert "source-ignore" not in file.read_text()
    assert "source-file-regex" not in file.read_text()
    assert "source-modified-after" not in file.read_text()
    assert "\nprompt = " not in file.read_text()
    assert 'prompt-mode = "source"\n\n# SOURCE MODE' in file.read_text()
    assert "\n\n# THEME MODE" in file.read_text()
    documented = file.read_text().replace("\n# ", " ")
    assert "SOURCE MODE: Git-style glob list where normal entries include" in documented
    assert "relative entries resolve from the process working directory" in documented
    assert "# increase VRAM usage" in file.read_text()
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
    cfg = config_from_args(args)
    assert 0 <= cfg.seed <= cli.MAX_SEED
    assert 0 <= cfg.usdu_seed <= cli.MAX_SEED
    assert 0 <= cfg.seedvr2.seed <= cli.MAX_SEEDVR2


def test_generate_config_refuses_to_overwrite(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text("steps = 4\n")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["--generate-config", str(file)])
    assert file.read_text() == "steps = 4\n"


def test_generate_config_requires_toml_filename(tmp_path):
    file = tmp_path / "krea2pipe.yaml"
    with pytest.raises(SystemExit, match=r"must use a \.toml filename"):
        main(["--generate-config", str(file)])
    assert not file.exists()


def test_generate_config_rejects_input_config(tmp_path):
    with pytest.raises(SystemExit, match="cannot be combined"):
        main([
            "--config",
            str(tmp_path / "input.toml"),
            "--generate-config",
            str(tmp_path / "output.toml"),
        ])


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


def test_toml_accepts_unified_source_patterns(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    second = tmp_path / "second.txt"
    second.write_text("prompt\n")
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'prompt-mode = "source"\n'
        f'sources = ["{first}/**/*.txt", "{second}", "!{first}/archive/**"]\n'
    )
    args = parse_args(["--config", str(file)])

    assert cli._resolve_input_mode(args) == "source"
    source_spec = cli._source_spec(args)
    assert source_spec.entries == (
        f"{first}/**/*.txt",
        str(second),
        f"!{first}/archive/**",
    )


def test_toml_accepts_legacy_source_and_watch_keys(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'prompt-mode = "source"\n'
        'source = "/data/prompts"\n'
        "watch = 45\n"
    )
    args = parse_args(["--config", str(file)])

    assert cli._resolve_input_mode(args) == "source"
    assert args.sources == ["/data/prompts"]
    assert args.reconcile_interval == 45


def test_toml_rejects_source_alias_with_sources(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'source = "/data/prompts"\n'
        'sources = ["/data/other"]\n'
    )
    with pytest.raises(SystemExit, match="same setting 'sources'"):
        parse_args(["--config", str(file)])


def test_source_mode_rejects_source_list_without_an_include(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'prompt-mode = "source"\n'
        f'sources = ["!{tmp_path}/archive/**"]\n'
    )
    args = parse_args(["--config", str(file)])
    cli._resolve_input_mode(args)

    with pytest.raises(SystemExit, match="at least one positive"):
        cli._source_spec(args)


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


def test_seedvr2_rejects_a_seed_above_the_numpy_limit(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(f"seedvr2-seed = {1 << 32}\n")
    with pytest.raises(SystemExit, match=r"seedvr2-seed: must be between 0 and 4294967295"):
        config_from_args(parse_args(["--config", str(file)]))


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

    with batch.SourceQueue(source, cfg.output_dir) as queue:
        queue.reconcile()
        monkeypatch.setattr(
            queue,
            "counts",
            lambda: pytest.fail("queue-wide count ran in the per-prompt path"),
        )
        assert cli._render_pending(cfg, queue) == 1
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
        lambda cfg, theme, seed: expanded.append(
            (theme, seed, cfg.theme_system_prompt)
        ) or f"prompt {seed}",
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
        theme_system_prompt="Use concise prompts.",
    )

    assert cli._render_theme(cfg, "quiet forest", 2) == 2
    assert [item[1] for item in expanded] == [100, 101]
    assert {item[2] for item in expanded} == {"Use concise prompts."}
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
        theme_system_prompt="Use concise prompts.",
    )
    assert cli._render_theme(resumed, "quiet forest", 3) == 1
    assert rendered[-1].prompt_index == 2
    assert rendered[-1].seed == 102
    assert rendered[-1].usdu_seed == 202
    assert rendered[-1].seedvr2.seed == 302


def test_prompt_mode_selects_source_and_ignores_theme_settings(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'prompt-mode = "source"\n'
        'source = "/data/prompts"\n'
        "reconcile-interval = 0\n"
        'theme = "forest"\n'
        "prompt-count = -1\n"
        "theme-system-prompt = 123\n"
    )
    args = parse_args(["--config", str(file)])

    assert cli._resolve_input_mode(args) == "source"
    assert args.sources == ["/data/prompts"]
    assert args.reconcile_interval == 0


def test_toml_rejects_cli_only_prompt(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text('prompt = "fox"\n')
    with pytest.raises(SystemExit, match="'prompt' is CLI-only"):
        parse_args(["--config", str(file)])


def test_one_shot_prompt_ignores_configured_source(tmp_path, monkeypatch):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'prompt-mode = "source"\n'
        'source = "/does/not/exist"\n'
        f'output-dir = "{tmp_path / "output"}"\n'
    )
    rendered = []
    monkeypatch.setattr(
        cli,
        "_render",
        lambda cfg: rendered.append(cfg) or SimpleNamespace(paths=["image.jpg"]),
    )
    monkeypatch.setattr(
        cli,
        "_render_pending",
        lambda *_args, **_kwargs: pytest.fail("configured source was not ignored"),
    )

    assert main(["--config", str(file), "--prompt", "  a fox  "]) == 0
    assert [cfg.prompt for cfg in rendered] == ["a fox"]


def test_one_shot_prompt_ignores_queue_only_settings(tmp_path, monkeypatch):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'prompt-mode = "invalid-but-ignored"\n'
        'theme = "forest"\n'
        "prompt-count = -1\n"
        "reconcile-interval = -1\n"
        f'output-dir = "{tmp_path / "output"}"\n'
    )
    monkeypatch.setattr(
        cli,
        "_render",
        lambda _cfg: SimpleNamespace(paths=["image.jpg"]),
    )

    assert main(["--config", str(file), "--prompt", "a fox"]) == 0


@pytest.mark.parametrize(
    ("mode_config", "expected_mode", "expected_value"),
    [
        (
            'prompt-mode = "source"\n'
            'source = "/data/prompts"\n'
            "reconcile-interval = 0\n"
            'theme = "ignored theme"\n'
            "prompt-count = -1\n",
            "source",
            ["/data/prompts"],
        ),
        (
            'prompt-mode = "theme"\n'
            'theme = "quiet forest"\n'
            "prompt-count = 1\n"
            "sources = 123\n"
            'reconcile-interval = "ignored"\n'
            "",
            "theme",
            "quiet forest",
        ),
    ],
)
def test_main_routes_configured_input_mode(
    tmp_path, monkeypatch, mode_config, expected_mode, expected_value
):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        mode_config + f'output-dir = "{tmp_path / "output"}"\n'
    )
    calls = []
    monkeypatch.setattr(
        cli,
        "_run_source_queue",
        lambda _cfg, source, interval: calls.append(
            ("source", source, interval)
        ) or 0,
    )
    monkeypatch.setattr(cli, "_source_spec", lambda args: args.sources)
    monkeypatch.setattr(
        cli,
        "_render_theme",
        lambda _cfg, theme, prompt_count: calls.append(
            ("theme", theme, prompt_count)
        ),
    )

    assert main(["--config", str(file)]) == 0
    assert calls[0][:2] == (expected_mode, expected_value)


def test_source_mode_reconciles_periodically_by_default(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text('prompt-mode = "source"\nsource = "/data/prompts"\n')
    args = parse_args(["--config", str(file)])

    assert cli._resolve_input_mode(args) == "source"
    assert (
        args.reconcile_interval
        == cli.DEFAULT_SOURCE_RECONCILE_INTERVAL
    )
    assert args.prompt_count is None


def test_source_fallback_reconciles_while_backlog_remains(monkeypatch):
    class StopFallback(Exception):
        pass

    class Queue:
        reconciliations = 0

        def reconcile(self):
            self.reconciliations += 1

    queue = Queue()
    clock = [0.0]
    render_calls = []

    def render_pending(_cfg, _queue, *, announce_empty, max_prompts):
        render_calls.append((announce_empty, max_prompts))
        if len(render_calls) == 3:
            raise StopFallback
        clock[0] += 6
        return 1

    monkeypatch.setattr(cli, "_render_pending", render_pending)
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])

    with pytest.raises(StopFallback):
        cli._run_reconciliation_fallback(SimpleNamespace(), queue, 10)

    assert queue.reconciliations == 2
    assert render_calls == [(True, 1), (False, 1), (False, 1)]


def test_theme_mode_runs_continuously_by_default(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text('prompt-mode = "theme"\ntheme = "forest"\n')
    args = parse_args(["--config", str(file)])

    assert cli._resolve_input_mode(args) == "theme"
    assert args.prompt_count == 0
    assert args.reconcile_interval is None


def test_generation_requires_a_config_file():
    with pytest.raises(SystemExit, match="generation requires --config"):
        main([])


def test_one_shot_prompt_requires_a_config_file():
    with pytest.raises(SystemExit, match="generation requires --config"):
        main(["--prompt", "a fox"])


def test_config_requires_a_mode_without_one_shot_prompt(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text("steps = 6\n")
    with pytest.raises(SystemExit, match="prompt-mode is 'source'"):
        main(["--config", str(file)])


def test_prompt_mode_rejects_unknown_value(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text('prompt-mode = "other"\nsource = "/data/prompts"\n')
    with pytest.raises(SystemExit, match="prompt-mode: unsupported value"):
        main(["--config", str(file)])


def test_theme_mode_ignores_source_settings(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'prompt-mode = "theme"\n'
        'theme = "forest"\n'
        "prompt-count = 1\n"
        "source = 123\n"
        'reconcile-interval = "not-a-number"\n'
    )
    args = parse_args(["--config", str(file)])

    assert cli._resolve_input_mode(args) == "theme"
    assert args.theme == "forest"
    assert args.prompt_count == 1


def test_theme_system_prompt_is_configurable(tmp_path):
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'prompt-mode = "theme"\n'
        'theme = "forest"\n'
        'theme-system-prompt = "Write a concise Chinese image prompt."\n'
    )
    args = parse_args(["--config", str(file)])

    assert cli._resolve_input_mode(args) == "theme"
    assert args.theme_system_prompt == "Write a concise Chinese image prompt."


def test_reset_status_cli_clears_source_completion(tmp_path):
    from krea2pipe import batch

    source = tmp_path / "prompts.txt"
    source.write_text("one\ntwo\n")
    output = tmp_path / "output"
    with batch.SourceQueue(source, output) as queue:
        queue.reconcile()
        queue.mark(queue.next_pending())

    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'prompt-mode = "source"\n'
        f'sources = ["{source}"]\n'
        f'output-dir = "{output}"\n'
    )

    assert main(["--config", str(file), "--reset-status"]) == 0
    with batch.SourceQueue(source, output) as queue:
        queue.reconcile()
        assert queue.counts() == (2, 0, 2)


def test_reset_status_cli_clears_theme_completion(tmp_path):
    from krea2pipe import batch
    from krea2pipe.prompting import EXPANSION_SYSTEM_PROMPT

    output = tmp_path / "output"
    seeds = {"base": 1, "usdu": 2, "seedvr2": 3}
    progress = batch.ThemeProgress(
        output,
        "quiet forest",
        seeds,
        EXPANSION_SYSTEM_PROMPT,
    )
    progress.mark_completed(0)
    file = tmp_path / "krea2pipe.toml"
    file.write_text(
        'prompt-mode = "theme"\n'
        'theme = "quiet forest"\n'
        f'output-dir = "{output}"\n'
    )

    assert main(["--config", str(file), "--reset-status"]) == 0
    restarted = batch.ThemeProgress(
        output,
        "quiet forest",
        seeds,
        EXPANSION_SYSTEM_PROMPT,
    )
    assert restarted.next_index == 0


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


def test_public_cli_rejects_lora_overrides():
    with pytest.raises(SystemExit):
        parse_args(["--add-lora", "first.safetensors", "0.4"])


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


def test_canonical_config_exposes_every_optional_stage():
    canonical = Path(__file__).parents[1] / "krea2pipe.toml"
    cfg = config_from_args(parse_args(["--config", str(canonical)]))
    assert cfg.run_usdu
    assert cfg.run_color_match
    assert cfg.run_seedvr2
    assert cfg.run_blend
    assert cfg.model_root == "/data/ComfyUI/models"
    assert cfg.seedvr2.model_dir == "/data/ComfyUI/models/SEEDVR2"
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
        main(["--config", str(file)])
