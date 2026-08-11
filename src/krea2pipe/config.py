"""Configuration file support and default TOML template generation.

A config file is a flat mapping of generation and queue settings, e.g.::

    sources = ["/data/prompts/**/*.txt", "!/data/prompts/archive/**"]
    state-dir = "/data/state"
    output-dir = "/data/renders"
    steps = 8
    reconcile-interval = 300 # Safety scan every five minutes

TOML is read with the standard library.  The public CLI only selects this file
or supplies a one-time prompt.
``krea2pipe --generate-config`` derives a complete template from the same
argument parser, keeping the template and runtime defaults in sync.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from textwrap import wrap
from typing import Any

__all__ = [
    "config_options",
    "load_config",
    "render_config_template",
    "write_config_template",
]

_TEMPLATE_SKIP = {"help", "config", "generate_config"}
_TEMPLATE_HIDE = {"lora_name", "lora_strength"}
_PLACEHOLDERS: dict[str, Any] = {
    "sources": [
        "/data/krea2/prompts/**/*.txt",
        "/data/krea2/special.prompts",
        "!/data/krea2/prompts/archive/**",
        "!/data/krea2/prompts/**/draft-?.txt",
    ],
    "theme": "A visual theme for Qwen to expand into varied prompts",
    "prompt_count": 0,
    "reconcile_interval": 300,
    "width": 1248,
    "height": 1248,
    "log_file": "/data/krea2/logs/krea2pipe.log",
}
_TEMPLATE_DEFAULTS: dict[str, Any] = {
    "loras": [
        {"name": "atmospheric photography.safetensors", "strength": 0.6},
    ],
}
_HELP_OVERRIDES = {
    "prompt_mode": "Select the active source or theme block; CLI --prompt ignores both",
    "sources": "SOURCE MODE: Git-style glob list where normal entries include and leading ! entries exclude; supports *, **, ?, and character classes; regex is not supported; concrete folders recursively include .txt, .text, .prompt, and .prompts files; concrete files are included regardless of extension; relative entries resolve from the process working directory",
    "reconcile_interval": "SOURCE MODE ONLY: filesystem events ingest new files immediately; this full metadata scan recovers missed events, and 0 consumes current files then exits",
    "theme": "THEME MODE: subject and requirements that resident Qwen expands into image prompts",
    "theme_system_prompt": "THEME MODE ONLY: editable Qwen expansion instructions; the generated value is Krea 2's official system prompt",
    "prompt_count": "THEME MODE ONLY: total expansion count; omitted or 0 continues until interrupted, while a positive value is resumable and finite",
    "batch_size": "Number of independent images generated together for each prompt; larger values increase VRAM usage",
    "model_root": "Absolute model-library root containing diffusion_models, text_encoders, vae, loras, upscale_models, and SEEDVR2",
    "loras": "LoRAs under model-root/loras, applied in listed order with independent strengths; use an empty array to disable all LoRAs",
    "state_dir": "Persistent operational state root containing the SQLite source queue, theme progress, and process lock; keep it separate from disposable image output; relative paths resolve from the working directory",
    "output_dir": "Image output root; it may be removed while the service is idle without losing queue completion state; relative paths resolve from the working directory",
    "filename": "Filename template supporting %time, %date, %width, %height, and %counter; collisions are resolved safely",
    "subdir": "Subdirectory below output-dir; an empty string writes directly to the output root",
    "quality": "JPEG/WebP quality from 1 to 100; PNG ignores this value",
    "save": "Write generated images to disk; source and theme modes require this for resumable completion state",
    "service_mode": "Run the loopback monitoring and generated-image management API alongside persistent source or theme processing; CLI --prompt and --reset-status do not start it",
    "api_host": "Numeric loopback IP address for the HTTP API, normally 127.0.0.1; hostnames and non-loopback bindings are rejected",
    "api_port": "Unused TCP port from 1 to 65535 for the HTTP API",
    "run_usdu": "Run UltimateSDUpscale after base generation",
    "run_color_match": "Run the standalone ColorMatch stage after UltimateSDUpscale",
    "run_seedvr2": "Run the SeedVR2 diffusion upscaler",
    "run_blend": "Run the separate model-upscale/Lanczos blend after SeedVR2",
    "dtype": "Model compute precision; bfloat16 is recommended for the target A100, float16 improves compatibility, and float32 is for debugging",
}
_GROUP_INTROS = {
    "input mode": (
        "Only the block selected by prompt-mode is validated and used.",
        "Source entries include paths; entries beginning with ! exclude paths and never re-include them.",
        "Source mode stores its queue in state-dir/.krea2pipe-source.sqlite3 using WAL synchronous=NORMAL.",
        "Use `krea2pipe --config FILE --reset-status` to clear completion state without deleting images.",
    ),
    "resolution": (
        "Width and height override aspect-ratio and megapixels only when both are set.",
        "Base dimensions are rounded to multiple-of before upscaling.",
    ),
    "sampling": (
        "Random seeds are resolved once per process and embedded in every output image.",
        "File prompts receive durable source/content identities; theme seeds resume from the atomic theme progress file.",
    ),
    "models": (
        "Model-root must be absolute; individual checkpoint values may be absolute or relative to their category folder.",
    ),
    "output": (
        "PNG, JPEG, and WebP include the versioned generation manifest and A1111-compatible parameters.",
        "Relative state and output paths resolve from the process working directory, not from the TOML file location.",
    ),
    "HTTP service": (
        "Service mode does not accept prompts; source files and theme configuration remain the only generation inputs.",
        "The API provides health, runtime status, image listing/download, thumbnails, embedded generation data, and deletion.",
        "No authentication or TLS is provided, so the server deliberately accepts only numeric loopback IP addresses.",
        "Source service mode requires a positive reconcile-interval so file watching remains active; a finite theme remains available for image management after reaching prompt-count.",
        "Thumbnail cache files live under state-dir/thumbnails and never pollute output-dir.",
    ),
    "stages / runtime": (
        "Disabled stages are skipped entirely, including their model validation and loading.",
    ),
}


def _canonical_key(dest: str) -> str:
    return dest.replace("_", "-")


def _preferred_key(action: argparse.Action) -> str:
    """Use the positive long CLI spelling, or the destination for ``--no-*``."""
    if action.dest == "loras":
        return "loras"
    if getattr(action, "const", None) is not False:
        long_options = [item[2:] for item in action.option_strings if item.startswith("--")]
        if long_options:
            return long_options[0]
    return _canonical_key(action.dest)


def config_options(parser: argparse.ArgumentParser) -> dict[str, str]:
    """Map accepted TOML keys to argparse destination names."""
    options: dict[str, str] = {}
    for action in parser._actions:
        if action.dest in _TEMPLATE_SKIP:
            continue
        options[_canonical_key(action.dest)] = action.dest
        options[_preferred_key(action)] = action.dest
        if action.dest == "sources":
            options["source"] = action.dest
        if action.dest == "reconcile_interval":
            options["watch"] = action.dest
    return options


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if "\n" in value and "'''" not in value:
            return f"'''\n{value}'''"
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, dict) for item in value):
            entries = "\n".join(f"  {_toml_value(item)}," for item in value)
            return f"[\n{entries}\n]"
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        fields = ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items())
        return f"{{ {fields} }}"
    raise TypeError(f"cannot write {type(value).__name__} as a TOML setting")


def _comment_lines(text: object) -> list[str]:
    value = str(text)
    value = value[:1].upper() + value[1:]
    return [f"# {line}" for line in wrap(value, width=84)]


def render_config_template(parser: argparse.ArgumentParser) -> str:
    """Return a documented flat TOML file containing every TOML default."""
    lines = [
        "# Krea2pipe configuration",
        "#",
        "# Quick start:",
        "#   1. Select `prompt-mode = \"source\"` or `prompt-mode = \"theme\"`.",
        "#   2. Configure the matching source or theme block below.",
        "#   3. Keep `state-dir` persistent and set `output-dir` for images.",
        "#   4. Run: krea2pipe --config krea2pipe.toml",
        "#",
        "# Only settings for the selected mode are used; the other mode is ignored.",
        "# For a one-time prompt, pass `--prompt` on the CLI to ignore both modes.",
        "# Source mode uses filesystem events; theme mode is continuous by default.",
        "# A full source reconciliation runs every 300 seconds as a safety net.",
        "#",
        "# Path resolution:",
        "#   - Relative source, state, and output paths use the process working directory.",
        "#   - The TOML file's directory does not affect relative paths.",
        "#   - Model-root must be absolute; checkpoint names may be relative to it.",
        "#   - The supplied systemd unit uses WorkingDirectory=/data/krea2.",
        "#",
        "# Operational state:",
        "#   - Source queue: STATE_DIR/.krea2pipe-source.sqlite3.",
        "#   - Theme progress: STATE_DIR/.krea2pipe-theme-progress.json.",
        "#   - Process lock: STATE_DIR/.krea2pipe.lock.",
        "#   - Thumbnail cache: STATE_DIR/thumbnails/.",
        "#   - WAL NORMAL avoids per-image fsync; abrupt power loss may replay work.",
        "#",
        "# Keep this file flat: use these keys rather than TOML [section] tables.",
    ]
    seen: set[str] = set()
    for group in parser._action_groups:
        actions = [
            action for action in group._group_actions
            if action.dest not in _TEMPLATE_SKIP
            and action.dest not in _TEMPLATE_HIDE
            and action.dest not in seen
        ]
        if not actions:
            continue
        title = group.title[:1].upper() + group.title[1:]
        lines.extend(("", f"# --- {title} ---"))
        introductions = _GROUP_INTROS.get(group.title, ())
        for introduction in introductions:
            lines.extend(_comment_lines(introduction))
        if introductions:
            lines.append("#")
        for action in actions:
            if action.dest in {"sources", "theme"}:
                lines.append("")
            seen.add(action.dest)
            key = _preferred_key(action)
            help_text = _HELP_OVERRIDES.get(action.dest, action.help)
            if help_text and help_text is not argparse.SUPPRESS:
                lines.extend(_comment_lines(help_text))
            value = action.default
            if action.dest in _TEMPLATE_DEFAULTS:
                value = _TEMPLATE_DEFAULTS[action.dest]
            if value is None:
                placeholder = _PLACEHOLDERS.get(action.dest)
                if placeholder is not None:
                    if action.dest == "sources":
                        lines.append(f"# {key} = [")
                        lines.extend(
                            f"#   {_toml_value(item)},"
                            for item in placeholder
                        )
                        lines.append("# ]")
                    else:
                        lines.append(f"# {key} = {_toml_value(placeholder)}")
                continue
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def write_config_template(path: str, parser: argparse.ArgumentParser) -> Path:
    """Create ``path`` with defaults, refusing to overwrite an existing file."""
    file = Path(path).expanduser()
    if file.suffix.lower() != ".toml":
        raise SystemExit(f"{file}: generated configuration must use a .toml filename")
    try:
        with file.open("x", encoding="utf-8") as fh:
            fh.write(render_config_template(parser))
    except FileExistsError as exc:
        raise SystemExit(f"refusing to overwrite existing config: {file}") from exc
    except OSError as exc:
        raise SystemExit(f"could not write config {file}: {exc}") from exc
    return file


def _read(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".toml":
        raise SystemExit(f"{path}: configuration must be a .toml file")
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def load_config(path: str, valid: set[str] | dict[str, str]) -> dict[str, Any]:
    """Read ``path`` and return it keyed by argparse destination names.

    ``valid`` is the set of known destinations; anything else is rejected so a
    typo in a service config fails loudly at startup instead of being ignored.
    """
    file = Path(path).expanduser()
    if not file.exists():
        raise SystemExit(f"config file not found: {file}")
    raw = _read(file)
    if not isinstance(raw, dict):
        raise SystemExit(f"{file}: expected a table of options at the top level")

    aliases = (
        {_canonical_key(item): item for item in valid}
        if isinstance(valid, set)
        else valid
    )
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            raise SystemExit(
                f"{file}: '{key}' is a table; the config is flat, use e.g. "
                f"'{key}-resolution' instead of a [{key}] section"
            )
        normalized = key.lstrip("-").replace("_", "-")
        if normalized == "prompt":
            raise SystemExit(
                f"{file}: 'prompt' is CLI-only; pass it with --prompt"
            )
        dest = aliases.get(normalized)
        if dest is None:
            raise SystemExit(f"{file}: unknown option '{key}'")
        if dest in out:
            raise SystemExit(
                f"{file}: options '{key}' and another alias configure "
                f"the same setting '{_canonical_key(dest)}'"
            )
        out[dest] = value
    return out
