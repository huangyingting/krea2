"""Configuration file support and default TOML template generation.

A config file is a flat mapping of generation and queue settings, e.g.::

    source     = "/data/prompts"
    output-dir = "/data/renders"
    steps      = 8
    watch      = 60          # rescan every minute instead of exiting

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
    "source": "/data/krea2/prompts",
    "theme": "A visual theme for Qwen to expand into varied prompts",
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
    "save": "write generated images to disk",
    "run_usdu": "run UltimateSDUpscale",
    "run_color_match": "run the standalone ColorMatch stage",
    "run_seedvr2": "run the SeedVR2 upscaler",
    "run_blend": "run the separate model-upscale/Lanczos/blend stage",
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
    return options


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
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


def render_config_template(parser: argparse.ArgumentParser) -> str:
    """Return a documented flat TOML file containing every CLI default."""
    lines = [
        "# krea2pipe configuration",
        "#",
        "# Quick start:",
        "#   1. Uncomment `source` and point it at a prompt file or directory.",
        "#   2. Set `output-dir` to a persistent directory.",
        "#   3. Set `watch` above zero for a continuously running service.",
        "#   4. Run: krea2pipe --config krea2pipe.toml",
        "#",
        "# Use exactly one input mode: `source` or `theme`. For a one-time prompt,",
        "# pass `--prompt` on the CLI; all other settings still come from this file.",
        "# Relative checkpoint names are",
        "# resolved below the absolute `model-root`; absolute checkpoints also work.",
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
        lines.extend(("", f"# --- {group.title} ---"))
        for action in actions:
            seen.add(action.dest)
            key = _preferred_key(action)
            help_text = _HELP_OVERRIDES.get(action.dest, action.help)
            if help_text and help_text is not argparse.SUPPRESS:
                for line in wrap(str(help_text), width=84):
                    lines.append(f"# {line}")
            value = action.default
            if action.dest in _TEMPLATE_DEFAULTS:
                value = _TEMPLATE_DEFAULTS[action.dest]
            if value is None:
                placeholder = _PLACEHOLDERS.get(action.dest)
                if placeholder is not None:
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
        out[dest] = value
    return out
