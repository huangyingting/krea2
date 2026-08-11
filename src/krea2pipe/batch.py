"""Prompt sources and resumable progress tracking.

A run is driven by a path: either a single text file or a folder of text files.
Every non-empty line is one image.  Finished lines are appended to a small log
next to the output, so re-running the same path after a crash or a reboot picks
up exactly where it stopped instead of regenerating what is already on disk.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Iterator

#: Text extensions considered prompt files when a folder is given.
PROMPT_SUFFIXES = (".txt", ".text", ".prompt", ".prompts")

#: Name of the resume log written inside the output directory.
PROGRESS_NAME = ".krea2pipe-progress.tsv"
THEME_PROGRESS_NAME = ".krea2pipe-theme-progress.json"

LOCK_NAME = ".krea2pipe.lock"

__all__ = [
    "AlreadyRunningError",
    "OutputLock",
    "Progress",
    "Prompt",
    "ThemeProgress",
    "ThemeProgressError",
    "iter_prompts",
]


class AlreadyRunningError(RuntimeError):
    """Raised when another renderer owns an output directory."""


class ThemeProgressError(RuntimeError):
    """Raised when persisted theme state is invalid or incompatible."""


class OutputLock:
    """Prevent concurrent renderers from corrupting one resume ledger."""

    def __init__(self, output_dir: str | os.PathLike):
        self.path = Path(output_dir).expanduser() / LOCK_NAME
        self._file = None

    def __enter__(self) -> OutputLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+", encoding="ascii")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._file.close()
            self._file = None
            raise AlreadyRunningError(
                f"another krea2pipe process is using output-dir {self.path.parent}"
            ) from exc
        self._file.seek(0)
        try:
            self._file.truncate()
            self._file.write(f"{os.getpid()}\n")
            self._file.flush()
            os.fsync(self._file.fileno())
        except OSError:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None
            raise
        return self

    def __exit__(self, *exc) -> None:
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None


@dataclass(frozen=True)
class Prompt:
    """One line of one prompt file."""

    file: Path
    line: int          # 1-based, matching what an editor shows
    text: str

    @property
    def key(self) -> str:
        content = blake2b(self.text.encode(), digest_size=8).hexdigest()
        return f"{self.file}\t{self.line}\t{content}"

    @property
    def seed_offset(self) -> int:
        """Stable seed component unique to this file and line."""
        digest = blake2b(self.key.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def prompt_files(path: str | os.PathLike) -> list[Path]:
    """The prompt files ``path`` refers to, sorted so a run is reproducible."""
    p = Path(path).expanduser()
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file()
                       and f.suffix.lower() in PROMPT_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no {'/'.join(PROMPT_SUFFIXES)} files under {p}")
        return files
    if not p.exists():
        raise FileNotFoundError(p)
    return [p]


def iter_prompts(path: str | os.PathLike) -> Iterator[Prompt]:
    """Yield every non-empty, non-comment line of every prompt file under ``path``."""
    for file in prompt_files(path):
        with open(file, encoding="utf-8", errors="replace") as fh:
            for number, raw in enumerate(fh, start=1):
                text = raw.strip()
                if text and not text.startswith("#"):
                    yield Prompt(file.resolve(), number, text)


class Progress:
    """Append-only record of the prompts that have already been rendered."""

    def __init__(self, output_dir: str | os.PathLike):
        self.path = Path(output_dir).expanduser() / PROGRESS_NAME
        self._done: set[str] = set()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                self._done = {line.rstrip("\n") for line in fh if line.strip()}

    def __contains__(self, prompt: Prompt) -> bool:
        return prompt.key in self._done

    def __len__(self) -> int:
        return len(self._done)

    def mark(self, prompt: Prompt) -> None:
        """Record ``prompt`` as done, durably enough to survive a power cut."""
        if prompt.key in self._done:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(prompt.key + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._done.add(prompt.key)


class ThemeProgress:
    """Atomic progress and resolved seeds for resumable theme generation."""

    def __init__(self, output_dir: str | os.PathLike, theme: str,
                 seeds: dict[str, int]):
        self.path = Path(output_dir).expanduser() / THEME_PROGRESS_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key = blake2b(theme.encode(), digest_size=16).hexdigest()
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ThemeProgressError(
                    f"invalid theme progress file {self.path}: {exc}"
                ) from exc
            if (
                not isinstance(self.state, dict)
                or self.state.get("schema_version") != 1
                or not isinstance(self.state.get("themes"), dict)
            ):
                raise ThemeProgressError(f"unsupported theme progress file: {self.path}")
        else:
            self.state = {"schema_version": 1, "themes": {}}

        entry = self.state["themes"].get(self.key)
        if entry is None:
            entry = {
                "theme": theme,
                "next_index": 0,
                "seeds": dict(seeds),
            }
            self.state["themes"][self.key] = entry
            self._write()
        elif entry.get("theme") != theme:
            raise ThemeProgressError(f"theme digest collision in {self.path}")
        self.entry = entry

    @property
    def next_index(self) -> int:
        value = self.entry.get("next_index")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ThemeProgressError(f"invalid theme prompt index in {self.path}")
        return value

    @property
    def seeds(self) -> dict[str, int]:
        values = self.entry.get("seeds")
        if not isinstance(values, dict):
            raise ThemeProgressError(f"invalid saved theme seeds in {self.path}")
        required = {"base", "usdu", "seedvr2"}
        if set(values) != required or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values.values()
        ):
            raise ThemeProgressError(f"invalid saved theme seeds in {self.path}")
        if not 0 <= values["base"] <= (1 << 64) - 1:
            raise ThemeProgressError(f"invalid saved base seed in {self.path}")
        if not 0 <= values["usdu"] <= (1 << 64) - 1:
            raise ThemeProgressError(f"invalid saved USDU seed in {self.path}")
        if not 0 <= values["seedvr2"] <= (1 << 32) - 1:
            raise ThemeProgressError(f"invalid saved SeedVR2 seed in {self.path}")
        return dict(values)

    def mark_completed(self, index: int) -> None:
        if index != self.next_index:
            raise ValueError(
                f"cannot complete theme prompt {index}; expected {self.next_index}"
            )
        self.entry["next_index"] = index + 1
        self._write()

    def _write(self) -> None:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}-",
            delete=False,
        ) as fh:
            temporary = Path(fh.name)
            json.dump(self.state, fh, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
