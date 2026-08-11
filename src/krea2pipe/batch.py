"""Prompt sources and resumable progress tracking.

A run is driven by a path: either a single text file or a folder of text files.
Every non-empty line is one image.  Finished lines are appended to a small log
next to the output, so re-running the same path after a crash or a reboot picks
up exactly where it stopped instead of regenerating what is already on disk.
"""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Iterator

#: Text extensions considered prompt files when a folder is given.
PROMPT_SUFFIXES = (".txt", ".text", ".prompt", ".prompts")

#: Name of the resume log written inside the output directory.
PROGRESS_NAME = ".krea2pipe-progress.tsv"

LOCK_NAME = ".krea2pipe.lock"

__all__ = ["AlreadyRunningError", "OutputLock", "Prompt", "iter_prompts", "Progress"]


class AlreadyRunningError(RuntimeError):
    """Raised when another renderer owns an output directory."""


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
