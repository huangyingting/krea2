"""Scalable prompt-source discovery and durable resume tracking.

Source mode consumes any mixture of files and folders. Filesystem events index
new content into SQLite, while periodic metadata reconciliation recovers missed
events without rereading unchanged prompt files.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path
from typing import Iterator

from pathspec import PathSpec
from watchfiles import watch

from .prompting import EXPANSION_SYSTEM_PROMPT

#: Text extensions considered prompt files when a folder is given.
PROMPT_SUFFIXES = (".txt", ".text", ".prompt", ".prompts")

#: Legacy resume log imported into the SQLite queue during upgrade.
PROGRESS_NAME = ".krea2pipe-progress.tsv"
SOURCE_QUEUE_NAME = ".krea2pipe-source.sqlite3"
THEME_PROGRESS_NAME = ".krea2pipe-theme-progress.json"

LOCK_NAME = ".krea2pipe.lock"
PromptRow = tuple[str, str, str, int, str]

__all__ = [
    "AlreadyRunningError",
    "OutputLock",
    "Progress",
    "Prompt",
    "SourceQueue",
    "SourceFilter",
    "SourceWatchError",
    "SourceWatcher",
    "ThemeProgress",
    "ThemeProgressError",
    "iter_prompts",
]


class AlreadyRunningError(RuntimeError):
    """Raised when another renderer owns an output directory."""


class ThemeProgressError(RuntimeError):
    """Raised when persisted theme state is invalid or incompatible."""


class SourceWatchError(RuntimeError):
    """Raised when filesystem event monitoring stops unexpectedly."""


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


class _FileChangedDuringIndex(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceFilter:
    """File, prompt-text, and modification-time filters for a source queue."""

    file_regex: str | None = None
    prompt_regex: str | None = None
    modified_after: str | None = None
    modified_before: str | None = None
    ignore: tuple[str, ...] = ()
    _ignore_spec: PathSpec = field(init=False, repr=False, compare=False)
    _file_pattern: re.Pattern | None = field(init=False, repr=False, compare=False)
    _prompt_pattern: re.Pattern | None = field(init=False, repr=False, compare=False)
    _after_ns: int | None = field(init=False, repr=False, compare=False)
    _before_ns: int | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for value, name in (
            (self.file_regex, "source-file-regex"),
            (self.prompt_regex, "source-prompt-regex"),
        ):
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{name} must be a non-empty string")
                try:
                    re.compile(value)
                except re.error as exc:
                    raise ValueError(f"{name}: invalid regular expression: {exc}") from exc
        after = self._parse_timestamp(self.modified_after, "source-modified-after")
        before = self._parse_timestamp(self.modified_before, "source-modified-before")
        if after is not None and before is not None and after >= before:
            raise ValueError(
                "source-modified-after must be earlier than source-modified-before"
            )
        if not isinstance(self.ignore, (list, tuple)) or any(
            not isinstance(pattern, str) or not pattern
            for pattern in self.ignore
        ):
            raise ValueError("source-ignore must be an array of non-empty patterns")
        object.__setattr__(self, "ignore", tuple(self.ignore))
        object.__setattr__(
            self,
            "_file_pattern",
            re.compile(self.file_regex) if self.file_regex is not None else None,
        )
        object.__setattr__(
            self,
            "_prompt_pattern",
            re.compile(self.prompt_regex) if self.prompt_regex is not None else None,
        )
        object.__setattr__(self, "_after_ns", after)
        object.__setattr__(self, "_before_ns", before)
        object.__setattr__(
            self,
            "_ignore_spec",
            PathSpec.from_lines("gitignore", self.ignore),
        )

    @staticmethod
    def _parse_timestamp(value: str | None, option: str) -> int | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{option} must be an ISO-8601 timestamp")
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"{option} must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1_000_000_000)

    @property
    def identity(self) -> dict[str, str | None]:
        return {
            "file_regex": self.file_regex,
            "prompt_regex": self.prompt_regex,
            "modified_after": self.modified_after,
            "modified_before": self.modified_before,
            "ignore": list(self.ignore),
        }

    def matches_file(
        self,
        file: Path,
        source: Path,
        stat: os.stat_result,
    ) -> bool:
        if self._file_pattern is not None:
            relative = file.name if source.is_file() else file.relative_to(source).as_posix()
            if self._file_pattern.search(relative) is None:
                return False
        relative = file.name if source.is_file() else file.relative_to(source).as_posix()
        if self._ignore_spec.match_file(relative):
            return False
        if self._after_ns is not None and stat.st_mtime_ns < self._after_ns:
            return False
        if self._before_ns is not None and stat.st_mtime_ns >= self._before_ns:
            return False
        return True

    def matches_prompt(self, text: str) -> bool:
        return self._prompt_pattern is None or self._prompt_pattern.search(text) is not None


class SourceQueue:
    """Incremental SQLite prompt index for one file or directory source."""

    _TAIL_BYTES = 4096
    _INSERT_BATCH = 1000
    _PROMPT_UPSERT = """
        INSERT INTO queue_prompts(
            source_id, prompt_key, file_path, line_number, text, active
        ) VALUES (?, ?, ?, ?, ?, 1)
        ON CONFLICT(source_id, prompt_key) DO UPDATE SET
            file_path = excluded.file_path,
            line_number = excluded.line_number,
            text = excluded.text,
            active = 1
    """

    def __init__(
        self,
        sources: str | os.PathLike | list[str] | tuple[str, ...],
        output_dir: str | os.PathLike,
        source_filter: SourceFilter | None = None,
    ):
        values = [sources] if isinstance(sources, (str, os.PathLike)) else list(sources)
        if not values:
            raise ValueError("sources must contain at least one file or directory")
        self.sources = tuple(dict.fromkeys(
            Path(value).expanduser().resolve() for value in values
        ))
        for source in self.sources:
            if not source.exists():
                raise FileNotFoundError(source)
        self.filter = source_filter or SourceFilter()
        identity = json.dumps(
            {
                "sources": sorted(str(source) for source in self.sources),
                "filter": self.filter.identity,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        self.source_id = blake2b(
            identity.encode(), digest_size=16
        ).hexdigest()
        output = Path(output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        self.path = output / SOURCE_QUEUE_NAME
        self._db = sqlite3.connect(self.path, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=30000")
        self._initialize()
        self._import_legacy_progress(output / PROGRESS_NAME)

    def __enter__(self) -> SourceQueue:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def _initialize(self) -> None:
        with self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS source_queue_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS source_files (
                    source_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    device INTEGER NOT NULL,
                    inode INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    line_count INTEGER NOT NULL,
                    ends_newline INTEGER NOT NULL,
                    tail_hash TEXT NOT NULL,
                    seen_scan INTEGER NOT NULL,
                    PRIMARY KEY (source_id, path)
                );
                CREATE TABLE IF NOT EXISTS queue_prompts (
                    source_id TEXT NOT NULL,
                    prompt_key TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (source_id, prompt_key)
                );
                CREATE INDEX IF NOT EXISTS queue_prompts_pending
                    ON queue_prompts (
                        source_id, active, file_path, line_number, prompt_key
                    );
                CREATE TABLE IF NOT EXISTS completed_prompts (
                    prompt_key TEXT PRIMARY KEY,
                    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def _import_legacy_progress(self, path: Path) -> None:
        imported = self._db.execute(
            "SELECT value FROM source_queue_state WHERE key = ?",
            ("legacy_progress_imported",),
        ).fetchone()
        if imported is not None:
            return
        with self._db:
            if path.exists():
                with path.open(encoding="utf-8") as fh:
                    for raw in fh:
                        key = raw.rstrip("\n")
                        if key:
                            self._db.execute(
                                "INSERT OR IGNORE INTO completed_prompts(prompt_key) "
                                "VALUES (?)",
                                (key,),
                            )
            self._db.execute(
                "INSERT INTO source_queue_state(key, value) VALUES (?, ?)",
                ("legacy_progress_imported", "1"),
            )

    @staticmethod
    def _snapshot(stat: os.stat_result) -> tuple[int, int, int, int]:
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    @staticmethod
    def _tail_hash(data: bytes) -> str:
        return blake2b(data, digest_size=16).hexdigest()

    def _stored_file(self, path: str) -> sqlite3.Row | None:
        return self._db.execute(
            "SELECT * FROM source_files WHERE source_id = ? AND path = ?",
            (self.source_id, path),
        ).fetchone()

    def _prompt_row(self, prompt: Prompt) -> PromptRow | None:
        if not self.filter.matches_prompt(prompt.text):
            return None
        return (
            self.source_id,
            prompt.key,
            str(prompt.file),
            prompt.line,
            prompt.text,
        )

    def _record_prompt_rows(self, rows: list[PromptRow]) -> None:
        if rows:
            self._db.executemany(self._PROMPT_UPSERT, rows)
            rows.clear()

    def _record_file(
        self,
        path: str,
        stat: os.stat_result,
        line_count: int,
        ends_newline: bool,
        tail_hash: str,
        seen_scan: int,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO source_files(
                source_id, path, device, inode, size, mtime_ns,
                line_count, ends_newline, tail_hash, seen_scan
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, path) DO UPDATE SET
                device = excluded.device,
                inode = excluded.inode,
                size = excluded.size,
                mtime_ns = excluded.mtime_ns,
                line_count = excluded.line_count,
                ends_newline = excluded.ends_newline,
                tail_hash = excluded.tail_hash,
                seen_scan = excluded.seen_scan
            """,
            (
                self.source_id,
                path,
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                line_count,
                int(ends_newline),
                tail_hash,
                seen_scan,
            ),
        )

    def _full_index(
        self,
        file: Path,
        before: os.stat_result,
        seen_scan: int,
    ) -> None:
        file_string = str(file)
        line_count = 0
        tail = b""
        last_line = b""
        rows: list[PromptRow] = []
        with self._db:
            self._db.execute(
                "UPDATE queue_prompts SET active = 0 "
                "WHERE source_id = ? AND file_path = ?",
                (self.source_id, file_string),
            )
            with file.open("rb") as fh:
                for line_count, raw in enumerate(fh, start=1):
                    last_line = raw
                    tail = (tail + raw)[-self._TAIL_BYTES:]
                    text = raw.decode("utf-8", errors="replace").strip()
                    if text and not text.startswith("#"):
                        row = self._prompt_row(Prompt(file, line_count, text))
                        if row is not None:
                            rows.append(row)
                            if len(rows) >= self._INSERT_BATCH:
                                self._record_prompt_rows(rows)
            self._record_prompt_rows(rows)
            after = file.stat()
            if self._snapshot(before) != self._snapshot(after):
                raise _FileChangedDuringIndex(file_string)
            self._db.execute(
                "DELETE FROM queue_prompts "
                "WHERE source_id = ? AND file_path = ? AND active = 0",
                (self.source_id, file_string),
            )
            self._record_file(
                file_string,
                after,
                line_count,
                not last_line or last_line.endswith(b"\n"),
                self._tail_hash(tail),
                seen_scan,
            )

    def _can_append(
        self,
        file: Path,
        stat: os.stat_result,
        stored: sqlite3.Row,
    ) -> bool:
        if (
            not stored["ends_newline"]
            or stat.st_dev != stored["device"]
            or stat.st_ino != stored["inode"]
            or stat.st_size <= stored["size"]
        ):
            return False
        start = max(0, stored["size"] - self._TAIL_BYTES)
        with file.open("rb") as fh:
            fh.seek(start)
            previous_tail = fh.read(stored["size"] - start)
        return self._tail_hash(previous_tail) == stored["tail_hash"]

    def _append_index(
        self,
        file: Path,
        before: os.stat_result,
        stored: sqlite3.Row,
        seen_scan: int,
    ) -> None:
        file_string = str(file)
        line_count = stored["line_count"]
        last_line = b""
        rows: list[PromptRow] = []
        with self._db:
            with file.open("rb") as fh:
                fh.seek(stored["size"])
                for line_count, raw in enumerate(
                    fh, start=stored["line_count"] + 1
                ):
                    last_line = raw
                    text = raw.decode("utf-8", errors="replace").strip()
                    if text and not text.startswith("#"):
                        row = self._prompt_row(Prompt(file, line_count, text))
                        if row is not None:
                            rows.append(row)
                            if len(rows) >= self._INSERT_BATCH:
                                self._record_prompt_rows(rows)
            self._record_prompt_rows(rows)
            after = file.stat()
            if self._snapshot(before) != self._snapshot(after):
                raise _FileChangedDuringIndex(file_string)
            with file.open("rb") as fh:
                start = max(0, after.st_size - self._TAIL_BYTES)
                fh.seek(start)
                tail = fh.read()
            self._record_file(
                file_string,
                after,
                line_count,
                not last_line or last_line.endswith(b"\n"),
                self._tail_hash(tail),
                seen_scan,
            )

    def index_file(self, file: Path, seen_scan: int = 0) -> bool:
        file = file.expanduser().resolve()
        if not file.is_file():
            return False
        source = self._root_for(file)
        if source is None:
            return False
        if (
            source.is_dir()
            and self.filter.file_regex is None
            and file.suffix.lower() not in PROMPT_SUFFIXES
        ):
            return False
        file_string = str(file)
        for _attempt in range(2):
            before = file.stat()
            if not self.filter.matches_file(file, source, before):
                return bool(self._remove_path(file))
            stored = self._stored_file(file_string)
            if (
                stored is not None
                and self._snapshot(before)
                == (
                    stored["device"],
                    stored["inode"],
                    stored["size"],
                    stored["mtime_ns"],
                )
            ):
                with self._db:
                    self._db.execute(
                        "UPDATE source_files SET seen_scan = ? "
                        "WHERE source_id = ? AND path = ?",
                        (seen_scan, self.source_id, file_string),
                    )
                return False
            try:
                if stored is not None and self._can_append(file, before, stored):
                    self._append_index(file, before, stored, seen_scan)
                else:
                    self._full_index(file, before, seen_scan)
                return True
            except _FileChangedDuringIndex:
                continue
        raise OSError(f"prompt file kept changing while indexing: {file}")

    def _remove_path(self, path: Path) -> int:
        value = str(path.expanduser().resolve())
        prefix = value.rstrip(os.sep) + os.sep
        rows = self._db.execute(
            "SELECT path FROM source_files WHERE source_id = ? "
            "AND (path = ? OR substr(path, 1, ?) = ?)",
            (self.source_id, value, len(prefix), prefix),
        ).fetchall()
        with self._db:
            for row in rows:
                self._db.execute(
                    "DELETE FROM queue_prompts "
                    "WHERE source_id = ? AND file_path = ?",
                    (self.source_id, row["path"]),
                )
                self._db.execute(
                    "DELETE FROM source_files "
                    "WHERE source_id = ? AND path = ?",
                    (self.source_id, row["path"]),
                )
        return len(rows)

    def reconcile(self) -> int:
        """Discover files and index only new, appended, or changed content."""
        scan = time.time_ns()
        changed = 0
        for source in self.sources:
            if source.is_dir():
                for file in source.rglob("*"):
                    if file.is_file() and (
                        self.filter.file_regex is not None
                        or file.suffix.lower() in PROMPT_SUFFIXES
                    ):
                        changed += int(self.index_file(file, scan))
            else:
                changed += int(self.index_file(source, scan))
        stale = self._db.execute(
            "SELECT path FROM source_files "
            "WHERE source_id = ? AND seen_scan != ?",
            (self.source_id, scan),
        ).fetchall()
        for row in stale:
            changed += self._remove_path(Path(row["path"]))
        return changed

    def _root_for(self, path: Path) -> Path | None:
        matching = [
            source
            for source in self.sources
            if path == source or (source.is_dir() and path.is_relative_to(source))
        ]
        if not matching:
            return None
        return max(matching, key=lambda item: len(item.parts))

    def update_paths(self, paths: set[Path]) -> int:
        """Apply filesystem changes without walking the full source tree."""
        changed = 0
        for path in sorted({item.expanduser().resolve() for item in paths}):
            if self._root_for(path) is None and not any(
                source.is_dir() and source.is_relative_to(path)
                for source in self.sources
            ):
                continue

            if path.is_dir():
                for file in path.rglob("*"):
                    if file.is_file() and (
                        self.filter.file_regex is not None
                        or file.suffix.lower() in PROMPT_SUFFIXES
                    ):
                        changed += int(self.index_file(file))
            elif path.is_file():
                changed += int(self.index_file(path))
            else:
                changed += self._remove_path(path)
        return changed

    def next_pending(self) -> Prompt | None:
        row = self._db.execute(
            """
            SELECT q.file_path, q.line_number, q.text
            FROM queue_prompts AS q
            LEFT JOIN completed_prompts AS c ON c.prompt_key = q.prompt_key
            WHERE q.source_id = ? AND q.active = 1
              AND c.prompt_key IS NULL
            ORDER BY q.file_path, q.line_number, q.prompt_key
            LIMIT 1
            """,
            (self.source_id,),
        ).fetchone()
        if row is None:
            return None
        return Prompt(Path(row["file_path"]), row["line_number"], row["text"])

    def counts(self) -> tuple[int, int, int]:
        row = self._db.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(c.prompt_key IS NOT NULL), 0) AS done
            FROM queue_prompts AS q
            LEFT JOIN completed_prompts AS c ON c.prompt_key = q.prompt_key
            WHERE q.source_id = ? AND q.active = 1
            """,
            (self.source_id,),
        ).fetchone()
        total = int(row["total"])
        done = int(row["done"])
        return total, done, total - done

    def mark(self, prompt: Prompt) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO completed_prompts(prompt_key) VALUES (?)",
                (prompt.key,),
            )

    def reset(self) -> int:
        """Clear completion records for prompts active in this filtered source."""
        with self._db:
            cursor = self._db.execute(
                """
                DELETE FROM completed_prompts
                WHERE prompt_key IN (
                    SELECT prompt_key FROM queue_prompts
                    WHERE source_id = ? AND active = 1
                )
                """,
                (self.source_id,),
            )
        return max(cursor.rowcount, 0)


class SourceWatcher:
    """Recursive event watcher with a bounded, coalescing path buffer."""

    def __init__(
        self,
        sources: str | os.PathLike | list[str] | tuple[str, ...],
    ):
        values = [sources] if isinstance(sources, (str, os.PathLike)) else list(sources)
        source_paths = [Path(value).expanduser().resolve() for value in values]
        self.paths = tuple(sorted({
            path if path.is_dir() else path.parent
            for path in source_paths
        }))
        self._condition = threading.Condition()
        self._pending: set[Path] = set()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="krea2-source-watcher",
            daemon=True,
        )

    def __enter__(self) -> SourceWatcher:
        self._thread.start()
        if not self._ready.wait(timeout=5):
            self._stop.set()
            self._thread.join(timeout=5)
            raise SourceWatchError(
                f"filesystem watcher did not start for {', '.join(map(str, self.paths))}"
            )
        if self._error is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            raise SourceWatchError(str(self._error)) from self._error
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            for changes in watch(
                *self.paths,
                recursive=True,
                stop_event=self._stop,
                debounce=500,
                step=100,
                rust_timeout=500,
                yield_on_timeout=True,
                raise_interrupt=False,
            ):
                self._ready.set()
                if not changes:
                    continue
                with self._condition:
                    self._pending.update(Path(path) for _change, path in changes)
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._ready.set()
                self._condition.notify_all()

    def wait(self, timeout: float) -> set[Path]:
        with self._condition:
            self._condition.wait_for(
                lambda: bool(self._pending) or self._error is not None,
                timeout=timeout,
            )
            if self._error is not None:
                raise SourceWatchError(str(self._error)) from self._error
            paths = set(self._pending)
            self._pending.clear()
            return paths


class ThemeProgress:
    """Atomic progress and resolved seeds for resumable theme generation."""

    def __init__(self, output_dir: str | os.PathLike, theme: str,
                 seeds: dict[str, int],
                 system_prompt: str = EXPANSION_SYSTEM_PROMPT):
        self.path = Path(output_dir).expanduser() / THEME_PROGRESS_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        identity = (
            theme
            if system_prompt == EXPANSION_SYSTEM_PROMPT
            else f"{theme}\0{system_prompt}"
        )
        self.key = blake2b(identity.encode(), digest_size=16).hexdigest()
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
                "system_prompt": system_prompt,
                "next_index": 0,
                "seeds": dict(seeds),
            }
            self.state["themes"][self.key] = entry
            self._write()
        elif entry.get("theme") != theme:
            raise ThemeProgressError(f"theme digest collision in {self.path}")
        elif entry.get("system_prompt", EXPANSION_SYSTEM_PROMPT) != system_prompt:
            raise ThemeProgressError(f"theme system prompt mismatch in {self.path}")
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

    def reset(self) -> int:
        completed = self.next_index
        del self.state["themes"][self.key]
        self._write()
        return completed

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
