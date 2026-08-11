"""Scalable prompt-source discovery and durable resume tracking.

Source mode consumes any mixture of files and folders. Filesystem events index
new content into SQLite, while periodic metadata reconciliation recovers missed
events without rereading unchanged prompt files.
"""

from __future__ import annotations

import fcntl
import glob
import json
import os
import sqlite3
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
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
PromptRow = tuple[bytes, bytes, int, int, int, int, bytes, int]
EMPTY_CONTENT_HASH = blake2b(b"", digest_size=16).digest()

__all__ = [
    "AlreadyRunningError",
    "OutputLock",
    "Progress",
    "Prompt",
    "SourceQueue",
    "SourceQueueSchemaError",
    "SourceSpec",
    "SourceWatchError",
    "SourceWatcher",
    "ThemeProgress",
    "ThemeProgressError",
    "iter_prompts",
]


class AlreadyRunningError(RuntimeError):
    """Raised when another renderer owns a state directory."""


class ThemeProgressError(RuntimeError):
    """Raised when persisted theme state is invalid or incompatible."""


class SourceWatchError(RuntimeError):
    """Raised when filesystem event monitoring stops unexpectedly."""


class SourceQueueSchemaError(RuntimeError):
    """Raised when persisted source state uses an unsupported schema."""


class OutputLock:
    """Prevent concurrent renderers from corrupting one resume ledger."""

    def __init__(self, state_dir: str | os.PathLike):
        self.path = Path(state_dir).expanduser() / LOCK_NAME
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
                f"another krea2pipe process is using state-dir {self.path.parent}"
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
    durable_id: bytes | None = None

    @property
    def key(self) -> str:
        content = blake2b(self.text.encode(), digest_size=8).hexdigest()
        return f"{self.file}\t{self.line}\t{content}"

    @property
    def identifier(self) -> bytes:
        """Compact durable identity, optionally supplied by the source queue."""
        return self.durable_id or blake2b(self.key.encode(), digest_size=16).digest()

    @property
    def seed_offset(self) -> int:
        """Stable seed component derived from this prompt's durable identity."""
        identity = self.durable_id if self.durable_id is not None else self.key.encode()
        digest = blake2b(identity, digest_size=8).digest()
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
class _SourcePattern:
    root: Path
    pattern: PathSpec
    explicit_file: Path | None = None
    default_extensions: bool = False

    def matches(self, file: Path) -> bool:
        if not file.is_relative_to(self.root):
            return False
        if self.explicit_file is not None:
            return file == self.explicit_file
        if self.default_extensions and file.suffix.lower() not in PROMPT_SUFFIXES:
            return False
        return self.pattern.match_file(file.relative_to(self.root).as_posix())


class SourceSpec:
    """One ordered source list with glob includes and leading-! exclusions."""

    def __init__(
        self,
        sources: (
            str
            | os.PathLike
            | list[str | os.PathLike]
            | tuple[str | os.PathLike, ...]
        ),
    ):
        raw_values = (
            [sources]
            if isinstance(sources, (str, os.PathLike))
            else list(sources)
        )
        try:
            values = [os.fspath(value) for value in raw_values]
        except TypeError as exc:
            raise ValueError(
                "sources must contain non-empty path or glob strings"
            ) from exc
        if not values or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError("sources must contain non-empty path or glob strings")
        self.entries = tuple(value.strip() for value in values)
        positives = [value for value in self.entries if not value.startswith("!")]
        if not positives:
            raise ValueError("sources must contain at least one positive entry")
        self.includes = tuple(self._include_pattern(value) for value in positives)
        self.absolute_excludes: list[_SourcePattern | Path] = []
        for value in self.entries:
            if value.startswith("!"):
                self._add_exclusion(value[1:])
        self.scan_roots = self._collapse_roots(
            pattern.root
            for pattern in self.includes
            if pattern.explicit_file is None
        )
        self.explicit_files = tuple(
            pattern.explicit_file
            for pattern in self.includes
            if pattern.explicit_file is not None
        )
        self.watch_roots = self._collapse_roots(
            list(self.scan_roots)
            + [file.parent for file in self.explicit_files]
        )

    @staticmethod
    def _absolute(value: str) -> str:
        expanded = os.path.expanduser(value)
        if not os.path.isabs(expanded):
            expanded = os.path.join(os.getcwd(), expanded)
        return os.path.normpath(expanded)

    @staticmethod
    def _collapse_roots(paths) -> tuple[Path, ...]:
        roots: list[Path] = []
        for path in sorted(set(paths), key=lambda item: (len(item.parts), str(item))):
            if not any(path == root or path.is_relative_to(root) for root in roots):
                roots.append(path)
        return tuple(roots)

    @staticmethod
    def _split_glob(value: str, *, require_root: bool = True) -> tuple[Path, str]:
        absolute = SourceSpec._absolute(value)
        parts = Path(absolute).parts
        prefix: list[str] = []
        pattern: list[str] = []
        found_magic = False
        for part in parts:
            if not found_magic and not glob.has_magic(part):
                prefix.append(part)
            else:
                found_magic = True
                pattern.append(part)
        root = Path(*prefix).resolve()
        if require_root and not root.is_dir():
            raise FileNotFoundError(
                f"source glob root not found or not a directory: {root}"
            )
        return root, "/".join(pattern)

    @staticmethod
    def _pathspec(pattern: str) -> PathSpec:
        return PathSpec.from_lines("gitignore", [f"/{pattern.lstrip('/')}"])

    def _include_pattern(self, value: str) -> _SourcePattern:
        absolute = self._absolute(value)
        if not glob.has_magic(absolute):
            path = Path(absolute).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            if path.is_file():
                return _SourcePattern(
                    path.parent,
                    self._pathspec(path.name),
                    explicit_file=path,
                )
            return _SourcePattern(
                path,
                self._pathspec("**"),
                default_extensions=True,
            )
        root, pattern = self._split_glob(value)
        return _SourcePattern(root, self._pathspec(pattern))

    def _add_exclusion(self, value: str) -> None:
        if not value:
            raise ValueError("sources exclusion after ! must not be empty")
        absolute = self._absolute(value)
        if not glob.has_magic(absolute):
            self.absolute_excludes.append(Path(absolute).resolve())
            return
        root, pattern = self._split_glob(value, require_root=False)
        self.absolute_excludes.append(_SourcePattern(root, self._pathspec(pattern)))

    def includes_file(self, file: Path) -> bool:
        file = file.expanduser().resolve()
        matching = [pattern for pattern in self.includes if pattern.matches(file)]
        if not matching:
            return False
        for exclusion in self.absolute_excludes:
            if isinstance(exclusion, Path):
                if file == exclusion or file.is_relative_to(exclusion):
                    return False
            elif exclusion.matches(file):
                return False
        return True

    def iter_files(self) -> Iterator[Path]:
        explicit = set(self.explicit_files)
        for file in self.explicit_files:
            if file.is_file() and self.includes_file(file):
                yield file
        for root in self.scan_roots:
            for file in root.rglob("*"):
                if file not in explicit and file.is_file() and self.includes_file(file):
                    yield file

    @property
    def identity(self) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in self.entries:
            excluded = value.startswith("!")
            path = value.removeprefix("!")
            absolute = self._absolute(path)
            if glob.has_magic(absolute):
                root, pattern = self._split_glob(
                    path,
                    require_root=not excluded,
                )
                canonical = str(root / pattern)
            else:
                canonical = str(Path(absolute).resolve())
            normalized.append(("!" if excluded else "") + canonical)
        return tuple(normalized)


class SourceQueue:
    """Compact incremental SQLite index backed by immutable source-file ranges."""

    # Bump this for every table or index change so incompatible state is rejected.
    _SCHEMA_VERSION = 2
    _TAIL_BYTES = 4096
    _INSERT_BATCH = 1000
    _PROMPT_UPSERT = """
        INSERT INTO queue_prompts(
            source_id, prompt_id, file_id, line_number,
            byte_offset, byte_length, text_hash, occurrence, active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(source_id, prompt_id) DO UPDATE SET
            file_id = excluded.file_id,
            line_number = excluded.line_number,
            byte_offset = excluded.byte_offset,
            byte_length = excluded.byte_length,
            text_hash = excluded.text_hash,
            occurrence = excluded.occurrence,
            active = 1
    """

    def __init__(
        self,
        sources: SourceSpec | str | os.PathLike | list[str] | tuple[str, ...],
        state_dir: str | os.PathLike,
    ):
        self.spec = sources if isinstance(sources, SourceSpec) else SourceSpec(sources)
        identity = json.dumps(
            {"sources": self.spec.identity},
            sort_keys=True,
            ensure_ascii=False,
        )
        self.source_id = blake2b(identity.encode(), digest_size=16).digest()
        state = Path(state_dir).expanduser()
        state.mkdir(parents=True, exist_ok=True)
        self.path = state / SOURCE_QUEUE_NAME
        self._db = sqlite3.connect(self.path, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.execute("PRAGMA foreign_keys=ON")
        try:
            self._initialize()
        except BaseException:
            self._db.close()
            raise

    def __enter__(self) -> SourceQueue:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def _initialize(self) -> None:
        version = int(self._db.execute("PRAGMA user_version").fetchone()[0])
        tables = {
            row["name"]
            for row in self._db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        application_tables = tables & {
            "source_files",
            "queue_prompts",
            "completed_prompts",
        }
        if version == 0 and application_tables:
            raise SourceQueueSchemaError(
                f"unversioned source queue schema in {self.path}; "
                "remove the development state database and restart"
            )
        if version not in {0, self._SCHEMA_VERSION}:
            raise SourceQueueSchemaError(
                f"unsupported source queue schema {version} in {self.path}; "
                f"expected {self._SCHEMA_VERSION}"
            )
        self._db.execute("BEGIN")
        try:
            self._create_schema()
            self._db.execute(f"PRAGMA user_version={self._SCHEMA_VERSION}")
            self._db.commit()
        except BaseException:
            self._db.rollback()
            raise

    def _create_schema(self) -> None:
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS source_files (
                file_id INTEGER PRIMARY KEY,
                source_id BLOB NOT NULL,
                file_key BLOB NOT NULL,
                path TEXT NOT NULL,
                device INTEGER NOT NULL,
                inode INTEGER NOT NULL,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                line_count INTEGER NOT NULL,
                ends_newline INTEGER NOT NULL,
                tail_hash BLOB NOT NULL,
                content_hash BLOB NOT NULL,
                seen_scan INTEGER NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                UNIQUE(source_id, path)
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS queue_prompts (
                source_id BLOB NOT NULL,
                prompt_id BLOB NOT NULL,
                file_id INTEGER NOT NULL REFERENCES source_files(file_id)
                    ON DELETE CASCADE,
                line_number INTEGER NOT NULL,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                text_hash BLOB NOT NULL,
                occurrence INTEGER NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (source_id, prompt_id)
            ) WITHOUT ROWID
            """
        )
        self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS source_files_inode
            ON source_files (source_id, active, device, inode)
            """
        )
        self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS source_files_size
            ON source_files (source_id, size)
            """
        )
        self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS queue_prompts_pending
            ON queue_prompts (
                source_id, active, file_id, line_number, prompt_id
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS completed_prompts (
                prompt_id BLOB PRIMARY KEY,
                completed_at INTEGER NOT NULL
            ) WITHOUT ROWID
            """
        )

    @staticmethod
    def _snapshot(stat: os.stat_result) -> tuple[int, int, int, int]:
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    @staticmethod
    def _tail_hash(data: bytes) -> bytes:
        return blake2b(data, digest_size=16).digest()

    @staticmethod
    def _extend_content_hash(digest: bytes, raw: bytes) -> bytes:
        return blake2b(digest + raw, digest_size=16).digest()

    @staticmethod
    def _text_hash(text: str) -> bytes:
        return blake2b(text.encode(), digest_size=16).digest()

    @staticmethod
    def _prompt_identifier(
        file_key: bytes,
        text_hash: bytes,
        occurrence: int,
    ) -> bytes:
        digest = blake2b(digest_size=16)
        digest.update(file_key)
        digest.update(text_hash)
        digest.update(occurrence.to_bytes(8, "big"))
        return digest.digest()

    def _stored_file(self, path: str) -> sqlite3.Row | None:
        return self._db.execute(
            "SELECT * FROM source_files WHERE source_id = ? AND path = ?",
            (self.source_id, path),
        ).fetchone()

    def _file_fingerprint(
        self,
        file: Path,
        before: os.stat_result,
    ) -> tuple[bytes, Counter[bytes]]:
        digest = EMPTY_CONTENT_HASH
        prompts: Counter[bytes] = Counter()
        with file.open("rb") as fh:
            for raw in fh:
                digest = self._extend_content_hash(digest, raw)
                text = raw.decode("utf-8", errors="replace").strip()
                if text and not text.startswith("#"):
                    prompts[self._text_hash(text)] += 1
        if self._snapshot(before) != self._snapshot(file.stat()):
            raise _FileChangedDuringIndex(str(file))
        return digest, prompts

    def _matching_moved_file(
        self,
        file: Path,
        stat: os.stat_result,
    ) -> sqlite3.Row | None:
        file_string = str(file)
        inode_matches = [
            row
            for row in self._db.execute(
                """
                SELECT * FROM source_files
                WHERE source_id = ? AND active = 1
                  AND device = ? AND inode = ? AND path != ?
                """,
                (self.source_id, stat.st_dev, stat.st_ino, file_string),
            )
            if not Path(row["path"]).exists()
        ]
        if len(inode_matches) == 1:
            return inode_matches[0]

        missing_candidates = [
            row
            for row in self._db.execute(
                """
                SELECT * FROM source_files
                WHERE source_id = ? AND path != ?
                """,
                (self.source_id, file_string),
            )
            if not Path(row["path"]).exists()
        ]
        if not missing_candidates:
            return None
        content_hash, new_prompts = self._file_fingerprint(file, stat)
        exact_matches = [
            row
            for row in missing_candidates
            if row["size"] == stat.st_size and row["content_hash"] == content_hash
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1 or not new_prompts:
            return None

        scores: list[tuple[float, int, sqlite3.Row]] = []
        new_total = new_prompts.total()
        for candidate in missing_candidates:
            old_prompts = Counter(
                {
                    bytes(row["text_hash"]): int(row["count"])
                    for row in self._db.execute(
                        """
                        SELECT text_hash, COUNT(*) AS count
                        FROM queue_prompts
                        WHERE source_id = ? AND file_id = ?
                        GROUP BY text_hash
                        """,
                        (self.source_id, candidate["file_id"]),
                    )
                }
            )
            old_total = old_prompts.total()
            overlap = sum((new_prompts & old_prompts).values())
            similarity = overlap / max(new_total, old_total, 1)
            if similarity >= 0.5:
                scores.append((similarity, overlap, candidate))
        if not scores:
            return None
        scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best = scores[0]
        if len(scores) > 1 and best[:2] == scores[1][:2]:
            return None
        return best[2]

    def _adopt_moved_file(
        self,
        stored: sqlite3.Row,
        path: str,
        seen_scan: int,
    ) -> sqlite3.Row:
        file_id = int(stored["file_id"])
        with self._db:
            self._db.execute(
                """
                UPDATE source_files
                SET path = ?, seen_scan = ?, active = 1
                WHERE file_id = ?
                """,
                (path, seen_scan, file_id),
            )
            self._db.execute(
                """
                UPDATE queue_prompts SET active = 1
                WHERE source_id = ? AND file_id = ?
                """,
                (self.source_id, file_id),
            )
        adopted = self._stored_file(path)
        if adopted is None:
            raise RuntimeError(f"failed to adopt moved source file: {path}")
        return adopted

    def _prompt_row(
        self,
        prompt: Prompt,
        file_id: int,
        file_key: bytes,
        text_hash: bytes,
        occurrence: int,
        byte_offset: int,
        byte_length: int,
    ) -> PromptRow:
        return (
            self.source_id,
            self._prompt_identifier(file_key, text_hash, occurrence),
            file_id,
            prompt.line,
            byte_offset,
            byte_length,
            text_hash,
            occurrence,
        )

    def _record_prompt_rows(self, rows: list[PromptRow]) -> None:
        if rows:
            self._db.executemany(self._PROMPT_UPSERT, rows)
            rows.clear()

    def _record_file(
        self,
        file_id: int,
        stat: os.stat_result,
        line_count: int,
        ends_newline: bool,
        tail_hash: bytes,
        content_hash: bytes,
        seen_scan: int,
    ) -> None:
        self._db.execute(
            """
            UPDATE source_files
            SET device = ?, inode = ?, size = ?, mtime_ns = ?,
                line_count = ?, ends_newline = ?, tail_hash = ?,
                content_hash = ?, seen_scan = ?, active = 1
            WHERE file_id = ?
            """,
            (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                line_count,
                int(ends_newline),
                tail_hash,
                content_hash,
                seen_scan,
                file_id,
            ),
        )

    def _ensure_file(
        self,
        path: str,
        stat: os.stat_result,
        seen_scan: int,
    ) -> sqlite3.Row:
        self._db.execute(
            """
            INSERT OR IGNORE INTO source_files(
                source_id, file_key, path, device, inode, size, mtime_ns,
                line_count, ends_newline, tail_hash, content_hash,
                seen_scan, active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1, ?, ?, ?, 1)
            """,
            (
                self.source_id,
                os.urandom(16),
                path,
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                b"",
                EMPTY_CONTENT_HASH,
                seen_scan,
            ),
        )
        row = self._stored_file(path)
        if row is None:
            raise RuntimeError(f"failed to register source file: {path}")
        return row

    def _full_index(
        self,
        file: Path,
        before: os.stat_result,
        seen_scan: int,
        stored: sqlite3.Row | None = None,
    ) -> None:
        file_string = str(file)
        line_count = 0
        byte_offset = 0
        tail = b""
        content_hash = EMPTY_CONTENT_HASH
        last_line = b""
        occurrences: Counter[bytes] = Counter()
        rows: list[PromptRow] = []
        with self._db:
            stored = (
                stored
                if stored is not None
                else self._ensure_file(file_string, before, seen_scan)
            )
            file_id = int(stored["file_id"])
            file_key = bytes(stored["file_key"])
            self._db.execute(
                "UPDATE queue_prompts SET active = 0 "
                "WHERE source_id = ? AND file_id = ?",
                (self.source_id, file_id),
            )
            with file.open("rb") as fh:
                for line_count, raw in enumerate(fh, start=1):
                    byte_length = len(raw)
                    last_line = raw
                    tail = (tail + raw)[-self._TAIL_BYTES:]
                    content_hash = self._extend_content_hash(content_hash, raw)
                    text = raw.decode("utf-8", errors="replace").strip()
                    if text and not text.startswith("#"):
                        text_hash = self._text_hash(text)
                        occurrences[text_hash] += 1
                        rows.append(
                            self._prompt_row(
                                Prompt(file, line_count, text),
                                file_id,
                                file_key,
                                text_hash,
                                occurrences[text_hash],
                                byte_offset,
                                byte_length,
                            )
                        )
                        if len(rows) >= self._INSERT_BATCH:
                            self._record_prompt_rows(rows)
                    byte_offset += byte_length
            self._record_prompt_rows(rows)
            after = file.stat()
            if self._snapshot(before) != self._snapshot(after):
                raise _FileChangedDuringIndex(file_string)
            self._db.execute(
                "DELETE FROM queue_prompts "
                "WHERE source_id = ? AND file_id = ? AND active = 0",
                (self.source_id, file_id),
            )
            self._record_file(
                file_id,
                after,
                line_count,
                not last_line or last_line.endswith(b"\n"),
                self._tail_hash(tail),
                content_hash,
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
        file_id = int(stored["file_id"])
        file_key = bytes(stored["file_key"])
        line_count = stored["line_count"]
        byte_offset = stored["size"]
        content_hash = bytes(stored["content_hash"])
        last_line = b""
        occurrences: dict[bytes, int] = {}
        rows: list[PromptRow] = []
        with self._db:
            with file.open("rb") as fh:
                fh.seek(stored["size"])
                for line_count, raw in enumerate(
                    fh, start=stored["line_count"] + 1
                ):
                    byte_length = len(raw)
                    last_line = raw
                    content_hash = self._extend_content_hash(content_hash, raw)
                    text = raw.decode("utf-8", errors="replace").strip()
                    if text and not text.startswith("#"):
                        text_hash = self._text_hash(text)
                        if text_hash not in occurrences:
                            row = self._db.execute(
                                """
                                SELECT COALESCE(MAX(occurrence), 0) AS occurrence
                                FROM queue_prompts
                                WHERE source_id = ? AND file_id = ?
                                  AND text_hash = ?
                                """,
                                (self.source_id, file_id, text_hash),
                            ).fetchone()
                            occurrences[text_hash] = int(row["occurrence"])
                        occurrences[text_hash] += 1
                        rows.append(
                            self._prompt_row(
                                Prompt(file, line_count, text),
                                file_id,
                                file_key,
                                text_hash,
                                occurrences[text_hash],
                                byte_offset,
                                byte_length,
                            )
                        )
                        if len(rows) >= self._INSERT_BATCH:
                            self._record_prompt_rows(rows)
                    byte_offset += byte_length
            self._record_prompt_rows(rows)
            after = file.stat()
            if self._snapshot(before) != self._snapshot(after):
                raise _FileChangedDuringIndex(file_string)
            with file.open("rb") as fh:
                start = max(0, after.st_size - self._TAIL_BYTES)
                fh.seek(start)
                tail = fh.read()
            self._record_file(
                file_id,
                after,
                line_count,
                not last_line or last_line.endswith(b"\n"),
                self._tail_hash(tail),
                content_hash,
                seen_scan,
            )

    def index_file(self, file: Path, seen_scan: int = 0) -> bool:
        file = file.expanduser().resolve()
        if not file.is_file():
            return False
        if not self.spec.includes_file(file):
            return bool(self._remove_path(file))
        file_string = str(file)
        for _attempt in range(2):
            before = file.stat()
            stored = self._stored_file(file_string)
            if stored is None:
                try:
                    moved = self._matching_moved_file(file, before)
                except _FileChangedDuringIndex:
                    continue
                if moved is not None:
                    stored = self._adopt_moved_file(moved, file_string, seen_scan)
            if (
                stored is not None
                and stored["active"]
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
                        "UPDATE source_files SET seen_scan = ?, active = 1 "
                        "WHERE source_id = ? AND path = ?",
                        (seen_scan, self.source_id, file_string),
                    )
                return False
            try:
                if stored is not None and self._can_append(file, before, stored):
                    self._append_index(file, before, stored, seen_scan)
                else:
                    self._full_index(file, before, seen_scan, stored)
                return True
            except _FileChangedDuringIndex:
                continue
        raise OSError(f"prompt file kept changing while indexing: {file}")

    def _remove_path(self, path: Path) -> int:
        value = str(path.expanduser().resolve())
        prefix = value.rstrip(os.sep) + os.sep
        rows = self._db.execute(
            "SELECT file_id, path FROM source_files WHERE source_id = ? "
            "AND active = 1 AND (path = ? OR substr(path, 1, ?) = ?)",
            (self.source_id, value, len(prefix), prefix),
        ).fetchall()
        with self._db:
            for row in rows:
                self._db.execute(
                    "UPDATE source_files SET active = 0 WHERE file_id = ?",
                    (row["file_id"],),
                )
                self._db.execute(
                    """
                    UPDATE queue_prompts SET active = 0
                    WHERE source_id = ? AND file_id = ?
                    """,
                    (self.source_id, row["file_id"]),
                )
        return len(rows)

    def reconcile(self) -> int:
        """Discover files and index only new, appended, or changed content."""
        scan = time.time_ns()
        changed = 0
        for file in self.spec.iter_files():
            changed += int(self.index_file(file, scan))
        stale = self._db.execute(
            "SELECT path FROM source_files "
            "WHERE source_id = ? AND active = 1 AND seen_scan != ?",
            (self.source_id, scan),
        ).fetchall()
        for row in stale:
            changed += self._remove_path(Path(row["path"]))
        return changed

    def update_paths(self, paths: set[Path]) -> int:
        """Apply filesystem changes without walking the full source tree."""
        changed = 0
        missing: list[Path] = []
        for path in sorted({item.expanduser().resolve() for item in paths}):
            if not any(
                path == root
                or path.is_relative_to(root)
                or root.is_relative_to(path)
                for root in self.spec.watch_roots
            ):
                continue

            if path.is_dir():
                for file in path.rglob("*"):
                    if file.is_file() and self.spec.includes_file(file):
                        changed += int(self.index_file(file))
            elif path.is_file():
                changed += int(self.index_file(path))
            else:
                missing.append(path)
        for path in missing:
            changed += self._remove_path(path)
        return changed

    def next_pending(self) -> Prompt | None:
        unstable_file: Path | None = None
        unstable_attempts = 0
        while True:
            row = self._db.execute(
                """
                SELECT f.path, f.file_id, f.file_key, q.line_number,
                       q.byte_offset, q.byte_length, q.text_hash,
                       q.occurrence, q.prompt_id
                FROM queue_prompts AS q
                JOIN source_files AS f ON f.file_id = q.file_id
                LEFT JOIN completed_prompts AS c ON c.prompt_id = q.prompt_id
                WHERE q.source_id = ? AND q.active = 1
                  AND c.prompt_id IS NULL
                ORDER BY f.path, q.line_number, q.prompt_id
                LIMIT 1
                """,
                (self.source_id,),
            ).fetchone()
            if row is None:
                return None
            file = Path(row["path"])
            try:
                with file.open("rb") as fh:
                    fh.seek(row["byte_offset"])
                    raw = fh.read(row["byte_length"])
            except OSError:
                self._remove_path(file)
                continue
            text = raw.decode("utf-8", errors="replace").strip()
            text_hash = self._text_hash(text)
            identifier = self._prompt_identifier(
                bytes(row["file_key"]), text_hash, row["occurrence"]
            )
            if (
                len(raw) == row["byte_length"]
                and text_hash == row["text_hash"]
                and identifier == row["prompt_id"]
            ):
                return Prompt(file, row["line_number"], text, identifier)
            if file == unstable_file:
                unstable_attempts += 1
            else:
                unstable_file = file
                unstable_attempts = 1
            if unstable_attempts >= 3:
                raise OSError(f"prompt source kept changing while reading {file}")
            stored = self._stored_file(str(file))
            if stored is None:
                continue
            try:
                self._full_index(file, file.stat(), 0, stored)
            except FileNotFoundError:
                self._remove_path(file)
            except _FileChangedDuringIndex:
                continue

    def counts(self) -> tuple[int, int, int]:
        row = self._db.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(c.prompt_id IS NOT NULL), 0) AS done
            FROM queue_prompts AS q
            LEFT JOIN completed_prompts AS c ON c.prompt_id = q.prompt_id
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
                """
                INSERT OR IGNORE INTO completed_prompts(prompt_id, completed_at)
                VALUES (?, ?)
                """,
                (prompt.identifier, int(time.time())),
            )

    def reset(self) -> int:
        """Clear completion records for prompts active in this filtered source."""
        with self._db:
            cursor = self._db.execute(
                """
                DELETE FROM completed_prompts
                WHERE prompt_id IN (
                    SELECT prompt_id FROM queue_prompts
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
        sources: SourceSpec | str | os.PathLike | list[str] | tuple[str, ...],
    ):
        self.spec = sources if isinstance(sources, SourceSpec) else SourceSpec(sources)
        self.paths = self.spec.watch_roots
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

    def __init__(self, state_dir: str | os.PathLike, theme: str,
                 seeds: dict[str, int],
                 system_prompt: str = EXPANSION_SYSTEM_PROMPT):
        self.path = Path(state_dir).expanduser() / THEME_PROGRESS_NAME
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
