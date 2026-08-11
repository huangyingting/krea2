"""Local HTTP monitoring and generated-image management."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import blake2b
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

from .metadata import read_generation_manifest

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
DEFAULT_THUMBNAIL_SIDE = 512
MAX_THUMBNAIL_SIDE = 1024


class ImageNotFoundError(FileNotFoundError):
    """Raised when an image ID is absent from the current output catalog."""


class InvalidImageRequest(ValueError):
    """Raised when an image API request is malformed."""


class GenerationDataNotFoundError(LookupError):
    """Raised when an image has no embedded krea2pipe manifest."""


class InvalidGenerationDataError(ValueError):
    """Raised when embedded krea2pipe metadata is invalid or unsupported."""


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    relative_path: str
    path: Path
    width: int
    height: int
    image_format: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    def as_json(self) -> dict[str, Any]:
        modified = datetime.fromtimestamp(
            self.mtime_ns / 1_000_000_000,
            tz=timezone.utc,
        ).isoformat()
        base = f"/v1/images/{self.image_id}"
        return {
            "id": self.image_id,
            "path": self.relative_path,
            "width": self.width,
            "height": self.height,
            "format": self.image_format.lower(),
            "size": self.size,
            "modified_at": modified,
            "urls": {
                "image": base,
                "thumbnail": f"{base}/thumbnail",
                "generation_data": f"{base}/generation-data",
            },
        }


class ImageCatalog:
    """Filesystem-backed catalog restricted to one configured output root."""

    def __init__(
        self,
        output_dir: str | os.PathLike,
        state_dir: str | os.PathLike,
    ):
        self.output_root = Path(output_dir).expanduser().resolve()
        self.thumbnail_root = (
            Path(state_dir).expanduser().resolve() / "thumbnails"
        )
        self._records: dict[str, ImageRecord] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _image_id(relative_path: str) -> str:
        return blake2b(relative_path.encode(), digest_size=16).hexdigest()

    def _safe_file(self, file: Path) -> Path | None:
        try:
            if file.is_symlink() or not file.is_file():
                return None
            resolved = file.resolve()
            resolved.relative_to(self.output_root)
        except (OSError, ValueError):
            return None
        return resolved

    def _record(self, file: Path, previous: ImageRecord | None) -> ImageRecord | None:
        safe = self._safe_file(file)
        if safe is None:
            return None
        try:
            relative = safe.relative_to(self.output_root)
        except ValueError:
            return None
        if any(part.startswith(".") for part in relative.parts):
            return None
        relative_path = relative.as_posix()
        try:
            stat = safe.stat()
        except OSError:
            return None
        if (
            previous is not None
            and previous.relative_path == relative_path
            and previous.size == stat.st_size
            and previous.mtime_ns == stat.st_mtime_ns
            and previous.ctime_ns == stat.st_ctime_ns
        ):
            return previous
        try:
            with Image.open(safe) as image:
                width, height = image.size
                image_format = image.format
        except (OSError, UnidentifiedImageError):
            return None
        if (
            image_format not in {"PNG", "JPEG", "WEBP"}
            or width < 1
            or height < 1
        ):
            return None
        return ImageRecord(
            image_id=self._image_id(relative_path),
            relative_path=relative_path,
            path=safe,
            width=width,
            height=height,
            image_format=image_format,
            device=stat.st_dev,
            inode=stat.st_ino,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            ctime_ns=stat.st_ctime_ns,
        )

    def reconcile(self) -> None:
        """Refresh the catalog and discard files removed outside the API."""
        with self._lock:
            discovered: dict[str, ImageRecord] = {}
            if self.output_root.is_dir():
                for file in self.output_root.rglob("*"):
                    if file.suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    if file.is_relative_to(self.thumbnail_root):
                        continue
                    relative_hint = file.relative_to(self.output_root).as_posix()
                    image_id = self._image_id(relative_hint)
                    record = self._record(file, self._records.get(image_id))
                    if record is None:
                        continue
                    previous = self._records.get(record.image_id)
                    if (
                        previous is not None
                        and (
                            previous.mtime_ns != record.mtime_ns
                            or previous.ctime_ns != record.ctime_ns
                            or previous.size != record.size
                        )
                    ):
                        self._remove_thumbnails(record.image_id)
                    existing = discovered.get(record.image_id)
                    if (
                        existing is not None
                        and existing.relative_path != record.relative_path
                    ):
                        logger.error(
                            "image ID collision between %s and %s",
                            existing.path,
                            record.path,
                        )
                        continue
                    discovered[record.image_id] = record
            removed = set(self._records) - set(discovered)
            self._records = discovered
            for image_id in removed:
                self._remove_thumbnails(image_id)

    def list(
        self,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
    ) -> tuple[list[ImageRecord], str | None]:
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise InvalidImageRequest(
                f"limit must be between 1 and {MAX_PAGE_SIZE}"
            )
        self.reconcile()
        with self._lock:
            records = sorted(
                self._records.values(),
                key=lambda item: (item.mtime_ns, item.relative_path),
                reverse=True,
            )
            start = 0
            if cursor is not None:
                if not IMAGE_ID_PATTERN.fullmatch(cursor):
                    raise InvalidImageRequest("cursor is invalid")
                try:
                    start = next(
                        index + 1
                        for index, item in enumerate(records)
                        if item.image_id == cursor
                    )
                except StopIteration as exc:
                    raise InvalidImageRequest("cursor no longer exists") from exc
            page = records[start : start + limit]
            next_cursor = (
                page[-1].image_id
                if page and start + len(page) < len(records)
                else None
            )
            return page, next_cursor

    def get(self, image_id: str) -> ImageRecord:
        if not IMAGE_ID_PATTERN.fullmatch(image_id):
            raise ImageNotFoundError(image_id)
        reconciled = False
        while True:
            with self._lock:
                record = self._records.get(image_id)
            if record is None:
                if reconciled:
                    raise ImageNotFoundError(image_id)
                self.reconcile()
                reconciled = True
                continue
            safe = self._safe_file(record.path)
            if safe is not None:
                try:
                    stat = safe.stat()
                except OSError:
                    pass
                else:
                    if (
                        stat.st_dev == record.device
                        and stat.st_ino == record.inode
                        and stat.st_size == record.size
                        and stat.st_mtime_ns == record.mtime_ns
                        and stat.st_ctime_ns == record.ctime_ns
                    ):
                        return record
            if reconciled:
                raise ImageNotFoundError(image_id)
            self.reconcile()
            reconciled = True

    def generation_data(self, image_id: str) -> dict[str, Any]:
        with self._lock:
            record = self.get(image_id)
            try:
                manifest = read_generation_manifest(record.path)
            except ValueError as exc:
                raise InvalidGenerationDataError(str(exc)) from exc
            if manifest is None:
                raise GenerationDataNotFoundError(image_id)
            return manifest

    def thumbnail(self, image_id: str, max_side: int) -> Path:
        if not 1 <= max_side <= MAX_THUMBNAIL_SIDE:
            raise InvalidImageRequest(
                f"max_side must be between 1 and {MAX_THUMBNAIL_SIDE}"
            )
        with self._lock:
            record = self.get(image_id)
            self.thumbnail_root.mkdir(parents=True, exist_ok=True)
            target = self.thumbnail_root / (
                f"{record.image_id}-{record.mtime_ns}-{record.ctime_ns}-"
                f"{record.size}-{max_side}.webp"
            )
            if target.is_file() and not target.is_symlink():
                return target
            with Image.open(record.path) as opened:
                image = ImageOps.exif_transpose(opened)
                image.thumbnail(
                    (max_side, max_side),
                    resample=Image.Resampling.LANCZOS,
                )
                if image.mode not in {"RGB", "RGBA"}:
                    image = image.convert("RGB")
                fd, temporary = tempfile.mkstemp(
                    prefix=f".{record.image_id}-",
                    suffix=".webp",
                    dir=self.thumbnail_root,
                )
                os.close(fd)
                try:
                    image.save(temporary, format="WEBP", quality=85, method=6)
                    with open(temporary, "rb") as fh:
                        os.fsync(fh.fileno())
                    os.replace(temporary, target)
                except BaseException:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
                    raise
            return target

    def delete(self, image_id: str) -> None:
        with self._lock:
            record = self.get(image_id)
            try:
                record.path.unlink()
            except FileNotFoundError as exc:
                raise ImageNotFoundError(image_id) from exc
            self._records.pop(image_id, None)
            self._remove_thumbnails(image_id)

    def _remove_thumbnails(self, image_id: str) -> None:
        if not self.thumbnail_root.is_dir():
            return
        for thumbnail in self.thumbnail_root.glob(f"{image_id}-*.webp"):
            try:
                thumbnail.unlink()
            except FileNotFoundError:
                pass


class RuntimeStatus:
    """Thread-safe snapshot of the source/theme worker's current activity."""

    def __init__(self, mode: str):
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._mode = mode
        self._ready = False
        self._state = "starting"
        self._stage: str | None = None
        self._current: dict[str, Any] | None = None
        self._queue: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._last_outputs: list[str] = []

    def ready(self) -> None:
        with self._lock:
            self._ready = True
            self._state = "idle"

    def begin(self, current: dict[str, Any], stage: str = "preparing") -> None:
        with self._lock:
            self._state = "running"
            self._stage = stage
            self._current = dict(current)
            self._last_error = None

    def set_prompt(self, prompt: str) -> None:
        with self._lock:
            if self._current is not None:
                self._current["prompt"] = prompt[:500]

    def progress(self, stage: str) -> None:
        with self._lock:
            self._stage = stage[:500]

    def complete(self, paths: list[str]) -> None:
        with self._lock:
            self._state = "idle"
            self._stage = None
            self._current = None
            self._last_outputs = list(paths)

    def fail(self, error: BaseException) -> None:
        with self._lock:
            self._ready = False
            self._state = "degraded"
            self._stage = None
            self._last_error = str(error)

    def set_queue(self, **values: Any) -> None:
        with self._lock:
            self._queue = dict(values)

    def decrement_pending(self) -> None:
        with self._lock:
            if self._queue is None:
                return
            pending = self._queue.get("pending")
            completed = self._queue.get("completed")
            if isinstance(pending, int) and pending > 0:
                self._queue["pending"] = pending - 1
            if isinstance(completed, int):
                self._queue["completed"] = completed + 1

    def stopping(self) -> None:
        with self._lock:
            self._ready = False
            self._state = "stopping"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self._mode,
                "state": self._state,
                "ready": self._ready,
                "uptime_seconds": max(0, round(time.time() - self._started_at, 3)),
                "stage": self._stage,
                "current": dict(self._current) if self._current is not None else None,
                "queue": dict(self._queue) if self._queue is not None else None,
                "last_error": self._last_error,
                "last_outputs": list(self._last_outputs),
            }


class _IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6


class HttpApiServer:
    """Context-managed loopback HTTP server running in a daemon thread."""

    def __init__(
        self,
        host: str,
        port: int,
        catalog: ImageCatalog,
        status: RuntimeStatus,
    ):
        self.host = host
        self.port = port
        self.catalog = catalog
        self.status = status
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            raise RuntimeError("HTTP API is not running")
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def __enter__(self) -> HttpApiServer:
        address = ipaddress.ip_address(self.host)
        if not address.is_loopback:
            raise ValueError("api-host must be a loopback IP address")
        server_type = (
            _IPv6ThreadingHTTPServer
            if address.version == 6
            else ThreadingHTTPServer
        )
        handler = self._handler_type()
        self._server = server_type((self.host, self.port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="krea2-http-api",
            daemon=True,
        )
        self._thread.start()
        host, port = self.address
        display_host = f"[{host}]" if ":" in host else host
        logger.info("HTTP API listening on http://%s:%d", display_host, port)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.status.stopping()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        catalog = self.catalog
        runtime = self.status

        class Handler(BaseHTTPRequestHandler):
            server_version = "krea2pipe"
            sys_version = ""

            def log_message(self, format: str, *args: object) -> None:
                logger.info("HTTP %s - %s", self.address_string(), format % args)

            def _headers(
                self,
                status: HTTPStatus,
                content_type: str,
                length: int,
                *,
                cache_control: str = "no-store",
                disposition: str | None = None,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(length))
                self.send_header("Cache-Control", cache_control)
                self.send_header("X-Content-Type-Options", "nosniff")
                if disposition is not None:
                    self.send_header("Content-Disposition", disposition)
                self.end_headers()

            def _json(
                self,
                status: HTTPStatus,
                body: dict[str, Any],
            ) -> None:
                raw = json.dumps(
                    body,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
                self._headers(status, "application/json; charset=utf-8", len(raw))
                self.wfile.write(raw)

            def _error(
                self,
                status: HTTPStatus,
                code: str,
                message: str,
            ) -> None:
                self._json(
                    status,
                    {"error": {"code": code, "message": message}},
                )

            def _file(
                self,
                path: Path,
                content_type: str,
                *,
                disposition: str | None = None,
            ) -> None:
                try:
                    size = path.stat().st_size
                    with path.open("rb") as fh:
                        self._headers(
                            HTTPStatus.OK,
                            content_type,
                            size,
                            disposition=disposition,
                        )
                        shutil.copyfileobj(fh, self.wfile)
                except FileNotFoundError:
                    self._error(
                        HTTPStatus.NOT_FOUND,
                        "image_not_found",
                        "image no longer exists",
                    )

            @staticmethod
            def _query(
                raw_query: str,
                allowed: set[str],
            ) -> dict[str, str]:
                parsed = parse_qs(raw_query, keep_blank_values=True)
                unknown = set(parsed) - allowed
                if unknown:
                    raise InvalidImageRequest(
                        f"unknown query parameter: {sorted(unknown)[0]}"
                    )
                if any(len(values) != 1 for values in parsed.values()):
                    raise InvalidImageRequest(
                        "query parameters may be supplied only once"
                    )
                return {key: values[0] for key, values in parsed.items()}

            def do_GET(self) -> None:
                target = urlsplit(self.path)
                parts = [part for part in target.path.split("/") if part]
                try:
                    if target.path == "/health":
                        if target.query:
                            raise InvalidImageRequest(
                                "health does not accept query parameters"
                            )
                        snapshot = runtime.snapshot()
                        status = (
                            HTTPStatus.OK
                            if snapshot["ready"]
                            else HTTPStatus.SERVICE_UNAVAILABLE
                        )
                        self._json(
                            status,
                            {
                                "status": (
                                    "ok"
                                    if snapshot["ready"]
                                    else snapshot["state"]
                                ),
                                "ready": snapshot["ready"],
                            },
                        )
                        return
                    if target.path == "/v1/status":
                        if target.query:
                            raise InvalidImageRequest(
                                "status does not accept query parameters"
                            )
                        self._json(HTTPStatus.OK, runtime.snapshot())
                        return
                    if target.path == "/v1/images":
                        query = self._query(target.query, {"limit", "cursor"})
                        try:
                            limit = int(query.get("limit", DEFAULT_PAGE_SIZE))
                        except ValueError as exc:
                            raise InvalidImageRequest(
                                "limit must be an integer"
                            ) from exc
                        records, next_cursor = catalog.list(
                            limit=limit,
                            cursor=query.get("cursor"),
                        )
                        self._json(
                            HTTPStatus.OK,
                            {
                                "images": [record.as_json() for record in records],
                                "next_cursor": next_cursor,
                            },
                        )
                        return
                    if len(parts) >= 3 and parts[:2] == ["v1", "images"]:
                        image_id = parts[2]
                        if len(parts) == 3:
                            if target.query:
                                raise InvalidImageRequest(
                                    "image downloads do not accept query parameters"
                                )
                            record = catalog.get(image_id)
                            media_type = {
                                "PNG": "image/png",
                                "JPEG": "image/jpeg",
                                "WEBP": "image/webp",
                            }[record.image_format]
                            disposition = (
                                "inline; filename*=UTF-8''"
                                + quote(record.path.name, safe="")
                            )
                            self._file(
                                record.path,
                                media_type,
                                disposition=disposition,
                            )
                            return
                        if len(parts) == 4 and parts[3] == "thumbnail":
                            query = self._query(target.query, {"max_side"})
                            try:
                                max_side = int(
                                    query.get(
                                        "max_side",
                                        DEFAULT_THUMBNAIL_SIDE,
                                    )
                                )
                            except ValueError as exc:
                                raise InvalidImageRequest(
                                    "max_side must be an integer"
                                ) from exc
                            self._file(
                                catalog.thumbnail(image_id, max_side),
                                "image/webp",
                            )
                            return
                        if len(parts) == 4 and parts[3] == "generation-data":
                            if target.query:
                                raise InvalidImageRequest(
                                    "generation-data does not accept query parameters"
                                )
                            self._json(
                                HTTPStatus.OK,
                                catalog.generation_data(image_id),
                            )
                            return
                    self._error(
                        HTTPStatus.NOT_FOUND,
                        "not_found",
                        "endpoint not found",
                    )
                except ImageNotFoundError as exc:
                    self._error(
                        HTTPStatus.NOT_FOUND,
                        "image_not_found",
                        str(exc),
                    )
                except GenerationDataNotFoundError:
                    self._error(
                        HTTPStatus.NOT_FOUND,
                        "generation_data_not_found",
                        "image has no embedded krea2pipe generation data",
                    )
                except InvalidImageRequest as exc:
                    self._error(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_request",
                        str(exc),
                    )
                except InvalidGenerationDataError as exc:
                    self._error(
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                        "invalid_generation_data",
                        str(exc),
                    )
                except FileNotFoundError:
                    self._error(
                        HTTPStatus.NOT_FOUND,
                        "image_not_found",
                        "image no longer exists",
                    )
                except OSError as exc:
                    logger.exception("image API request failed")
                    self._error(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "io_error",
                        str(exc),
                    )

            def do_DELETE(self) -> None:
                target = urlsplit(self.path)
                parts = [part for part in target.path.split("/") if part]
                if (
                    len(parts) != 3
                    or parts[:2] != ["v1", "images"]
                    or target.query
                ):
                    self._error(
                        HTTPStatus.NOT_FOUND,
                        "not_found",
                        "endpoint not found",
                    )
                    return
                try:
                    catalog.delete(parts[2])
                    self._headers(
                        HTTPStatus.NO_CONTENT,
                        "application/json",
                        0,
                    )
                except ImageNotFoundError as exc:
                    self._error(
                        HTTPStatus.NOT_FOUND,
                        "image_not_found",
                        str(exc),
                    )
                except OSError as exc:
                    logger.exception("image deletion failed")
                    self._error(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "io_error",
                        str(exc),
                    )

            def do_POST(self) -> None:
                self._error(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    "method_not_allowed",
                    "the API is read-only except for image deletion",
                )

            do_PUT = do_POST
            do_PATCH = do_POST

        return Handler
