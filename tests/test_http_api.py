"""Local monitoring and generated-image management API."""

from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from krea2pipe import metadata
from krea2pipe.http_api import (
    HttpApiServer,
    ImageCatalog,
    ImageNotFoundError,
    InvalidImageRequest,
    RuntimeStatus,
)


def _image(
    path: Path,
    size: tuple[int, int] = (1600, 800),
    *,
    manifest: bool = True,
) -> dict | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngInfo()
    generation_data = None
    if manifest:
        generation_data = {
            "schema": metadata.SCHEMA,
            "schema_version": metadata.SCHEMA_VERSION,
            "prompt": {"positive": "test prompt", "negative": ""},
            "image": {"width": size[0], "height": size[1]},
        }
        info.add_itxt(
            metadata.PNG_KEY,
            metadata.encode_manifest(generation_data),
            zip=True,
        )
    Image.new("RGB", size, "navy").save(path, pnginfo=info)
    return generation_data


def _request(
    base: str,
    path: str,
    *,
    method: str = "GET",
) -> tuple[int, str, bytes]:
    request = Request(base + path, method=method)
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as exc:
        return exc.code, exc.headers.get_content_type(), exc.read()
    with response:
        return (
            response.status,
            response.headers.get_content_type(),
            response.read(),
        )


def test_image_catalog_lists_only_safe_supported_images(tmp_path):
    output = tmp_path / "output"
    state = tmp_path / "state"
    first = output / "nested" / "first.png"
    second = output / "second.jpg"
    _image(first)
    Image.new("RGB", (640, 480), "red").save(second)
    _image(output / ".temporary.png")
    (output / "notes.txt").write_text("not an image")
    outside = tmp_path / "outside.png"
    _image(outside)
    (output / "escape.png").symlink_to(outside)

    catalog = ImageCatalog(output, state)
    records, cursor = catalog.list()

    assert cursor is None
    assert {record.relative_path for record in records} == {
        "nested/first.png",
        "second.jpg",
    }
    assert all(len(record.image_id) == 32 for record in records)
    first_record = next(
        record for record in records if record.relative_path == "nested/first.png"
    )
    assert (first_record.width, first_record.height) == (1600, 800)
    assert first_record.as_json()["urls"]["thumbnail"].endswith("/thumbnail")


def test_image_catalog_paginates_with_stable_opaque_cursors(tmp_path):
    output = tmp_path / "output"
    for index in range(3):
        path = output / f"{index}.png"
        _image(path, (32, 32))
        path.touch()

    catalog = ImageCatalog(output, tmp_path / "state")
    first, cursor = catalog.list(limit=2)
    second, final_cursor = catalog.list(limit=2, cursor=cursor)

    assert len(first) == 2
    assert cursor == first[-1].image_id
    assert len(second) == 1
    assert final_cursor is None
    assert not ({item.image_id for item in first} & {item.image_id for item in second})
    try:
        catalog.list(cursor="not-an-id")
    except InvalidImageRequest as exc:
        assert "cursor is invalid" in str(exc)
    else:
        raise AssertionError("invalid cursor was accepted")


def test_image_catalog_builds_bounded_cached_thumbnails(tmp_path):
    output = tmp_path / "output"
    _image(output / "wide.png", (2400, 1200))
    catalog = ImageCatalog(output, tmp_path / "state")
    record = catalog.list()[0][0]

    thumbnail = catalog.thumbnail(record.image_id, 512)
    repeated = catalog.thumbnail(record.image_id, 512)

    assert repeated == thumbnail
    with Image.open(thumbnail) as image:
        assert image.format == "WEBP"
        assert image.size == (512, 256)
    try:
        catalog.thumbnail(record.image_id, 1025)
    except InvalidImageRequest as exc:
        assert "between 1 and 1024" in str(exc)
    else:
        raise AssertionError("oversized thumbnail was accepted")

    _image(output / "wide.png", (1200, 1200))
    catalog.reconcile()
    replacement = catalog.thumbnail(record.image_id, 512)
    assert replacement != thumbnail
    assert not thumbnail.exists()
    with Image.open(replacement) as image:
        assert image.size == (512, 512)


def test_thumbnail_cache_is_never_cataloged_inside_output(tmp_path):
    output = tmp_path / "output"
    _image(output / "generated.png", (1000, 500))
    catalog = ImageCatalog(output, output)
    record = catalog.list()[0][0]

    catalog.thumbnail(record.image_id, 128)
    records, _cursor = catalog.list()

    assert [item.relative_path for item in records] == ["generated.png"]


def test_image_catalog_extracts_generation_data_and_deletes_image(tmp_path):
    output = tmp_path / "output"
    expected = _image(output / "generated.png")
    catalog = ImageCatalog(output, tmp_path / "state")
    record = catalog.list()[0][0]
    thumbnail = catalog.thumbnail(record.image_id, 128)

    assert catalog.generation_data(record.image_id) == expected
    catalog.delete(record.image_id)

    assert not record.path.exists()
    assert not thumbnail.exists()
    try:
        catalog.get(record.image_id)
    except ImageNotFoundError:
        pass
    else:
        raise AssertionError("deleted image remained in the catalog")


def test_cached_image_get_avoids_a_full_catalog_rescan(tmp_path, monkeypatch):
    output = tmp_path / "output"
    _image(output / "generated.png")
    catalog = ImageCatalog(output, tmp_path / "state")
    record = catalog.list()[0][0]
    monkeypatch.setattr(
        catalog,
        "reconcile",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected full scan")),
    )

    assert catalog.get(record.image_id) == record


def test_http_api_exposes_status_images_metadata_and_deletion(tmp_path):
    output = tmp_path / "output"
    expected = _image(output / "generated.png", (1600, 800))
    runtime = RuntimeStatus("source")
    catalog = ImageCatalog(output, tmp_path / "state")

    with HttpApiServer("127.0.0.1", 0, catalog, runtime) as server:
        host, port = server.address
        base = f"http://{host}:{port}"

        code, _content_type, body = _request(base, "/health")
        assert code == HTTPStatus.SERVICE_UNAVAILABLE
        assert json.loads(body)["ready"] is False

        runtime.ready()
        runtime.set_queue(total=4, completed=3, pending=1)
        code, content_type, body = _request(base, "/health")
        assert (code, content_type) == (HTTPStatus.OK, "application/json")
        assert json.loads(body) == {"status": "ok", "ready": True}

        code, _content_type, body = _request(base, "/v1/status")
        status = json.loads(body)
        assert status["mode"] == "source"
        assert status["queue"]["pending"] == 1

        runtime.begin(
            {"source": "file", "file": "/data/prompts/a.txt", "line": 4},
            stage="sampling",
        )
        code, _content_type, body = _request(base, "/v1/status")
        status = json.loads(body)
        assert status["state"] == "running"
        assert status["stage"] == "sampling"
        assert status["current"]["line"] == 4
        runtime.fail(RuntimeError("worker failed"))
        code, _content_type, body = _request(base, "/health")
        assert code == HTTPStatus.SERVICE_UNAVAILABLE
        assert json.loads(body) == {"status": "degraded", "ready": False}
        runtime.ready()

        code, _content_type, body = _request(base, "/v1/images?limit=1")
        listing = json.loads(body)
        image = listing["images"][0]
        image_id = image["id"]
        assert image["path"] == "generated.png"

        code, content_type, body = _request(base, f"/v1/images/{image_id}")
        assert (code, content_type) == (HTTPStatus.OK, "image/png")
        assert body.startswith(b"\x89PNG")

        code, content_type, body = _request(
            base,
            f"/v1/images/{image_id}/thumbnail?max_side=256",
        )
        assert (code, content_type) == (HTTPStatus.OK, "image/webp")
        thumbnail = tmp_path / "thumbnail.webp"
        thumbnail.write_bytes(body)
        with Image.open(thumbnail) as opened:
            assert opened.size == (256, 128)

        code, _content_type, body = _request(
            base,
            f"/v1/images/{image_id}/generation-data",
        )
        assert code == HTTPStatus.OK
        assert json.loads(body) == expected

        code, _content_type, body = _request(
            base,
            f"/v1/images/{image_id}",
            method="DELETE",
        )
        assert (code, body) == (HTTPStatus.NO_CONTENT, b"")
        assert not (output / "generated.png").exists()

        code, _content_type, body = _request(base, f"/v1/images/{image_id}")
        assert code == HTTPStatus.NOT_FOUND
        assert json.loads(body)["error"]["code"] == "image_not_found"


def test_http_api_distinguishes_missing_and_invalid_generation_data(tmp_path):
    output = tmp_path / "output"
    _image(output / "plain.png", manifest=False)
    invalid = output / "invalid.png"
    info = PngInfo()
    info.add_text(metadata.PNG_KEY, "{}")
    Image.new("RGB", (32, 32)).save(invalid, pnginfo=info)
    runtime = RuntimeStatus("source")
    catalog = ImageCatalog(output, tmp_path / "state")

    with HttpApiServer("127.0.0.1", 0, catalog, runtime) as server:
        runtime.ready()
        host, port = server.address
        base = f"http://{host}:{port}"
        listing = json.loads(_request(base, "/v1/images")[2])
        image_ids = {item["path"]: item["id"] for item in listing["images"]}

        code, _content_type, body = _request(
            base,
            f"/v1/images/{image_ids['plain.png']}/generation-data",
        )
        assert code == HTTPStatus.NOT_FOUND
        assert json.loads(body)["error"]["code"] == "generation_data_not_found"

        code, _content_type, body = _request(
            base,
            f"/v1/images/{image_ids['invalid.png']}/generation-data",
        )
        assert code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert json.loads(body)["error"]["code"] == "invalid_generation_data"

        code, _content_type, body = _request(base, "/health?verbose=true")
        assert code == HTTPStatus.BAD_REQUEST
        assert json.loads(body)["error"]["code"] == "invalid_request"


def test_http_api_rejects_mutation_and_non_loopback_binding(tmp_path):
    runtime = RuntimeStatus("source")
    catalog = ImageCatalog(tmp_path / "output", tmp_path / "state")
    with HttpApiServer("127.0.0.1", 0, catalog, runtime) as server:
        host, port = server.address
        code, _content_type, body = _request(
            f"http://{host}:{port}",
            "/v1/jobs",
            method="POST",
        )
        assert code == HTTPStatus.METHOD_NOT_ALLOWED
        assert json.loads(body)["error"]["code"] == "method_not_allowed"

    try:
        with HttpApiServer("0.0.0.0", 0, catalog, runtime):
            pass
    except ValueError as exc:
        assert "loopback" in str(exc)
    else:
        raise AssertionError("non-loopback API binding was accepted")
