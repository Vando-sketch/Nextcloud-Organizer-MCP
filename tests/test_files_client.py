"""Unit tests for FilesService with WebDAV mocked via httpx.MockTransport."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from nextcloud_organizer_mcp.errors import (
    AuthenticationFailedError,
    CloudFileNotFoundError,
    ConnectionFailedError,
    FileAccessDeniedError,
    TaskMcpError,
    TransientServerError,
    UnsupportedFileError,
)
from nextcloud_organizer_mcp.files_client import (
    MAX_READ_BYTES,
    FilesService,
    normalize_path,
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _service(
    handler: Callable[[httpx.Request], httpx.Response],
    allowed: tuple[str, ...] = ("/Schule", "/Ausbildung/Berichte"),
    base_url: str = "https://cloud.example.com",
) -> FilesService:
    return FilesService(
        base_url=base_url,
        username="elias",
        password="p",
        allowed_paths=allowed,
        transport=httpx.MockTransport(handler),
    )


def _entry(href: str, *, folder: bool = False, size: int = 10, ctype: str = "text/markdown") -> str:
    resourcetype = "<d:collection/>" if folder else ""
    extra = "" if folder else f"<d:getcontenttype>{ctype}</d:getcontenttype>"
    return f"""
  <d:response>
    <d:href>{href}</d:href>
    <d:propstat>
      <d:prop>
        <d:getlastmodified>Wed, 11 Nov 2026 10:00:00 GMT</d:getlastmodified>
        <d:resourcetype>{resourcetype}</d:resourcetype>
        <oc:size>{size}</oc:size>
        {extra}
      </d:prop>
      <d:status>HTTP/1.1 200 OK</d:status>
    </d:propstat>
    <d:propstat>
      <d:prop><d:getcontentlength/></d:prop>
      <d:status>HTTP/1.1 404 Not Found</d:status>
    </d:propstat>
  </d:response>"""


def _multistatus(*entries: str) -> httpx.Response:
    body = (
        '<?xml version="1.0"?>'
        '<d:multistatus xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">'
        + "".join(entries)
        + "</d:multistatus>"
    )
    return httpx.Response(207, content=body.encode())


def _never(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
    raise AssertionError(f"unexpected request {request.method} {request.url}")


# --- path normalization / allow-list ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "/"),
        ("", "/"),
        ("Schule", "/Schule"),
        ("/Schule/", "/Schule"),
        ("//Schule//LF04", "/Schule/LF04"),
        ("/Schule/./LF04", "/Schule/LF04"),
        ("/Schule/../Privat", "/Privat"),
        ("\\Schule\\LF04", "/Schule/LF04"),
    ],
)
def test_normalize_path(raw, expected):
    assert normalize_path(raw) == expected


@pytest.mark.parametrize(
    "path",
    [
        "/Privat",
        "/Schule/../Privat/geheim.md",
        "/SchuleXYZ/datei.md",  # prefix match must respect folder boundaries
        "/Ausbildung",  # parent of an allowed folder
        "/",
    ],
)
def test_paths_outside_allow_list_are_refused_without_a_request(path):
    service = _service(_never)
    with pytest.raises(FileAccessDeniedError):
        _run(service.list_files(path))
    with pytest.raises(FileAccessDeniedError):
        _run(service.read_file(path))


def test_root_entry_allows_everything():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"hi", headers={"content-type": "text/plain"})

    service = _service(handler, allowed=("/",))
    assert _run(service.read_file("/Privat/x.txt"))["content"] == "hi"


# --- list_files ---


def test_list_files_without_path_returns_allowed_roots():
    service = _service(_never)
    result = _run(service.list_files())
    assert result["path"] is None
    assert [e["path"] for e in result["entries"]] == ["/Schule", "/Ausbildung/Berichte"]
    assert service.allowed_paths == ("/Schule", "/Ausbildung/Berichte")


def test_list_files_propfind_parses_entries_and_drops_self():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _multistatus(
            _entry("/remote.php/dav/files/elias/Schule/", folder=True, size=300),
            _entry("/remote.php/dav/files/elias/Schule/z%C3%BCrich.md", size=5),
            _entry("/remote.php/dav/files/elias/Schule/LF04/", folder=True, size=99),
            _entry("/somewhere/else/ignored.md"),
        )

    service = _service(handler)
    result = _run(service.list_files("Schule/"))

    request = captured[0]
    assert request.method == "PROPFIND"
    assert request.headers["Depth"] == "1"
    assert str(request.url) == "https://cloud.example.com/remote.php/dav/files/elias/Schule"
    assert result["path"] == "/Schule"
    assert result["entries"] == [
        {
            "path": "/Schule/LF04",
            "name": "LF04",
            "type": "folder",
            "size": 99,
            "content_type": None,
            "modified": "Wed, 11 Nov 2026 10:00:00 GMT",
        },
        {
            "path": "/Schule/zürich.md",
            "name": "zürich.md",
            "type": "file",
            "size": 5,
            "content_type": "text/markdown",
            "modified": "Wed, 11 Nov 2026 10:00:00 GMT",
        },
    ]


def test_list_files_handles_subdirectory_install_and_special_chars():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _multistatus(
            _entry("/nextcloud/remote.php/dav/files/elias/Schule/Lern%20Zettel/a.md"),
        )

    service = _service(handler, base_url="https://cloud.example.com/nextcloud/")
    result = _run(service.list_files("/Schule/Lern Zettel"))
    assert captured[0].url.raw_path == b"/nextcloud/remote.php/dav/files/elias/Schule/Lern%20Zettel"
    assert result["entries"][0]["path"] == "/Schule/Lern Zettel/a.md"


def test_list_files_tolerates_bad_size_and_missing_ok_propstat():
    body = """
  <d:response>
    <d:href>/remote.php/dav/files/elias/Schule/a.md</d:href>
    <d:propstat><d:prop><oc:size>abc</oc:size><d:resourcetype/></d:prop>
    <d:status>HTTP/1.1 200 OK</d:status></d:propstat>
  </d:response>
  <d:response>
    <d:href>/remote.php/dav/files/elias/Schule/b.md</d:href>
    <d:propstat><d:prop/><d:status>HTTP/1.1 404 Not Found</d:status></d:propstat>
  </d:response>
  <d:response><d:propstat/></d:response>"""

    service = _service(lambda request: _multistatus(body))
    entries = _run(service.list_files("/Schule"))["entries"]
    assert [(e["name"], e["size"]) for e in entries] == [("a.md", None)]


def test_unparseable_multistatus_is_a_clean_error():
    service = _service(lambda request: httpx.Response(207, content=b"<not xml"))
    with pytest.raises(TaskMcpError, match="unreadable"):
        _run(service.list_files("/Schule"))


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, AuthenticationFailedError),
        (404, CloudFileNotFoundError),
        (502, TransientServerError),
        (500, TaskMcpError),
    ],
)
def test_http_errors_are_translated(status, error):
    service = _service(lambda request: httpx.Response(status))
    with pytest.raises(error):
        _run(service.list_files("/Schule"))
    with pytest.raises(error):
        _run(service.read_file("/Schule/a.md"))


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (httpx.ConnectTimeout("t"), "timed out"),
        (httpx.ConnectError("c"), "connection failed"),
        (httpx.DecodingError("d"), "unexpected"),
    ],
)
def test_transport_errors_are_translated(exc, message):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    service = _service(handler)
    with pytest.raises(TaskMcpError, match=message):
        _run(service.list_files("/Schule"))
    with pytest.raises(TaskMcpError, match=message):
        _run(service.read_file("/Schule/a.md"))


def test_connection_errors_use_connection_failed_type():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(ConnectionFailedError):
        _run(_service(handler).list_files("/Schule"))


# --- read_file ---


def test_read_file_returns_text():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content="# RAID\nSpiegelung ä".encode(),
            headers={"content-type": "text/markdown; charset=utf-8"},
        )

    result = _run(_service(handler).read_file("/Schule/LF04/RAID.md"))
    assert captured[0].method == "GET"
    assert result == {
        "path": "/Schule/LF04/RAID.md",
        "content_type": "text/markdown",
        "content": "# RAID\nSpiegelung ä",
        "truncated": False,
    }


def test_read_file_empty_file():
    result = _run(_service(lambda r: httpx.Response(200, content=b"")).read_file("/Schule/a"))
    assert result["content"] == ""
    assert result["content_type"] is None


def test_read_file_truncates_and_keeps_utf8_boundaries():
    content = ("ä" * 10).encode()  # 2 bytes each

    result = _run(
        _service(lambda r: httpx.Response(200, content=content)).read_file("/Schule/a", 5)
    )
    assert result["truncated"] is True
    assert result["content"] == "ää"


def test_read_file_caps_max_bytes():
    content = b"a" * (MAX_READ_BYTES + 10)
    result = _run(
        _service(lambda r: httpx.Response(200, content=content)).read_file("/Schule/a", 10**9)
    )
    assert len(result["content"]) == MAX_READ_BYTES
    assert result["truncated"] is True


@pytest.mark.parametrize("content", [b"\x89PNG\r\n\x1a\n\x00\x00", b"\xff\xfe\xfa abc"])
def test_read_file_refuses_binary(content):
    service = _service(lambda r: httpx.Response(200, content=content))
    with pytest.raises(UnsupportedFileError):
        _run(service.read_file("/Schule/bild.jpg"))


def test_read_file_refuses_binary_even_when_truncated():
    service = _service(lambda r: httpx.Response(200, content=b"\xff" * 50))
    with pytest.raises(UnsupportedFileError):
        _run(service.read_file("/Schule/bild.jpg", 20))


def test_read_file_refuses_folder():
    service = _service(
        lambda r: httpx.Response(200, content=b"", headers={"content-type": "httpd/unix-directory"})
    )
    with pytest.raises(UnsupportedFileError, match="folder"):
        _run(service.read_file("/Schule/LF04"))


# --- search_files ---


def test_search_files_sends_basicsearch_per_root_and_filters():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        hits = [
            _entry("/remote.php/dav/files/elias/Schule/LF04/RAID.md"),
            _entry("/remote.php/dav/files/elias/Privat/RAID.md"),  # outside: dropped
        ]
        if "/files/elias/Schule<" in request.content.decode():
            hits.insert(0, _entry("/remote.php/dav/files/elias/Schule/", folder=True))
        return _multistatus(*hits)

    results = _run(_service(handler).search_files("RAID & Co"))

    assert [r.method for r in captured] == ["SEARCH", "SEARCH"]
    assert str(captured[0].url) == "https://cloud.example.com/remote.php/dav/"
    body = captured[0].content.decode()
    assert "<d:href>/files/elias/Schule</d:href>" in body
    assert "<d:literal>%RAID &amp; Co%</d:literal>" in body
    assert "<d:nresults>50</d:nresults>" in body
    assert "/files/elias/Ausbildung/Berichte" in captured[1].content.decode()
    # The scope folder itself is dropped; a hit returned twice is reported once.
    assert [r["path"] for r in results] == ["/Schule/LF04/RAID.md"]


def test_search_files_in_subfolder_and_limit():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _multistatus(
            *[_entry(f"/remote.php/dav/files/elias/Schule/LF04/{i}.md") for i in range(5)]
        )

    results = _run(_service(handler).search_files("md", path="/Schule/LF04", limit=3))
    assert len(captured) == 1
    assert "<d:nresults>3</d:nresults>" in captured[0].content.decode()
    assert len(results) == 3


def test_search_files_root_scope():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _multistatus()

    _run(_service(handler, allowed=("/",)).search_files("x"))
    assert "<d:href>/files/elias</d:href>" in captured[0].content.decode()


def test_search_files_validation():
    service = _service(_never)
    with pytest.raises(UnsupportedFileError):
        _run(service.search_files("  "))
    with pytest.raises(FileAccessDeniedError):
        _run(service.search_files("x", path="/Privat"))


def test_aclose():
    _run(_service(_never).aclose())
