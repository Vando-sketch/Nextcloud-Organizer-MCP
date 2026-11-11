"""Read-only access to Nextcloud Files over WebDAV, limited to an allow-list.

Talks to `<base>/remote.php/dav/files/<user>/` with `httpx`, like
notes_client.py talks to the Notes REST API. Every path a caller passes is
normalized first and must lie inside one of the configured allowed folders
(NEXTCLOUD_FILES_ALLOWED_PATHS); anything else is refused before a request is
sent. There are deliberately no write operations here.
"""

from __future__ import annotations

import logging
import posixpath
from typing import Any
from urllib.parse import quote, unquote, urlparse
from xml.etree import ElementTree
from xml.sax.saxutils import escape

import httpx

from .errors import (
    AuthenticationFailedError,
    CloudFileNotFoundError,
    ConnectionFailedError,
    FileAccessDeniedError,
    TaskMcpError,
    TransientServerError,
    UnsupportedFileError,
)

logger = logging.getLogger(__name__)

_DAV = "{DAV:}"
_OC = "{http://owncloud.org/ns}"

_PROPS = """
    <d:getlastmodified/>
    <d:getcontentlength/>
    <d:getcontenttype/>
    <d:resourcetype/>
    <oc:size/>
"""

_PROPFIND_BODY = f"""<?xml version="1.0" encoding="utf-8"?>
<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
  <d:prop>{_PROPS}</d:prop>
</d:propfind>"""

#: Hard ceiling for read_file, whatever the caller asks for.
MAX_READ_BYTES = 1_000_000
#: Default read_file size when the caller does not pass one.
DEFAULT_READ_BYTES = 200_000
#: Hard ceiling for search_files results per allowed folder.
MAX_SEARCH_RESULTS = 200


def _translate(exc: Exception) -> TaskMcpError:
    """Turn an httpx transport exception into a clean TaskMcpError (no raw text)."""
    if isinstance(exc, httpx.TimeoutException):
        return ConnectionFailedError("Could not reach Nextcloud Files (request timed out).")
    if isinstance(exc, httpx.TransportError):
        return ConnectionFailedError("Could not reach Nextcloud Files (connection failed).")
    logger.warning("Unexpected error talking to Nextcloud Files", exc_info=exc)
    return TaskMcpError("An unexpected error occurred talking to Nextcloud Files.")


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 401:
        raise AuthenticationFailedError(
            "Nextcloud rejected the Files credentials (check username/app password)."
        )
    if response.status_code == 404:
        raise CloudFileNotFoundError("The requested file or folder was not found.")
    if response.status_code in (502, 503, 504):
        raise TransientServerError(
            "Nextcloud Files gave no answer (gateway error) - try again shortly."
        )
    if response.is_error:
        logger.warning("Files request failed: %s", response.status_code)
        raise TaskMcpError("The Nextcloud Files request failed on the server.")


def _decode_utf8(data: bytes, truncated: bool) -> str | None:
    """Strict UTF-8 decode; None if `data` is not UTF-8 text.

    A truncated read can end inside a multi-byte character, so up to three
    trailing bytes may be dropped - but only then, and only at the very end.
    """
    for cut in range(4 if truncated else 1):
        try:
            return data[: len(data) - cut].decode("utf-8")
        except UnicodeDecodeError:
            continue
    return None


def normalize_path(path: str | None) -> str:
    """Absolute, normalized form of a caller-supplied path ("" and None mean "/")."""
    cleaned = (path or "").strip().replace("\\", "/")
    normalized = posixpath.normpath("/" + cleaned)
    # normpath keeps a leading "//" as-is (POSIX allows it to be special).
    return "/" + normalized.lstrip("/")


class FilesService:
    """One reused HTTP client plus list/read/search over the allowed folders."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        allowed_paths: tuple[str, ...],
        timeout: int = 30,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._username = username
        self._allowed = tuple(allowed_paths)
        dav_root = f"{base_url.rstrip('/')}/remote.php/dav/"
        # Path part of the files root, to turn returned hrefs back into paths.
        self._files_prefix = urlparse(dav_root).path + f"files/{quote(username, safe='')}"
        self._client = httpx.AsyncClient(
            base_url=dav_root,
            auth=(username, password),
            timeout=timeout,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def allowed_paths(self) -> tuple[str, ...]:
        return self._allowed

    # --- path handling ---

    def _is_allowed(self, path: str) -> bool:
        return any(
            root == "/" or path == root or path.startswith(root + "/") for root in self._allowed
        )

    def _check(self, path: str | None) -> str:
        normalized = normalize_path(path)
        if not self._is_allowed(normalized):
            raise FileAccessDeniedError(
                f"Path {normalized!r} is outside the allowed folders: {', '.join(self._allowed)}."
            )
        return normalized

    def _url(self, path: str) -> str:
        return f"files/{quote(self._username, safe='')}{quote(path)}"

    def _href_to_path(self, href: str) -> str | None:
        decoded = unquote(urlparse(href).path)
        prefix = unquote(self._files_prefix)
        if not decoded.startswith(prefix):
            return None
        return normalize_path(decoded[len(prefix) :])

    # --- requests ---

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise _translate(exc) from exc
        _raise_for_status(response)
        return response

    def _parse_multistatus(self, body: bytes) -> list[dict[str, Any]]:
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as exc:
            raise TaskMcpError("Nextcloud Files returned an unreadable response.") from exc
        entries: list[dict[str, Any]] = []
        for response in root.iter(f"{_DAV}response"):
            href = response.findtext(f"{_DAV}href")
            path = self._href_to_path(href) if href else None
            if path is None:
                continue
            prop: ElementTree.Element | None = None
            for propstat in response.iter(f"{_DAV}propstat"):
                status = propstat.findtext(f"{_DAV}status") or ""
                if " 200 " in f"{status} ":
                    prop = propstat.find(f"{_DAV}prop")
                    break
            if prop is None:
                continue
            resourcetype = prop.find(f"{_DAV}resourcetype")
            is_dir = resourcetype is not None and resourcetype.find(f"{_DAV}collection") is not None
            size_text = prop.findtext(f"{_OC}size") or prop.findtext(f"{_DAV}getcontentlength")
            try:
                size = int(size_text) if size_text else None
            except ValueError:
                size = None
            entries.append(
                {
                    "path": path,
                    "name": posixpath.basename(path) or "/",
                    "type": "folder" if is_dir else "file",
                    "size": size,
                    "content_type": None if is_dir else prop.findtext(f"{_DAV}getcontenttype"),
                    "modified": prop.findtext(f"{_DAV}getlastmodified"),
                }
            )
        return entries

    # --- public operations ---

    async def list_files(self, path: str | None = None) -> dict[str, Any]:
        """Entries directly inside `path`; with no path, the allowed folders themselves."""
        if path is None or not path.strip():
            return {
                "path": None,
                "entries": [
                    {"path": root, "name": posixpath.basename(root) or "/", "type": "folder"}
                    for root in self._allowed
                ],
            }
        folder = self._check(path)
        response = await self._request(
            "PROPFIND",
            self._url(folder),
            headers={"Depth": "1", "Content-Type": "application/xml; charset=utf-8"},
            content=_PROPFIND_BODY.encode(),
        )
        entries = [e for e in self._parse_multistatus(response.content) if e["path"] != folder]
        entries.sort(key=lambda e: (e["type"] != "folder", e["name"].casefold()))
        return {"path": folder, "entries": entries}

    async def read_file(self, path: str, max_bytes: int | None = None) -> dict[str, Any]:
        """Text content of one file, cut at `max_bytes` (UTF-8 text only)."""
        file_path = self._check(path)
        limit = DEFAULT_READ_BYTES if max_bytes is None else max_bytes
        limit = max(1, min(limit, MAX_READ_BYTES))
        try:
            async with self._client.stream("GET", self._url(file_path)) as response:
                if response.is_error:
                    await response.aread()
                    _raise_for_status(response)
                content_type = response.headers.get("content-type", "")
                if content_type.startswith("httpd/unix-directory"):
                    raise UnsupportedFileError(f"{file_path!r} is a folder - use list_files.")
                chunks: list[bytes] = []
                received = 0
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                    received += len(chunk)
                    if received > limit:
                        break
        except httpx.HTTPError as exc:
            raise _translate(exc) from exc
        data = b"".join(chunks)
        truncated = len(data) > limit
        data = data[:limit]
        if b"\x00" in data:
            raise UnsupportedFileError(f"{file_path!r} is not a text file.")
        text = _decode_utf8(data, truncated)
        if text is None:
            raise UnsupportedFileError(f"{file_path!r} is not a UTF-8 text file.")
        return {
            "path": file_path,
            "content_type": content_type.split(";")[0].strip() or None,
            "content": text,
            "truncated": truncated,
        }

    async def search_files(
        self, name: str, path: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Files and folders whose name contains `name`, via Nextcloud's WebDAV SEARCH."""
        term = name.strip()
        if not term:
            raise UnsupportedFileError("search_files needs a non-empty name to search for.")
        scopes = [self._check(path)] if path and path.strip() else list(self._allowed)
        limit = max(1, min(limit, MAX_SEARCH_RESULTS))
        results: list[dict[str, Any]] = []
        for scope in scopes:
            body = f"""<?xml version="1.0" encoding="UTF-8"?>
<d:searchrequest xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
  <d:basicsearch>
    <d:select><d:prop>{_PROPS}</d:prop></d:select>
    <d:from><d:scope>
      <d:href>/files/{escape(self._username)}{escape(scope if scope != "/" else "")}</d:href>
      <d:depth>infinity</d:depth>
    </d:scope></d:from>
    <d:where><d:like>
      <d:prop><d:displayname/></d:prop>
      <d:literal>%{escape(term)}%</d:literal>
    </d:like></d:where>
    <d:orderby/>
    <d:limit><d:nresults>{limit}</d:nresults></d:limit>
  </d:basicsearch>
</d:searchrequest>"""
            response = await self._request(
                "SEARCH",
                "",
                headers={"Content-Type": "text/xml; charset=utf-8"},
                content=body.encode(),
            )
            for entry in self._parse_multistatus(response.content):
                # Defence in depth: never hand back anything outside the list.
                if entry["path"] != scope and self._is_allowed(entry["path"]):
                    results.append(entry)
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for entry in results:
            if entry["path"] not in seen:
                seen.add(entry["path"])
                unique.append(entry)
        return unique[:limit]
