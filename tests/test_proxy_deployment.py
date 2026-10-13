"""Real socket tests for Uvicorn trust and optional Caddy TLS termination.

Covered through the proxy: discovery, OAuth routing and consent, redirect
allow-list, MCP requests, incremental SSE delivery, backend restart and client
IP handling. See docs/deployment-verification.md for what these do not prove.

The Caddy tests run only with RUN_PROXY_TESTS=1 and CADDY_BIN pointing to a
locally installed binary (or ``caddy`` on PATH). They use an isolated static
certificate trusted by this test's HTTP client only and mock all Nextcloud I/O.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
import pytest
import uvicorn
from conftest import TEST_OAUTH_PASSWORD
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastmcp import Context, FastMCP
from mcp.client.auth.utils import build_oauth_authorization_server_metadata_discovery_urls
from test_oauth_e2e import (
    REDIRECT_URI,
    _authorize,
    _consent,
    _exchange_code,
    _pkce,
    _refresh,
    _register,
)

from nextcloud_organizer_mcp.caldav_client import CalDavService
from nextcloud_organizer_mcp.config import Settings
from nextcloud_organizer_mcp.personal_auth import CONSENT_MAX_FAILURES_PER_IP
from nextcloud_organizer_mcp.server import build_server

PROXY_TESTS_ENABLED = os.environ.get("RUN_PROXY_TESTS") == "1"
requires_caddy = pytest.mark.skipif(
    not PROXY_TESTS_ENABLED, reason="set RUN_PROXY_TESTS=1 to run local Caddy TLS tests"
)
TASK_LISTS = [{"name": "Proxy test", "url": "https://cloud.example.com/dav/proxy-test/"}]


@dataclass
class RunningBackend:
    base_url: str
    stop: Callable[[], None]


@dataclass
class ProxyDeployment:
    base_url: str
    tls_context: ssl.SSLContext
    service: MagicMock
    stop_backend: Callable[[], None]
    restart_backend: Callable[[], None]
    release: threading.Event
    tool_finished: threading.Event
    log_path: Path


def _start_backend(
    settings: Settings,
    service: MagicMock,
    public_base_url: str | None = None,
    port: int = 0,
    configure: Callable[[FastMCP], None] | None = None,
) -> RunningBackend:
    """Run real Uvicorn; leave its documented FORWARDED_ALLOW_IPS default intact."""
    sock = socket.socket()
    # A restart rebinds the port while connections from the previous run linger.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    base_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    local = replace(
        settings,
        public_base_url=public_base_url or base_url,
        host="127.0.0.1",
        oauth_allowed_redirect_domains=["claude.ai"],
    )
    mcp = build_server(local, service=service)
    if configure is not None:
        configure(mcp)
    server = uvicorn.Server(uvicorn.Config(mcp.http_app(), log_level="warning", access_log=False))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve([sock])), daemon=True)
    thread.start()

    def stop() -> None:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        assert not thread.is_alive(), "backend did not stop"

    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline, "backend did not start"
            time.sleep(0.02)
    except BaseException:
        stop()
        raise
    return RunningBackend(base_url, stop)


@contextmanager
def _backend(
    settings: Settings,
    service: MagicMock,
    public_base_url: str | None = None,
    configure: Callable[[FastMCP], None] | None = None,
) -> Iterator[RunningBackend]:
    backend = _start_backend(settings, service, public_base_url, configure=configure)
    try:
        yield backend
    finally:
        backend.stop()


def _certificate(tmp_path: Path) -> tuple[Path, Path]:
    """Make a short-lived localhost certificate without installing a trust root."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certificate_path = tmp_path / "localhost.pem"
    key_path = tmp_path / "localhost-key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    return certificate_path, key_path


def _caddy_binary() -> str:
    binary = shutil.which(os.environ.get("CADDY_BIN", "caddy"))
    assert binary is not None, "RUN_PROXY_TESTS=1 requires CADDY_BIN or caddy on PATH"
    return binary


def _free_port() -> int:
    # Caddy cannot adopt a pre-bound socket, so release it before starting the proxy.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _site_config(example: str, site: str, certificate: Path, key: Path) -> str:
    """Make a documented example Caddyfile serve `site` with the test certificate."""
    config = (Path(__file__).resolve().parents[1] / "examples" / example).read_text()
    block = f"{site} {{"
    assert block in config, f"examples/{example} must contain the documented site block"
    config = config.replace(block, f'{block}\n    bind 127.0.0.1\n    tls "{certificate}" "{key}"')
    return config.replace("{\n", "{\n    admin off\n    auto_https off\n", 1)


@contextmanager
def _caddy(
    binary: str,
    config: str,
    base_url: str,
    context: ssl.SSLContext,
    tmp_path: Path,
    ready_path: str = "/.well-known/oauth-authorization-server",
) -> Iterator[Path]:
    """Run Caddy with `config` until `ready_path` answers 200; yield its log path."""
    config_path = tmp_path / "Caddyfile"
    config_path.write_text(config)
    log_path = tmp_path / "caddy.log"
    with log_path.open("w+") as log:
        process = subprocess.Popen(
            [binary, "run", "--config", str(config_path), "--adapter", "caddyfile"],
            stdout=log,
            stderr=log,
            cwd=tmp_path,
            env={
                **os.environ,
                "XDG_DATA_HOME": str(tmp_path / "data"),
                "XDG_CONFIG_HOME": str(tmp_path / "config"),
            },
        )
        try:
            deadline = time.monotonic() + 10
            with httpx.Client(base_url=base_url, verify=context, trust_env=False) as http:
                while True:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        log.flush()
                        pytest.fail(f"Caddy did not start:\n{log_path.read_text()}")
                    try:
                        if http.get(ready_path, timeout=0.5).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.02)
            yield log_path
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


@pytest.fixture
def proxy_deployment(settings: Settings, tmp_path: Path, monkeypatch) -> Iterator[ProxyDeployment]:
    binary = _caddy_binary()
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "127.0.0.1")
    port = _free_port()
    base_url = f"https://localhost:{port}"
    certificate, key = _certificate(tmp_path)
    context = ssl.create_default_context(cafile=str(certificate))
    service = MagicMock(spec=CalDavService)
    service.list_task_lists.return_value = TASK_LISTS
    release = threading.Event()
    tool_finished = threading.Event()

    def add_hold_open_tool(mcp: FastMCP) -> None:
        # Test-only tool: emits one progress event, then stays open until the
        # test releases it, so the test controls when the response completes.
        @mcp.tool
        async def hold_open(ctx: Context) -> str:
            await ctx.report_progress(progress=1, total=2, message="started")
            await asyncio.to_thread(release.wait, 20)
            tool_finished.set()
            return "released"

    current: list[RunningBackend] = []

    def restart_backend() -> None:
        # Same port and OAuth state directory, as after `systemctl restart`.
        current.append(
            _start_backend(
                settings,
                service,
                base_url,
                port=urlparse(current[0].base_url).port or 0,
                configure=add_hold_open_tool,
            )
        )

    with _backend(settings, service, base_url, configure=add_hold_open_tool) as backend:
        current.append(backend)
        config = _site_config("Caddyfile", "organizer.example.com", certificate, key)
        config = config.replace("organizer.example.com", f"localhost:{port}")
        config = config.replace("127.0.0.1:8000", urlparse(backend.base_url).netloc)
        with _caddy(binary, config, base_url, context, tmp_path) as log_path:
            try:
                yield ProxyDeployment(
                    base_url,
                    context,
                    service,
                    backend.stop,
                    restart_backend,
                    release,
                    tool_finished,
                    log_path,
                )
            finally:
                for restarted in current[1:]:
                    restarted.stop()


def _exhaust_password_budget(http: httpx.Client, client_id: str) -> None:
    # Each attempt has a new authorization so the per-key limit cannot mask
    # whether the actual per-IP budget is enforced.
    for _ in range(CONSENT_MAX_FAILURES_PER_IP):
        _, challenge = _pkce()
        pending = _authorize(http, client_id, challenge)
        assert _consent(http, pending, "deliberately-wrong").status_code == 401


@pytest.mark.parametrize("trust_proxy", [True, False])
def test_uvicorn_only_trusts_forwarded_client_ips_from_configured_peers(
    settings: Settings, monkeypatch, trust_proxy: bool
):
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "127.0.0.1" if trust_proxy else "192.0.2.123")
    service = MagicMock(spec=CalDavService)
    with _backend(settings, service) as backend:
        with (
            httpx.Client(
                base_url=backend.base_url,
                headers={"X-Forwarded-For": "198.51.100.10"},
                trust_env=False,
            ) as first,
            httpx.Client(
                base_url=backend.base_url,
                headers={"X-Forwarded-For": "198.51.100.11", "X-Real-IP": "198.51.100.12"},
                trust_env=False,
            ) as second,
        ):
            client_id = _register(first)
            _exhaust_password_budget(first, client_id)
            _, challenge = _pkce()
            pending = _authorize(first, client_id, challenge)
            assert _consent(first, pending, TEST_OAUTH_PASSWORD).status_code == 429
            # Trusted Uvicorn forwarding separates clients. Without that trust,
            # forged headers cannot change the socket peer's exhausted budget.
            response = _consent(second, pending, TEST_OAUTH_PASSWORD)
            assert response.status_code == (302 if trust_proxy else 429)


def _rpc(http: httpx.Client, message: dict[str, Any]) -> dict[str, Any]:
    """Read a genuine Streamable HTTP SSE response through Caddy."""
    with http.stream("POST", "/mcp", json=message) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        if "mcp-session-id" in response.headers:
            http.headers["Mcp-Session-Id"] = response.headers["mcp-session-id"]
        for line in response.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event.get("id") == message["id"]:
                    assert "error" not in event, event
                    return event["result"]
    raise AssertionError("MCP stream ended without the requested JSON-RPC result")


@requires_caddy
def test_caddy_https_discovery_consent_and_streamed_read_only_mcp_call(proxy_deployment):
    deployment: ProxyDeployment = proxy_deployment
    with httpx.Client(
        base_url=deployment.base_url,
        verify=deployment.tls_context,
        trust_env=False,
        timeout=10,
    ) as http:
        meta = http.get("/.well-known/oauth-authorization-server").json()
        assert meta["issuer"] == deployment.base_url + "/"
        for field, path in (
            ("authorization_endpoint", "/authorize"),
            ("registration_endpoint", "/register"),
            ("token_endpoint", "/token"),
        ):
            assert meta[field] == deployment.base_url + path
        resource = http.get("/.well-known/oauth-protected-resource/mcp")
        assert resource.status_code == 200
        assert resource.json()["resource"] == deployment.base_url + "/mcp"
        assert resource.json()["authorization_servers"] == [deployment.base_url + "/"]
        challenge_response = http.post("/mcp", json={})
        assert challenge_response.status_code == 401
        assert (
            f'resource_metadata="{deployment.base_url}/.well-known/oauth-protected-resource/mcp"'
            in challenge_response.headers["www-authenticate"]
        )

        client_id = _register(http)
        verifier, challenge = _pkce()
        pending = _authorize(http, client_id, challenge)
        consent_form = http.get("/consent", params={"pending": pending})
        assert consent_form.status_code == 200
        assert 'type="password"' in consent_form.text
        assert consent_form.headers["cache-control"] == "no-store"
        assert _consent(http, pending, "deliberately-wrong").status_code == 401
        consent = _consent(http, pending, TEST_OAUTH_PASSWORD)
        assert consent.status_code == 302
        redirect = urlparse(consent.headers["location"])
        assert f"{redirect.scheme}://{redirect.netloc}{redirect.path}" == REDIRECT_URI
        query = parse_qs(redirect.query)
        assert query["state"] == ["csrf-state"]
        token = _exchange_code(http, client_id, query["code"][0], verifier)
        assert token.status_code == 200
        refreshed = _refresh(http, client_id, token.json()["refresh_token"])
        assert refreshed.status_code == 200
        http.headers.update(
            {
                "Authorization": f"Bearer {refreshed.json()['access_token']}",
                "Accept": "application/json, text/event-stream",
            }
        )
        initialized = _rpc(
            http,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "proxy-test", "version": "1"},
                },
            },
        )
        http.headers["Mcp-Protocol-Version"] = initialized["protocolVersion"]
        notification = http.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        assert notification.status_code == 202
        tools = _rpc(http, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tool = next(tool for tool in tools["tools"] if tool["name"] == "list_task_lists")
        assert tool["annotations"]["readOnlyHint"] is True
        result = _rpc(
            http,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "list_task_lists", "arguments": {}},
            },
        )
        assert not result.get("isError", False)
        text = "".join(item["text"] for item in result["content"] if item["type"] == "text")
        assert json.loads(text) == TASK_LISTS
        deployment.service.list_task_lists.assert_called_once_with()


@requires_caddy
def test_caddy_discards_spoofed_client_ip_headers(proxy_deployment):
    deployment: ProxyDeployment = proxy_deployment
    with httpx.Client(
        base_url=deployment.base_url,
        verify=deployment.tls_context,
        trust_env=False,
    ) as http:
        client_id = _register(http)
        for attempt in range(CONSENT_MAX_FAILURES_PER_IP):
            http.headers.update(
                {
                    "X-Forwarded-For": f"198.51.100.{attempt + 1}",
                    "X-Real-IP": f"198.51.100.{attempt + 1}",
                    "Forwarded": f"for=198.51.100.{attempt + 1};proto=https",
                }
            )
            _, challenge = _pkce()
            pending = _authorize(http, client_id, challenge)
            assert _consent(http, pending, "deliberately-wrong").status_code == 401
        http.headers["X-Forwarded-For"] = "203.0.113.200"
        _, challenge = _pkce()
        pending = _authorize(http, client_id, challenge)
        assert _consent(http, pending, TEST_OAUTH_PASSWORD).status_code == 429
        deployment.service.list_task_lists.assert_not_called()


@requires_caddy
def test_caddy_upstream_failure_logs_redact_oauth_urls(proxy_deployment):
    deployment: ProxyDeployment = proxy_deployment
    # Exercise Caddy's default runtime/error logger, which records requests on
    # upstream failures even when no access log is configured for the site.
    deployment.stop_backend()
    pending_marker = "disposable-pending-marker"
    referer_marker = "disposable-referrer-marker"
    with httpx.Client(
        base_url=deployment.base_url,
        verify=deployment.tls_context,
        trust_env=False,
        timeout=10,
    ) as http:
        response = http.get(
            "/consent",
            params={"pending": pending_marker},
            headers={"Referer": f"{deployment.base_url}/consent?pending={referer_marker}"},
        )
    assert response.status_code == 502
    deadline = time.monotonic() + 2
    while True:
        logs = deployment.log_path.read_text()
        if '"logger":"http.log.error"' in logs:
            break
        assert time.monotonic() < deadline, "Caddy did not emit its upstream failure diagnostic"
        time.sleep(0.02)
    assert pending_marker not in logs
    assert referer_marker not in logs
    errors = [json.loads(line) for line in logs.splitlines() if '"logger":"http.log.error"' in line]
    assert any(error["status"] == 502 for error in errors)
    for error in errors:
        assert "uri" not in error["request"]
        assert "Referer" not in error["request"]["headers"]


def _issue_token(http: httpx.Client) -> tuple[str, dict[str, Any]]:
    client_id = _register(http)
    verifier, challenge = _pkce()
    pending = _authorize(http, client_id, challenge)
    consent = _consent(http, pending, TEST_OAUTH_PASSWORD)
    assert consent.status_code == 302
    code = parse_qs(urlparse(consent.headers["location"]).query)["code"][0]
    token = _exchange_code(http, client_id, code, verifier)
    assert token.status_code == 200
    return client_id, token.json()


def _open_mcp_session(http: httpx.Client, access_token: str) -> None:
    http.headers.update(
        {"Authorization": f"Bearer {access_token}", "Accept": "application/json, text/event-stream"}
    )
    http.headers.pop("Mcp-Session-Id", None)
    initialized = _rpc(
        http,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "proxy-test", "version": "1"},
            },
        },
    )
    http.headers["Mcp-Protocol-Version"] = initialized["protocolVersion"]
    notification = http.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert notification.status_code == 202


def _events(response: httpx.Response) -> Iterator[dict[str, Any]]:
    for line in response.iter_lines():
        if line.startswith("data: "):
            yield json.loads(line[6:])


@requires_caddy
def test_caddy_delivers_sse_events_before_the_response_completes(proxy_deployment):
    deployment: ProxyDeployment = proxy_deployment
    with httpx.Client(
        base_url=deployment.base_url,
        verify=deployment.tls_context,
        trust_env=False,
        timeout=10,
    ) as http:
        _, token = _issue_token(http)
        _open_mcp_session(http, token["access_token"])
        call = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "hold_open",
                "arguments": {},
                "_meta": {"progressToken": "proxy-progress"},
            },
        }
        with http.stream("POST", "/mcp", json=call) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = _events(response)
            # The tool is still blocked, so this event can only arrive if the
            # proxy forwards SSE incrementally instead of buffering the response.
            assert next(events)["method"] == "notifications/progress"
            assert not deployment.tool_finished.is_set()
            # An idle stream must survive a pause without being closed or flushed.
            time.sleep(2)
            assert not deployment.tool_finished.is_set()
            deployment.release.set()
            final = next(event for event in events if event.get("id") == 7)
        assert deployment.tool_finished.is_set()
        assert "error" not in final, final


@requires_caddy
def test_caddy_opens_the_standalone_event_stream(proxy_deployment):
    deployment: ProxyDeployment = proxy_deployment
    with httpx.Client(
        base_url=deployment.base_url,
        verify=deployment.tls_context,
        trust_env=False,
        timeout=5,
    ) as http:
        _, token = _issue_token(http)
        _open_mcp_session(http, token["access_token"])
        with http.stream("GET", "/mcp", headers={"Accept": "text/event-stream"}) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")


@requires_caddy
def test_caddy_rejects_redirects_outside_the_allow_list(proxy_deployment):
    deployment: ProxyDeployment = proxy_deployment
    with httpx.Client(
        base_url=deployment.base_url,
        verify=deployment.tls_context,
        trust_env=False,
        timeout=10,
    ) as http:
        evil = "https://attacker.example.net/callback"
        registration = http.post(
            "/register",
            json={
                "client_name": "attacker",
                "redirect_uris": [evil],
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
            },
        )
        assert registration.status_code == 201, registration.text
        _, challenge = _pkce()
        params = {
            "redirect_uri": evil,
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        # Registration accepts any URI; the allow-list gates /authorize, so no
        # consent page and no code may be issued for a disallowed domain.
        denied = http.get(
            "/authorize", params={**params, "client_id": registration.json()["client_id"]}
        )
        assert denied.status_code == 302, denied.text
        location = urlparse(denied.headers["location"])
        assert location.path != "/consent"
        assert parse_qs(location.query)["error"] == ["access_denied"]
        assert "code" not in parse_qs(location.query)
        # A registered client cannot switch to a redirect it never registered.
        unregistered = http.get("/authorize", params={**params, "client_id": _register(http)})
        assert unregistered.status_code == 400, unregistered.text
        assert "location" not in unregistered.headers


@requires_caddy
def test_caddy_reaches_a_restarted_backend_and_keeps_issued_tokens(proxy_deployment):
    deployment: ProxyDeployment = proxy_deployment
    with httpx.Client(
        base_url=deployment.base_url,
        verify=deployment.tls_context,
        trust_env=False,
        timeout=10,
    ) as http:
        _, token = _issue_token(http)
        _open_mcp_session(http, token["access_token"])
        stale_session = http.headers["Mcp-Session-Id"]
        deployment.stop_backend()
        assert http.get("/.well-known/oauth-authorization-server").status_code == 502
        deployment.restart_backend()
        # Persisted token, but the in-memory MCP session is gone.
        stale = http.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        )
        assert stale.status_code == 404
        _open_mcp_session(http, token["access_token"])
        assert http.headers["Mcp-Session-Id"] != stale_session
        result = _rpc(
            http,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "list_task_lists", "arguments": {}},
            },
        )
        assert not result.get("isError", False)
        deployment.service.list_task_lists.assert_called_once_with()


@contextmanager
def _placeholder_site() -> Iterator[str]:
    """Stand in for the Nextcloud upstream that owns the rest of the domain."""

    class Handler(BaseHTTPRequestHandler):
        def _answer(self) -> None:
            body = b"placeholder nextcloud"
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _answer

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@requires_caddy
def test_caddy_subpath_example_completes_oauth_and_a_read_only_call(
    settings: Settings, tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "127.0.0.1")
    binary = _caddy_binary()
    port = _free_port()
    origin = f"https://localhost:{port}"
    base = f"{origin}/organizer"
    certificate, key = _certificate(tmp_path)
    context = ssl.create_default_context(cafile=str(certificate))
    service = MagicMock(spec=CalDavService)
    service.list_task_lists.return_value = TASK_LISTS
    with _placeholder_site() as nextcloud, _backend(settings, service, base) as backend:
        config = _site_config("Caddyfile.subpath", "cloud.example.com", certificate, key)
        config = config.replace("cloud.example.com", f"localhost:{port}")
        config = config.replace("127.0.0.1:8000", urlparse(backend.base_url).netloc)
        config = config.replace("127.0.0.1:8080", nextcloud)
        ready = "/.well-known/oauth-authorization-server/organizer"
        with (
            _caddy(binary, config, origin, context, tmp_path, ready),
            httpx.Client(verify=context, trust_env=False, timeout=10) as http,
        ):
            # Discovery as the MCP SDK client derives it from the challenge.
            challenge = http.post(f"{base}/mcp", json={})
            assert challenge.status_code == 401
            match = re.search(r'resource_metadata="([^"]+)"', challenge.headers["www-authenticate"])
            assert match is not None
            resource_url = match.group(1)
            assert resource_url == f"{origin}/.well-known/oauth-protected-resource/organizer/mcp"
            resource = http.get(resource_url).json()
            assert resource["resource"] == f"{base}/mcp"
            assert resource["authorization_servers"] == [base]
            metadata_url = build_oauth_authorization_server_metadata_discovery_urls(
                base, f"{base}/mcp"
            )[0]
            meta = http.get(metadata_url).json()
            assert meta["issuer"] == base

            registered = http.post(
                meta["registration_endpoint"],
                json={
                    "client_name": "subpath-proxy",
                    "redirect_uris": [REDIRECT_URI],
                    "token_endpoint_auth_method": "none",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                },
            )
            assert registered.status_code == 201, registered.text
            client_id = registered.json()["client_id"]
            verifier, code_challenge = _pkce()
            authorized = http.get(
                meta["authorization_endpoint"],
                params={
                    "client_id": client_id,
                    "redirect_uri": REDIRECT_URI,
                    "response_type": "code",
                    "code_challenge": code_challenge,
                    "code_challenge_method": "S256",
                    "state": "csrf-state",
                },
            )
            assert authorized.status_code == 302
            consent_url = authorized.headers["location"]
            assert consent_url.startswith(f"{base}/consent?pending=")
            page = http.get(consent_url)
            assert page.status_code == 200
            action = re.search(r'<form method="POST" action="([^"]+)"', page.text)
            assert action is not None
            submit_url = urljoin(consent_url, action.group(1))
            assert submit_url == f"{base}/consent"
            pending = parse_qs(urlparse(consent_url).query)["pending"][0]
            consent = http.post(
                submit_url, data={"pending": pending, "password": TEST_OAUTH_PASSWORD}
            )
            assert consent.status_code == 302, consent.text
            code = parse_qs(urlparse(consent.headers["location"]).query)["code"][0]
            token = http.post(
                meta["token_endpoint"],
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": REDIRECT_URI,
                    "client_id": client_id,
                    "code_verifier": verifier,
                },
            )
            assert token.status_code == 200, token.text

            # The rest of the domain still belongs to the other site.
            assert http.get(f"{origin}/status.php").text == "placeholder nextcloud"
            assert http.post(f"{origin}/consent").text == "placeholder nextcloud"

            with httpx.Client(base_url=base, verify=context, trust_env=False, timeout=10) as mcp:
                _open_mcp_session(mcp, token.json()["access_token"])
                result = _rpc(
                    mcp,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": "list_task_lists", "arguments": {}},
                    },
                )
            assert not result.get("isError", False)
            text = "".join(item["text"] for item in result["content"] if item["type"] == "text")
            assert json.loads(text) == TASK_LISTS
