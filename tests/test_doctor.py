"""Deployment diagnostics (`nextcloud-organizer-mcp-doctor`, #74).

The healthy cases run against the real app over real HTTP, at the root and
behind a prefix-stripping proxy. The broken deployments are simulated with
`httpx.MockTransport`, each modelled on what the misconfigured proxy, tunnel
or server actually answers.
"""

from __future__ import annotations

import socket
import ssl
from collections.abc import Callable
from unittest.mock import MagicMock

import httpx
import pytest
from test_oauth_e2e import running_prefixed_server, running_server

from nextcloud_organizer_mcp import doctor
from nextcloud_organizer_mcp.caldav_client import CalDavService
from nextcloud_organizer_mcp.doctor import FAIL, OK, WARN, Check, parse_target, run_checks

BASE = "https://organizer.example.com"
PREFIXED = "https://cloud.example.com/organizer"
SECRET_MARKER = "do-not-print-this-body"


def _public(_host: str) -> list[str]:
    return ["93.184.215.14"]


def _json(data: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=data)


def _healthy(base: str) -> dict[str, httpx.Response]:
    """What a correctly routed server at `base` answers, keyed by path."""
    origin = "/".join(base.split("/")[:3])
    prefix = base[len(origin) :]
    meta = {
        "issuer": base,
        "authorization_endpoint": f"{base}/authorize",
        "token_endpoint": f"{base}/token",
        "registration_endpoint": f"{base}/register",
    }
    prm_path = f"/.well-known/oauth-protected-resource{prefix}/mcp"
    as_path = f"/.well-known/oauth-authorization-server{prefix}"
    return {
        f"{prefix}/mcp": httpx.Response(
            401, headers={"WWW-Authenticate": f'Bearer resource_metadata="{origin}{prm_path}"'}
        ),
        prm_path: _json({"resource": f"{base}/mcp", "authorization_servers": [base]}),
        as_path: _json(meta),
        f"{prefix}/register": httpx.Response(405, text="Method Not Allowed"),
        f"{prefix}/token": httpx.Response(405, text="Method Not Allowed"),
        f"{prefix}/authorize": _json({"error": "invalid_request"}, 400),
        f"{prefix}/consent": httpx.Response(400, html="<p>Authorization failed</p>"),
    }


def _nextcloud_404() -> httpx.Response:
    return httpx.Response(
        404,
        html=f"<title>Nextcloud</title>{SECRET_MARKER}",
        headers={"Set-Cookie": "nc_sameSiteCookielax=true; path=/"},
    )


def _client(routes: dict[str, httpx.Response]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return routes.get(request.url.path) or httpx.Response(404, text="not found")

    return httpx.Client(transport=httpx.MockTransport(handler))


def _run(
    url: str,
    routes: dict[str, httpx.Response],
    resolve: Callable[[str], list[str]] = _public,
) -> list[Check]:
    with _client(routes) as client:
        return run_checks(url, client, resolve).checks


def _by_name(checks: list[Check]) -> dict[str, Check]:
    return {check.name: check for check in checks}


@pytest.mark.parametrize(
    ("url", "base", "mcp"),
    [
        (BASE, BASE, f"{BASE}/mcp"),
        (f"{BASE}/", BASE, f"{BASE}/mcp"),
        (f"{BASE}/mcp", BASE, f"{BASE}/mcp"),
        (f"{PREFIXED}/mcp/", PREFIXED, f"{PREFIXED}/mcp"),
    ],
)
def test_parse_target_accepts_base_or_connector_url(url, base, mcp):
    target = parse_target(url)
    assert (target.base, target.mcp) == (base, mcp)


@pytest.mark.parametrize("url", ["organizer.example.com", "ftp://x.example", f"{BASE}/?a=1"])
def test_unusable_urls_fail_before_any_request(url):
    checks = _run(url, {})
    assert [(check.name, check.status) for check in checks] == [("URL", FAIL)]


def test_plain_http_is_rejected_for_a_public_host():
    checks = _run("http://organizer.example.com", _healthy("http://organizer.example.com"))
    assert [(check.name, check.status) for check in checks] == [("URL", FAIL)]
    assert "HTTPS" in (checks[0].hint or "")


@pytest.mark.parametrize("base", [BASE, PREFIXED])
def test_healthy_deployment_passes_every_check(base):
    checks = _run(base, _healthy(base))
    assert all(check.status == OK for check in checks), checks
    assert [check.name for check in checks] == [
        "URL",
        "DNS",
        "TLS",
        "MCP endpoint",
        "Protected resource metadata",
        "Authorization server metadata",
        "OAuth endpoints",
    ]


def test_real_root_deployment_passes(settings):
    with running_server(settings) as base, httpx.Client() as client:
        report = run_checks(base, client)
    assert not report.failed, report.checks
    assert all(check.status == OK for check in report.checks)


def test_real_subpath_deployment_passes(settings):
    with (
        running_prefixed_server(settings, MagicMock(spec=CalDavService)) as base,
        httpx.Client() as client,
    ):
        report = run_checks(f"{base}/mcp", client)
    assert all(check.status == OK for check in report.checks), report.checks


def test_dns_failure_stops_the_checks():
    def unresolvable(host: str) -> list[str]:
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    checks = _run(BASE, _healthy(BASE), unresolvable)
    assert checks[-1].name == "DNS" and checks[-1].status == FAIL


def test_private_dns_answers_only_warn():
    checks = _by_name(_run(BASE, _healthy(BASE), lambda host: ["192.168.1.20", "100.101.102.103"]))
    assert checks["DNS"].status == WARN
    assert checks["OAuth endpoints"].status == OK


def test_certificate_errors_are_reported_as_tls():
    def handler(request: httpx.Request) -> httpx.Response:
        error = ssl.SSLCertVerificationError("certificate verify failed: hostname mismatch")
        raise httpx.ConnectError("handshake failed") from error

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        checks = run_checks(BASE, client, _public).checks
    assert (checks[-1].name, checks[-1].status) == ("TLS", FAIL)
    assert "hostname mismatch" in checks[-1].detail


def test_unreachable_backend_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        checks = run_checks(BASE, client, _public).checks
    assert (checks[-1].name, checks[-1].status) == ("Reachability", FAIL)


def test_localhost_public_base_url_is_detected():
    # PUBLIC_BASE_URL left at the local default behind a proxy.
    routes = _healthy(BASE)
    routes["/mcp"] = httpx.Response(
        401,
        headers={
            "WWW-Authenticate": 'Bearer resource_metadata="http://127.0.0.1:8000/'
            '.well-known/oauth-protected-resource/mcp"'
        },
    )
    check = _by_name(_run(BASE, routes))["MCP endpoint"]
    assert check.status == FAIL
    assert "localhost address" in check.detail
    assert f"set it to exactly {BASE}" in (check.hint or "")


def test_resource_mismatch_is_detected():
    routes = _healthy(BASE)
    routes["/.well-known/oauth-protected-resource/mcp"] = _json(
        {"resource": "https://other.example.com/mcp", "authorization_servers": [BASE]}
    )
    assert _by_name(_run(BASE, routes))["Protected resource metadata"].status == FAIL


def test_endpoints_outside_the_checked_url_are_detected():
    routes = _healthy(BASE)
    routes["/.well-known/oauth-authorization-server"] = _json(
        {
            "issuer": BASE,
            "authorization_endpoint": "http://localhost:8000/authorize",
            "token_endpoint": f"{BASE}/token",
            "registration_endpoint": f"{BASE}/register",
        }
    )
    check = _by_name(_run(BASE, routes))["Authorization server metadata"]
    assert check.status == FAIL
    assert "authorization_endpoint" in check.detail


def test_cloudflare_challenge_is_named():
    routes = _healthy(BASE)
    routes["/mcp"] = httpx.Response(
        403,
        html=f"<title>Just a moment...</title>{SECRET_MARKER}",
        headers={"cf-mitigated": "challenge", "server": "cloudflare"},
    )
    check = _by_name(_run(BASE, routes))["MCP endpoint"]
    assert check.status == FAIL
    assert "Cloudflare" in check.detail


def test_route_to_nextcloud_is_named():
    routes = _healthy(BASE)
    routes["/mcp"] = _nextcloud_404()
    check = _by_name(_run(BASE, routes))["MCP endpoint"]
    assert "Nextcloud" in check.detail


def test_proxy_basic_auth_is_named():
    routes = _healthy(BASE)
    routes["/mcp"] = httpx.Response(401, headers={"WWW-Authenticate": 'Basic realm="x"'})
    check = _by_name(_run(BASE, routes))["MCP endpoint"]
    assert check.status == FAIL
    assert "Basic" in check.detail


def test_missing_challenge_fails():
    routes = _healthy(BASE)
    routes["/mcp"] = _json({"jsonrpc": "2.0", "id": 1, "result": {}})
    check = _by_name(_run(BASE, routes))["MCP endpoint"]
    assert check.status == FAIL
    assert "without asking for authentication" in check.detail


def test_redirect_is_reported_with_its_target():
    routes = _healthy(BASE)
    routes["/mcp"] = httpx.Response(307, headers={"Location": f"{BASE}/mcp/"})
    check = _by_name(_run(BASE, routes))["MCP endpoint"]
    assert check.status == FAIL
    assert f"{BASE}/mcp/" in check.detail


def test_bad_gateway_points_at_the_backend():
    routes = _healthy(BASE)
    routes["/mcp"] = httpx.Response(502, text="")
    check = _by_name(_run(BASE, routes))["MCP endpoint"]
    assert "MCP_PORT" in (check.hint or "")


def test_unknown_404_points_at_the_route():
    routes = _healthy(BASE)
    del routes["/mcp"]
    check = _by_name(_run(BASE, routes))["MCP endpoint"]
    assert check.status == FAIL
    assert "Nothing serves /mcp" in (check.hint or "")


def test_challenge_without_resource_metadata_warns_and_falls_back():
    routes = _healthy(BASE)
    routes["/mcp"] = httpx.Response(401, headers={"WWW-Authenticate": "Bearer"})
    checks = _by_name(_run(BASE, routes))
    assert checks["MCP endpoint"].status == WARN
    assert checks["Protected resource metadata"].status == OK


def test_subpath_without_forwarded_discovery_explains_the_root_paths():
    # The proxy strips /organizer, but leaves the root-level discovery paths to Nextcloud.
    routes = _healthy(PREFIXED)
    routes["/.well-known/oauth-authorization-server/organizer"] = _nextcloud_404()
    check = _by_name(_run(PREFIXED, routes))["Authorization server metadata"]
    assert check.status == FAIL
    assert "/.well-known/oauth-authorization-server/organizer" in (check.hint or "")


def test_missing_protected_resource_metadata_fails():
    routes = _healthy(BASE)
    del routes["/.well-known/oauth-protected-resource/mcp"]
    assert _by_name(_run(BASE, routes))["Protected resource metadata"].status == FAIL


def test_issuer_mismatch_fails():
    routes = _healthy(BASE)
    routes["/.well-known/oauth-authorization-server"] = _json({"issuer": "https://x.example"})
    assert _by_name(_run(BASE, routes))["Authorization server metadata"].status == FAIL


def test_consent_routed_elsewhere_fails():
    # e.g. a subpath proxy that forwards only /mcp and the OAuth paths.
    routes = _healthy(PREFIXED)
    routes["/organizer/consent"] = _nextcloud_404()
    check = _by_name(_run(PREFIXED, routes))["OAuth endpoints"]
    assert check.status == FAIL
    assert "consent /organizer/consent answered 404 from Nextcloud" in check.detail


def test_main_exit_status_and_output(monkeypatch, capsys):
    routes = _healthy(BASE)
    routes["/mcp"] = _nextcloud_404()
    monkeypatch.setattr(doctor, "_default_resolve", _public)
    monkeypatch.setattr(doctor, "_make_client", lambda timeout: _client(routes))
    assert doctor.main([BASE]) == 1
    output = capsys.readouterr().out
    assert "[FAIL] MCP endpoint" in output
    assert "do not prove that OAuth consent" in output
    assert SECRET_MARKER not in output

    routes.update(_healthy(BASE))
    assert doctor.main([BASE]) == 0
    assert "Basic deployment checks passed." in capsys.readouterr().out


def test_client_does_not_follow_redirects():
    with doctor._make_client(5) as client:
        assert client.follow_redirects is False


def test_usage_errors_exit_2():
    with pytest.raises(SystemExit) as exc:
        doctor.main([])
    assert exc.value.code == 2
