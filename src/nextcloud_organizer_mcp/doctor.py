"""Read-only deployment diagnostics for a public MCP URL (#74).

Checks what a remote MCP client sees before it can connect: DNS, TLS,
routing to `/mcp`, the OAuth `401` challenge, protected-resource and
authorization-server discovery, and whether the advertised URLs match the
URL that was checked. Every request is unauthenticated and read-only: the
checks never register a client, start an authorization or submit the consent
form, so they cannot consume rate limits or create OAuth state.

Passing does not prove that OAuth consent or Nextcloud access work; the
summary says so. Nothing secret is ever handled: the checks send no
credentials and print only URLs, status codes and header-derived
classifications, never response bodies.

Usage:
    nextcloud-organizer-mcp-doctor https://organizer.example.com
    nextcloud-organizer-mcp-doctor https://cloud.example.com/organizer/mcp
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import socket
import ssl
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import is_local_hostname

try:
    _VERSION = version("nextcloud-organizer-mcp")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    _VERSION = "unknown"

OK = "ok"
WARN = "warn"
FAIL = "fail"

_LABELS = {OK: "PASS", WARN: "WARN", FAIL: "FAIL"}

#: A minimal initialize request: what a client sends first, so proxies and
#: challenge pages answer it the way they would answer a real client.
_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "nextcloud-organizer-mcp-doctor", "version": _VERSION},
    },
}

Resolver = Callable[[str], list[str]]


@dataclass
class Check:
    name: str
    status: str
    detail: str
    hint: str | None = None


@dataclass
class Target:
    """The URLs a client derives from what the operator typed."""

    base: str  # PUBLIC_BASE_URL: origin plus optional path prefix, no trailing slash
    mcp: str  # the connector URL
    origin: str
    prefix: str  # "" or "/organizer"
    host: str


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, hint: str | None = None) -> Check:
        check = Check(name, status, detail, hint)
        self.checks.append(check)
        return check

    @property
    def failed(self) -> bool:
        return any(check.status == FAIL for check in self.checks)


def parse_target(url: str) -> Target:
    """Accept PUBLIC_BASE_URL or the connector URL; raise ValueError if unusable."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("expected an http(s) URL such as https://organizer.example.com")
    if parsed.query or parsed.fragment:
        raise ValueError("the URL must not contain a query string or fragment")
    prefix = parsed.path.rstrip("/")
    if prefix.endswith("/mcp"):
        prefix = prefix[: -len("/mcp")]
    origin = f"{parsed.scheme}://{parsed.netloc}"
    base = f"{origin}{prefix}"
    return Target(base, f"{base}/mcp", origin, prefix, parsed.hostname)


def _default_resolve(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _is_html(response: httpx.Response) -> bool:
    return response.headers.get("content-type", "").startswith("text/html")


def _describe_foreign_response(response: httpx.Response) -> tuple[str, str]:
    """Name whatever answered instead of this server, and what to do about it."""
    headers = response.headers
    server = headers.get("server", "").lower()
    if "cf-mitigated" in headers or (
        server == "cloudflare" and response.status_code in (403, 503) and _is_html(response)
    ):
        return (
            "a Cloudflare challenge or block page",
            "Disable bot protection, challenges and Access for this hostname, or skip them "
            "for its paths; MCP clients cannot solve browser challenges.",
        )
    cookies = " ".join(headers.get_list("set-cookie"))
    if "nc_sameSiteCookie" in cookies or "oc_sessionPassphrase" in cookies:
        return (
            "Nextcloud",
            "The proxy sends this path to Nextcloud. Route the whole MCP hostname, or for a "
            "subpath deployment the prefix and its root-level discovery paths, to this server.",
        )
    scheme = headers.get("www-authenticate", "").split(" ", 1)[0]
    if scheme and scheme.lower() != "bearer":
        return (
            f"another authentication layer ({scheme})",
            "Remove proxy-level authentication in front of /mcp; the server's own OAuth "
            "protects it and clients cannot pass a second login.",
        )
    if _is_html(response):
        return (
            "an HTML page (login, challenge or another site)",
            "Something in front of the server answers instead of it; check the proxy route "
            "and any access-control layer for this hostname.",
        )
    return (
        "something other than this server",
        "Check that the proxy forwards this path, method and body to the server unchanged.",
    )


def _redirect_hint(target: Target, response: httpx.Response) -> str:
    location = response.headers.get("location", "")
    if target.origin.startswith("http://") and location.startswith("https://"):
        return "The server redirects to HTTPS: check again with the https:// URL."
    return (
        f"MCP clients may not follow a redirect here. Use the final URL ({location}) "
        "as the connector URL if it is this server, or remove the redirect."
    )


def _advertised_url_problems(target: Target, url: str) -> list[str]:
    """Reasons an advertised URL cannot be the deployment that was checked."""
    parsed = urlparse(url)
    problems = []
    if f"{parsed.scheme}://{parsed.netloc}" != target.origin:
        problems.append(f"origin {parsed.scheme}://{parsed.netloc} instead of {target.origin}")
    if is_local_hostname(parsed.hostname) and not is_local_hostname(target.host):
        problems.append("a localhost address")
    return problems


_MISMATCH_HINT = (
    "The server builds these URLs from PUBLIC_BASE_URL alone: set it to exactly {base} "
    "(no /mcp) and restart the server."
)


def _get_json(client: httpx.Client, url: str) -> tuple[httpx.Response | None, Any, str | None]:
    """GET `url`; return the response, its JSON (or None) and a transport error, if any."""
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:
        return None, None, _transport_error(exc)[0]
    if response.status_code != 200 or "json" not in response.headers.get("content-type", ""):
        return response, None, None
    try:
        return response, response.json(), None
    except ValueError:
        return response, None, None


def _transport_error(exc: httpx.HTTPError) -> tuple[str, str, str]:
    """Classify a failed request as (detail, hint, check name)."""
    text = str(exc)
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, ssl.SSLError):
            text = str(cause)
            break
        cause = cause.__cause__ or cause.__context__
    if isinstance(cause, ssl.SSLError) or "CERTIFICATE" in text or "SSL" in text:
        return (
            f"TLS handshake failed: {text}",
            "Check that the certificate is valid, unexpired and issued for this hostname, "
            "and that the full chain is served. A stale AAAA record can point at a host "
            "with another certificate; WRONG_VERSION_NUMBER means the port speaks plain HTTP.",
            "TLS",
        )
    if isinstance(exc, httpx.TimeoutException):
        return (
            "the request timed out",
            "Check firewalls and that inbound port 443 (or the tunnel) reaches the proxy.",
            "Reachability",
        )
    return (
        f"connection failed: {text}",
        "Check that the proxy or tunnel is running and listening on this host and port.",
        "Reachability",
    )


def _protected_resource_urls(target: Target, challenge_url: str | None) -> list[str]:
    urls = [challenge_url] if challenge_url else []
    # RFC 9728 path insertion, then the root, as MCP clients fall back.
    for path in (f"{target.prefix}/mcp", ""):
        url = f"{target.origin}/.well-known/oauth-protected-resource{path}"
        if url not in urls:
            urls.append(url)
    return urls


def _authorization_server_urls(issuer: str) -> list[str]:
    """Where MCP clients look for metadata of `issuer` (RFC 8414 and OIDC, in client order)."""
    parsed = urlparse(issuer)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")
    if path:
        return [
            f"{origin}/.well-known/oauth-authorization-server{path}",
            f"{origin}/.well-known/openid-configuration{path}",
            f"{origin}{path}/.well-known/openid-configuration",
        ]
    return [
        f"{origin}/.well-known/oauth-authorization-server",
        f"{origin}/.well-known/openid-configuration",
    ]


def run_checks(url: str, client: httpx.Client, resolve: Resolver | None = None) -> Report:
    """Run every check in order, stopping where later checks cannot mean anything."""
    resolve = resolve or _default_resolve
    report = Report()
    try:
        target = parse_target(url)
    except ValueError as exc:
        report.add("URL", FAIL, str(exc))
        return report

    local = is_local_hostname(target.host)
    if target.origin.startswith("http://") and not local:
        report.add(
            "URL",
            FAIL,
            f"{target.mcp} uses plain http://",
            "Remote MCP clients require HTTPS. Terminate TLS at a proxy or tunnel and "
            "check the https:// URL.",
        )
        return report
    report.add("URL", OK, f"connector URL {target.mcp}, PUBLIC_BASE_URL {target.base}")

    if not _is_ip_literal(target.host):
        try:
            addresses = resolve(target.host)
        except OSError as exc:
            report.add(
                "DNS",
                FAIL,
                f"{target.host} does not resolve: {exc}",
                "Create A/AAAA records pointing at the proxy (or the tunnel's CNAME), "
                "then wait for them to propagate.",
            )
            return report
        # A scope id ("fe80::1%eth0") is not part of the address.
        public = [a for a in addresses if ipaddress.ip_address(a.split("%")[0]).is_global]
        if not public and not local:
            report.add(
                "DNS",
                WARN,
                f"{target.host} resolves only to non-public addresses here: {', '.join(addresses)}",
                "Cloud-hosted clients such as claude.ai cannot reach these addresses. "
                "This is fine if public DNS answers differently; run the check from outside "
                "your network to be sure.",
            )
        else:
            report.add("DNS", OK, f"{target.host} resolves to {', '.join(addresses)}")

    try:
        challenge = client.post(
            target.mcp,
            json=_INITIALIZE,
            headers={"Accept": "application/json, text/event-stream"},
        )
    except httpx.HTTPError as exc:
        detail, hint, name = _transport_error(exc)
        report.add(name, FAIL, detail, hint)
        return report
    if target.origin.startswith("https://"):
        report.add("TLS", OK, f"certificate for {target.host} accepted")

    passed, challenge_url = _check_challenge(report, target, challenge)
    if not passed:
        return report

    issuer = _check_protected_resource(report, client, target, challenge_url)
    if issuer is None:
        return report
    metadata = _check_authorization_server(report, client, target, issuer)
    if metadata is None:
        return report
    _check_endpoints(report, client, target, metadata)
    return report


def _check_challenge(
    report: Report, target: Target, response: httpx.Response
) -> tuple[bool, str | None]:
    """Check /mcp answers with a Bearer challenge; return (passed, its metadata URL)."""
    name = "MCP endpoint"
    authenticate = response.headers.get("www-authenticate", "")
    if response.status_code == 401 and authenticate.lower().startswith("bearer"):
        match = re.search(r'resource_metadata="([^"]+)"', authenticate)
        if not match:
            report.add(
                name,
                WARN,
                "401 Bearer challenge without resource_metadata",
                "Current MCP clients find the authorization server through resource_metadata; "
                "check that the proxy passes the WWW-Authenticate header through unchanged.",
            )
            return True, None
        url = match.group(1)
        problems = _advertised_url_problems(target, url)
        if problems:
            report.add(
                name,
                FAIL,
                f"the challenge advertises {url}: {', '.join(problems)}",
                _MISMATCH_HINT.format(base=target.base),
            )
            return False, None
        report.add(name, OK, f"{target.mcp} answers 401 with an OAuth challenge")
        return True, url
    if response.is_redirect:
        report.add(
            name,
            FAIL,
            f"{response.status_code} redirect to {response.headers.get('location', '?')}",
            _redirect_hint(target, response),
        )
        return False, None
    if response.is_success:
        report.add(
            name,
            FAIL,
            f"{target.mcp} answered {response.status_code} without asking for authentication",
            "This server always requires OAuth over HTTP. Something else answers here, or "
            "the URL points at an unprotected server.",
        )
        return False, None
    if response.status_code in (502, 503, 504) and not _is_html(response):
        report.add(
            name,
            FAIL,
            f"the proxy answered {response.status_code}",
            "The proxy cannot reach the server: check that it is running and that the "
            "proxy's upstream address and port match MCP_HOST and MCP_PORT.",
        )
        return False, None
    what, hint = _describe_foreign_response(response)
    if response.status_code == 404 and what == "something other than this server":
        hint = (
            "Nothing serves /mcp at this URL. Check the connector URL, and that the proxy "
            "forwards it (and strips the prefix for a subpath deployment)."
        )
    report.add(
        name, FAIL, f"{response.status_code} from {what} instead of an OAuth challenge", hint
    )
    return False, None


def _check_protected_resource(
    report: Report, client: httpx.Client, target: Target, challenge_url: str | None
) -> str | None:
    name = "Protected resource metadata"
    tried = []
    for url in _protected_resource_urls(target, challenge_url):
        response, data, error = _get_json(client, url)
        tried.append(f"{url} ({error or (response.status_code if response else '?')})")
        if isinstance(data, dict):
            break
    else:
        report.add(
            name,
            FAIL,
            "not found at " + "; ".join(tried),
            "Forward /.well-known/oauth-protected-resource/... to the server unchanged; for "
            "a subpath deployment this path is at the domain root, outside the prefix.",
        )
        return None
    resource = data.get("resource")
    servers = data.get("authorization_servers") or []
    if resource != target.mcp:
        report.add(
            name,
            FAIL,
            f"resource is {resource!r}, expected {target.mcp}",
            _MISMATCH_HINT.format(base=target.base),
        )
        return None
    if not servers or not isinstance(servers[0], str):
        report.add(name, FAIL, "no authorization_servers listed")
        return None
    problems = _advertised_url_problems(target, servers[0])
    if problems:
        report.add(
            name,
            FAIL,
            f"authorization server {servers[0]}: {', '.join(problems)}",
            _MISMATCH_HINT.format(base=target.base),
        )
        return None
    report.add(name, OK, f"resource {resource}, authorization server {servers[0]}")
    return servers[0]


def _check_authorization_server(
    report: Report, client: httpx.Client, target: Target, issuer: str
) -> dict[str, Any] | None:
    name = "Authorization server metadata"
    tried = []
    for url in _authorization_server_urls(issuer):
        response, data, error = _get_json(client, url)
        tried.append(f"{url} ({error or (response.status_code if response else '?')})")
        if isinstance(data, dict):
            break
    else:
        hint = "Forward the server's /.well-known/ discovery paths to it unchanged."
        if urlparse(issuer).path.rstrip("/"):
            hint = (
                "For a subpath deployment the proxy must forward "
                f"/.well-known/oauth-authorization-server{urlparse(issuer).path.rstrip('/')} "
                "from the domain root to the server, ahead of Nextcloud's /.well-known rules."
            )
        report.add(name, FAIL, "not found at " + "; ".join(tried), hint)
        return None
    if str(data.get("issuer", "")).rstrip("/") != issuer.rstrip("/"):
        report.add(
            name,
            FAIL,
            f"issuer is {data.get('issuer')!r}, expected {issuer}",
            _MISMATCH_HINT.format(base=target.base),
        )
        return None
    problems = []
    for key in ("authorization_endpoint", "token_endpoint", "registration_endpoint"):
        endpoint = data.get(key)
        if not isinstance(endpoint, str):
            problems.append(f"{key} missing")
            continue
        problems += [f"{key} has {issue}" for issue in _advertised_url_problems(target, endpoint)]
        if not endpoint.startswith(f"{target.base}/"):
            problems.append(f"{key} {endpoint} is outside {target.base}")
    if problems:
        report.add(name, FAIL, "; ".join(problems), _MISMATCH_HINT.format(base=target.base))
        return None
    report.add(name, OK, f"issuer {issuer} with matching endpoint URLs")
    return data


def _check_endpoints(
    report: Report, client: httpx.Client, target: Target, metadata: dict[str, Any]
) -> None:
    """Probe each OAuth endpoint with a harmless request only this server answers as expected."""
    probes = [
        # (label, url, expected status, expected content type prefix)
        ("registration", metadata["registration_endpoint"], 405, ""),
        ("token", metadata["token_endpoint"], 405, ""),
        # No parameters: an invalid_request error, before any client lookup.
        ("authorization", metadata["authorization_endpoint"], 400, "application/json"),
        # An unknown key shows the expired-request page; GETs never count
        # against the consent rate limits.
        ("consent", f"{target.base}/consent?pending=doctor-check", 400, "text/html"),
    ]
    broken = []
    for label, url, status, content_type in probes:
        try:
            response = client.get(url)
        except httpx.HTTPError as exc:
            broken.append(f"{label}: {_transport_error(exc)[0]}")
            continue
        if response.status_code == status and response.headers.get("content-type", "").startswith(
            content_type
        ):
            continue
        what, _ = _describe_foreign_response(response)
        broken.append(f"{label} {urlparse(url).path} answered {response.status_code} from {what}")
    if broken:
        report.add(
            "OAuth endpoints",
            FAIL,
            "; ".join(broken),
            "Route every path under the server's base URL to it unchanged, including "
            "/register, /authorize, /token and /consent (GET and POST).",
        )
        return
    report.add("OAuth endpoints", OK, "/register, /token, /authorize and /consent reach the server")


_SCOPE_NOTE = (
    "These checks are read-only and unauthenticated. They do not prove that OAuth consent "
    "or Nextcloud access work: add the connector in your client, enter MCP_OAUTH_PASSWORD "
    "on the consent page and run a read-only tool such as list_task_lists."
)


def format_report(report: Report) -> str:
    lines = []
    for check in report.checks:
        lines.append(f"[{_LABELS[check.status]}] {check.name}: {check.detail}")
        if check.hint:
            lines.append(f"       -> {check.hint}")
    if report.failed:
        lines.append("\nSome checks failed; fix the first failure and run the check again.")
    else:
        lines.append("\nBasic deployment checks passed.")
    lines.append(_SCOPE_NOTE)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nextcloud-organizer-mcp-doctor",
        description=(
            "Check a deployment's public URL, TLS and OAuth discovery the way a remote "
            "MCP client sees them. Read-only and unauthenticated."
        ),
    )
    parser.add_argument(
        "url",
        help="PUBLIC_BASE_URL or the connector URL, e.g. https://organizer.example.com",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="seconds per request (default: 10)"
    )
    return parser


def _make_client(timeout: float) -> httpx.Client:
    # Redirects are reported, not followed: clients do not follow them for POST.
    return httpx.Client(
        timeout=timeout,
        follow_redirects=False,
        headers={"User-Agent": f"nextcloud-organizer-mcp-doctor/{_VERSION}"},
    )


def main(argv: list[str] | None = None) -> int:
    """Exit 0 when every check passes or only warns, 1 when one fails, 2 on usage errors."""
    args = build_parser().parse_args(argv)
    with _make_client(args.timeout) as client:
        report = run_checks(args.url, client)
    print(format_report(report))
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
