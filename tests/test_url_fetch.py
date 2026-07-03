"""Tests for detectors/url_fetch.py — SSRF guard, extraction, fetch (all offline).

DNS resolution and the ``requests`` module are mocked; no test touches the network.
The SSRF tests assert BOTH that a blocked URL returns ``None`` and that NO outbound
request is attempted.
"""

from __future__ import annotations

import socket as _socket

import pytest
from pydantic import ValidationError

import detectors.url_fetch as uf
from models.request import AnalyzeRequest

_ADDR = "0x" + "6" * 40


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
def _gai(ip_by_host: dict):
    """Fake socket.getaddrinfo: map host -> ip (IP-literal hosts resolve to self)."""
    def fake(host, port, *args, **kwargs):
        ip = ip_by_host.get(host, host)
        return [(_socket.AF_INET, _socket.SOCK_STREAM, _socket.IPPROTO_TCP, "", (ip, port))]
    return fake


class _FakeResp:
    def __init__(self, status=200, headers=None, chunks=(b"",)):
        self.status_code = status
        self.headers = headers if headers is not None else {"content-type": "text/html"}
        self._chunks = list(chunks)
        self.closed = False

    def iter_content(self, chunk_size=8192):
        for c in self._chunks:
            yield c

    def close(self):
        self.closed = True


class _FakeRequests:
    """Stand-in for the requests module; dispatches .get(url) through a handler."""
    def __init__(self, handler):
        self._handler = handler
        self.calls: list[str] = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return self._handler(url)


def _no_fetch() -> _FakeRequests:
    def handler(url):
        raise AssertionError(f"must not make an outbound request to {url!r}")
    return _FakeRequests(handler)


# --------------------------------------------------------------------------- #
# IP classification (the SSRF core) — no mocking needed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ip", [
    "127.0.0.1", "127.10.20.30",             # loopback
    "10.1.2.3", "172.16.5.5", "172.31.255.255", "192.168.0.1",  # private
    "169.254.0.1", "169.254.169.254",        # link-local incl. cloud metadata
    "0.0.0.0",                               # unspecified
    "::1", "fc00::1", "fd00:1234::1", "fe80::abcd",              # ipv6 loopback/private/link-local
    "::ffff:127.0.0.1",                      # ipv4-mapped loopback
    "224.0.0.1",                             # multicast
    "garbage-not-an-ip",                     # unparseable → refuse
])
def test_blocked_ips(ip):
    assert uf._is_blocked_ip(ip) is True


@pytest.mark.parametrize("ip", [
    "8.8.8.8", "1.1.1.1", "93.184.216.34", "172.32.0.1",
    "2606:2800:220:1:248:1893:25c8:1946", "2001:4860:4860::8888",
])
def test_public_ips_allowed(ip):
    assert uf._is_blocked_ip(ip) is False


# --------------------------------------------------------------------------- #
# SSRF guard at fetch level — blocked URLs never reach the network
# --------------------------------------------------------------------------- #
def test_fetch_blocks_cloud_metadata_ip(monkeypatch):
    monkeypatch.setattr(uf.socket, "getaddrinfo", _gai({}))  # IP literal → itself
    req = _no_fetch()
    monkeypatch.setattr(uf, "requests", req)
    assert uf.fetch_project_text("http://169.254.169.254/latest/meta-data/") is None
    assert req.calls == []


def test_fetch_blocks_private_dns(monkeypatch):
    monkeypatch.setattr(uf.socket, "getaddrinfo", _gai({"intranet.example.com": "10.0.0.5"}))
    req = _no_fetch()
    monkeypatch.setattr(uf, "requests", req)
    assert uf.fetch_project_text("http://intranet.example.com/") is None
    assert req.calls == []


def test_fetch_blocks_localhost_even_if_dns_public(monkeypatch):
    # The hostname blocklist wins even when DNS would resolve to a public address.
    monkeypatch.setattr(uf.socket, "getaddrinfo", _gai({"localhost": "8.8.8.8"}))
    req = _no_fetch()
    monkeypatch.setattr(uf, "requests", req)
    assert uf.fetch_project_text("http://localhost:8000/admin") is None
    assert req.calls == []


def test_fetch_rejects_non_http_schemes(monkeypatch):
    monkeypatch.setattr(uf.socket, "getaddrinfo", _gai({}))
    req = _no_fetch()
    monkeypatch.setattr(uf, "requests", req)
    assert uf.fetch_project_text("file:///etc/passwd") is None
    assert uf.fetch_project_text("gopher://evil/") is None
    assert req.calls == []


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def test_extract_strips_scripts_styles_head_and_collapses():
    html = (
        "<html><head><title>HEADTITLE</title>"
        "<style>.a{color:red} STYLEBODY</style></head>"
        "<body><h1>Hello   World</h1>"
        "<script>var SCRIPTBODY = 1;</script>"
        "<p>We\n  build   things.</p></body></html>"
    )
    text = uf._collapse_ws(uf._extract_text(html))
    assert "Hello World" in text
    assert "We build things." in text
    assert "HEADTITLE" not in text   # <title> is inside <head>, skipped
    assert "STYLEBODY" not in text
    assert "SCRIPTBODY" not in text


def test_fetch_success_extracts_visible_text(monkeypatch):
    monkeypatch.setattr(uf.socket, "getaddrinfo", _gai({"example.com": "93.184.216.34"}))
    html = b"<html><body><h1>Cool Project</h1><script>bad()</script><p>We build things.</p></body></html>"
    resp = _FakeResp(headers={"content-type": "text/html; charset=utf-8"}, chunks=[html])
    monkeypatch.setattr(uf, "requests", _FakeRequests(lambda url: resp))
    text = uf.fetch_project_text("http://example.com/")
    assert text is not None
    assert "Cool Project" in text and "We build things." in text
    assert "bad()" not in text


def test_fetch_truncates_to_max_chars(monkeypatch):
    monkeypatch.setattr(uf.socket, "getaddrinfo", _gai({"example.com": "93.184.216.34"}))
    body = b"<body>" + b"word " * 5000 + b"</body>"  # ~25 KB of text
    resp = _FakeResp(headers={"content-type": "text/html"}, chunks=[body])
    monkeypatch.setattr(uf, "requests", _FakeRequests(lambda url: resp))
    text = uf.fetch_project_text("http://example.com/")
    assert text is not None and len(text) <= 8000


# --------------------------------------------------------------------------- #
# Fetch failure modes → None, never a crash
# --------------------------------------------------------------------------- #
def test_fetch_oversized_body_rejected(monkeypatch):
    monkeypatch.setattr(uf.socket, "getaddrinfo", _gai({"example.com": "93.184.216.34"}))
    big = b"<body>" + b"A" * 2_000_000 + b"</body>"  # > 1.5 MB cap
    chunks = [big[i:i + 8192] for i in range(0, len(big), 8192)]
    resp = _FakeResp(headers={"content-type": "text/html"}, chunks=chunks)
    monkeypatch.setattr(uf, "requests", _FakeRequests(lambda url: resp))
    assert uf.fetch_project_text("http://example.com/") is None


def test_fetch_non_200_rejected(monkeypatch):
    monkeypatch.setattr(uf.socket, "getaddrinfo", _gai({"example.com": "93.184.216.34"}))
    resp = _FakeResp(status=404, headers={"content-type": "text/html"}, chunks=[b"nope"])
    monkeypatch.setattr(uf, "requests", _FakeRequests(lambda url: resp))
    assert uf.fetch_project_text("http://example.com/") is None


def test_fetch_non_html_content_type_rejected(monkeypatch):
    monkeypatch.setattr(uf.socket, "getaddrinfo", _gai({"example.com": "93.184.216.34"}))
    resp = _FakeResp(headers={"content-type": "application/pdf"}, chunks=[b"%PDF-1.7 binary"])
    monkeypatch.setattr(uf, "requests", _FakeRequests(lambda url: resp))
    assert uf.fetch_project_text("http://example.com/") is None


def test_fetch_timeout_returns_none(monkeypatch):
    monkeypatch.setattr(uf.socket, "getaddrinfo", _gai({"example.com": "93.184.216.34"}))
    def boom(url):
        raise TimeoutError("timed out")
    monkeypatch.setattr(uf, "requests", _FakeRequests(boom))
    assert uf.fetch_project_text("http://example.com/") is None


# --------------------------------------------------------------------------- #
# Redirects — re-guarded on each hop
# --------------------------------------------------------------------------- #
def test_redirect_into_private_is_blocked(monkeypatch):
    monkeypatch.setattr(uf.socket, "getaddrinfo", _gai({"public.example.com": "93.184.216.34"}))

    def handler(url):
        if url == "http://public.example.com/":
            return _FakeResp(status=302, headers={"location": "http://169.254.169.254/latest/"}, chunks=[b""])
        raise AssertionError(f"redirect target must not be fetched: {url!r}")

    req = _FakeRequests(handler)
    monkeypatch.setattr(uf, "requests", req)
    assert uf.fetch_project_text("http://public.example.com/") is None
    assert req.calls == ["http://public.example.com/"]  # target never requested


def test_redirect_to_public_is_followed(monkeypatch):
    monkeypatch.setattr(uf.socket, "getaddrinfo", _gai({"public.example.com": "93.184.216.34"}))

    def handler(url):
        if url == "http://public.example.com/":
            return _FakeResp(status=301, headers={"location": "https://public.example.com/home"}, chunks=[b""])
        if url == "https://public.example.com/home":
            return _FakeResp(headers={"content-type": "text/html"}, chunks=[b"<body><p>Landed OK</p></body>"])
        raise AssertionError(url)

    req = _FakeRequests(handler)
    monkeypatch.setattr(uf, "requests", req)
    text = uf.fetch_project_text("http://public.example.com/")
    assert text is not None and "Landed OK" in text
    assert req.calls == ["http://public.example.com/", "https://public.example.com/home"]


def test_too_many_redirects_returns_none(monkeypatch):
    monkeypatch.setattr(uf.socket, "getaddrinfo", _gai({"public.example.com": "93.184.216.34"}))
    # Always redirects → the redirect cap (default 3) stops it.
    handler = lambda url: _FakeResp(status=302, headers={"location": "https://public.example.com/next"}, chunks=[b""])
    req = _FakeRequests(handler)
    monkeypatch.setattr(uf, "requests", req)
    assert uf.fetch_project_text("http://public.example.com/") is None
    assert len(req.calls) == 4  # initial + 3 redirect hops


# --------------------------------------------------------------------------- #
# Request-model validation for project_url (syntactic http/https only)
# --------------------------------------------------------------------------- #
def test_request_accepts_valid_url():
    r = AnalyzeRequest(contract_address=_ADDR, project_url="https://foo.example.com/wp")
    assert r.project_url == "https://foo.example.com/wp"


def test_request_url_optional_and_blank_is_none():
    assert AnalyzeRequest(contract_address=_ADDR).project_url is None
    assert AnalyzeRequest(contract_address=_ADDR, project_url="   ").project_url is None


@pytest.mark.parametrize("bad", ["ftp://x", "not a url", "file:///etc/passwd", "javascript:alert(1)"])
def test_request_rejects_non_http_url(bad):
    with pytest.raises(ValidationError):
        AnalyzeRequest(contract_address=_ADDR, project_url=bad)


def test_request_accepts_wellformed_but_internal_url():
    # A well-formed internal URL is syntactically valid — SSRF is enforced later,
    # at fetch time, so this must NOT raise here.
    r = AnalyzeRequest(contract_address=_ADDR, project_url="http://169.254.169.254/")
    assert r.project_url == "http://169.254.169.254/"
