"""Fetch a project URL and extract readable text — with SSRF protection.

Used only by the AI-content detector: when a caller supplies ``project_url``
instead of ``project_text``, we fetch the page, strip it to visible text, and feed
that to the existing detector. This is a **public endpoint fetching user-supplied
URLs**, i.e. a Server-Side Request Forgery (SSRF) surface, so before every request
*and every redirect hop* we:

  * allow only ``http`` / ``https`` schemes (reject ``file:``, ``gopher:``, …),
  * resolve the hostname and reject if ANY resolved IP is loopback / private /
    link-local (incl. the cloud-metadata IP ``169.254.169.254``) / reserved /
    multicast / unspecified — and reject ``localhost`` outright,
  * follow redirects manually, re-running the guard on each hop, so a public URL
    can't ``302`` into an internal address,
  * cap the downloaded size and require an HTML/text content type.

Never raises: returns the extracted text, or ``None`` on any block/failure/empty
page, so the caller degrades to ``checked = False``. Stdlib only, plus the
``requests`` already used by the collectors.

TOCTOU note: we validate the resolved IP(s) and then let ``requests`` re-resolve
on connect, so a hostile authoritative DNS could in principle rebind between the
two lookups. The size cap and scheme/redirect guards bound the blast radius, and
we only ever *read* text; pinning the socket to the vetted IP is intentionally out
of scope for this helper.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin, urlsplit

try:  # requests is a hard dep in practice; degrade gracefully if absent
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

logger = logging.getLogger("token_trust.url_fetch")

_UA = (
    "Mozilla/5.0 (compatible; TokenTrustAnalyzer/1.0; "
    "+https://github.com/Zarethustra-code/token-trust-analyzer)"
)

_ALLOWED_SCHEMES = {"http", "https"}
# Hostnames we refuse before we even resolve them.
_BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}
_REDIRECT_CODES = {301, 302, 303, 307, 308}
_ACCEPTABLE_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}
_SKIP_TAGS = {"script", "style", "head", "noscript", "template", "svg", "iframe"}

_DEFAULT_TIMEOUT = 8.0
_MAX_BYTES = 1_500_000  # ~1.5 MB hard cap on the downloaded body
_MAX_CHARS = 8000       # chars of extracted text passed on to the detector
_MAX_REDIRECTS = 3


# --- SSRF guard ------------------------------------------------------------- #

def _is_blocked_ip(ip_str: str) -> bool:
    """True if ``ip_str`` is a non-public address we must never connect to."""
    try:
        ip = ipaddress.ip_address(ip_str.split("%", 1)[0])  # drop any IPv6 zone id
    except ValueError:
        return True  # unparseable → refuse
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        # e.g. ::ffff:127.0.0.1 — evaluate the embedded IPv4 address
        return _is_blocked_ip(str(ip.ipv4_mapped))
    return (
        ip.is_private       # 10/8, 172.16/12, 192.168/16, fc00::/7, …
        or ip.is_loopback   # 127/8, ::1
        or ip.is_link_local # 169.254/16 (incl. 169.254.169.254), fe80::/10
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified  # 0.0.0.0, ::
    )


def _ssrf_reason(url: str) -> Optional[str]:
    """Return a human-readable reason the URL must NOT be fetched, else ``None``."""
    try:
        parts = urlsplit(url)
    except Exception:
        return "malformed URL"

    scheme = (parts.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        return f"scheme {scheme or '(none)'!r} is not allowed"

    host = parts.hostname
    if not host:
        return "URL has no host"
    if host.lower() in _BLOCKED_HOSTNAMES:
        return "host is not routable (localhost)"

    port = parts.port or (443 if scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except Exception:
        return "host could not be resolved"

    ips = {info[4][0] for info in infos}
    if not ips:
        return "host did not resolve to any address"
    for ip in ips:
        if _is_blocked_ip(ip):
            return f"host resolves to a non-public address ({ip})"
    return None


# --- text extraction (stdlib html.parser) ----------------------------------- #

class _TextExtractor(HTMLParser):
    """Collect visible text; skip <script>/<style>/<head>/<noscript>/etc. content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_startendtag(self, tag, attrs):
        # Self-closing tag (<br/>, <img/>, <script .../>): no enclosed content,
        # so it must not change the skip depth.
        pass

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0 and data:
            self._parts.append(data)

    def text(self) -> str:
        return " ".join(self._parts)


def _extract_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # HTMLParser is lenient, but never let a parse error escape
        pass
    return parser.text()


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _acceptable_content_type(ctype: str) -> bool:
    """Accept HTML/plain-text bodies; a missing header is tolerated (size-capped)."""
    if not ctype:
        return True
    return ctype.split(";", 1)[0].strip().lower() in _ACCEPTABLE_CONTENT_TYPES


def _charset_from_ctype(ctype: str) -> Optional[str]:
    match = re.search(r"charset=([\w\-]+)", ctype or "", re.IGNORECASE)
    return match.group(1) if match else None


def _read_capped(resp, max_bytes: int) -> Optional[bytes]:
    """Read the body in chunks, returning ``None`` if it exceeds ``max_bytes``."""
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_content(chunk_size=8192):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


# --- public entry point ----------------------------------------------------- #

def fetch_project_text(
    url: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    max_bytes: int = _MAX_BYTES,
    max_chars: int = _MAX_CHARS,
    max_redirects: int = _MAX_REDIRECTS,
) -> Optional[str]:
    """Fetch ``url`` (SSRF-guarded) and return its readable text, or ``None``.

    ``None`` means "no usable text" — a blocked, oversized, or failed fetch, a
    non-HTML body, or an empty page. Callers must treat that as *not checked*,
    never as an error.
    """
    if requests is None or not url:
        return None

    current = url
    for _ in range(max_redirects + 1):
        blocked = _ssrf_reason(current)
        if blocked:
            logger.info("Refusing to fetch %r: %s", current, blocked)
            return None

        try:
            resp = requests.get(
                current,
                stream=True,
                timeout=timeout,
                headers={"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"},
                allow_redirects=False,  # follow manually so each hop is re-guarded
            )
        except Exception as exc:
            logger.info("Fetch of %r failed: %s", current, exc)
            return None

        try:
            if resp.status_code in _REDIRECT_CODES:
                location = resp.headers.get("location")
                if not location:
                    return None
                current = urljoin(current, location)  # resolve relative redirects
                continue  # re-guarded at the top of the loop
            if resp.status_code != 200:
                logger.info("Fetch of %r returned HTTP %s", current, resp.status_code)
                return None
            ctype = resp.headers.get("content-type", "")
            if not _acceptable_content_type(ctype):
                logger.info("Fetch of %r has non-HTML content type %r", current, ctype)
                return None
            clen = resp.headers.get("content-length")
            if clen and clen.isdigit() and int(clen) > max_bytes:
                return None
            body = _read_capped(resp, max_bytes)
        finally:
            resp.close()

        if body is None:  # oversized
            return None
        encoding = _charset_from_ctype(ctype) or "utf-8"
        text = _collapse_ws(_extract_text(body.decode(encoding, errors="replace")))
        text = text[:max_chars].strip()
        return text or None

    logger.info("Too many redirects while fetching %r", url)
    return None
