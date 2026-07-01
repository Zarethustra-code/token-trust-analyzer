"""Tests for collectors/honeypot_source.py — HTTP mocked, fully offline."""

from __future__ import annotations

from collectors import honeypot_source
from collectors.honeypot_source import fetch_honeypot_check

_ADDR = "0x6B175474E89094C44Da98b954EedeAC495271d0F"


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_parses_honeypot_true_with_taxes(monkeypatch):
    payload = {
        "honeypotResult": {"isHoneypot": True, "honeypotReason": "cannot sell"},
        "simulationResult": {"buyTax": 5.0, "sellTax": 99.0},
    }
    monkeypatch.setattr(honeypot_source.requests, "get", lambda url, **kw: _Resp(payload))
    r = fetch_honeypot_check("ethereum", _ADDR)
    assert r["is_honeypot"] is True
    assert r["reason"] == "cannot sell"
    assert r["buy_tax"] == 5.0 and r["sell_tax"] == 99.0


def test_parses_honeypot_false(monkeypatch):
    payload = {"honeypotResult": {"isHoneypot": False}, "simulationResult": {"buyTax": 0, "sellTax": 0}}
    monkeypatch.setattr(honeypot_source.requests, "get", lambda url, **kw: _Resp(payload))
    r = fetch_honeypot_check("ethereum", _ADDR)
    assert r["is_honeypot"] is False
    assert r["reason"] is None
    assert r["buy_tax"] == 0.0


def test_missing_honeypot_result_returns_none(monkeypatch):
    # Token couldn't be checked -> no honeypotResult -> no signal (not "clean").
    payload = {"simulationResult": {"buyTax": 1}}
    monkeypatch.setattr(honeypot_source.requests, "get", lambda url, **kw: _Resp(payload))
    assert fetch_honeypot_check("ethereum", _ADDR) is None


def test_request_failure_returns_none(monkeypatch):
    def boom(url, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(honeypot_source.requests, "get", boom)
    assert fetch_honeypot_check("ethereum", _ADDR) is None


def test_error_status_returns_none(monkeypatch):
    monkeypatch.setattr(honeypot_source.requests, "get", lambda url, **kw: _Resp({}, status=500))
    assert fetch_honeypot_check("ethereum", _ADDR) is None


def test_unsupported_chain_returns_none():
    assert fetch_honeypot_check("solana", _ADDR) is None


def test_sends_required_chainid(monkeypatch):
    captured = {}

    def fake(url, params=None, **kwargs):
        captured.update(params or {})
        return _Resp({"honeypotResult": {"isHoneypot": False}})

    monkeypatch.setattr(honeypot_source.requests, "get", fake)
    fetch_honeypot_check("base", _ADDR)
    assert captured["chainID"] == 8453
    assert captured["address"] == _ADDR


def test_api_key_sent_as_header(monkeypatch):
    monkeypatch.setenv("HONEYPOT_IS_API_KEY", "secret-key")
    captured = {}

    def fake(url, params=None, headers=None, **kwargs):
        captured["headers"] = headers or {}
        return _Resp({"honeypotResult": {"isHoneypot": False}})

    monkeypatch.setattr(honeypot_source.requests, "get", fake)
    fetch_honeypot_check("ethereum", _ADDR)
    assert captured["headers"].get("X-API-KEY") == "secret-key"
