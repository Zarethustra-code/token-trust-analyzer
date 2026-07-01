"""Tests for collectors/dex_source.py — HTTP mocked, fully offline."""

from __future__ import annotations

from collectors import dex_source
from collectors.dex_source import fetch_dex_metrics

_ADDR = "0x6B175474E89094C44Da98b954EedeAC495271d0F"

# DexScreener: two ethereum pairs (aggregated) + one base pair (ignored on eth).
_DEXSCREENER = {
    "pairs": [
        {"chainId": "ethereum", "liquidity": {"usd": 1_000_000}, "marketCap": 5_000_000,
         "fdv": 6_000_000, "volume": {"h24": 200_000}, "txns": {"h24": {"buys": 120, "sells": 100}}},
        {"chainId": "ethereum", "liquidity": {"usd": 500_000}, "marketCap": 5_000_000,
         "volume": {"h24": 100_000}, "txns": {"h24": {"buys": 80, "sells": 60}}},
        {"chainId": "base", "liquidity": {"usd": 9_999_999}, "marketCap": 1,
         "txns": {"h24": {"buys": 1, "sells": 1}}},
    ]
}

# GeckoTerminal: one pool.
_GECKOTERMINAL = {
    "data": [
        {"attributes": {"reserve_in_usd": "800000", "market_cap_usd": "4000000",
                        "fdv_usd": "4500000", "volume_usd": {"h24": "150000"},
                        "transactions": {"h24": {"buys": 90, "sells": 75}}}},
    ]
}


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_dexscreener_primary(monkeypatch):
    def fake_get(url, **kwargs):
        assert "dexscreener" in url
        return _Resp(_DEXSCREENER)

    monkeypatch.setattr(dex_source.requests, "get", fake_get)
    m = fetch_dex_metrics("ethereum", _ADDR)
    assert m["source"] == "dexscreener"
    # matched eth pairs: liquidity 1.5M / mcap 5M = 0.3 ; buys 200 / sells 160 = 1.25
    assert m["liquidity_to_mcap_ratio"] == 0.3
    assert m["buy_sell_ratio"] == 1.25
    assert m["liquidity_usd"] == 1_500_000


def test_geckoterminal_fallback(monkeypatch):
    def fake_get(url, **kwargs):
        if "dexscreener" in url:
            return _Resp({"pairs": []})          # no matching pairs -> triggers fallback
        assert "geckoterminal" in url and "/eth/" in url  # eth network id, not "ethereum"
        return _Resp(_GECKOTERMINAL)

    monkeypatch.setattr(dex_source.requests, "get", fake_get)
    m = fetch_dex_metrics("ethereum", _ADDR)
    assert m["source"] == "geckoterminal"
    assert m["liquidity_to_mcap_ratio"] == 0.2   # 800k / 4M
    assert m["buy_sell_ratio"] == 1.2            # 90 / 75


def test_both_sources_fail_returns_none(monkeypatch):
    def fake_get(url, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(dex_source.requests, "get", fake_get)
    assert fetch_dex_metrics("ethereum", _ADDR) is None


def test_dexscreener_error_status_falls_back(monkeypatch):
    def fake_get(url, **kwargs):
        if "dexscreener" in url:
            return _Resp({}, status=429)         # rate-limited
        return _Resp(_GECKOTERMINAL)

    monkeypatch.setattr(dex_source.requests, "get", fake_get)
    m = fetch_dex_metrics("ethereum", _ADDR)
    assert m["source"] == "geckoterminal"


def test_buy_sell_zero_sells_is_guarded(monkeypatch):
    payload = {"pairs": [{"chainId": "ethereum", "liquidity": {"usd": 1000},
                          "marketCap": 10000, "txns": {"h24": {"buys": 5, "sells": 0}},
                          "volume": {"h24": 100}}]}
    monkeypatch.setattr(dex_source.requests, "get", lambda url, **kw: _Resp(payload))
    m = fetch_dex_metrics("ethereum", _ADDR)
    assert m["buy_sell_ratio"] is None           # divide-by-zero guarded
    assert m["liquidity_to_mcap_ratio"] == 0.1   # 1000 / 10000


def test_liquidity_falls_back_to_fdv_when_no_mcap(monkeypatch):
    payload = {"pairs": [{"chainId": "ethereum", "liquidity": {"usd": 300},
                          "marketCap": None, "fdv": 3000,
                          "txns": {"h24": {"buys": 10, "sells": 10}}, "volume": {"h24": 1}}]}
    monkeypatch.setattr(dex_source.requests, "get", lambda url, **kw: _Resp(payload))
    m = fetch_dex_metrics("ethereum", _ADDR)
    assert m["liquidity_to_mcap_ratio"] == 0.1   # 300 / fdv 3000
