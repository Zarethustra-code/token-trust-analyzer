"""Tests for response caching and the /analyze/batch endpoint (offline, mocked)."""

from __future__ import annotations

import app as app_module
from tests.conftest import make_raw_token_data

_DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
_USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"


def _install_counting_collector(monkeypatch, raw: dict, calls: list):
    class _FakeCollector:
        def __init__(self, **kwargs):
            pass

        def collect(self, address):
            calls.append(address)  # list.append is atomic under the GIL
            return dict(raw)

    monkeypatch.setattr(app_module, "OnChainCollector", _FakeCollector)


# --- caching ---------------------------------------------------------------- #
def test_analyze_cache_hit(client, monkeypatch, healthy_features):
    monkeypatch.setenv("CACHE_TTL_SECONDS", "600")
    calls = []
    _install_counting_collector(monkeypatch, make_raw_token_data(healthy_features), calls)

    body = {"contract_address": _DAI, "chain": "ethereum"}
    r1 = client.post("/analyze", json=body)
    r2 = client.post("/analyze", json=body)

    assert r1.status_code == 200 and r2.status_code == 200
    assert len(calls) == 1                      # collector invoked exactly once
    assert r1.json()["cached"] is False
    assert r2.json()["cached"] is True          # served from cache


def test_analyze_cache_disabled(client, monkeypatch, healthy_features):
    monkeypatch.setenv("CACHE_TTL_SECONDS", "0")  # disable
    calls = []
    _install_counting_collector(monkeypatch, make_raw_token_data(healthy_features), calls)

    body = {"contract_address": _DAI, "chain": "ethereum"}
    r1 = client.post("/analyze", json=body)
    r2 = client.post("/analyze", json=body)

    assert len(calls) == 2                       # collector invoked every time
    assert r1.json()["cached"] is False
    assert r2.json()["cached"] is False


def test_health_reports_cache_state(client, monkeypatch):
    monkeypatch.setenv("CACHE_TTL_SECONDS", "0")
    assert client.get("/health").json()["cache_enabled"] is False
    monkeypatch.setenv("CACHE_TTL_SECONDS", "600")
    assert client.get("/health").json()["cache_enabled"] is True


# --- batch ------------------------------------------------------------------ #
def test_batch_ordered_reports(client, monkeypatch, healthy_features):
    calls = []
    _install_counting_collector(monkeypatch, make_raw_token_data(healthy_features), calls)

    body = {"tokens": [
        {"contract_address": _DAI, "chain": "ethereum"},
        {"contract_address": _USDC, "chain": "ethereum"},
    ]}
    resp = client.post("/analyze/batch", json=body)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["contract_address"] for r in results] == [_DAI, _USDC]  # order preserved
    assert all(r["report"] is not None and r["error"] is None for r in results)
    assert len(calls) == 2


def test_batch_isolates_per_token_errors(client, monkeypatch, healthy_features):
    calls = []
    _install_counting_collector(monkeypatch, make_raw_token_data(healthy_features), calls)

    body = {"tokens": [
        {"contract_address": _DAI, "chain": "ethereum"},
        {"contract_address": "0xNOTVALID", "chain": "ethereum"},  # bad -> error entry
        {"contract_address": _USDC, "chain": "ethereum"},
    ]}
    resp = client.post("/analyze/batch", json=body)
    assert resp.status_code == 200                # batch itself never 500s
    results = resp.json()["results"]
    assert len(results) == 3
    assert results[0]["report"] is not None
    assert results[1]["report"] is None and results[1]["error"]
    assert results[2]["report"] is not None
    assert len(calls) == 2                        # invalid token never reached the collector


def test_batch_dedupes_identical_tokens(client, monkeypatch, healthy_features):
    calls = []
    _install_counting_collector(monkeypatch, make_raw_token_data(healthy_features), calls)

    body = {"tokens": [
        {"contract_address": _DAI, "chain": "ethereum"},
        {"contract_address": _DAI.lower(), "chain": "Ethereum"},  # same token, different case
    ]}
    resp = client.post("/analyze/batch", json=body)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert all(r["report"] is not None for r in results)
    assert len(calls) == 1                        # analyzed once, reused


def test_batch_over_cap_is_422(client):
    tokens = [{"contract_address": _DAI, "chain": "ethereum"} for _ in range(26)]
    assert client.post("/analyze/batch", json={"tokens": tokens}).status_code == 422


def test_batch_empty_is_422(client):
    assert client.post("/analyze/batch", json={"tokens": []}).status_code == 422
