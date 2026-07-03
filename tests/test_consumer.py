"""Tests for cap/consumer.py — the requester agent's simulation path (offline).

No test builds a real croo AgentClient or makes any CROO network call: the live
path needs credentials + a funded wallet and is exercised manually (see the
README "A2A composability" section). Here we cover the simulation requester and
the live/simulation mode selection.
"""

from __future__ import annotations

import app as app_module
from tests.conftest import make_raw_token_data

_DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"


def _fake_collector(raw: dict):
    class _FakeCollector:
        def __init__(self, **kwargs):
            pass

        def collect(self, address):
            return dict(raw)

    return _FakeCollector


# --- simulation requester end-to-end (against the local API via TestClient) --- #
def test_simulation_consumer_end_to_end(client, monkeypatch, healthy_features):
    raw = make_raw_token_data(healthy_features)
    monkeypatch.setattr(app_module, "OnChainCollector", _fake_collector(raw))

    from cap import consumer

    lines: list[str] = []
    # Hire the analyzer over the in-process TestClient — the "POST /analyze" hop,
    # with zero CROO network involvement.
    fetcher = lambda token, chain: client.post(
        "/analyze", json={"contract_address": token, "chain": chain}
    ).json()

    result = consumer.run_simulation(_DAI, "ethereum", report_fetcher=fetcher, log=lines.append)

    assert result["mode"] == "simulation"
    report = result["report"]
    assert report["contract_address"] == _DAI
    assert report["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert 0 <= report["trust_score"] <= 100

    out = "\n".join(lines)
    assert "[SIMULATION]" in out
    for stage in ("POST", "LOCK", "DELIVER", "CLEAR"):
        assert stage in out              # the full Post->Lock->Deliver->Clear narration
    assert "Trust Report received" in out
    assert "trust_score" in out


def test_run_consumer_routes_to_simulation(monkeypatch):
    # force_mode='simulation' must take the simulation path; stub the local fetch
    # so there's no HTTP call and no in-process pipeline import.
    from cap import consumer

    monkeypatch.setattr(
        consumer, "_default_local_fetch",
        lambda token, chain, base=None: {
            "contract_address": token, "chain": chain,
            "trust_score": 5, "risk_level": "LOW", "flags": [],
            "token": {"symbol": "DAI"}, "score_breakdown": {"confidence": "HIGH"},
        },
    )
    lines: list[str] = []
    result = consumer.run_consumer(_DAI, "ethereum", force_mode="simulation", log=lines.append)
    assert result["mode"] == "simulation"
    assert result["report"]["risk_level"] == "LOW"
    assert any("SIMULATION" in ln for ln in lines)


# --- mode selection / requester identity ------------------------------------ #
def test_config_without_croo_env_is_not_live(monkeypatch):
    for var in ("CROO_SDK_KEY", "CONSUMER_CROO_SDK_KEY", "CROO_API_URL", "CROO_WS_URL", "CROO_SERVICE_ID"):
        monkeypatch.delenv(var, raising=False)
    from cap import consumer

    cfg = consumer.consumer_config()
    assert consumer._is_live(cfg) is False


def test_consumer_key_prefers_consumer_env(monkeypatch):
    monkeypatch.setenv("CROO_SDK_KEY", "provider_key")
    monkeypatch.setenv("CONSUMER_CROO_SDK_KEY", "consumer_key")
    from cap import consumer

    assert consumer.consumer_config().sdk_key == "consumer_key"


def test_consumer_key_falls_back_to_croo_key(monkeypatch):
    monkeypatch.setenv("CROO_SDK_KEY", "provider_key")
    monkeypatch.delenv("CONSUMER_CROO_SDK_KEY", raising=False)
    from cap import consumer

    assert consumer.consumer_config().sdk_key == "provider_key"
