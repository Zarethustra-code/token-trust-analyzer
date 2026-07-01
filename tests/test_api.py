"""Endpoint tests via FastAPI TestClient. Every external call is mocked (offline)."""

from __future__ import annotations

import app as app_module
from models.response import AIContentResult
from tests.conftest import make_raw_token_data

_DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"


def _fake_collector(raw: dict):
    class _FakeCollector:
        def __init__(self, **kwargs):
            pass

        def collect(self, address):
            return dict(raw)

    return _FakeCollector


# --- /ui -------------------------------------------------------------------- #
def test_ui_serves_html(client):
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Token Trust Analyzer" in resp.text
    assert "/analyze" in resp.text  # the page calls the analyze endpoint


# --- /health ---------------------------------------------------------------- #
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["anomaly_model_ready"] is True


# --- /score ----------------------------------------------------------------- #
def test_score_healthy(client, healthy_features):
    resp = client.post("/score", json={"features": healthy_features})
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["trust_score"] <= 100
    assert body["risk_level"] == "LOW"
    breakdown = body["score_breakdown"]
    for key in ("data_completeness", "completeness_factor", "confidence"):
        assert key in breakdown


def test_score_scam(client, scam_features):
    resp = client.post("/score", json={"features": scam_features})
    assert resp.status_code == 200
    assert resp.json()["risk_level"] == "HIGH"


def test_score_unknown_feature_key(client):
    # NOTE: the app deliberately returns 400 (not 422) for a semantically-valid
    # body whose feature keys aren't in FEATURE_ORDER.
    resp = client.post("/score", json={"features": {"not_a_feature": 1.0}})
    assert resp.status_code == 400


def test_score_malformed_body(client):
    resp = client.post("/score", json={"nope": 1})  # missing required 'features'
    assert resp.status_code == 422


# --- /analyze --------------------------------------------------------------- #
def test_analyze_with_mocked_collector(client, monkeypatch, healthy_features):
    raw = make_raw_token_data(healthy_features)
    monkeypatch.setattr(app_module, "OnChainCollector", _fake_collector(raw))

    resp = client.post("/analyze", json={"contract_address": _DAI, "chain": "ethereum"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["contract_address"] == _DAI
    assert body["chain"] == "ethereum"
    assert 0 <= body["trust_score"] <= 100
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert body["token"]["symbol"] == "TST"
    assert "data_completeness" in body["score_breakdown"]


# --- /detect-ai ------------------------------------------------------------- #
def test_detect_ai_with_mocked_detector(client, monkeypatch):
    class _FakeDetector:
        def detect(self, project_text):
            return AIContentResult(checked=True, is_ai_generated=False, reason="mock")

    monkeypatch.setattr(app_module, "get_detector", lambda: _FakeDetector())

    resp = client.post("/detect-ai", json={"project_text": "our revolutionary synergistic protocol"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["checked"] is True
    assert body["is_ai_generated"] is False


# --- /cap/analyze ----------------------------------------------------------- #
def test_cap_analyze_simulates_lifecycle(client, monkeypatch, healthy_features):
    raw = make_raw_token_data(healthy_features)
    monkeypatch.setattr(app_module, "OnChainCollector", _fake_collector(raw))

    resp = client.post("/cap/analyze", json={"contract_address": _DAI})
    assert resp.status_code == 200
    body = resp.json()
    assert "cap" in body and "report" in body
    assert body["cap"]["settled"] is False
    stages = [step["stage"] for step in body["cap"]["lifecycle"]]
    assert stages == ["POST", "LOCK", "DELIVER", "CLEAR"]
    assert body["report"]["risk_level"] in ("LOW", "MEDIUM", "HIGH")
