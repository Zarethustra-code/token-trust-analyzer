"""Tests for the SLM-written risk narrative.

Fully OFFLINE: the SLM is mocked (``narrative._run_slm`` or, for the unavailable
path, the ``transformers`` import seam) — no model downloads, ever. These tests
prove the narrative is *grounded* (facts-only prompt), correctly *gated*
(env + per-request override), *bounded* (empty/short/long -> null), and that it
never touches the deterministic score / flags / explanation.
"""

from __future__ import annotations

import pytest

import app as app_module
import detectors.ai_content_detector as det_mod
import detectors.local_ai_detector as local_mod
import ml.narrative as narrative_mod
from models.response import (
    AIContentResult,
    DataQuality,
    RiskLevel,
    RulePenalty,
    ScoreBreakdown,
    TokenInfo,
    TokenMetrics,
    TrustReport,
)
from tests.conftest import make_raw_token_data

_DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"

_GOOD_NARRATIVE = (
    "Although liquidity is locked and ownership is renounced, the elevated top-10 "
    "concentration alongside the contract's young age keeps this token in "
    "medium-risk territory."
)


@pytest.fixture(autouse=True)
def _reset_shared_slm():
    """Drop the shared SLM + detector cache so mocked import failures don't leak."""
    det_mod._DETECTORS.clear()
    local_mod.reset_shared_local_detector()
    yield
    det_mod._DETECTORS.clear()
    local_mod.reset_shared_local_detector()


def _fake_collector(raw: dict):
    class _FakeCollector:
        def __init__(self, **kwargs):
            pass

        def collect(self, address):
            return dict(raw)

    return _FakeCollector


def _make_report(
    trust_score: int = 40,
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    fired: list[RulePenalty] | None = None,
    **metric_overrides,
) -> TrustReport:
    """A representative finished report the narrative can be grounded on."""
    if fired is None:
        fired = [
            RulePenalty(
                rule="liquidity_not_locked", points=12.0,
                flag="Liquidity is not locked", feature="liquidity_locked",
            ),
            RulePenalty(
                rule="high_top10_concentration", points=8.0,
                flag="Top-10 holders control a large share of supply",
                feature="top10_holder_pct",
            ),
        ]
    metrics_kwargs = dict(
        top_holder_pct=6.0, top10_holder_pct=72.0, holder_count=1200,
        liquidity_locked=True, ownership_renounced=True, contract_age_days=15,
        is_honeypot=False, has_mint=False, buy_tax=0.0, sell_tax=0.0,
    )
    metrics_kwargs.update(metric_overrides)
    breakdown = ScoreBreakdown(
        rule_penalties=fired,
        rule_penalty_total=sum(p.points for p in fired),
        anomaly_score=55.0, anomaly_weight=0.35, anomaly_contribution=6.2,
        data_completeness=0.95, completeness_factor=0.95, confidence="HIGH",
        raw_total=float(trust_score),
    )
    return TrustReport(
        contract_address="0x" + "1" * 40, chain="ethereum",
        token=TokenInfo(name="Demo", symbol="DEMO"),
        trust_score=trust_score, risk_level=risk_level,
        flags=[p.flag for p in fired],
        metrics=TokenMetrics(**metrics_kwargs),
        score_breakdown=breakdown,
        ai_generated_content=AIContentResult(checked=False),
        data_quality=DataQuality(sources_used=["goplus"]),
        explanation=f"Risk score {trust_score}/100 ({risk_level.value}). Higher means riskier.",
    )


# --- facts extraction --------------------------------------------------------- #
def test_facts_carry_score_rules_and_unknowns():
    report = _make_report(top_holder_pct=None, contract_age_days=None)
    facts = narrative_mod.build_narrative_facts(report)

    assert facts["trust_score"] == 40
    assert facts["risk_level"] == "MEDIUM"
    assert facts["confidence"] == "HIGH"
    # nulls surface as "unknown" — never invented
    assert facts["metrics"]["top holder share"] == "unknown"
    assert facts["metrics"]["contract age (days)"] == "unknown"
    # booleans render yes/no
    assert facts["metrics"]["liquidity locked"] == "yes"
    assert facts["metrics"]["honeypot"] == "no"
    # every fired rule carried through with its points
    flags = {r["flag"]: r["points"] for r in facts["fired_rules"]}
    assert flags["Liquidity is not locked"] == 12.0
    assert flags["Top-10 holders control a large share of supply"] == 8.0


# --- grounding: prompt is facts-only ------------------------------------------ #
def test_narrative_prompt_is_grounded(monkeypatch):
    """With a mocked generate, the prompt must carry the fired flags, the
    trust_score and the risk_level, and forbid inventing anything else."""
    report = _make_report()
    captured: dict = {}

    def fake_run(prompt: str):
        captured["prompt"] = prompt
        return _GOOD_NARRATIVE

    monkeypatch.setattr(narrative_mod, "_run_slm", fake_run)
    result = narrative_mod.generate_narrative(report)

    assert result == _GOOD_NARRATIVE
    prompt = captured["prompt"]
    assert "Liquidity is not locked" in prompt                     # fired flag 1
    assert "Top-10 holders control a large share of supply" in prompt  # fired flag 2
    assert "40 out of 100" in prompt                                # trust_score
    assert "MEDIUM" in prompt                                       # risk_level
    assert "Use ONLY the facts" in prompt                          # facts-only guardrail
    assert "do NOT contradict the stated risk level" in prompt.replace("\n", " ")


def test_prompt_lists_no_rules_when_none_fired():
    report = _make_report(trust_score=5, risk_level=RiskLevel.LOW, fired=[])
    prompt = narrative_mod.build_narrative_prompt(
        narrative_mod.build_narrative_facts(report)
    )
    assert "Rules that fired: none." in prompt
    assert "5 out of 100" in prompt
    assert "LOW" in prompt


# --- output sanity bounds ----------------------------------------------------- #
@pytest.mark.parametrize(
    "raw",
    ["", "   ", "\n\n", "too short to keep", "x" * 901],
    ids=["empty", "blank", "newlines", "under-30", "over-900"],
)
def test_out_of_bounds_output_becomes_null(monkeypatch, raw):
    report = _make_report()
    monkeypatch.setattr(narrative_mod, "_run_slm", lambda _p: raw)
    assert narrative_mod.generate_narrative(report) is None


def test_whitespace_is_collapsed(monkeypatch):
    report = _make_report()
    monkeypatch.setattr(
        narrative_mod, "_run_slm",
        lambda _p: "  Liquidity is locked,\n\n   yet   concentration   stays high.  ",
    )
    out = narrative_mod.generate_narrative(report)
    assert out == "Liquidity is locked, yet concentration stays high."


# --- graceful degradation: SLM/deps unavailable ------------------------------- #
def test_generate_narrative_slm_unavailable_returns_none(monkeypatch):
    """transformers import forced to fail while requested -> null, no crash."""
    monkeypatch.setattr(
        local_mod,
        "_import_transformers",
        lambda: (_ for _ in ()).throw(ImportError("No module named 'transformers'")),
    )
    assert narrative_mod.generate_narrative(_make_report()) is None


def test_narrative_and_detector_share_one_slm(monkeypatch):
    """The never-a-second-copy guarantee: the local detector and the narrative
    writer resolve to the SAME LocalAIDetector instance."""
    monkeypatch.delenv("AI_DETECTOR_BACKEND", raising=False)
    from detectors.ai_content_detector import get_detector

    assert get_detector() is local_mod.get_shared_local_detector()


# --- gating matrix (env + per-request override) via /analyze ------------------ #
def _analyze(client, monkeypatch, healthy_features, **body_extra):
    raw = make_raw_token_data(healthy_features)
    monkeypatch.setattr(app_module, "OnChainCollector", _fake_collector(raw))
    body = {"contract_address": _DAI, "chain": "ethereum", **body_extra}
    resp = client.post("/analyze", json=body)
    assert resp.status_code == 200
    return resp.json()


def test_gate_env_off_yields_null(client, monkeypatch, healthy_features):
    monkeypatch.delenv("RISK_NARRATIVE", raising=False)  # default is off
    called: list = []
    monkeypatch.setattr(
        app_module, "generate_narrative",
        lambda r: called.append(1) or _GOOD_NARRATIVE,
    )
    body = _analyze(client, monkeypatch, healthy_features)
    assert body["narrative"] is None
    assert not called  # gate closed -> generator never even invoked (keeps it fast)


def test_gate_env_on_populates(client, monkeypatch, healthy_features):
    monkeypatch.setenv("RISK_NARRATIVE", "on")
    monkeypatch.setattr(app_module, "generate_narrative", lambda r: _GOOD_NARRATIVE)
    body = _analyze(client, monkeypatch, healthy_features)
    assert body["narrative"] == _GOOD_NARRATIVE


def test_request_override_true_beats_env_off(client, monkeypatch, healthy_features):
    monkeypatch.setenv("RISK_NARRATIVE", "off")
    monkeypatch.setattr(app_module, "generate_narrative", lambda r: _GOOD_NARRATIVE)
    body = _analyze(client, monkeypatch, healthy_features, include_narrative=True)
    assert body["narrative"] == _GOOD_NARRATIVE


def test_request_override_false_beats_env_on(client, monkeypatch, healthy_features):
    monkeypatch.setenv("RISK_NARRATIVE", "on")
    called: list = []
    monkeypatch.setattr(
        app_module, "generate_narrative",
        lambda r: called.append(1) or _GOOD_NARRATIVE,
    )
    body = _analyze(client, monkeypatch, healthy_features, include_narrative=False)
    assert body["narrative"] is None
    assert not called


# --- graceful path + explanation untouched, end-to-end ------------------------ #
def test_unavailable_slm_notes_and_keeps_report(client, monkeypatch, healthy_features):
    """Requested but SLM unavailable -> narrative null + a data_quality note, and
    the deterministic report is fully intact."""
    monkeypatch.setenv("RISK_NARRATIVE", "on")
    monkeypatch.setattr(
        local_mod,
        "_import_transformers",
        lambda: (_ for _ in ()).throw(ImportError("transformers unavailable")),
    )
    body = _analyze(client, monkeypatch, healthy_features)

    assert body["narrative"] is None
    assert "narrative unavailable" in body["data_quality"]["notes"]
    # the deterministic pipeline is untouched
    assert 0 <= body["trust_score"] <= 100
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert body["explanation"].startswith("Risk score")


def test_explanation_unchanged_when_narrative_present(client, monkeypatch, healthy_features):
    monkeypatch.setenv("RISK_NARRATIVE", "on")
    monkeypatch.setattr(app_module, "generate_narrative", lambda r: _GOOD_NARRATIVE)
    body = _analyze(client, monkeypatch, healthy_features)
    # narrative is additive: the templated explanation is still the source of truth
    assert body["narrative"] == _GOOD_NARRATIVE
    assert body["explanation"].startswith("Risk score")
    assert "Higher means riskier" in body["explanation"]


def test_cap_analyze_gets_narrative(client, monkeypatch, healthy_features):
    """The CAP-delivered report (paid product) inherits the same gate."""
    monkeypatch.setenv("RISK_NARRATIVE", "on")
    monkeypatch.setattr(app_module, "generate_narrative", lambda r: _GOOD_NARRATIVE)
    raw = make_raw_token_data(healthy_features)
    monkeypatch.setattr(app_module, "OnChainCollector", _fake_collector(raw))
    resp = client.post("/cap/analyze", json={"contract_address": _DAI})
    assert resp.status_code == 200
    assert resp.json()["report"]["narrative"] == _GOOD_NARRATIVE


# --- /score stays untouched (pure-ML, no SLM) --------------------------------- #
def test_score_endpoint_has_no_narrative(client, monkeypatch, healthy_features):
    monkeypatch.setenv("RISK_NARRATIVE", "on")  # must not affect /score
    monkeypatch.setattr(
        app_module, "generate_narrative",
        lambda r: (_ for _ in ()).throw(AssertionError("/score must not build a narrative")),
    )
    resp = client.post("/score", json={"features": healthy_features})
    assert resp.status_code == 200
    assert resp.json()["narrative"] is None
