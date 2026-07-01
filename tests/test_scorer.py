"""Tests for ml/scorer.py, including the data-completeness anomaly weighting."""

from __future__ import annotations

import dataclasses

from features.feature_extractor import FEATURE_ORDER, FeatureExtractor
from ml.scorer import (
    ANOMALY_COMPLETENESS_FLOOR,
    W_ANOMALY,
    TrustScorer,
    _confidence_label,
)
from models.response import RiskLevel


class _StubAnomaly:
    """Returns a fixed anomaly score regardless of the vector (deterministic)."""

    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, vector) -> float:
        return self._score


class _StubSupervised:
    def predict_proba(self, vector):
        return None


def _scorer(score: float = 50.0) -> TrustScorer:
    return TrustScorer(anomaly_model=_StubAnomaly(score), supervised_model=_StubSupervised())


def _with_imputed(fs, names):
    """Same token/vector, but marked as having `names` imputed (isolates completeness)."""
    return dataclasses.replace(fs, imputed_features=list(names))


# --- risk levels ------------------------------------------------------------ #
def test_healthy_is_low_high_confidence(healthy_features):
    fs = FeatureExtractor().extract(healthy_features)  # 0 imputed
    res = _scorer().score(fs)
    assert res.risk_level is RiskLevel.LOW
    assert res.breakdown.confidence == "HIGH"
    assert res.breakdown.data_completeness == 1.0


def test_scam_is_high(scam_features):
    fs = FeatureExtractor().extract(scam_features)
    res = _scorer().score(fs)
    assert res.risk_level is RiskLevel.HIGH
    assert res.trust_score == 100  # rules alone clear the clamp


# --- completeness weighting (the new behavior) ------------------------------ #
def test_completeness_down_weights_anomaly(healthy_features):
    base = FeatureExtractor().extract(healthy_features)
    scorer = _scorer(50.0)

    partial = _with_imputed(base, ["buy_sell_ratio", "recent_tx_count", "gini"])  # 3/20 imputed
    bd = scorer.score(partial).breakdown

    assert bd.data_completeness < 1.0
    assert bd.data_completeness == 1.0 - 3 / len(FEATURE_ORDER)
    assert bd.completeness_factor == max(ANOMALY_COMPLETENESS_FLOOR, bd.data_completeness)
    # anomaly_contribution == weight * score * factor (the documented formula)
    assert bd.anomaly_weight == W_ANOMALY
    assert bd.anomaly_contribution == round(
        bd.anomaly_weight * bd.anomaly_score * bd.completeness_factor, 2
    )


def test_confidence_band_matches_labels(healthy_features):
    base = FeatureExtractor().extract(healthy_features)
    scorer = _scorer()
    for n_imputed in (0, 2, 8, 14):
        fs = _with_imputed(base, FEATURE_ORDER[:n_imputed])
        bd = scorer.score(fs).breakdown
        assert bd.confidence == _confidence_label(bd.data_completeness)
    # spot-check the thresholds directly
    assert _confidence_label(1.0) == "HIGH"
    assert _confidence_label(0.85) == "HIGH"
    assert _confidence_label(0.75) == "MEDIUM"
    assert _confidence_label(0.60) == "MEDIUM"
    assert _confidence_label(0.59) == "LOW"


def test_completeness_floor_applies(healthy_features):
    base = FeatureExtractor().extract(healthy_features)
    fs = _with_imputed(base, FEATURE_ORDER[:18])  # completeness 0.1 -> below floor
    bd = _scorer().score(fs).breakdown
    assert bd.data_completeness == 0.1
    assert bd.completeness_factor == ANOMALY_COMPLETENESS_FLOOR
    assert bd.confidence == "LOW"


def test_more_imputed_means_smaller_contribution(healthy_features):
    base = FeatureExtractor().extract(healthy_features)
    scorer = _scorer(80.0)  # same fixed anomaly score, same vector
    few = scorer.score(_with_imputed(base, FEATURE_ORDER[:2])).breakdown   # 90% complete
    many = scorer.score(_with_imputed(base, FEATURE_ORDER[:12])).breakdown  # 40% complete
    assert few.anomaly_score == many.anomaly_score          # same token
    assert few.anomaly_contribution > many.anomaly_contribution


# --- totals / clamp / explanation ------------------------------------------ #
def test_raw_total_and_clamp(scam_features, healthy_features):
    for feats in (scam_features, healthy_features):
        res = _scorer().score(FeatureExtractor().extract(feats))
        bd = res.breakdown
        assert bd.raw_total == round(
            bd.rule_penalty_total + bd.anomaly_contribution + bd.supervised_contribution, 2
        )
        assert res.trust_score == int(round(min(100.0, max(0.0, bd.raw_total))))
        assert 0 <= res.trust_score <= 100


def test_explanation_mentions_completeness_when_imputed(healthy_features):
    base = FeatureExtractor().extract(healthy_features)
    scorer = _scorer()

    imputed = _with_imputed(base, ["buy_sell_ratio", "recent_tx_count"])
    expl = scorer.score(imputed).explanation.lower()
    assert "completeness" in expl
    assert "buy_sell_ratio" in expl

    full = scorer.score(base).explanation.lower()  # 0 imputed
    assert "completeness" not in full
