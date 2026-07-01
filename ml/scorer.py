"""Hybrid trust scorer.

Combines the three signals into a single 0-100 risk score:

    final = clamp( W_ANOMALY * anomaly_score          # Isolation Forest, 0-100
                 + rule_penalty_total                 # interpretable heuristics
                 + W_SUPERVISED * supervised_prob*100  # XGBoost, 0 if no model
                 , 0, 100 )

Higher score = riskier. The scorer emits a full :class:`ScoreBreakdown` and a
plain-language ``explanation`` so every point in the final number is traceable
to a specific feature or model.
"""

from __future__ import annotations

from dataclasses import dataclass

from features.feature_extractor import FeatureSet
from models.response import RiskLevel, RulePenalty, ScoreBreakdown
from ml import rules as rules_mod
from ml.anomaly_model import AnomalyModel, get_anomaly_model
from ml.supervised_model import SupervisedModel, get_supervised_model

# Blend weights. Rule penalties enter at full strength (they are already in
# "risk points"); the two ML signals are down-weighted so no single model
# dominates. Tunable without touching the report contract.
W_ANOMALY = 0.35
W_SUPERVISED = 0.35


@dataclass
class ScoringResult:
    trust_score: int
    risk_level: RiskLevel
    flags: list[str]
    breakdown: ScoreBreakdown
    explanation: str


class TrustScorer:
    def __init__(
        self,
        anomaly_model: AnomalyModel | None = None,
        supervised_model: SupervisedModel | None = None,
    ) -> None:
        self._anomaly = anomaly_model
        self._supervised = supervised_model

    @property
    def anomaly(self) -> AnomalyModel:
        return self._anomaly or get_anomaly_model()

    @property
    def supervised(self) -> SupervisedModel:
        return self._supervised or get_supervised_model()

    def score(self, feature_set: FeatureSet) -> ScoringResult:
        # 1) Interpretable rule penalties.
        fired = rules_mod.evaluate_rules(feature_set.raw)
        rule_total = rules_mod.total_penalty(fired)
        rule_penalties = [
            RulePenalty(rule=r.rule, points=r.points, flag=r.flag, feature=r.feature)
            for r in fired
        ]

        # 2) Isolation Forest anomaly contribution.
        anomaly_score = self.anomaly.score(feature_set.vector)
        anomaly_contribution = W_ANOMALY * anomaly_score

        # 3) Optional supervised probability.
        supervised_prob01 = self.supervised.predict_proba(feature_set.vector)
        if supervised_prob01 is None:
            supervised_prob100 = None
            supervised_contribution = 0.0
        else:
            supervised_prob100 = supervised_prob01 * 100.0
            supervised_contribution = W_SUPERVISED * supervised_prob100

        raw_total = rule_total + anomaly_contribution + supervised_contribution
        final = int(round(max(0.0, min(100.0, raw_total))))
        risk_level = RiskLevel.from_score(final)

        breakdown = ScoreBreakdown(
            rule_penalties=rule_penalties,
            rule_penalty_total=rule_total,
            anomaly_score=round(anomaly_score, 2),
            anomaly_weight=W_ANOMALY,
            anomaly_contribution=round(anomaly_contribution, 2),
            supervised_prob=None if supervised_prob100 is None else round(supervised_prob100, 2),
            supervised_weight=W_SUPERVISED,
            supervised_contribution=round(supervised_contribution, 2),
            raw_total=round(raw_total, 2),
        )

        flags = [r.flag for r in fired]
        explanation = self._explain(
            final, risk_level, fired, anomaly_score, anomaly_contribution,
            supervised_prob100, feature_set,
        )
        return ScoringResult(
            trust_score=final,
            risk_level=risk_level,
            flags=flags,
            breakdown=breakdown,
            explanation=explanation,
        )

    @staticmethod
    def _explain(
        final: int,
        level: RiskLevel,
        fired: list[rules_mod.FiredRule],
        anomaly_score: float,
        anomaly_contribution: float,
        supervised_prob100: float | None,
        feature_set: FeatureSet,
    ) -> str:
        parts: list[str] = [
            f"Risk score {final}/100 ({level.value}). Higher means riskier."
        ]
        if fired:
            top = sorted(fired, key=lambda r: r.points, reverse=True)
            drivers = "; ".join(f"{r.flag} (+{r.points:g})" for r in top)
            parts.append(f"Rule-based risk drivers: {drivers}.")
        else:
            parts.append("No rule-based risk conditions were triggered.")

        parts.append(
            f"The Isolation Forest rated this token {anomaly_score:.0f}/100 on how "
            f"unusual its on-chain profile is versus healthy tokens, adding "
            f"{anomaly_contribution:.1f} points."
        )
        if supervised_prob100 is not None:
            parts.append(
                f"The supervised classifier estimated a {supervised_prob100:.0f}% "
                "probability of being a scam."
            )
        if feature_set.imputed_features:
            parts.append(
                "Note: the following features were unavailable and imputed to a "
                "healthy-token baseline (so they neither raised nor lowered the "
                f"score): {', '.join(feature_set.imputed_features)}."
            )
        return " ".join(parts)
