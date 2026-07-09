"""Hybrid trust scorer.

Combines the three signals into a single 0-100 risk score:

    final = clamp( W_ANOMALY * anomaly_score * completeness_factor  # Isolation Forest, 0-100
                 + rule_penalty_total                 # interpretable heuristics
                 + W_SUPERVISED * supervised_prob*100  # XGBoost, 0 if no model
                 , 0, 100 )

Higher score = riskier. The scorer emits a full :class:`ScoreBreakdown` and a
plain-language ``explanation`` so every point in the final number is traceable
to a specific feature or model.

The anomaly term is **down-weighted by data completeness**: when many features
were unavailable and imputed, the Isolation Forest is working on a partly-guessed
profile, so its contribution is scaled by ``completeness_factor`` (floored at
``ANOMALY_COMPLETENESS_FLOOR``). Rule penalties are left at full strength — they
rest on directly-observed GoPlus facts, not imputed values.
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

# The anomaly contribution is scaled by data completeness, but never below this
# floor — even a sparse profile carries *some* anomaly signal.
ANOMALY_COMPLETENESS_FLOOR = 0.35

# Below this completeness AND with no rule fired, we have observed too little
# to classify — verdict is INCONCLUSIVE, never a misleading "LOW risk".
INCONCLUSIVE_COMPLETENESS = 0.4


def _confidence_label(completeness: float) -> str:
    """Map data completeness (0..1) to a confidence band."""
    if completeness >= 0.85:
        return "HIGH"
    if completeness >= 0.60:
        return "MEDIUM"
    return "LOW"


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

        # 2) Isolation Forest anomaly contribution, down-weighted by how much of
        #    the feature vector was directly observed (vs imputed).
        anomaly_score = self.anomaly.score(feature_set.vector)
        data_completeness = 1.0 - len(feature_set.imputed_features) / len(feature_set.vector)
        completeness_factor = max(ANOMALY_COMPLETENESS_FLOOR, data_completeness)
        anomaly_contribution = W_ANOMALY * anomaly_score * completeness_factor
        confidence = _confidence_label(data_completeness)

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

        # A data-starved token with no rule fired must not be labelled "LOW risk":
        # a low score there reflects missing data, not verified safety. A fired
        # rule is hard, directly-observed evidence, so it overrides the guard and
        # honeypots etc. still surface with their real level.
        if data_completeness < INCONCLUSIVE_COMPLETENESS and not fired:
            risk_level = RiskLevel.INCONCLUSIVE
        else:
            risk_level = RiskLevel.from_score(final)

        breakdown = ScoreBreakdown(
            rule_penalties=rule_penalties,
            rule_penalty_total=rule_total,
            anomaly_score=round(anomaly_score, 2),
            anomaly_weight=W_ANOMALY,
            anomaly_contribution=round(anomaly_contribution, 2),
            data_completeness=round(data_completeness, 3),
            completeness_factor=round(completeness_factor, 3),
            confidence=confidence,
            supervised_prob=None if supervised_prob100 is None else round(supervised_prob100, 2),
            supervised_weight=W_SUPERVISED,
            supervised_contribution=round(supervised_contribution, 2),
            raw_total=round(raw_total, 2),
        )

        flags = [r.flag for r in fired]
        explanation = self._explain(
            final, risk_level, fired, anomaly_score, anomaly_contribution,
            supervised_prob100, feature_set, data_completeness, confidence,
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
        data_completeness: float,
        confidence: str,
    ) -> str:
        if level == RiskLevel.INCONCLUSIVE:
            parts: list[str] = [
                f"INCONCLUSIVE: only {data_completeness * 100:.0f}% of the expected "
                "on-chain data could be observed and no risk rule was triggered, so "
                "there is not enough reliable information to classify this token. A "
                "low score here reflects missing data, not verified safety — do not "
                "treat this token as safe."
            ]
        else:
            parts = [f"Risk score {final}/100 ({level.value}). Higher means riskier."]
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
                f"Data completeness {data_completeness * 100:.0f}% ({confidence} "
                "confidence): the following features were unavailable and imputed to "
                "a healthy-token baseline, so the anomaly contribution was "
                "down-weighted accordingly: "
                f"{', '.join(feature_set.imputed_features)}."
            )
        return " ".join(parts)
