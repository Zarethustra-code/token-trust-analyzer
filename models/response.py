"""Output schema: the structured Trust Report.

Every field the pipeline can surface lives here. The report is intentionally
*explainable*: `flags` and `score_breakdown` tie the final number back to the
specific features that produced it.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """LOW < 30 <= MEDIUM < 60 <= HIGH (higher score = riskier)."""
        if score < 30:
            return cls.LOW
        if score < 60:
            return cls.MEDIUM
        return cls.HIGH


class TokenInfo(BaseModel):
    """Basic on-chain identity of the token (best-effort; fields may be null)."""

    model_config = {"extra": "forbid"}

    name: Optional[str] = None
    symbol: Optional[str] = None
    decimals: Optional[int] = None
    total_supply: Optional[float] = None


class TokenMetrics(BaseModel):
    """Human-facing key metrics that back the score (raw, un-imputed values)."""

    model_config = {"extra": "forbid"}

    top_holder_pct: Optional[float] = Field(
        None, description="Share of supply held by the single largest holder (%)."
    )
    top10_holder_pct: Optional[float] = Field(
        None, description="Share of supply held by the top 10 holders (%)."
    )
    holder_count: Optional[int] = None
    gini: Optional[float] = Field(
        None, description="Gini coefficient of holder distribution (0=even, 1=concentrated)."
    )
    creator_percent: Optional[float] = Field(
        None, description="Share of supply still held by the contract creator (%)."
    )
    liquidity_locked: Optional[bool] = None
    liquidity_to_mcap_ratio: Optional[float] = None
    source_verified: Optional[bool] = None
    has_mint: Optional[bool] = None
    ownership_renounced: Optional[bool] = None
    has_blacklist: Optional[bool] = None
    is_honeypot: Optional[bool] = Field(
        None, description="Whether the token behaves as a honeypot (buyers can't sell)."
    )
    buy_tax: Optional[float] = Field(None, description="Buy tax as a percentage (0-100).")
    sell_tax: Optional[float] = Field(None, description="Sell tax as a percentage (0-100).")
    hidden_owner: Optional[bool] = None
    can_take_back_ownership: Optional[bool] = None
    is_anti_whale: Optional[bool] = None
    contract_age_days: Optional[float] = None
    recent_tx_count: Optional[int] = None
    buy_sell_ratio: Optional[float] = None


class RulePenalty(BaseModel):
    """One fired heuristic rule and the points it contributed."""

    model_config = {"extra": "forbid"}

    rule: str = Field(..., description="Machine name of the rule, e.g. 'liquidity_not_locked'.")
    points: float = Field(..., description="Penalty points this rule added to the risk score.")
    flag: str = Field(..., description="Human-readable explanation of what tripped the rule.")
    feature: str = Field(..., description="The feature this rule inspected (traceability).")


class ScoreBreakdown(BaseModel):
    """Exactly how the final score was assembled, for full explainability."""

    model_config = {"extra": "forbid"}

    rule_penalties: list[RulePenalty] = Field(default_factory=list)
    rule_penalty_total: float = 0.0
    anomaly_score: float = Field(
        0.0, description="Isolation Forest anomaly contribution, normalized 0-100."
    )
    anomaly_weight: float = 0.0
    anomaly_contribution: float = Field(
        0.0, description="anomaly_score * anomaly_weight * completeness_factor (points added)."
    )
    data_completeness: float = Field(
        1.0, description="Fraction of features directly observed (1 - imputed/total)."
    )
    completeness_factor: float = Field(
        1.0, description="Down-weight applied to the anomaly contribution (>= floor)."
    )
    confidence: str = Field(
        "HIGH", description="Confidence band from data completeness: HIGH/MEDIUM/LOW."
    )
    supervised_prob: Optional[float] = Field(
        None, description="XGBoost P(scam) in [0,100], or null if no model is loaded."
    )
    supervised_weight: float = 0.0
    supervised_contribution: float = 0.0
    raw_total: float = Field(
        0.0, description="Sum of all contributions before clamping to [0,100]."
    )


class AIContentResult(BaseModel):
    """Result of the (optional) AI-generated-marketing-text detector."""

    model_config = {"extra": "forbid"}

    checked: bool = Field(False, description="Whether the detector actually ran.")
    is_ai_generated: Optional[bool] = None
    confidence: Optional[float] = Field(None, description="Detector confidence 0-1, if available.")
    reason: Optional[str] = None
    source: Optional[str] = Field(
        None,
        description="Origin of the analyzed text: 'provided_text', 'fetched_url', or null.",
    )


class DataQuality(BaseModel):
    """Transparency about what could and couldn't be fetched."""

    model_config = {"extra": "forbid"}

    sources_used: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class TrustReport(BaseModel):
    """The full, structured Trust Report returned to the caller (and delivered on CAP)."""

    model_config = {"extra": "forbid"}

    contract_address: str
    chain: str
    token: TokenInfo = Field(default_factory=TokenInfo)

    trust_score: int = Field(..., ge=0, le=100, description="Risk score 0 (safe) - 100 (dangerous).")
    risk_level: RiskLevel
    flags: list[str] = Field(
        default_factory=list, description="Flat list of human-readable anomaly flags."
    )

    metrics: TokenMetrics = Field(default_factory=TokenMetrics)
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    ai_generated_content: AIContentResult = Field(default_factory=AIContentResult)
    data_quality: DataQuality = Field(default_factory=DataQuality)

    explanation: str = Field(..., description="Plain-language justification for the score.")
    narrative: Optional[str] = Field(
        default=None,
        description=(
            "Optional analyst-style risk summary (2–4 sentences) written by the "
            "local SLM, phrasing the deterministic signals. Null when disabled "
            "(RISK_NARRATIVE=off) or the SLM is unavailable; the scores, flags and "
            "explanation are always produced deterministically regardless."
        ),
    )
    generated_at: Optional[str] = Field(
        None, description="ISO-8601 UTC timestamp of when the report was produced."
    )
    cached: bool = Field(
        False, description="True when the underlying on-chain data was served from cache."
    )


class BatchResultItem(BaseModel):
    """One entry in a batch response — either a report or an error, never both."""

    model_config = {"extra": "forbid"}

    contract_address: str
    chain: str
    report: Optional[TrustReport] = None
    error: Optional[str] = Field(
        None, description="Error message if this token could not be analyzed."
    )


class AnalyzeBatchResponse(BaseModel):
    """Batch response: one result per requested token, in the same order."""

    model_config = {"extra": "forbid"}

    results: list[BatchResultItem] = Field(default_factory=list)
