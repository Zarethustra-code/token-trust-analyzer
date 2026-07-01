"""Rule-based heuristic penalties.

These are the *interpretable* half of the hybrid scorer. Each rule inspects one
feature and, if a confirmed risk is present, returns a penalty and a
human-readable flag. Rules are deliberately conservative: if the feature is
``None`` (unknown), the rule is **skipped** — we never penalize on missing data
(that neutrality is handled by imputation on the ML side instead).

Every penalty is traceable back to the exact feature that produced it, which is
what makes the final score explainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class FiredRule:
    rule: str
    points: float
    flag: str
    feature: str


@dataclass(frozen=True)
class _RuleDef:
    name: str
    feature: str
    points: float
    # predicate(value) -> True if the risk condition holds. Only called for non-None.
    predicate: Callable[[float], bool]
    flag: str


# Point values match the specification.
_RULES: list[_RuleDef] = [
    _RuleDef(
        name="liquidity_not_locked",
        feature="liquidity_locked",
        points=30.0,
        predicate=lambda v: v < 0.5,
        flag="Liquidity is not locked — the team could pull liquidity (rug pull risk).",
    ),
    _RuleDef(
        name="whale_top_holder",
        feature="top_holder_pct",
        points=25.0,
        predicate=lambda v: v > 50.0,
        flag="A single holder controls more than 50% of supply (extreme concentration).",
    ),
    _RuleDef(
        name="source_not_verified",
        feature="source_verified",
        points=20.0,
        predicate=lambda v: v < 0.5,
        flag="Contract source code is not verified — its behavior can't be audited.",
    ),
    _RuleDef(
        name="active_mint_function",
        feature="has_mint",
        points=15.0,
        predicate=lambda v: v >= 0.5,
        flag="Contract exposes a mint function — supply can be inflated.",
    ),
    _RuleDef(
        name="ownership_not_renounced",
        feature="ownership_renounced",
        points=10.0,
        predicate=lambda v: v < 0.5,
        flag="Ownership is not renounced — the owner retains privileged control.",
    ),
    _RuleDef(
        name="very_new_contract",
        feature="contract_age_days",
        points=10.0,
        predicate=lambda v: v < 3.0,
        flag="Contract is less than 3 days old — very little track record.",
    ),
    # A couple of additional interpretable checks that complement the required set.
    _RuleDef(
        name="blacklist_capability",
        feature="has_blacklist",
        points=15.0,
        predicate=lambda v: v >= 0.5,
        flag="Contract can blacklist/restrict transfers — holders may be blocked from selling.",
    ),
    _RuleDef(
        name="top10_concentration",
        feature="top10_holder_pct",
        points=10.0,
        predicate=lambda v: v > 80.0,
        flag="Top 10 holders control more than 80% of supply.",
    ),
    # --- GoPlus-derived signals (fed as raw features; these rules add the
    #     interpretable, per-feature penalties on top of the anomaly model). ---
    _RuleDef(
        name="honeypot",
        feature="is_honeypot",
        points=40.0,
        predicate=lambda v: v >= 0.5,
        flag="Token behaves as a HONEYPOT — buyers are unable to sell (funds trapped).",
    ),
    _RuleDef(
        name="high_sell_tax",
        feature="sell_tax",
        points=20.0,
        predicate=lambda v: v > 10.0,
        flag="Sell tax exceeds 10% — a large cut is taken on every sell (soft-rug risk).",
    ),
    _RuleDef(
        name="high_buy_tax",
        feature="buy_tax",
        points=10.0,
        predicate=lambda v: v > 10.0,
        flag="Buy tax exceeds 10% — an unusually large cut is taken on every buy.",
    ),
    _RuleDef(
        name="creator_concentration",
        feature="creator_percent",
        points=15.0,
        predicate=lambda v: v > 20.0,
        flag="The contract creator still holds more than 20% of supply.",
    ),
    _RuleDef(
        name="hidden_owner",
        feature="hidden_owner",
        points=20.0,
        predicate=lambda v: v >= 0.5,
        flag="Contract has a hidden owner — privileged control is concealed.",
    ),
    _RuleDef(
        name="can_take_back_ownership",
        feature="can_take_back_ownership",
        points=20.0,
        predicate=lambda v: v >= 0.5,
        flag="Ownership can be reclaimed after being renounced (renounce is not final).",
    ),
]


def evaluate_rules(raw_features: dict[str, Optional[float]]) -> list[FiredRule]:
    """Return the list of rules that fired, given raw (possibly-None) features."""
    fired: list[FiredRule] = []
    for rule in _RULES:
        value = raw_features.get(rule.feature)
        if value is None:
            continue  # unknown -> skip (never penalize on missing data)
        try:
            if rule.predicate(float(value)):
                fired.append(
                    FiredRule(
                        rule=rule.name,
                        points=rule.points,
                        flag=rule.flag,
                        feature=rule.feature,
                    )
                )
        except (TypeError, ValueError):
            continue
    return fired


def total_penalty(fired: list[FiredRule]) -> float:
    return float(sum(r.points for r in fired))
