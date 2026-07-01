"""Tests for ml/rules.py — interpretable heuristic penalties."""

from __future__ import annotations

from features.feature_extractor import FEATURE_ORDER
from ml.rules import FiredRule, evaluate_rules, total_penalty


def test_healthy_fires_no_rules(healthy_features):
    fired = evaluate_rules(healthy_features)
    assert fired == []
    assert total_penalty(fired) == 0.0


def test_scam_fires_expected_rules(scam_features):
    fired = evaluate_rules(scam_features)
    by_name = {r.rule: r for r in fired}

    # The headline scam signals must all fire.
    expected = {
        "honeypot": "is_honeypot",
        "high_sell_tax": "sell_tax",
        "high_buy_tax": "buy_tax",
        "hidden_owner": "hidden_owner",
        "can_take_back_ownership": "can_take_back_ownership",
        "source_not_verified": "source_verified",
        "active_mint_function": "has_mint",
        "blacklist_capability": "has_blacklist",
        "liquidity_not_locked": "liquidity_locked",
        "whale_top_holder": "top_holder_pct",
        "creator_concentration": "creator_percent",
        "ownership_not_renounced": "ownership_renounced",
        "very_new_contract": "contract_age_days",
        "top10_concentration": "top10_holder_pct",
    }
    for rule_name, feature in expected.items():
        assert rule_name in by_name, f"expected rule {rule_name!r} to fire"
        assert by_name[rule_name].feature == feature


def test_fired_rules_are_well_formed(scam_features):
    fired = evaluate_rules(scam_features)
    assert fired, "scam profile should fire rules"
    for r in fired:
        assert isinstance(r, FiredRule)
        assert isinstance(r.flag, str) and r.flag.strip()          # non-empty flag
        assert r.feature in FEATURE_ORDER                          # traceable feature
        assert r.points > 0


def test_total_penalty_equals_sum(scam_features):
    fired = evaluate_rules(scam_features)
    assert total_penalty(fired) == sum(r.points for r in fired)
    # honeypot alone is +40, so the scam profile clears 100 easily.
    assert total_penalty(fired) >= 100


def test_rules_skip_unknown_features():
    # Every feature None => nothing can be confirmed => no rule fires.
    raw = {name: None for name in FEATURE_ORDER}
    assert evaluate_rules(raw) == []
