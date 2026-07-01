"""Tests for features/feature_extractor.py."""

from __future__ import annotations

from features.feature_extractor import FEATURE_ORDER, FeatureExtractor, compute_gini


def test_full_dict_produces_full_vector(healthy_features):
    fs = FeatureExtractor().extract(healthy_features)
    assert len(fs.vector) == len(FEATURE_ORDER)
    assert all(isinstance(x, float) for x in fs.vector)
    assert fs.imputed_features == []                      # nothing was missing
    # raw view mirrors the schema exactly.
    assert set(fs.raw) == set(FEATURE_ORDER)


def test_missing_keys_are_imputed_and_tracked(healthy_features):
    partial = dict(healthy_features)
    for key in ("buy_sell_ratio", "recent_tx_count", "liquidity_to_mcap_ratio"):
        partial.pop(key)
    partial["contract_age_days"] = None                  # explicit None also counts

    fs = FeatureExtractor().extract(partial)

    for key in ("buy_sell_ratio", "recent_tx_count", "liquidity_to_mcap_ratio", "contract_age_days"):
        assert key in fs.imputed_features
        assert fs.raw[key] is None                       # raw preserves the gap
    # vector is still full length and all-float (imputed values filled in).
    assert len(fs.vector) == len(FEATURE_ORDER)
    assert all(isinstance(x, float) for x in fs.vector)
    # imputed view has no None.
    assert all(v is not None for v in fs.imputed.values())


def test_gini_derived_from_balances():
    even = compute_gini([10, 10, 10, 10, 10])
    concentrated = compute_gini([95, 2, 1, 1, 1])
    assert even < 0.05
    assert concentrated > 0.7
    assert compute_gini([]) is None

    # extractor derives gini from a balances list when gini isn't supplied.
    raw = {"top_holder_balances": [95, 2, 1, 1, 1]}
    fs = FeatureExtractor().extract(raw)
    assert fs.raw["gini"] is not None and fs.raw["gini"] > 0.7
