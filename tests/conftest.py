"""Shared fixtures + helpers for the Token Trust Analyzer test suite.

Everything here is OFFLINE — no test hits the network. External calls (GoPlus,
Etherscan, Web3, Anthropic, CROO) are stubbed by the individual tests via
``monkeypatch``; this module only provides deterministic inputs and a
``TestClient`` whose lifespan fits the Isolation Forest on the committed
150-row training set.
"""

from __future__ import annotations

import copy

import pytest

from features.feature_extractor import FEATURE_ORDER

# --------------------------------------------------------------------------- #
# Canonical feature dicts (all 20 features in FEATURE_ORDER)
# --------------------------------------------------------------------------- #
_HEALTHY = {
    "top_holder_pct": 6.0,
    "top10_holder_pct": 28.0,
    "holder_count": 120000,
    "gini": 0.58,
    "creator_percent": 1.5,
    "liquidity_locked": 1,
    "liquidity_to_mcap_ratio": 0.12,
    "source_verified": 1,
    "has_mint": 0,
    "ownership_renounced": 1,
    "has_blacklist": 0,
    "is_honeypot": 0,
    "buy_tax": 0.0,
    "sell_tax": 0.0,
    "hidden_owner": 0,
    "can_take_back_ownership": 0,
    "is_anti_whale": 0,
    "contract_age_days": 1200,
    "recent_tx_count": 600,
    "buy_sell_ratio": 1.0,
}

_SCAM = {
    "top_holder_pct": 85.0,
    "top10_holder_pct": 97.0,
    "holder_count": 50,
    "gini": 0.95,
    "creator_percent": 40.0,
    "liquidity_locked": 0,
    "liquidity_to_mcap_ratio": 0.01,
    "source_verified": 0,
    "has_mint": 1,
    "ownership_renounced": 0,
    "has_blacklist": 1,
    "is_honeypot": 1,
    "buy_tax": 20.0,
    "sell_tax": 60.0,
    "hidden_owner": 1,
    "can_take_back_ownership": 1,
    "is_anti_whale": 0,
    "contract_age_days": 1,
    "recent_tx_count": 30,
    "buy_sell_ratio": 3.0,
}

# Sanity: both dicts cover exactly the canonical feature schema.
assert set(_HEALTHY) == set(FEATURE_ORDER)
assert set(_SCAM) == set(FEATURE_ORDER)


@pytest.fixture
def healthy_features() -> dict:
    return copy.deepcopy(_HEALTHY)


@pytest.fixture
def scam_features() -> dict:
    return copy.deepcopy(_SCAM)


# --------------------------------------------------------------------------- #
# Fake GoPlus response (the raw per-token result shape the API returns)
# --------------------------------------------------------------------------- #
def make_goplus_entry(**overrides) -> dict:
    """Build a fake GoPlus ``result[<addr>]`` entry (all values are strings)."""
    entry = {
        "token_name": "Test Token",
        "token_symbol": "TST",
        "total_supply": "1000000000",
        "holder_count": "12345",
        # percents are FRACTIONS in GoPlus (0.05 == 5%)
        "holders": [
            {"percent": "0.05"}, {"percent": "0.04"}, {"percent": "0.03"},
            {"percent": "0.02"}, {"percent": "0.01"},
        ],
        "creator_percent": "0.02",
        "is_honeypot": "0",
        "is_mintable": "1",
        "is_blacklisted": "0",
        "is_open_source": "1",
        "hidden_owner": "0",
        "can_take_back_ownership": "0",
        "is_anti_whale": "0",
        "buy_tax": "0.01",
        "sell_tax": "0.02",
        "owner_address": "0x0000000000000000000000000000000000000000",  # renounced
        "lp_holders": [
            {"is_locked": 1, "percent": "0.9",
             "address": "0x000000000000000000000000000000000000dEaD"},
        ],
    }
    entry.update(overrides)
    return entry


def make_raw_token_data(features: dict, **extra) -> dict:
    """Build a collector-style RawTokenData dict from a feature dict.

    This is what ``OnChainCollector.collect`` returns and what the pipeline feeds
    to the extractor: feature keys + token_info + provenance fields.
    """
    raw = dict(features)
    raw.setdefault("top_holder_balances", None)
    raw["token_info"] = extra.get("token_info", {
        "name": "Test Token", "symbol": "TST", "decimals": 18, "total_supply": 1_000_000_000.0,
    })
    raw["sources_used"] = extra.get("sources_used", ["goplus"])
    raw["missing_fields"] = extra.get("missing_fields", [])
    raw["notes"] = extra.get("notes", [])
    return raw


# --------------------------------------------------------------------------- #
# FastAPI test client (lifespan fits the Isolation Forest once, offline)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as test_client:
        yield test_client
