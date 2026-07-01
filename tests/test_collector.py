"""Tests for collectors/onchain_collector.py — GoPlus/Etherscan mocked (offline)."""

from __future__ import annotations

import pytest

from collectors.onchain_collector import OnChainCollector
from features.feature_extractor import FEATURE_ORDER
from tests.conftest import make_goplus_entry

_DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"  # valid, checksummed


def _collector(monkeypatch, goplus_entry, etherscan_result=None, dex_metrics=None):
    """An OnChainCollector with GoPlus + Etherscan + DEX stubbed (no network, no keys)."""
    col = OnChainCollector(chain="ethereum")
    monkeypatch.setattr(col, "_goplus", lambda checksum: goplus_entry)
    monkeypatch.setattr(col, "_etherscan", lambda *a, **k: etherscan_result)
    # DEX source defaults to unavailable so tests stay offline unless they opt in.
    monkeypatch.setattr("collectors.onchain_collector.fetch_dex_metrics", lambda *a, **k: dex_metrics)
    return col


def test_goplus_fields_map_onto_features(monkeypatch):
    col = _collector(monkeypatch, make_goplus_entry())
    data = col.collect(_DAI)

    assert "goplus" in data["sources_used"]
    # boolean/privilege flags
    assert data["is_honeypot"] == 0.0
    assert data["has_mint"] == 1.0            # is_mintable -> has_mint
    assert data["has_blacklist"] == 0.0       # is_blacklisted -> has_blacklist
    assert data["hidden_owner"] == 0.0
    assert data["can_take_back_ownership"] == 0.0
    assert data["is_anti_whale"] == 0.0
    assert data["source_verified"] is True    # is_open_source -> source_verified
    assert data["ownership_renounced"] is True  # zero owner_address -> renounced
    # taxes are fractions in GoPlus -> percentages here
    assert data["buy_tax"] == 1.0
    assert data["sell_tax"] == 2.0
    # holder distribution (percents are fractions -> *100)
    assert data["top_holder_pct"] == 5.0
    assert data["top10_holder_pct"] == 15.0
    assert data["holder_count"] == 12345
    assert data["creator_percent"] == 2.0
    # LP locked -> liquidity_locked
    assert data["liquidity_locked"] is True


def test_no_false_undetermined_note_when_goplus_set_flags(monkeypatch):
    """Regression: Etherscan-unavailable note must not contradict GoPlus-set flags."""
    col = _collector(monkeypatch, make_goplus_entry(), etherscan_result=None)
    data = col.collect(_DAI)

    # GoPlus set source_verified / has_mint / has_blacklist, so nothing is undetermined.
    assert data["source_verified"] is not None
    assert data["has_mint"] is not None
    assert data["has_blacklist"] is not None
    joined = " ".join(data["notes"]).lower()
    assert "could not be independently confirmed" not in joined
    assert "could not be determined" not in joined  # old, contradictory wording


def test_undetermined_note_lists_only_still_none_fields(monkeypatch):
    """When GoPlus is missing has_mint, the note may mention mint but not verification."""
    entry = make_goplus_entry()
    entry.pop("is_mintable")          # GoPlus can't confirm mint
    col = _collector(monkeypatch, entry, etherscan_result=None)
    data = col.collect(_DAI)

    assert data["has_mint"] is None            # genuinely unknown
    note = next((n for n in data["notes"] if "could not be independently confirmed" in n), "")
    assert "mint" in note
    assert "verification" not in note          # GoPlus set source_verified


def test_graceful_degradation_when_goplus_unavailable(monkeypatch):
    col = _collector(monkeypatch, goplus_entry=None, etherscan_result=None)
    data = col.collect(_DAI)  # must not raise

    for name in FEATURE_ORDER:
        assert name in data
        assert data[name] is None              # nothing could be fetched
    assert any("goplus" in n.lower() for n in data["notes"])
    assert data["missing_fields"] == list(FEATURE_ORDER) or set(data["missing_fields"]) == set(FEATURE_ORDER)


def test_dex_fills_market_features(monkeypatch):
    """DEX source fills the two market features GoPlus can't provide."""
    dex = {"source": "dexscreener", "liquidity_to_mcap_ratio": 0.3, "buy_sell_ratio": 1.25}
    col = _collector(monkeypatch, make_goplus_entry(), dex_metrics=dex)
    data = col.collect(_DAI)

    assert data["liquidity_to_mcap_ratio"] == 0.3
    assert data["buy_sell_ratio"] == 1.25
    assert "dexscreener" in data["sources_used"]
    # filled -> no longer imputed/missing
    assert "liquidity_to_mcap_ratio" not in data["missing_fields"]
    assert "buy_sell_ratio" not in data["missing_fields"]


def test_dex_does_not_overwrite_and_notes_on_failure(monkeypatch):
    # DEX unavailable -> fields stay None, a note explains, nothing raised.
    col = _collector(monkeypatch, make_goplus_entry(), dex_metrics=None)
    data = col.collect(_DAI)
    assert data["liquidity_to_mcap_ratio"] is None
    assert data["buy_sell_ratio"] is None
    assert any("DEX market data unavailable" in n for n in data["notes"])


def test_invalid_address_raises_value_error(monkeypatch):
    col = _collector(monkeypatch, make_goplus_entry())
    with pytest.raises(ValueError):
        col.collect("0x123")  # too short to be a valid address
