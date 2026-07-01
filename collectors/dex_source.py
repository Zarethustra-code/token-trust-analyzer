"""DEX market-data source: DexScreener (primary) + GeckoTerminal (fallback).

GoPlus doesn't expose market/liquidity/trading data, so this module fills the two
market features from live pool data:

  * ``liquidity_to_mcap_ratio`` = total pool liquidity / market cap (or FDV)
  * ``buy_sell_ratio``          = total 24h buys / total 24h sells

Both APIs are free and keyless (with rate limits). A token usually trades in
several pools, so metrics are aggregated across the token's matched pools (capped
to the top-N by liquidity). Never raises — returns ``None`` when neither source
yields usable data, so the caller simply leaves the fields imputed.

Attribution: pool data courtesy of DexScreener (https://dexscreener.com) and
GeckoTerminal (https://www.geckoterminal.com).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

try:  # requests is a hard dep in practice; degrade gracefully if absent
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

logger = logging.getLogger("token_trust.dex")

DEXSCREENER_URL = "https://api.dexscreener.com/latest/dex/tokens/{address}"
GECKOTERMINAL_URL = (
    "https://api.geckoterminal.com/api/v2/networks/{network}/tokens/{address}/pools"
)

# app chain -> DexScreener chainId
_DEXSCREENER_CHAIN = {"ethereum": "ethereum", "base": "base"}
# app chain -> GeckoTerminal network id (differs from DexScreener!)
_GECKOTERMINAL_NETWORK = {"ethereum": "eth", "base": "base"}

_MAX_POOLS = 10  # aggregate only the top-N pools by liquidity


def _f(value: Any) -> Optional[float]:
    """Parse a possibly-stringified numeric field to float; None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_dex_metrics(chain: str, address: str, timeout: float = 8.0) -> Optional[dict]:
    """Return ``{source, liquidity_to_mcap_ratio, buy_sell_ratio, ...}`` or ``None``.

    Tries DexScreener first; on any failure (non-200, rate-limited, no matching
    pairs, timeout) falls back to GeckoTerminal. Returns ``None`` if neither
    source produces a usable ratio.
    """
    chain = (chain or "").lower()

    aggregate = _from_dexscreener(chain, address, timeout)
    source = "dexscreener"
    if aggregate is None:
        aggregate = _from_geckoterminal(chain, address, timeout)
        source = "geckoterminal"
    if aggregate is None:
        return None

    ratios = _compute_ratios(aggregate)
    if ratios["liquidity_to_mcap_ratio"] is None and ratios["buy_sell_ratio"] is None:
        return None  # reached a source, but nothing usable came back

    return {
        "source": source,
        "liquidity_to_mcap_ratio": ratios["liquidity_to_mcap_ratio"],
        "buy_sell_ratio": ratios["buy_sell_ratio"],
        "liquidity_usd": aggregate.get("liquidity_usd"),
        "volume_h24": aggregate.get("volume_h24"),
        "market_cap": aggregate.get("market_cap") or aggregate.get("fdv"),
    }


def _compute_ratios(agg: dict) -> dict:
    liquidity = agg.get("liquidity_usd")
    denom = agg.get("market_cap") or agg.get("fdv")  # prefer mcap, fall back to FDV
    liquidity_to_mcap = None
    if liquidity is not None and denom and denom > 0:
        liquidity_to_mcap = round(liquidity / denom, 4)

    buys = agg.get("buys") or 0.0
    sells = agg.get("sells") or 0.0
    buy_sell = round(buys / sells, 3) if sells > 0 else None  # guard divide-by-zero

    return {"liquidity_to_mcap_ratio": liquidity_to_mcap, "buy_sell_ratio": buy_sell}


def _from_dexscreener(chain: str, address: str, timeout: float) -> Optional[dict]:
    if requests is None:
        return None
    chain_id = _DEXSCREENER_CHAIN.get(chain)
    if not chain_id:
        return None
    try:
        resp = requests.get(DEXSCREENER_URL.format(address=address), timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.info("DexScreener request failed: %s", exc)
        return None

    pairs = payload.get("pairs") or []
    matched = [p for p in pairs if str(p.get("chainId", "")).lower() == chain_id]
    if not matched:
        return None

    def liq(pair: dict) -> float:
        return _f((pair.get("liquidity") or {}).get("usd")) or 0.0

    matched.sort(key=liq, reverse=True)
    matched = matched[:_MAX_POOLS]

    total_liquidity = total_volume = buys = sells = 0.0
    market_cap = fdv = None
    for pair in matched:
        total_liquidity += liq(pair)
        total_volume += _f((pair.get("volume") or {}).get("h24")) or 0.0
        txns = (pair.get("txns") or {}).get("h24") or {}
        buys += _f(txns.get("buys")) or 0.0
        sells += _f(txns.get("sells")) or 0.0
        market_cap = _max(market_cap, _f(pair.get("marketCap")))
        fdv = _max(fdv, _f(pair.get("fdv")))

    return {
        "liquidity_usd": total_liquidity, "volume_h24": total_volume,
        "buys": buys, "sells": sells, "market_cap": market_cap, "fdv": fdv,
    }


def _from_geckoterminal(chain: str, address: str, timeout: float) -> Optional[dict]:
    if requests is None:
        return None
    network = _GECKOTERMINAL_NETWORK.get(chain)
    if not network:
        return None
    try:
        resp = requests.get(
            GECKOTERMINAL_URL.format(network=network, address=address),
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.info("GeckoTerminal request failed: %s", exc)
        return None

    pools = payload.get("data") or []
    if not pools:
        return None

    def liq(pool: dict) -> float:
        return _f((pool.get("attributes") or {}).get("reserve_in_usd")) or 0.0

    pools.sort(key=liq, reverse=True)
    pools = pools[:_MAX_POOLS]

    total_liquidity = total_volume = buys = sells = 0.0
    market_cap = fdv = None
    for pool in pools:
        attrs = pool.get("attributes") or {}
        total_liquidity += _f(attrs.get("reserve_in_usd")) or 0.0
        total_volume += _f((attrs.get("volume_usd") or {}).get("h24")) or 0.0
        txns = (attrs.get("transactions") or {}).get("h24") or {}
        buys += _f(txns.get("buys")) or 0.0
        sells += _f(txns.get("sells")) or 0.0
        market_cap = _max(market_cap, _f(attrs.get("market_cap_usd")))
        fdv = _max(fdv, _f(attrs.get("fdv_usd")))

    if total_liquidity == 0.0 and buys == 0.0 and sells == 0.0:
        return None
    return {
        "liquidity_usd": total_liquidity, "volume_h24": total_volume,
        "buys": buys, "sells": sells, "market_cap": market_cap, "fdv": fdv,
    }


def _max(current: Optional[float], candidate: Optional[float]) -> Optional[float]:
    """Return the larger of two optional floats (mcap/fdv are token-level, so we
    take the max non-null across pools to be robust to per-pair nulls)."""
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate)
