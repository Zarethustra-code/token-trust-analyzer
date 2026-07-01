"""Honeypot.is cross-check — an independent, simulation-based honeypot second opinion.

GoPlus's ``is_honeypot`` comes from static analysis; Honeypot.is actually *simulates*
a buy and a sell. Combining the two conservatively (in the collector) catches
honeypots that either source alone would miss. Free and keyless (set the optional
``HONEYPOT_IS_API_KEY`` env for higher limits, sent as ``X-API-KEY``).

Never raises; returns ``None`` on any failure, an unsupported chain, or when the API
returns no ``honeypotResult`` — absence of a result is treated as **no signal**, never
as "clean".
"""

from __future__ import annotations

import logging
import os
from typing import Optional

try:  # requests is a hard dep in practice; degrade gracefully if absent
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

logger = logging.getLogger("token_trust.honeypot")

HONEYPOT_URL = "https://api.honeypot.is/v2/IsHoneypot"

# app chain -> Honeypot.is chainID (required by the API)
_CHAIN_ID = {"ethereum": 1, "base": 8453}


def _f(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_honeypot_check(chain: str, address: str, timeout: float = 8.0) -> Optional[dict]:
    """Return ``{is_honeypot, reason, buy_tax, sell_tax}`` or ``None``.

    ``None`` means "no usable signal" (request failed, unsupported chain, or the API
    couldn't check the token) — callers must not read that as a clean result.
    """
    if requests is None:
        return None
    chain_id = _CHAIN_ID.get((chain or "").lower())
    if not chain_id:
        return None

    headers = {}
    api_key = os.getenv("HONEYPOT_IS_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key

    try:
        resp = requests.get(
            HONEYPOT_URL,
            params={"address": address, "chainID": chain_id},
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.info("Honeypot.is request failed: %s", exc)
        return None

    honeypot = payload.get("honeypotResult")
    if not isinstance(honeypot, dict) or "isHoneypot" not in honeypot:
        return None  # no result => no signal (do NOT treat as clean)

    simulation = payload.get("simulationResult") or {}
    return {
        "is_honeypot": bool(honeypot.get("isHoneypot")),
        "reason": honeypot.get("honeypotReason") or None,
        # Honeypot.is reports taxes as percentages (same unit we store).
        "buy_tax": _f(simulation.get("buyTax")),
        "sell_tax": _f(simulation.get("sellTax")),
    }
