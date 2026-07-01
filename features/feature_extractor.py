"""Feature extraction.

Turns the raw, best-effort output of the on-chain collector into a *fixed-order*
numeric feature vector suitable for the ML models, while also preserving the
raw (possibly-missing) values for the human-facing report.

Design notes
------------
* ``FEATURE_ORDER`` is the single source of truth for feature identity and order.
  The collector, the rules, the Isolation Forest and the report all agree on it.
* Missing values are imputed to a **healthy-token prior** (``IMPUTATION_DEFAULTS``)
  so that *absence of data is treated neutrally* by the ML model rather than as a
  risk signal. The rule engine (ml/rules.py) takes the opposite, conservative
  stance: it only penalizes *confirmed* risks and skips unknown fields entirely.
* Booleans are encoded as 0.0 / 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# --- Canonical feature schema -------------------------------------------------

FEATURE_ORDER: list[str] = [
    "top_holder_pct",          # % of supply held by the single largest holder
    "top10_holder_pct",        # % of supply held by the top 10 holders
    "holder_count",            # number of distinct holders
    "gini",                    # Gini coefficient of holder distribution (0..1)
    "creator_percent",         # % of supply still held by the contract creator
    "liquidity_locked",        # bool: LP tokens locked/burned
    "liquidity_to_mcap_ratio", # liquidity value / market cap
    "source_verified",         # bool: contract source verified / open source
    "has_mint",                # bool: contract exposes a callable mint path
    "ownership_renounced",     # bool: owner() is zero/dead address
    "has_blacklist",           # bool: contract can block/restrict transfers
    "is_honeypot",             # bool: buyers cannot sell (GoPlus)
    "buy_tax",                 # buy tax as a percentage (0..100)
    "sell_tax",                # sell tax as a percentage (0..100)
    "hidden_owner",            # bool: contract has a concealed owner (GoPlus)
    "can_take_back_ownership", # bool: renounced ownership can be reclaimed (GoPlus)
    "is_anti_whale",           # bool: per-wallet/tx limits present (GoPlus)
    "contract_age_days",       # days since contract creation
    "recent_tx_count",         # recent transfer/tx count (activity)
    "buy_sell_ratio",          # approx buys/sells over the recent window
]

# The set of features that are semantically boolean (encoded 0/1).
BOOLEAN_FEATURES: set[str] = {
    "liquidity_locked",
    "source_verified",
    "has_mint",
    "ownership_renounced",
    "has_blacklist",
    "is_honeypot",
    "hidden_owner",
    "can_take_back_ownership",
    "is_anti_whale",
}

# Healthy-token priors used to impute missing values for the ML vector.
# Chosen so that "unknown" looks like a typical legitimate token, not an outlier.
IMPUTATION_DEFAULTS: dict[str, float] = {
    "top_holder_pct": 15.0,
    "top10_holder_pct": 40.0,
    "holder_count": 1500.0,
    "gini": 0.60,
    "creator_percent": 2.0,
    "liquidity_locked": 1.0,
    "liquidity_to_mcap_ratio": 0.10,
    "source_verified": 1.0,
    "has_mint": 0.0,
    "ownership_renounced": 1.0,
    "has_blacklist": 0.0,
    "is_honeypot": 0.0,
    "buy_tax": 1.0,
    "sell_tax": 1.0,
    "hidden_owner": 0.0,
    "can_take_back_ownership": 0.0,
    "is_anti_whale": 0.0,
    "contract_age_days": 365.0,
    "recent_tx_count": 500.0,
    "buy_sell_ratio": 1.0,
}


def compute_gini(balances: list[float]) -> Optional[float]:
    """Gini coefficient of a list of holder balances.

    0 => perfectly even distribution, ->1 => extreme concentration.
    Returns ``None`` if the input is empty or non-positive. Only the balances we
    actually have are used (typically the top-N holders), so this is a lower-bound
    estimate of true concentration, which is fine as a comparative feature.
    """
    if not balances:
        return None
    values = sorted(float(b) for b in balances if b is not None and b > 0)
    n = len(values)
    if n == 0:
        return None
    if n == 1:
        return 1.0
    total = sum(values)
    if total <= 0:
        return None
    # Standard mean-absolute-difference formulation via sorted cumulative sums.
    cumulative = 0.0
    weighted = 0.0
    for i, v in enumerate(values, start=1):
        cumulative += v
        weighted += i * v
    gini = (2.0 * weighted) / (n * total) - (n + 1.0) / n
    # Clamp tiny negative float error to 0.
    return max(0.0, min(1.0, gini))


def _to_number(value: Any) -> Optional[float]:
    """Coerce bools/ints/floats to float; leave None as None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class FeatureSet:
    """The extracted features in three views.

    raw:      feature name -> value or None (exactly what we could observe)
    imputed:  feature name -> value (never None; used to build the vector)
    vector:   list[float] in FEATURE_ORDER (the ML input)
    imputed_features: names that were filled from the prior (for transparency)
    """

    raw: dict[str, Optional[float]]
    imputed: dict[str, float]
    vector: list[float]
    imputed_features: list[str] = field(default_factory=list)

    def as_metrics_dict(self) -> dict[str, Optional[float]]:
        """Raw values keyed for the TokenMetrics report model."""
        return dict(self.raw)


class FeatureExtractor:
    """Builds a :class:`FeatureSet` from raw collector output."""

    def extract(self, raw_data: dict[str, Any]) -> FeatureSet:
        """Map raw collector output onto the canonical feature schema.

        ``raw_data`` may contain any subset of the feature keys, plus the extra
        key ``top_holder_balances`` (a list) from which ``gini`` is derived when
        not supplied directly.
        """
        raw: dict[str, Optional[float]] = {}

        # Derive gini from a balances list if we weren't handed it directly.
        gini_value = _to_number(raw_data.get("gini"))
        if gini_value is None:
            balances = raw_data.get("top_holder_balances")
            if isinstance(balances, list) and balances:
                gini_value = compute_gini([float(b) for b in balances if b is not None])

        for name in FEATURE_ORDER:
            if name == "gini":
                raw[name] = gini_value
            else:
                raw[name] = _to_number(raw_data.get(name))

        imputed: dict[str, float] = {}
        imputed_features: list[str] = []
        for name in FEATURE_ORDER:
            value = raw[name]
            if value is None:
                value = IMPUTATION_DEFAULTS[name]
                imputed_features.append(name)
            elif name in BOOLEAN_FEATURES:
                value = 1.0 if value >= 0.5 else 0.0
            imputed[name] = float(value)

        vector = [imputed[name] for name in FEATURE_ORDER]
        return FeatureSet(
            raw=raw,
            imputed=imputed,
            vector=vector,
            imputed_features=imputed_features,
        )
