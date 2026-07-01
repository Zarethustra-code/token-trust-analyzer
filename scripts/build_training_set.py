#!/usr/bin/env python3
"""Build the Isolation Forest training set (``data/training_tokens.json``).

The anomaly model learns "normal" from a set of *healthy* token feature vectors.
A small, narrow set makes the model over-flag legitimate tokens, so this script
produces a larger, more diverse set (~150-250 rows) with two strategies:

1. **Real tokens (primary).** For a curated list of well-known, legitimate ERC-20
   contracts (Ethereum + Base), fetch features through the *exact same* path the
   app uses at inference time — ``OnChainCollector`` (GoPlus primary) →
   ``FeatureExtractor`` — so training vectors are produced identically (same
   imputation, same ``FEATURE_ORDER``). No feature logic is duplicated here.

2. **Synthetic (fallback / top-up).** When offline, or to reach ``--target`` rows,
   sample each feature from a realistic healthy distribution derived from the
   immutable seed set (``data/seed_tokens.json``): per-feature mean/std for
   continuous features, observed 0/1 frequency for booleans, plus mild noise and
   range clamping. Synthetic rows are forced clearly-healthy (no honeypot, low
   taxes, verified, reasonable holder spread) — they represent the normal class.

Output matches the existing file format: a JSON list of objects keyed by feature
name, 20 features per row. Re-runnable and idempotent (fixed seed; synthetic
distributions always come from the immutable seed file, not the previous output).

Usage:
    python scripts/build_training_set.py --target 200          # real + synthetic
    python scripts/build_training_set.py --synthetic-only      # offline
    python scripts/build_training_set.py --target 200 --out /tmp/preview.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from statistics import mean, pstdev
from typing import Optional

# Make the project root importable when run as `python scripts/build_training_set.py`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Feature schema — the single source of truth (no external deps, safe to import
# even offline / without the ML stack installed).
from features.feature_extractor import (  # noqa: E402
    BOOLEAN_FEATURES,
    FEATURE_ORDER,
    FeatureExtractor,
)

logger = logging.getLogger("build_training_set")

DATA_DIR = os.path.join(_ROOT, "data")
SEED_PATH = os.path.join(DATA_DIR, "seed_tokens.json")
DEFAULT_OUT = os.path.join(DATA_DIR, "training_tokens.json")
MODEL_PATH = os.path.join(DATA_DIR, "anomaly_model.joblib")

# Features rendered as integers in the output (to match the seed file's style).
INT_FEATURES = {"holder_count", "recent_tx_count", "contract_age_days"}
CONTINUOUS_FEATURES = [f for f in FEATURE_ORDER if f not in BOOLEAN_FEATURES]

# Booleans that must stay 0/1 = healthy for every synthetic row, regardless of the
# observed seed frequency (these define the "normal" class).
_HEALTHY_BOOL_OVERRIDES = {
    "is_honeypot": 0,
    "has_blacklist": 0,
    "hidden_owner": 0,
    "can_take_back_ownership": 0,
    "source_verified": 1,
}

# Valid ranges for synthetic continuous features. Chosen to keep synthetic rows
# unambiguously healthy (e.g. top holder < 50% and creator < 20% so they don't
# even brush the rule thresholds), while still spanning a realistic spread.
_RANGES = {
    "top_holder_pct": (0.5, 40.0),
    "top10_holder_pct": (5.0, 80.0),
    "holder_count": (100.0, 500_000.0),
    "gini": (0.30, 0.85),
    "creator_percent": (0.0, 15.0),
    "liquidity_to_mcap_ratio": (0.01, 0.60),
    "buy_tax": (0.0, 8.0),
    "sell_tax": (0.0, 8.0),
    "contract_age_days": (30.0, 3000.0),
    "recent_tx_count": (20.0, 1000.0),
    "buy_sell_ratio": (0.60, 1.50),
}


# --------------------------------------------------------------------------- #
# Curated real tokens as (address, expected_symbol) pairs.
#
# Addresses are transcribed from memory, so each carries its expected symbol.
# After fetching a token, the builder compares GoPlus's returned symbol against
# the expected one and DROPS the address on any mismatch — so a mis-transcribed
# address can never silently pollute the set with a different token. Addresses
# that fail, return no GoPlus data, or don't match are simply skipped (and made
# up for by synthetic top-up), so the list needs no manual verification.
# --------------------------------------------------------------------------- #
CURATED_ADDRESSES: dict[str, list[tuple[str, str]]] = {
    "ethereum": [
        ("0x6B175474E89094C44Da98b954EedeAC495271d0F", "DAI"),
        ("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "USDC"),
        ("0xdAC17F958D2ee523a2206206994597C13D831ec7", "USDT"),
        ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "WETH"),
        ("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "WBTC"),
        ("0x514910771AF9Ca656af840dff83E8264EcF986CA", "LINK"),
        ("0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984", "UNI"),
        ("0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9", "AAVE"),
        ("0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2", "MKR"),
        ("0xc00e94Cb662C3520282E6f5717214004A7f26888", "COMP"),
        ("0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F", "SNX"),
        ("0xD533a949740bb3306d119CC777fa900bA034cd52", "CRV"),
        ("0x5A98FcBEA516Cf06857215779Fd812CA3beF1B32", "LDO"),
        ("0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE", "SHIB"),
        ("0x6982508145454Ce325dDbE47a25d4ec3d2311933", "PEPE"),
        ("0x4d224452801ACEd8B2F0aebE155379bb5D594381", "APE"),
        ("0x3845badAde8e6dFF049820680d1F14bD3903a5d0", "SAND"),
        ("0x0F5D2fB29fb7d3CFeE444a200298f468908cC942", "MANA"),
        ("0xc944E90C64B2c07662A292be6244BDf05Cda44a7", "GRT"),
        ("0x0D8775F648430679A709E98d2b0Cb6250d2887EF", "BAT"),
        ("0x111111111117dC0aa78b770fA6A738034120C302", "1INCH"),
        ("0x6B3595068778DD592e39A122f4f5a5cF09C90fE2", "SUSHI"),
        ("0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e", "YFI"),
        ("0x853d955aCEf822Db058eb8505911ED77F175b99e", "FRAX"),
        ("0x5f98805A4E8be255a32880FDeC7F6728C6568bA0", "LUSD"),
        ("0x0000000000085d4780B73119b644AE5ecd22b376", "TUSD"),
        ("0x8E870D67F660D95d5be530380D0eC0bd388289E1", "USDP"),
        ("0xC18360217D8F7Ab5e7c516566761Ea12Ce7F9D72", "ENS"),
        ("0xD33526068D116cE69F19A9ee46F0bd304F21A51f", "RPL"),
        ("0x3432B6A60D23Ca0dFCa7761B7ab56459D9C964D0", "FXS"),
        ("0xba100000625a3754423978a60c9317c58a424e3D", "BAL"),
        ("0x4e3FBD56CD56c3e72c1403e103b45Db9da5B9D2B", "CVX"),
        ("0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84", "stETH"),
        ("0xae78736Cd615f374D3085123A210448E74Fc6393", "rETH"),
        ("0xBe9895146f7AF43049ca1c1AE358B0541Ea49704", "cbETH"),
        ("0x40D16FC0246aD3160Ccc09B8D0D3A2cD28aE6C2f", "GHO"),
        ("0x75231F58b43240C9718Dd58B4967c5114342a86c", "OKB"),
        ("0x2AF5D2aD76741191D15Dfe7bF6aC92d4Bd912Ca3", "LEO"),
        ("0x4E15361FD6b4BB609Fa63C81A2be19d873717870", "FTM"),
        ("0x3506424F91fD33084466F402d5D97f05F8e3b4AF", "CHZ"),
        ("0xE41d2489571d322189246DaFA5ebDe1F4699F498", "ZRX"),
        ("0xF57e7e7C23978C3cAEC3C3548E3D615c346e79fF", "IMX"),
        ("0xe28b3B32B6c345A34Ff64674606124Dd5Aceca30", "INJ"),
        ("0xBB0E17EF65F82Ab018d8EDd776e8DD940327B28b", "AXS"),
        ("0x6De037ef9aD2725EB40118Bb1702EBb27e4Aeb24", "RNDR"),
        ("0x92D6C1e31e14520e676a687F0a93788B716BEff5", "DYDX"),
        ("0x58b6A8A3302369DAEc383334672404Ee733aB239", "LPT"),
        ("0xdeFA4e8a7bcBA345F687a2f1456F5Edd9CE97202", "KNC"),
        ("0x111111517e4929D3dcbdfa7CCe55d30d4B6BC4d6", "ICHI"),
        ("0x6c6EE5e31d828De241282B9606C8e98Ea48526E2", "HOT"),
    ],
    "base": [
        ("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", "USDC"),
        ("0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb", "DAI"),
        ("0x4200000000000000000000000000000000000006", "WETH"),
        ("0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22", "cbETH"),
        ("0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf", "cbBTC"),
        ("0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA", "USDbC"),
        ("0x940181a94A35A4569E4529A3CDfB74e38FD98631", "AERO"),
        ("0xB6fe221Fe9EeF5aBa221c348bA20A1Bf5e73624c", "rETH"),
        ("0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed", "DEGEN"),
        ("0x532f27101965dd16442E59d40670FaF5eBB142E4", "BRETT"),
        ("0xc1CBa3fCea344f92D9239c08C0568f6F2F0ee452", "wstETH"),
        ("0x60a3E35Cc302bFA44Cb288Bc5a4F316Fdb1adb42", "EURC"),
        ("0xA88594D404727625A9437C3f886C7643872296AE", "WELL"),
        ("0x9e1028F5F1D5eDE59748FFceE5532509976840E0", "COMP"),
        ("0x78a087d713Be963Bf307b18F2Ff8122EF9A63ae9", "BSWAP"),
    ],
}


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _format_row(values: dict) -> dict:
    """Render a full feature dict into the output row format (matches seed file)."""
    row: dict = {}
    for feature in FEATURE_ORDER:
        value = float(values[feature])
        if feature in BOOLEAN_FEATURES:
            row[feature] = 1 if value >= 0.5 else 0
        elif feature in INT_FEATURES:
            row[feature] = int(round(value))
        elif feature in ("gini", "liquidity_to_mcap_ratio"):
            row[feature] = round(value, 3)
        else:
            row[feature] = round(value, 2)
    return row


# --------------------------------------------------------------------------- #
# Real-token collection (reuses the app's collector + extractor)
# --------------------------------------------------------------------------- #
def collect_real_rows(delay: float, retries: int) -> list[dict]:
    """Collect healthy feature rows for the curated addresses via GoPlus."""
    try:
        from collectors.onchain_collector import OnChainCollector
    except Exception as exc:  # missing deps / offline
        logger.warning("Cannot import OnChainCollector (%s) — skipping real collection.", exc)
        return []

    extractor = FeatureExtractor()
    goplus_key = os.getenv("GOPLUS_API_KEY")  # optional; public endpoint needs none
    rows: list[dict] = []
    attempted = 0

    for chain, pairs in CURATED_ADDRESSES.items():
        try:
            collector = OnChainCollector(chain=chain, goplus_api_key=goplus_key)
        except Exception as exc:
            logger.warning("Collector init failed for chain %s: %s", chain, exc)
            continue
        for address, expected_symbol in pairs:
            attempted += 1
            row = _collect_one(collector, extractor, chain, address, expected_symbol, retries, delay)
            if row is not None:
                rows.append(row)
            time.sleep(delay)  # be gentle with the GoPlus public endpoint

    logger.info("Real collection: %d/%d addresses yielded verified healthy rows.", len(rows), attempted)
    return rows


def _collect_one(collector, extractor, chain, address, expected_symbol, retries, delay) -> Optional[dict]:
    """Fetch one address with retry/backoff; return a row or None (never raises).

    Enforces an identity check: the symbol GoPlus returns must match the curated
    expectation, so a mis-transcribed address is dropped instead of contributing a
    *different* token's features.
    """
    for attempt in range(retries + 1):
        try:
            raw = collector.collect(address)
        except Exception as exc:
            logger.info("collect(%s) errored (attempt %d): %s", address, attempt + 1, exc)
            time.sleep(delay * (attempt + 1))
            continue

        if "goplus" not in raw.get("sources_used", []):
            # No GoPlus data — likely rate-limited or unsupported; back off and retry.
            logger.info("GoPlus miss for %s (attempt %d) — backing off.", address, attempt + 1)
            time.sleep(delay * (attempt + 2))
            continue

        # Identity check: the returned symbol must match the curated expectation.
        got_symbol = ((raw.get("token_info") or {}).get("symbol") or "").strip()
        if got_symbol.upper() != expected_symbol.strip().upper():
            logger.warning(
                "Symbol mismatch for %s on %s: expected %r, GoPlus returned %r — dropping.",
                address, chain, expected_symbol, got_symbol or "<none>",
            )
            return None

        feature_set = extractor.extract(raw)
        # A curated blue-chip flagged as a honeypot means bad/garbled data — drop it.
        if float(feature_set.raw.get("is_honeypot") or 0.0) >= 0.5:
            logger.warning("Skipping %s (%s): unexpectedly flagged honeypot.", address, chain)
            return None
        logger.info("collected %s / %s (%s)", chain, address, got_symbol)
        return _format_row(feature_set.imputed)

    logger.warning("Gave up on %s after %d attempts.", address, retries + 1)
    return None


# --------------------------------------------------------------------------- #
# Synthetic generation
# --------------------------------------------------------------------------- #
def load_seed_rows() -> list[dict]:
    with open(SEED_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def derive_distributions(seed_rows: list[dict]) -> dict:
    """Per-feature (mean, std) for continuous features + P(1) for booleans."""
    continuous: dict[str, tuple[float, float]] = {}
    for feature in CONTINUOUS_FEATURES:
        xs = [float(r[feature]) for r in seed_rows if r.get(feature) is not None]
        mu = mean(xs) if xs else 0.0
        sigma = pstdev(xs) if len(xs) > 1 else 0.0
        continuous[feature] = (mu, sigma)

    boolean: dict[str, float] = {}
    for feature in BOOLEAN_FEATURES:
        xs = [1 if float(r.get(feature, 0)) >= 0.5 else 0 for r in seed_rows]
        boolean[feature] = (sum(xs) / len(xs)) if xs else 0.0

    return {"continuous": continuous, "boolean": boolean}


def _synthetic_row(dists: dict, rng: random.Random) -> dict:
    values: dict = {}
    for feature in CONTINUOUS_FEATURES:
        mu, sigma = dists["continuous"][feature]
        # Mild extra noise so synthetic rows aren't a tight cluster on the seed mean.
        noise = sigma if sigma > 0 else max(1.0, abs(mu) * 0.1)
        value = rng.gauss(mu, noise * 1.15)
        lo, hi = _RANGES.get(feature, (0.0, None))
        value = max(lo, value)
        if hi is not None:
            value = min(hi, value)
        values[feature] = value

    # Keep the holder metrics coherent: top10 >= top1, and <= 80.
    values["top10_holder_pct"] = min(
        80.0, max(values["top10_holder_pct"], values["top_holder_pct"])
    )

    for feature in BOOLEAN_FEATURES:
        if feature in _HEALTHY_BOOL_OVERRIDES:
            values[feature] = _HEALTHY_BOOL_OVERRIDES[feature]
        else:
            values[feature] = 1 if rng.random() < dists["boolean"][feature] else 0

    return _format_row(values)


def generate_synthetic(count: int, dists: dict, rng: random.Random) -> list[dict]:
    return [_synthetic_row(dists, rng) for _ in range(max(0, count))]


# --------------------------------------------------------------------------- #
# Summary + main
# --------------------------------------------------------------------------- #
def summarize(rows: list[dict]) -> None:
    logger.info("Per-feature min / mean / max across %d rows:", len(rows))
    for feature in FEATURE_ORDER:
        xs = [float(r[feature]) for r in rows if r.get(feature) is not None]
        if not xs:
            continue
        logger.info("  %-24s %10.3f %10.3f %10.3f", feature, min(xs), mean(xs), max(xs))


def build(target: int, synthetic_only: bool, delay: float, retries: int,
          min_real: int, seed: int, out_path: str) -> None:
    rng = random.Random(seed)

    seed_rows = load_seed_rows()
    dists = derive_distributions(seed_rows)
    logger.info("Derived synthetic distributions from %d immutable seed rows.", len(seed_rows))

    real_rows: list[dict] = []
    if synthetic_only:
        logger.info("--synthetic-only: skipping real GoPlus collection.")
    else:
        real_rows = collect_real_rows(delay, retries)
        if len(real_rows) < min_real:
            logger.warning(
                "Only %d real rows (< --min-real %d) — topping up with synthetic to reach target.",
                len(real_rows), min_real,
            )

    need = max(0, target - len(real_rows))
    synthetic_rows = generate_synthetic(need, dists, rng)

    rows = real_rows + synthetic_rows
    rng.shuffle(rows)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("[\n")
        fh.write(",\n".join("  " + json.dumps(r) for r in rows))
        fh.write("\n]\n")

    logger.info("Wrote %d rows to %s (%d real + %d synthetic).",
                len(rows), out_path, len(real_rows), len(synthetic_rows))

    # Refresh the cached model so the next app start refits on the new set.
    if os.path.abspath(out_path) == os.path.abspath(DEFAULT_OUT) and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        logger.info("Removed cached %s — the Isolation Forest will refit on next startup.",
                    os.path.basename(MODEL_PATH))

    summarize(rows)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build the Isolation Forest training set.")
    parser.add_argument("--target", type=int, default=200, help="Target total rows (default 200).")
    parser.add_argument("--synthetic-only", action="store_true", help="Skip real collection (offline).")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between GoPlus calls.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per address on a GoPlus miss.")
    parser.add_argument("--min-real", type=int, default=40, help="Warn if fewer real rows collected.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (reproducibility).")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output path (default data/training_tokens.json).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build(
        target=args.target,
        synthetic_only=args.synthetic_only,
        delay=args.delay,
        retries=args.retries,
        min_real=args.min_real,
        seed=args.seed,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
