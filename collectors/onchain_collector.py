"""On-chain data collection for an ERC-20 token.

Fetches everything the feature extractor needs from two sources:

* **Etherscan V2** (multichain REST API, keyed by ``chainid``) for verified-source
  analysis, contract-creation time, token supply, holder list (pro), and recent
  transfer activity.
* **Web3.py** (JSON-RPC) for direct ERC-20 reads (name/symbol/decimals/supply)
  and the ``owner()`` call used to decide whether ownership is renounced.

Guiding rule: **graceful degradation.** Any single data point that cannot be
fetched becomes ``None`` and is recorded in ``missing_fields`` / ``notes`` — the
collector never raises for a missing datum. Only a hard input error (bad address)
is surfaced to the caller.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

try:  # optional at import time; degrades if absent
    import requests
except Exception:  # pragma: no cover - requests is a hard dep in practice
    requests = None  # type: ignore

try:
    from web3 import Web3
except Exception:  # pragma: no cover
    Web3 = None  # type: ignore

from features.feature_extractor import FEATURE_ORDER as FEATURE_FIELDS
from models.request import SUPPORTED_CHAINS

logger = logging.getLogger("token_trust.collector")

ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"
# GoPlus Token Security API — public, no key required for single-token queries.
# Docs for agents: https://docs.gopluslabs.io/llms.txt
GOPLUS_URL = "https://api.gopluslabs.io/api/v1/token_security/{chain_id}"


def _to_float(value: Any) -> Optional[float]:
    """Parse a GoPlus string/number field to float; blank/None/invalid -> None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool01(value: Any) -> Optional[float]:
    """Parse a GoPlus "0"/"1" flag to 0.0/1.0; blank/None/unknown -> None."""
    f = _to_float(value)
    if f is None:
        return None
    return 1.0 if f >= 0.5 else 0.0

# Addresses that mean "no owner" when returned by owner()/getOwner().
_DEAD_ADDRESSES = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
}

# Minimal ERC-20 (+ Ownable) ABI for direct reads.
_ERC20_ABI = json.loads(
    """
[
 {"constant":true,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"stateMutability":"view","type":"function"},
 {"constant":true,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"stateMutability":"view","type":"function"},
 {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"stateMutability":"view","type":"function"},
 {"constant":true,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
 {"constant":true,"inputs":[],"name":"owner","outputs":[{"name":"","type":"address"}],"stateMutability":"view","type":"function"},
 {"constant":true,"inputs":[],"name":"getOwner","outputs":[{"name":"","type":"address"}],"stateMutability":"view","type":"function"}
]
"""
)

# Substrings (lowercased) in ABI function names that indicate mint / blacklist paths.
_MINT_HINTS = ("mint",)
_BLACKLIST_HINTS = ("blacklist", "blocklist", "denylist", "banned", "isbot", "setbots")


class RawTokenData(dict):
    """A plain dict subclass so callers can treat it as the feature-name mapping.

    In addition to feature keys it carries: ``token_info`` (dict),
    ``top_holder_balances`` (list | None), ``missing_fields`` (list),
    ``sources_used`` (list), ``notes`` (list).
    """


class OnChainCollector:
    def __init__(
        self,
        chain: str = "ethereum",
        rpc_url: Optional[str] = None,
        etherscan_api_key: Optional[str] = None,
        goplus_api_key: Optional[str] = None,
        timeout: float = 15.0,
    ) -> None:
        self.chain = chain.lower()
        if self.chain not in SUPPORTED_CHAINS:
            raise ValueError(f"Unsupported chain: {chain!r}")
        self.chain_id = SUPPORTED_CHAINS[self.chain]
        self.rpc_url = rpc_url
        self.api_key = etherscan_api_key
        self.goplus_api_key = goplus_api_key  # optional; raises rate limits only
        self.timeout = timeout

        self._w3 = None
        if Web3 is not None and rpc_url:
            try:
                self._w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout}))
            except Exception as exc:  # pragma: no cover - construction rarely fails
                logger.warning("Web3 init failed: %s", exc)
                self._w3 = None

    # -- public API ------------------------------------------------------------

    def collect(self, address: str) -> RawTokenData:
        """Collect all available on-chain features for ``address``.

        Raises ``ValueError`` only for a structurally invalid address. Everything
        else degrades to ``None`` + a note.
        """
        checksum = self._checksum(address)  # may raise ValueError (bad address)

        data = RawTokenData()
        data["token_info"] = {"name": None, "symbol": None, "decimals": None, "total_supply": None}
        data["top_holder_balances"] = None
        data["missing_fields"] = []
        data["sources_used"] = []
        data["notes"] = []
        # Pre-seed all feature keys as None so the extractor sees a full schema.
        for key in FEATURE_FIELDS:
            data[key] = None

        # GoPlus is the PRIMARY raw-signal source (holders, honeypot, taxes,
        # privileges, liquidity). Etherscan then supplies verified-source and
        # contract age (which it owns), and only *fills gaps* for anything GoPlus
        # could not provide. Web3 is the direct-read fallback.
        self._collect_goplus(checksum, data)
        self._collect_metadata(checksum, data)
        self._collect_source_analysis(checksum, data)
        self._collect_creation_age(checksum, data)
        self._collect_holders(checksum, data)
        self._collect_activity(checksum, data)
        self._collect_liquidity(checksum, data)

        # Finalize missing-field list from whatever stayed None.
        data["missing_fields"] = [k for k in FEATURE_FIELDS if data.get(k) is None]
        data["sources_used"] = sorted(set(data["sources_used"]))
        return data

    @staticmethod
    def _set_if_none(data: RawTokenData, key: str, value: Any) -> None:
        """Assign only if not already provided by a higher-priority source."""
        if value is not None and data.get(key) is None:
            data[key] = value

    # -- helpers ---------------------------------------------------------------

    def _checksum(self, address: str) -> str:
        address = address.strip()
        if Web3 is not None:
            try:
                return Web3.to_checksum_address(address)
            except Exception as exc:
                raise ValueError(f"Invalid EVM address: {address!r}") from exc
        # Web3 absent: fall back to a light format check.
        if not (address.startswith("0x") and len(address) == 42):
            raise ValueError(f"Invalid EVM address: {address!r}")
        return address

    def _etherscan(self, module: str, action: str, **params: Any) -> Optional[Any]:
        """Call the Etherscan V2 API; return ``result`` or ``None`` on any failure."""
        if requests is None or not self.api_key:
            return None
        query = {
            "chainid": self.chain_id,
            "module": module,
            "action": action,
            "apikey": self.api_key,
            **params,
        }
        try:
            resp = requests.get(ETHERSCAN_V2_URL, params=query, timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.info("Etherscan %s/%s failed: %s", module, action, exc)
            return None
        # status "1" => OK. tokensupply returns status "1" with a string result.
        if str(payload.get("status")) == "1":
            return payload.get("result")
        # Some endpoints (e.g. tokensupply) put the value directly in result with
        # message "OK" but status "0" for "no records"; treat non-1 as no-data.
        logger.info(
            "Etherscan %s/%s no data: %s", module, action, payload.get("message")
        )
        return None

    def _contract(self, checksum: str):
        if self._w3 is None:
            return None
        try:
            return self._w3.eth.contract(address=checksum, abi=_ERC20_ABI)
        except Exception as exc:
            logger.info("contract() failed: %s", exc)
            return None

    # -- GoPlus (primary raw-signal source) ------------------------------------

    def _goplus(self, checksum: str) -> Optional[dict]:
        """Query the GoPlus Token Security API; return the token's result dict."""
        if requests is None:
            return None
        url = GOPLUS_URL.format(chain_id=self.chain_id)
        headers = {}
        if self.goplus_api_key:
            headers["Authorization"] = self.goplus_api_key
        try:
            resp = requests.get(
                url,
                params={"contract_addresses": checksum},
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.info("GoPlus request failed: %s", exc)
            return None
        if str(payload.get("code")) != "1":
            logger.info("GoPlus non-OK response: %s", payload.get("message"))
        result = payload.get("result") or {}
        # Result is keyed by the lowercased address; fall back to the first entry.
        entry = result.get(checksum.lower())
        if entry is None and result:
            entry = next(iter(result.values()), None)
        return entry if isinstance(entry, dict) else None

    def _collect_goplus(self, checksum: str, data: RawTokenData) -> None:
        """Populate the primary risk features from GoPlus Token Security.

        Every value goes in as a *raw* feature (numeric or 0/1) so it feeds both
        the rule engine and the Isolation Forest — GoPlus flags are inputs to the
        combined score, never the verdict on their own.
        """
        entry = self._goplus(checksum)
        if not entry:
            data["notes"].append(
                "GoPlus token-security data unavailable; primary signals "
                "(honeypot, taxes, holder distribution, privileges) were imputed."
            )
            return
        data["sources_used"].append("goplus")

        # Token identity (fallback metadata).
        info = data["token_info"]
        if info.get("name") is None and entry.get("token_name"):
            info["name"] = entry.get("token_name")
        if info.get("symbol") is None and entry.get("token_symbol"):
            info["symbol"] = entry.get("token_symbol")
        total_supply = _to_float(entry.get("total_supply"))
        if info.get("total_supply") is None and total_supply is not None:
            info["total_supply"] = total_supply

        # Holder distribution.
        holder_count = _to_float(entry.get("holder_count"))
        if holder_count is not None:
            data["holder_count"] = int(holder_count)
        holders = entry.get("holders")
        if isinstance(holders, list) and holders:
            percents = []
            for h in holders:
                p = _to_float(h.get("percent"))
                if p is not None:
                    percents.append(p * 100.0)  # GoPlus percents are fractions 0..1
            if percents:
                percents.sort(reverse=True)
                data["top_holder_pct"] = percents[0]
                data["top10_holder_pct"] = sum(percents[:10])
                data["top_holder_balances"] = percents  # -> Gini downstream
        creator_percent = _to_float(entry.get("creator_percent"))
        if creator_percent is not None:
            data["creator_percent"] = creator_percent * 100.0

        # Contract-privilege & trading flags (raw 0/1 signals).
        data["is_honeypot"] = _to_bool01(entry.get("is_honeypot"))
        data["has_mint"] = _to_bool01(entry.get("is_mintable"))
        data["has_blacklist"] = _to_bool01(entry.get("is_blacklisted"))
        data["hidden_owner"] = _to_bool01(entry.get("hidden_owner"))
        data["can_take_back_ownership"] = _to_bool01(entry.get("can_take_back_ownership"))
        data["is_anti_whale"] = _to_bool01(entry.get("is_anti_whale"))

        buy_tax = _to_float(entry.get("buy_tax"))
        if buy_tax is not None:
            data["buy_tax"] = buy_tax * 100.0  # fraction -> percentage
        sell_tax = _to_float(entry.get("sell_tax"))
        if sell_tax is not None:
            data["sell_tax"] = sell_tax * 100.0

        # Ownership renounced: GoPlus leaves owner_address blank when renounced/none.
        owner = (entry.get("owner_address") or "").strip().lower()
        if owner == "":
            data["ownership_renounced"] = True
        else:
            data["ownership_renounced"] = owner in _DEAD_ADDRESSES

        # is_open_source is a *fallback* for verified source (Etherscan overrides).
        open_source = _to_bool01(entry.get("is_open_source"))
        if open_source is not None:
            data["source_verified"] = bool(open_source)

        self._parse_goplus_liquidity(entry, data)

    def _parse_goplus_liquidity(self, entry: dict, data: RawTokenData) -> None:
        """Derive liquidity-lock status from GoPlus LP-holder data."""
        lp_holders = entry.get("lp_holders")
        if isinstance(lp_holders, list) and lp_holders:
            locked_fraction = 0.0
            for h in lp_holders:
                is_locked = _to_bool01(h.get("is_locked")) or 0.0
                addr = (h.get("address") or "").lower()
                pct = _to_float(h.get("percent")) or 0.0
                if is_locked >= 0.5 or addr in _DEAD_ADDRESSES:
                    locked_fraction += pct  # fraction 0..1
            data["liquidity_locked"] = bool(locked_fraction >= 0.5)
        # liquidity_to_mcap_ratio needs price/market-cap (not in GoPlus); left None.

    def _collect_metadata(self, checksum: str, data: RawTokenData) -> None:
        contract = self._contract(checksum)
        info = data["token_info"]
        decimals: Optional[int] = None
        if contract is not None:
            for field_name, method in (
                ("name", "name"), ("symbol", "symbol"), ("decimals", "decimals"),
            ):
                try:
                    info[field_name] = getattr(contract.functions, method)().call()
                    data["sources_used"].append("web3")
                except Exception:
                    pass
            decimals = info.get("decimals")
            try:
                raw_supply = contract.functions.totalSupply().call()
                if decimals is not None:
                    info["total_supply"] = raw_supply / (10 ** decimals)
                else:
                    info["total_supply"] = float(raw_supply)
                data["sources_used"].append("web3")
            except Exception:
                pass

        # Fallback / cross-check supply via Etherscan stats.
        if info.get("total_supply") is None:
            supply = self._etherscan("stats", "tokensupply", contractaddress=checksum)
            if supply is not None:
                try:
                    raw = float(supply)
                    info["total_supply"] = raw / (10 ** decimals) if decimals else raw
                    data["sources_used"].append("etherscan")
                except Exception:
                    pass

        if all(info.get(k) is None for k in ("name", "symbol", "decimals")):
            data["notes"].append("Token metadata unavailable (no RPC/explorer data).")

    def _collect_source_analysis(self, checksum: str, data: RawTokenData) -> None:
        """Verified-source flag + ABI-based mint/blacklist/ownership detection."""
        result = self._etherscan("contract", "getsourcecode", address=checksum)
        if not result or not isinstance(result, list):
            undetermined = [
                label
                for label, key in (
                    ("verification", "source_verified"),
                    ("mint", "has_mint"),
                    ("blacklist", "has_blacklist"),
                )
                if data.get(key) is None
            ]
            if undetermined:
                data["notes"].append(
                    "Etherscan source/ABI unavailable; "
                    + ", ".join(undetermined)
                    + " flag(s) could not be independently confirmed."
                )
            return
        entry = result[0] if result else {}
        source_code = (entry.get("SourceCode") or "").strip()
        abi_raw = entry.get("ABI") or ""
        verified = bool(source_code) and abi_raw not in ("", "Contract source code not verified")
        data["source_verified"] = verified
        data["sources_used"].append("etherscan")

        if not verified:
            # Unverified: we cannot inspect the ABI, so mint/blacklist stay None.
            data["notes"].append("Contract source not verified; mint/blacklist "
                                 "detection skipped.")
            return

        # Parse the ABI to find state-changing mint / blacklist functions.
        function_names: list[str] = []
        try:
            abi = json.loads(abi_raw)
            for item in abi:
                if item.get("type") == "function":
                    function_names.append((item.get("name") or "").lower())
        except Exception:
            function_names = []

        # Also scan raw source text as a backstop for obfuscated ABIs. These only
        # FILL GAPS — GoPlus is the primary source for mint/blacklist flags.
        haystack = " ".join(function_names) + " " + source_code.lower()
        self._set_if_none(data, "has_mint", any(h in haystack for h in _MINT_HINTS))
        self._set_if_none(data, "has_blacklist", any(h in haystack for h in _BLACKLIST_HINTS))

        self._collect_ownership(checksum, data)

    def _collect_ownership(self, checksum: str, data: RawTokenData) -> None:
        if data.get("ownership_renounced") is not None:
            return  # already resolved by GoPlus (primary)
        contract = self._contract(checksum)
        if contract is None:
            return
        owner: Optional[str] = None
        for method in ("owner", "getOwner"):
            try:
                owner = getattr(contract.functions, method)().call()
                break
            except Exception:
                continue
        if owner is None:
            # No owner function exposed -> commonly means non-Ownable / renounced.
            data["notes"].append("No owner() function found; ownership treated as unknown.")
            return
        self._set_if_none(data, "ownership_renounced", str(owner).lower() in _DEAD_ADDRESSES)
        data["sources_used"].append("web3")

    def _collect_creation_age(self, checksum: str, data: RawTokenData) -> None:
        result = self._etherscan(
            "contract", "getcontractcreation", contractaddresses=checksum
        )
        if not result or not isinstance(result, list):
            data["notes"].append("Contract creation time unavailable; age unknown.")
            return
        entry = result[0]
        tx_hash = entry.get("txHash")
        # V2 sometimes returns a 'timestamp' directly; use it if present.
        ts = entry.get("timestamp") or entry.get("blockTimestamp")
        block_time: Optional[int] = None
        if ts is not None:
            try:
                block_time = int(ts)
            except Exception:
                block_time = None
        if block_time is None and tx_hash and self._w3 is not None:
            try:
                receipt = self._w3.eth.get_transaction(tx_hash)
                block = self._w3.eth.get_block(receipt["blockNumber"])
                block_time = int(block["timestamp"])
            except Exception:
                block_time = None
        if block_time is None:
            data["notes"].append("Could not resolve contract creation timestamp.")
            return
        try:
            now = int(self._w3.eth.get_block("latest")["timestamp"]) if self._w3 else None
        except Exception:
            now = None
        if now is None:
            data["notes"].append("Could not read latest block time for age calc.")
            return
        data["contract_age_days"] = max(0.0, (now - block_time) / 86400.0)
        data["sources_used"].append("etherscan")

    def _collect_holders(self, checksum: str, data: RawTokenData) -> None:
        """Holder distribution FALLBACK via Etherscan's (pro) token holder list.

        GoPlus is the primary source; this only runs if GoPlus didn't supply
        holder data. On the free explorer tier this endpoint is usually
        unavailable, so the fields stay ``None`` and are imputed.
        """
        if data.get("top_holder_pct") is not None:
            return  # already provided by GoPlus (primary)
        result = self._etherscan(
            "token", "tokenholderlist", contractaddress=checksum, page=1, offset=100
        )
        if not result or not isinstance(result, list):
            data["notes"].append(
                "Holder distribution unavailable from GoPlus and the explorer "
                "(pro endpoint); top-holder %, holder count and Gini were imputed."
            )
            return
        balances: list[float] = []
        for row in result:
            try:
                balances.append(float(row.get("TokenHolderQuantity")))
            except Exception:
                continue
        if not balances:
            return
        total = sum(balances)
        balances.sort(reverse=True)
        if total > 0:
            data["top_holder_pct"] = 100.0 * balances[0] / total
            data["top10_holder_pct"] = 100.0 * sum(balances[:10]) / total
        data["holder_count"] = len(balances)  # note: capped by the page offset
        data["top_holder_balances"] = balances
        data["sources_used"].append("etherscan")
        data["notes"].append(
            "Holder metrics computed from the top holders page (count is a lower bound)."
        )

    def _collect_activity(self, checksum: str, data: RawTokenData) -> None:
        """Recent transfer count as an activity proxy (buy/sell ratio needs a DEX pair)."""
        result = self._etherscan(
            "account", "tokentx", contractaddress=checksum,
            page=1, offset=1000, sort="desc",
        )
        if isinstance(result, list):
            data["recent_tx_count"] = len(result)
            data["sources_used"].append("etherscan")
            if len(result) >= 1000:
                data["notes"].append("recent_tx_count capped at the 1000-row page limit.")
        else:
            data["notes"].append("Recent activity unavailable; recent_tx_count imputed.")
        # buy/sell classification requires identifying the DEX pair address; out of
        # scope for the free tier — left as None so it is imputed neutrally.
        data["notes"].append("buy_sell_ratio requires DEX pair data (not collected).")

    def _collect_liquidity(self, checksum: str, data: RawTokenData) -> None:
        """Liquidity-lock comes from GoPlus LP data; note the gap only if unknown."""
        if data.get("liquidity_locked") is None:
            data["notes"].append(
                "Liquidity-lock status unavailable from GoPlus; left unknown (imputed)."
            )
        if data.get("liquidity_to_mcap_ratio") is None:
            data["notes"].append(
                "liquidity/mcap ratio needs price + market-cap data (not collected); imputed."
            )
