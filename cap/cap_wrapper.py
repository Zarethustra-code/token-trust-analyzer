"""CROO Agent Protocol (CAP) integration.

This adapter connects the Token Trust Analyzer to the CROO Agent Protocol so the
agent can be **hired and paid on-chain**. It has two modes:

* **Live provider** (``run_worker`` / ``python -m cap.cap_wrapper``) — connects to
  the CROO event stream over a websocket and serves real, paid orders. This is the
  actual on-chain integration.
* **Local simulation** (``simulate_cap_cycle``) — walks the full
  **Post → Lock → Deliver → Clear** lifecycle locally, running the real analysis
  pipeline, so the agent is fully testable *without* ``croo-sdk`` or credentials.
  This backs the ``/cap/analyze`` endpoint.

NOTE on packaging: this package is named ``cap`` (not ``croo``) on purpose — a
local package named ``croo`` would shadow the installed ``croo-sdk`` and make
``from croo import AgentClient`` resolve to *this* code, silently disabling the
integration. With the package named ``cap``, the import below resolves to the
real SDK.

SDK surface (verified against ``croo-sdk`` 0.2.1 / ``examples/provider.py``):

    from croo import AgentClient, Config, EventType, DeliverableType, DeliverOrderRequest
    client = AgentClient(Config(base_url, ws_url, rpc_url), CROO_SDK_KEY)
    stream = await client.connect_websocket()   # already starts read/ping loops
    stream.on(EventType.NEGOTIATION_CREATED, cb) # e.negotiation_id
    stream.on(EventType.ORDER_PAID, cb)          # e.order_id
    neg    = await client.get_negotiation(e.negotiation_id)   # .requirements (str) + .metadata
    result = await client.accept_negotiation(e.negotiation_id)  # -> result.order.order_id
    order  = await client.get_order(order_id)                 # -> Order
    await client.reject_order(order_id, reason)               # -> None
    await client.deliver_order(order_id, DeliverOrderRequest(
        deliverable_type=DeliverableType.TEXT, deliverable_text=<report JSON>))

Registration and service PRICING happen on the CROO Agent Store dashboard, not in
code — there is **no** ``register_agent()``. If the service is configured with
``require_fund_transfer=true`` on the dashboard, use
``accept_negotiation_with_fund_address(...)`` instead of ``accept_negotiation()``;
for a flat-priced analysis service, plain accept (below) is correct.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from models.request import AnalyzeRequest
from models.response import TrustReport

logger = logging.getLogger("croo.cap")

_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")

# --- Defensive SDK import (the app/worker must import cleanly with or without it) ---
try:  # pragma: no cover - depends on optional dependency
    from croo import (  # type: ignore
        AgentClient,
        Config,
        DeliverableType,
        DeliverOrderRequest,
        EventType,
    )

    CROO_AVAILABLE = True

    try:  # error types are used in `except` clauses below
        from croo import APIError, InsufficientBalanceError  # type: ignore
    except Exception:  # pragma: no cover
        class APIError(Exception):  # type: ignore
            """Stand-in when croo.APIError is unavailable."""

        class InsufficientBalanceError(APIError):  # type: ignore
            """Stand-in when croo.InsufficientBalanceError is unavailable."""

except Exception:  # pragma: no cover
    AgentClient = Config = EventType = DeliverableType = DeliverOrderRequest = None  # type: ignore
    CROO_AVAILABLE = False

    class APIError(Exception):  # type: ignore
        """Stand-in for croo.APIError when croo-sdk is not installed."""

    class InsufficientBalanceError(APIError):  # type: ignore
        """Stand-in for croo.InsufficientBalanceError when croo-sdk is missing."""


class CAPError(Exception):
    """Configuration / adapter error (not an SDK API error)."""


@dataclass
class CAPConfig:
    """CAP configuration, read from environment variables."""

    sdk_key: Optional[str] = field(default_factory=lambda: os.getenv("CROO_SDK_KEY"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("CROO_API_URL"))
    ws_url: str = field(default_factory=lambda: os.getenv("CROO_WS_URL", ""))
    rpc_url: str = field(default_factory=lambda: os.getenv("BASE_RPC_URL", ""))
    service_id: Optional[str] = field(default_factory=lambda: os.getenv("CROO_SERVICE_ID"))
    wallet_address: Optional[str] = field(
        default_factory=lambda: os.getenv("CROO_WALLET_ADDRESS")
    )

    @property
    def is_configured(self) -> bool:
        """True when the SDK is installed and live credentials are present.

        ``ws_url`` is required by the SDK ``Config`` for the websocket, so it is
        part of the readiness check.
        """
        return bool(CROO_AVAILABLE and self.sdk_key and self.base_url and self.ws_url)


def build_client(config: CAPConfig):
    """Construct a real ``croo.AgentClient`` (positional args, per provider.py)."""
    if not CROO_AVAILABLE:
        raise CAPError("croo-sdk is not installed. Run: pip install 'croo-sdk==0.2.1'.")
    if not config.sdk_key:
        raise CAPError("CROO_SDK_KEY is not set.")
    if not config.base_url:
        raise CAPError("CROO_API_URL is not set.")
    if not config.ws_url:
        raise CAPError("CROO_WS_URL is not set (required for the websocket).")
    return AgentClient(
        Config(base_url=config.base_url, ws_url=config.ws_url, rpc_url=config.rpc_url),
        config.sdk_key,
    )


# --------------------------------------------------------------------------- #
# Running the analysis pipeline
#
# The pipeline (analyze_token) is synchronous (requests + scikit-learn), so we
# run it in a worker thread to avoid blocking the event loop. Imported lazily to
# avoid a circular import with app.py.
# --------------------------------------------------------------------------- #
async def _run_pipeline(request: AnalyzeRequest) -> TrustReport:
    from app import analyze_token  # lazy import

    return await asyncio.to_thread(analyze_token, request)


def _coerce_params(raw: Any) -> dict:
    """Turn a requirements/metadata value (JSON string, bare address, or dict) into a dict."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            # A plain address string is a valid, minimal "requirements".
            return {"contract_address": text} if _ADDRESS_RE.match(text) else {}
    return {}


def _request_from_negotiation(negotiation: Any) -> AnalyzeRequest:
    """Build an AnalyzeRequest from a Negotiation's buyer inputs.

    Verified against croo-sdk 0.2.1: the buyer's parameters live on
    ``Negotiation.requirements`` (a string) and ``Negotiation.metadata`` — NOT on
    the Order or the event. ``requirements`` is expected to be JSON like
    ``{"contract_address": "0x..", "chain": "base", "project_text": "..."}`` (a
    bare address string is also accepted); ``metadata`` supplements missing keys.
    """
    data = _coerce_params(getattr(negotiation, "requirements", None))
    meta = _coerce_params(getattr(negotiation, "metadata", None))
    for key in ("contract_address", "chain", "project_text"):
        if not data.get(key) and meta.get(key):
            data[key] = meta[key]

    addr = (data.get("contract_address") or "").strip()
    if not addr:
        raise CAPError(
            "Negotiation.requirements/metadata did not contain a contract_address."
        )
    return AnalyzeRequest(
        contract_address=addr,
        chain=(data.get("chain") or "ethereum"),
        project_text=data.get("project_text"),
    )


def _deliverable(report: TrustReport):
    """Build the DeliverOrderRequest carrying the Trust Report as JSON text."""
    # provider.py uses DeliverableType.TEXT + deliverable_text; the report body is
    # JSON so downstream agents can parse it.
    return DeliverOrderRequest(
        deliverable_type=DeliverableType.TEXT,
        deliverable_text=report.model_dump_json(),
    )


# --------------------------------------------------------------------------- #
# Live provider event handlers
#
# The buyer's inputs arrive with the *negotiation*, but the pipeline runs when the
# *order* is paid — so we cache the parsed request by the order_id returned from
# accept_negotiation, keyed in `pending`.
# --------------------------------------------------------------------------- #
async def handle_negotiation(client, event, pending: Dict[str, AnalyzeRequest]) -> None:
    """NEGOTIATION_CREATED → read buyer requirements → accept → cache by order_id."""
    negotiation_id = getattr(event, "negotiation_id", None)
    if not negotiation_id:
        logger.error("NEGOTIATION_CREATED without a negotiation_id: %r", event)
        return

    try:
        negotiation = await client.get_negotiation(negotiation_id)
        request = _request_from_negotiation(negotiation)
    except CAPError as exc:
        logger.error("Negotiation %s has no usable inputs: %s — not accepting.",
                     negotiation_id, exc)
        return
    except (APIError, InsufficientBalanceError) as exc:
        logger.error("get_negotiation(%s) failed: %s", negotiation_id, exc)
        return

    logger.info("NEGOTIATION_CREATED %s (token=%s) — accepting",
                negotiation_id, request.contract_address)
    try:
        # Flat-priced service → plain accept. For require_fund_transfer=true
        # services, use client.accept_negotiation_with_fund_address(...) instead.
        result = await client.accept_negotiation(negotiation_id)
    except (APIError, InsufficientBalanceError) as exc:
        logger.error("accept_negotiation(%s) failed: %s", negotiation_id, exc)
        return

    order = getattr(result, "order", None)   # AcceptNegotiationResult.order
    order_id = getattr(order, "order_id", None)
    if order_id:
        pending[order_id] = request
        logger.info("Order created: %s (awaiting payment)", order_id)
    else:
        logger.warning("accept_negotiation(%s) returned no order_id: %r",
                       negotiation_id, result)


async def handle_paid_order(client, event, pending: Dict[str, AnalyzeRequest]) -> None:
    """ORDER_PAID (escrow locked) → run pipeline → deliver_order."""
    order_id = getattr(event, "order_id", None)
    if not order_id:
        logger.error("ORDER_PAID without an order_id: %r", event)
        return

    request = pending.pop(order_id, None)
    if request is None:
        # Cache miss (e.g. worker restarted between accept and payment): recover
        # the negotiation via the order, then re-parse its requirements.
        request = await _recover_request(client, order_id)
    if request is None:
        reason = "provider lost the buyer requirements for this order"
        logger.error("Order %s: %s — rejecting.", order_id, reason)
        await _safe_reject_order(client, order_id, reason)
        return

    logger.info("ORDER_PAID %s — analyzing %s", order_id, request.contract_address)
    try:
        report = await _run_pipeline(request)
    except Exception as exc:
        logger.exception("Pipeline failed for order %s: %s", order_id, exc)
        await _safe_reject_order(client, order_id, f"provider execution error: {exc}")
        return

    try:
        await client.deliver_order(order_id, _deliverable(report))
        logger.info(
            "Delivered order %s (trust_score=%s, risk=%s)",
            order_id, report.trust_score, report.risk_level.value,
        )
    except (APIError, InsufficientBalanceError) as exc:
        logger.error("deliver_order(%s) failed: %s", order_id, exc)


async def _recover_request(client, order_id: str) -> Optional[AnalyzeRequest]:
    """Best-effort recovery of the buyer request from a paid order."""
    try:
        order = await client.get_order(order_id)
    except (APIError, InsufficientBalanceError) as exc:
        logger.info("get_order(%s) failed during recovery: %s", order_id, exc)
        return None
    negotiation_id = getattr(order, "negotiation_id", None)
    if not negotiation_id:
        return None
    try:
        negotiation = await client.get_negotiation(negotiation_id)
        return _request_from_negotiation(negotiation)
    except (CAPError, APIError, InsufficientBalanceError) as exc:
        logger.info("Could not recover request for order %s: %s", order_id, exc)
        return None


async def handle_order_completed(_client, event) -> None:
    """ORDER_COMPLETED → escrow cleared, settled on-chain (USDC on Base)."""
    logger.info(
        "ORDER_COMPLETED %s — escrow cleared, funds settled.",
        getattr(event, "order_id", "<unknown>"),
    )


async def _safe_reject_order(client, order_id: str, reason: str) -> None:
    try:
        await client.reject_order(order_id, reason)  # confirmed: async, returns None
    except (APIError, InsufficientBalanceError) as exc:
        logger.error("reject_order(%s) failed: %s", order_id, exc)


# --------------------------------------------------------------------------- #
# Live provider worker
# --------------------------------------------------------------------------- #
async def run_worker(config: Optional[CAPConfig] = None) -> None:
    """Connect to CAP and serve paid orders until interrupted."""
    config = config or CAPConfig()
    client = build_client(config)
    logger.info(
        "CAP worker starting (service_id=%s, wallet=%s)",
        config.service_id or "<unset>", config.wallet_address or "<unset>",
    )

    # connect_websocket() already calls stream.connect() and starts the background
    # read/ping loops — do NOT call it again (double-dial). We just register
    # handlers and keep the process alive.
    stream = await client.connect_websocket()

    # order_id -> parsed buyer request, populated on accept, consumed on payment.
    pending: Dict[str, AnalyzeRequest] = {}

    # EventStream.on() dispatches handlers synchronously (it does not await them),
    # so register plain callbacks that schedule the async work.
    def on_negotiation(event) -> None:
        asyncio.create_task(handle_negotiation(client, event, pending))

    def on_order_paid(event) -> None:
        asyncio.create_task(handle_paid_order(client, event, pending))

    def on_order_completed(event) -> None:
        asyncio.create_task(handle_order_completed(client, event))

    stream.on(EventType.NEGOTIATION_CREATED, on_negotiation)
    stream.on(EventType.ORDER_PAID, on_order_paid)
    stream.on(EventType.ORDER_COMPLETED, on_order_completed)

    logger.info("Listening for negotiations and paid orders. Ctrl-C to stop.")
    try:
        await asyncio.Event().wait()  # idle forever; the SDK drives the stream
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                await close()
            except Exception:
                pass
        logger.info("CAP worker stopped.")


# --------------------------------------------------------------------------- #
# Local simulation (no credentials required) — backs POST /cap/analyze
# --------------------------------------------------------------------------- #
async def simulate_cap_cycle(
    request: AnalyzeRequest, config: Optional[CAPConfig] = None
) -> dict:
    """Run the real pipeline wrapped in a simulated Post→Lock→Deliver→Clear cycle.

    Nothing settles on-chain; tx hashes are placeholders. Returns the lifecycle
    trace plus the delivered Trust Report.
    """
    config = config or CAPConfig()
    report = await _run_pipeline(request)

    lifecycle = [
        {"stage": "POST", "event": "NEGOTIATION_CREATED",
         "detail": f"buyer requests service_id={config.service_id or '<unset>'}; "
                   f"requirements carry contract_address={request.contract_address}"},
        {"stage": "LOCK", "event": "ORDER_PAID",
         "detail": "buyer USDC escrow-locked on Base", "tx_hash": "0xSIMULATED_PAY"},
        {"stage": "DELIVER", "event": "deliver_order",
         "detail": f"Trust Report delivered (score={report.trust_score}, "
                   f"risk={report.risk_level.value})", "tx_hash": "0xSIMULATED_DELIVER"},
        {"stage": "CLEAR", "event": "ORDER_COMPLETED",
         "detail": f"escrow cleared → settled to {config.wallet_address or '<unset>'}",
         "tx_hash": "0xSIMULATED_CLEAR"},
    ]

    return {
        "cap": {
            "mode": "simulation" if not config.is_configured else "simulation (credentials present; "
                    "live settlement runs via the CAP worker, not this endpoint)",
            "settled": False,
            "service_id": config.service_id,
            "wallet_address": config.wallet_address,
            "deliverable_type": "TEXT (JSON body)",
            "lifecycle": lifecycle,
        },
        "report": report,
    }


async def _main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = CAPConfig()
    if config.is_configured:
        await run_worker(config)
    else:
        logger.warning(
            "CAP not configured (need croo-sdk + CROO_SDK_KEY + CROO_API_URL + "
            "CROO_WS_URL) — running a local Post→Lock→Deliver→Clear SIMULATION."
        )
        sample = AnalyzeRequest(
            contract_address="0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
            chain="ethereum",
        )
        result = await simulate_cap_cycle(sample, config)
        for step in result["cap"]["lifecycle"]:
            logger.info("[SIM] %-8s %s — %s", step["stage"], step["event"], step["detail"])
        logger.info("[SIM] delivered report: %s", result["report"].model_dump_json()[:160])


if __name__ == "__main__":
    asyncio.run(_main())
