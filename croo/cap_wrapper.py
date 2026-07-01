"""CROO Agent Protocol (CAP) integration.

This adapter connects the Token Trust Analyzer to the CROO Agent Protocol so the
agent can be **hired and paid on-chain**. It has two modes:

* **Live provider** (``run_worker`` / ``python -m croo.cap_wrapper``) — connects to
  the CROO event stream over a websocket and serves real, paid orders. This is the
  actual on-chain integration.
* **Local simulation** (``simulate_cap_cycle``) — walks the full
  **Post → Lock → Deliver → Clear** lifecycle locally, running the real analysis
  pipeline, so the agent is fully testable *without* ``croo-sdk`` or credentials.
  This backs the ``/cap/analyze`` endpoint.

The SDK surface below was validated against the reference provider implementation
(`CROO-Network/python-sdk` → ``examples/provider.py``):

    from croo import AgentClient, Config, EventType, DeliverableType, DeliverOrderRequest, Event
    client = AgentClient(Config(base_url, ws_url, rpc_url), CROO_SDK_KEY)
    stream = await client.connect_websocket()
    stream.on(EventType.NEGOTIATION_CREATED, cb)   # e.negotiation_id
    stream.on(EventType.ORDER_PAID, cb)            # e.order_id
    result = await client.accept_negotiation(e.negotiation_id)   # -> result.order.order_id
    await client.deliver_order(e.order_id, DeliverOrderRequest(
        deliverable_type=DeliverableType.TEXT, deliverable_text=...))

Registration and service PRICING happen on the CROO Agent Store dashboard, not in
code — there is **no** ``register_agent()``. Spots that still need confirmation
against the live event/order payload shape are marked ``# TODO(CAP)``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from models.request import AnalyzeRequest
from models.response import TrustReport

logger = logging.getLogger("croo.cap")

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
        """True when the SDK is installed and live credentials are present."""
        return bool(CROO_AVAILABLE and self.sdk_key and self.base_url)


def build_client(config: CAPConfig):
    """Construct a real ``croo.AgentClient`` (positional args, per provider.py)."""
    if not CROO_AVAILABLE:
        raise CAPError("croo-sdk is not installed. Run: pip install 'croo-sdk==0.2.1'.")
    if not config.sdk_key:
        raise CAPError("CROO_SDK_KEY is not set.")
    if not config.base_url:
        raise CAPError("CROO_API_URL is not set.")
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


def _request_from_order(event: Any, order: Any = None) -> AnalyzeRequest:
    """Build an AnalyzeRequest from the buyer's paid order.

    # TODO(CAP): confirm where the buyer's inputs live on a paid order. The
    #            reference provider.py exposes e.order_id but does not show the
    #            buyer-parameter shape. We probe the most likely locations
    #            (event/order attributes, then a JSON 'requirements'/'params'
    #            field) and fall back to raising so failures are loud.
    """
    candidates = (order, event)
    data: dict = {}
    for obj in candidates:
        if obj is None:
            continue
        raw = None
        for attr in ("requirements", "params", "input", "metadata"):
            raw = getattr(obj, attr, None)
            if raw:
                break
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {}
        elif isinstance(raw, dict):
            data = raw
        # Direct attributes as a last resort.
        addr = data.get("contract_address") or getattr(obj, "contract_address", None)
        if addr:
            return AnalyzeRequest(
                contract_address=addr,
                chain=data.get("chain") or getattr(obj, "chain", None) or "ethereum",
                project_text=data.get("project_text"),
            )
    raise CAPError(
        "Could not find a 'contract_address' on the paid order. Adjust "
        "_request_from_order() to the real Order payload (see examples/provider.py)."
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
# --------------------------------------------------------------------------- #
async def handle_negotiation(client, event) -> None:
    """NEGOTIATION_CREATED → accept (creates an Order)."""
    negotiation_id = getattr(event, "negotiation_id", None)
    if not negotiation_id:
        logger.error("NEGOTIATION_CREATED without a negotiation_id: %r", event)
        return
    logger.info("NEGOTIATION_CREATED %s — accepting", negotiation_id)
    try:
        result = await client.accept_negotiation(negotiation_id)
    except (APIError, InsufficientBalanceError) as exc:
        logger.error("accept_negotiation(%s) failed: %s", negotiation_id, exc)
        return
    order = getattr(result, "order", None)
    logger.info("Order created: %s", getattr(order, "order_id", "<unknown>"))


async def handle_paid_order(client, event) -> None:
    """ORDER_PAID (escrow locked) → run pipeline → deliver_order."""
    order_id = getattr(event, "order_id", None)
    if not order_id:
        logger.error("ORDER_PAID without an order_id: %r", event)
        return

    # Fetch full order details if the SDK exposes it (reference didn't confirm).
    order = None
    get_order = getattr(client, "get_order", None)
    if callable(get_order):  # TODO(CAP): confirm get_order exists / its return shape
        try:
            order = await get_order(order_id)
        except Exception as exc:
            logger.info("get_order(%s) unavailable: %s", order_id, exc)

    try:
        request = _request_from_order(event, order)
    except CAPError as exc:
        logger.error("Cannot build request for order %s: %s", order_id, exc)
        await _safe_reject_order(client, order_id, str(exc))
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


async def handle_order_completed(_client, event) -> None:
    """ORDER_COMPLETED → escrow cleared, settled on-chain (USDC on Base)."""
    logger.info(
        "ORDER_COMPLETED %s — escrow cleared, funds settled.",
        getattr(event, "order_id", "<unknown>"),
    )


async def _safe_reject_order(client, order_id: str, reason: str) -> None:
    reject = getattr(client, "reject_order", None)  # TODO(CAP): confirm reject_order
    if not callable(reject):
        logger.warning("No reject_order on client; order %s left unhandled.", order_id)
        return
    try:
        await reject(order_id, reason)
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

    stream = await client.connect_websocket()

    # EventStream.on() dispatches handlers synchronously (it does not await them),
    # so register plain callbacks that schedule the async work — the provider.py
    # pattern.
    def on_negotiation(event) -> None:
        asyncio.create_task(handle_negotiation(client, event))

    def on_order_paid(event) -> None:
        asyncio.create_task(handle_paid_order(client, event))

    def on_order_completed(event) -> None:
        asyncio.create_task(handle_order_completed(client, event))

    stream.on(EventType.NEGOTIATION_CREATED, on_negotiation)
    stream.on(EventType.ORDER_PAID, on_order_paid)
    stream.on(EventType.ORDER_COMPLETED, on_order_completed)

    logger.info("Listening for negotiations and paid orders. Ctrl-C to stop.")
    # TODO(CAP): confirm the run/dispatch call. examples/provider.py opens the
    #            stream and registers handlers; the process then stays alive
    #            dispatching events. We try the likely blocking calls, else idle.
    run = getattr(stream, "connect", None) or getattr(stream, "run", None) \
        or getattr(stream, "listen", None)
    try:
        if callable(run):
            await run()
        else:
            while True:
                await asyncio.sleep(3600)
    finally:
        for closer in ("close",):
            fn = getattr(stream, closer, None)
            if callable(fn):
                try:
                    await fn()
                except Exception:
                    pass
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
         "detail": f"buyer requests service_id={config.service_id or '<unset>'}"},
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
            "CAP not configured (need croo-sdk + CROO_SDK_KEY + CROO_API_URL) — "
            "running a local Post→Lock→Deliver→Clear SIMULATION."
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
