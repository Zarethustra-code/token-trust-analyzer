"""CROO consumer (requester) agent — hire the Token Trust Analyzer over CAP.

This is the **buyer** side of an agent-to-agent (A2A) trade: it hires the analyzer
(the provider in ``cap/cap_wrapper.py``) over the CROO Agent Protocol, pays in USDC
on Base, and receives the delivered **Trust Report**. Two modes:

* **Live CROO** (``run_live_requester``) — negotiate → pay → await delivery over the
  real ``croo-sdk`` websocket + REST API. This is the actual on-chain A2A loop and
  the marketplace's core differentiator.
* **Simulation** (``run_simulation``) — when ``croo-sdk`` / the CROO env isn't set,
  hire the *local* analyzer API (``POST /analyze``) and narrate the same
  Post → Lock → Deliver → Clear story, tagged ``[SIMULATION]``, so the full loop
  runs (and tests) with no keys and no funded wallet.

Run it: ``python -m cap.consumer 0x6B175474E89094C44Da98b954EedeAC495271d0F``

Requester SDK surface — verified against the installed ``croo-sdk`` 0.2.1
(``croo/agent_client.py`` + ``croo/types.py``):

    client = AgentClient(Config(base_url, ws_url, rpc_url), CONSUMER_SDK_KEY)
    neg    = await client.negotiate_order(NegotiateOrderRequest(
                 service_id=SERVICE_ID, requirements=<JSON w/ contract_address+chain>))
    # provider accepts -> Order is created; discover it via ORDER_CREATED or list_orders
    pay    = await client.pay_order(order_id)          # checks USDC balance, escrows on Base
    # provider delivers -> ORDER_COMPLETED
    dv     = await client.get_delivery(order_id)       # dv.deliverable_text == report JSON

The provider reads the buyer's inputs from ``Negotiation.requirements`` (see
``cap_wrapper._request_from_negotiation``), so we send the token there as JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
from typing import Any, Callable, Optional

from cap.cap_wrapper import CAPConfig, CAPError, CROO_AVAILABLE, build_client

logger = logging.getLogger("croo.consumer")

# Requester-only SDK symbols, imported defensively so the module loads with or
# without croo-sdk (the simulation path never touches these).
if CROO_AVAILABLE:  # pragma: no cover - depends on the optional dependency
    from croo import (  # type: ignore
        APIError,
        EventType,
        InsufficientBalanceError,
        ListOptions,
        NegotiateOrderRequest,
        OrderStatus,
    )
else:  # pragma: no cover
    EventType = NegotiateOrderRequest = ListOptions = OrderStatus = None  # type: ignore

    class APIError(Exception):  # type: ignore
        """Stand-in for croo.APIError when croo-sdk is missing."""

    class InsufficientBalanceError(APIError):  # type: ignore
        """Stand-in for croo.InsufficientBalanceError when croo-sdk is missing."""


DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"  # a nice, healthy demo token
CONSUMER_KEY_ENV = "CONSUMER_CROO_SDK_KEY"

# Order statuses that end the wait (verified against croo.types.OrderStatus).
_OK_STATUSES = {"completed"}
_BAD_STATUSES = {"rejected", "expired", "create_failed", "pay_failed", "deliver_failed"}


# --------------------------------------------------------------------------- #
# Config — reuse the provider's env pattern, but with the requester's identity
# --------------------------------------------------------------------------- #
def consumer_config() -> CAPConfig:
    """CAP config for the requester.

    Reuses ``CAPConfig`` (``CROO_API_URL`` / ``CROO_WS_URL`` / ``CROO_SERVICE_ID`` /
    ``BASE_RPC_URL``) but signs as a *second* identity: ``CONSUMER_CROO_SDK_KEY`` if
    set, otherwise falling back to ``CROO_SDK_KEY``.
    """
    cfg = CAPConfig()
    consumer_key = os.getenv(CONSUMER_KEY_ENV)
    if consumer_key:
        cfg.sdk_key = consumer_key
    return cfg


def _is_live(cfg: CAPConfig) -> bool:
    """Live requires the SDK, live creds (base_url/ws_url/sdk_key) AND a service_id."""
    return bool(cfg.is_configured and cfg.service_id)


# --------------------------------------------------------------------------- #
# Reporting helpers (shared by both modes)
# --------------------------------------------------------------------------- #
def _as_dict(report: Any) -> dict:
    if isinstance(report, dict):
        return report
    if hasattr(report, "model_dump"):
        return report.model_dump(mode="json")
    return dict(report)


def _summarize_report(report: Any, *, mode: str = "", log: Callable[[str], None] = print) -> None:
    """Print a clean Trust Report summary: score, risk, confidence, top flags."""
    report = _as_dict(report)
    tag = f"[{mode}] " if mode else ""
    token = report.get("token") or {}
    symbol = token.get("symbol") or "?"
    breakdown = report.get("score_breakdown") or {}
    flags = report.get("flags") or []

    log(f"{tag}╭─ Trust Report received ─────────────────────────────")
    log(f"{tag}│ token       : {symbol}  ({report.get('contract_address')})")
    log(f"{tag}│ trust_score : {report.get('trust_score')}/100  (higher = riskier)")
    log(f"{tag}│ risk_level  : {report.get('risk_level')}")
    log(f"{tag}│ confidence  : {breakdown.get('confidence')}")
    if flags:
        log(f"{tag}│ top flags   :")
        for flag in flags[:5]:
            log(f"{tag}│   • {flag}")
    else:
        log(f"{tag}│ flags       : (none — looks clean)")
    log(f"{tag}╰─────────────────────────────────────────────────────")


# --------------------------------------------------------------------------- #
# Live CROO requester
# --------------------------------------------------------------------------- #
def _match(fut: "asyncio.Future", negotiation_id: str, attr: str) -> Callable[[Any], None]:
    """WS handler that resolves ``fut`` with ``event.<attr>`` for our negotiation."""
    def handler(event: Any) -> None:
        ev_neg = getattr(event, "negotiation_id", "") or ""
        if negotiation_id and ev_neg and ev_neg != negotiation_id:
            return
        if not fut.done():
            fut.set_result(getattr(event, attr, "") or "")
    return handler


def _fail(fut: "asyncio.Future", negotiation_id: str) -> Callable[[Any], None]:
    """WS handler that fails ``fut`` on a rejection/expiry for our negotiation."""
    def handler(event: Any) -> None:
        ev_neg = getattr(event, "negotiation_id", "") or ""
        if negotiation_id and ev_neg and ev_neg != negotiation_id:
            return
        if not fut.done():
            reason = getattr(event, "reason", "") or "no reason given"
            fut.set_exception(RuntimeError(f"{getattr(event, 'type', 'failed')}: {reason}"))
    return handler


async def _wait_resolved(
    fut: "asyncio.Future",
    poll: Callable[[], "asyncio.Future"],
    *,
    timeout: float,
    interval: float = 3.0,
) -> Any:
    """Resolve via the WS future OR by polling, whichever happens first; time-boxed.

    A poll that raises (terminal order status) or a future set with an exception
    (rejection event) propagates out — we never hang forever.
    """
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while True:
        remaining = end - loop.time()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for the CROO order to progress")
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=min(interval, remaining))
        except asyncio.TimeoutError:
            pass  # WS hasn't fired yet — fall back to polling
        result = await poll()
        if result is not None:
            return result


async def _find_order(client: Any, negotiation_id: str) -> Optional[str]:
    """Poll fallback for order discovery: find our order via list_orders."""
    # TODO(CAP): the role filter value ("requester") is unconfirmed against the
    # live API — try it, then fall back to an unfiltered listing.
    orders: list = []
    for opts in (ListOptions(role="requester"), ListOptions()):
        try:
            orders = await client.list_orders(opts)
            break
        except (APIError, InsufficientBalanceError, Exception):
            continue
    for order in orders:
        if getattr(order, "negotiation_id", "") == negotiation_id:
            return order.order_id
    return None


async def _await_completion(client: Any, order_id: str) -> Optional[str]:
    """Poll fallback for delivery: return order_id once completed, raise if it failed."""
    order = await client.get_order(order_id)
    status = getattr(order, "status", "")
    if status in _OK_STATUSES:
        return order_id
    if status in _BAD_STATUSES:
        raise RuntimeError(f"order {order_id} ended in status {status!r}")
    return None


def _parse_deliverable(delivery: Any) -> dict:
    """Extract the Trust Report JSON from a delivery (DeliverableType.TEXT)."""
    text = getattr(delivery, "deliverable_text", "") or ""
    if not text:
        # TODO(CAP): SCHEMA / object-storage deliverables would be read via
        # get_download_url(delivery.deliverable_schema); the analyzer delivers TEXT.
        raise RuntimeError("delivery carried no deliverable_text")
    return json.loads(text)


async def run_live_requester(
    token_address: str,
    chain: str,
    config: CAPConfig,
    *,
    log: Callable[[str], None] = print,
    accept_timeout: float = 120.0,
    deliver_timeout: float = 180.0,
) -> dict:
    """Hire the analyzer over live CROO: negotiate → pay → await delivery → read."""
    client = build_client(config)
    loop = asyncio.get_running_loop()
    order_created: "asyncio.Future[str]" = loop.create_future()
    order_completed: "asyncio.Future[str]" = loop.create_future()

    log("[LIVE CROO] Hiring the Token Trust Analyzer over the CROO Agent Protocol.")
    log(f"[LIVE CROO]   service_id={config.service_id}  token={token_address}  chain={chain}")

    # The SDK's connect_websocket() already starts the read/ping loops. If it fails
    # we degrade to pure polling rather than aborting the trade.
    stream = None
    try:
        stream = await client.connect_websocket()
    except Exception as exc:  # noqa: BLE001 - WS is best-effort; we can still poll
        log(f"[LIVE CROO]   (websocket unavailable: {exc}; will poll order status)")

    try:
        # 1. NEGOTIATE — the token address rides in `requirements`, exactly where
        #    the provider reads it (cap_wrapper._request_from_negotiation).
        requirements = json.dumps({"contract_address": token_address, "chain": chain})
        log("[LIVE CROO] → NEGOTIATE  posting an order for the analyzer service")
        neg = await client.negotiate_order(
            NegotiateOrderRequest(service_id=config.service_id, requirements=requirements)
        )
        negotiation_id = neg.negotiation_id
        log(f"[LIVE CROO] ✓ NEGOTIATION_CREATED  negotiation_id={negotiation_id} status={neg.status}")

        # Register handlers now that we know our negotiation_id.
        if stream is not None:
            stream.on(EventType.ORDER_CREATED, _match(order_created, negotiation_id, "order_id"))
            stream.on(EventType.ORDER_COMPLETED, _match(order_completed, negotiation_id, "order_id"))
            stream.on(EventType.NEGOTIATION_REJECTED, _fail(order_created, negotiation_id))
            stream.on(EventType.NEGOTIATION_EXPIRED, _fail(order_created, negotiation_id))
            stream.on(EventType.ORDER_REJECTED, _fail(order_completed, negotiation_id))
            stream.on(EventType.ORDER_EXPIRED, _fail(order_completed, negotiation_id))

        # 2. Wait for the provider to ACCEPT (an Order is created).
        log("[LIVE CROO] … awaiting provider acceptance")
        order_id = await _wait_resolved(
            order_created,
            poll=lambda: _find_order(client, negotiation_id),
            timeout=accept_timeout,
        )
        log(f"[LIVE CROO] ✓ ORDER_ACCEPTED  order_id={order_id}")

        # 3. PAY (USDC escrow-locked on Base). Fails fast on insufficient balance.
        log(f"[LIVE CROO] → PAY  paying order {order_id} (USDC on Base)")
        try:
            pay = await client.pay_order(order_id)
        except InsufficientBalanceError as exc:
            raise CAPError(
                "Consumer wallet has insufficient USDC on Base to pay the order. "
                f"Fund the requester wallet and retry. ({exc})"
            ) from exc
        log(f"[LIVE CROO] ✓ ORDER_PAID  tx={pay.tx_hash or '<pending>'} status={pay.order.status}")

        # 4. Wait for DELIVERY + settlement.
        log("[LIVE CROO] … awaiting delivery of the Trust Report")
        await _wait_resolved(
            order_completed,
            poll=lambda: _await_completion(client, order_id),
            timeout=deliver_timeout,
        )
        log(f"[LIVE CROO] ✓ ORDER_COMPLETED  order_id={order_id} (delivered + settled on Base)")

        # 5. Read the deliverable (the Trust Report JSON) and summarize.
        delivery = await client.get_delivery(order_id)
        report = _parse_deliverable(delivery)
        _summarize_report(report, mode="LIVE CROO", log=log)
        return {"mode": "live", "order_id": order_id, "report": report}
    finally:
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.close()
        with contextlib.suppress(Exception):
            await client.close()


# --------------------------------------------------------------------------- #
# Simulation requester (no CROO, no wallet) — hires the local analyzer API
# --------------------------------------------------------------------------- #
def _default_local_fetch(token: str, chain: str, app_base_url: Optional[str] = None) -> dict:
    """Get a Trust Report from the local analyzer.

    Prefers a POST to a running local API (the honest "hire over HTTP" story); if
    no server is reachable, runs the pipeline in-process so the demo still completes
    with a single command.
    """
    base = (app_base_url or os.getenv("APP_BASE_URL") or "http://localhost:8000").rstrip("/")
    try:
        import requests

        resp = requests.post(
            f"{base}/analyze",
            json={"contract_address": token, "chain": chain},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 - fall back to in-process so the demo runs
        logger.info("Local API at %s unavailable (%s); running the pipeline in-process.", base, exc)
        from app import analyze_token  # lazy: importing app fits the model
        from models.request import AnalyzeRequest

        return analyze_token(AnalyzeRequest(contract_address=token, chain=chain)).model_dump(mode="json")


def run_simulation(
    token_address: str,
    chain: str = "ethereum",
    *,
    report_fetcher: Optional[Callable[[str, str], Any]] = None,
    app_base_url: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Narrate an A2A hire (Post→Lock→Deliver→Clear) against the local analyzer."""
    log("[SIMULATION] No live CROO config — simulating an A2A hire against the local analyzer.")
    log("[SIMULATION] → POST     requester posts an order for the Token Trust Analyzer service")
    log(f"[SIMULATION]            requirements: contract_address={token_address}, chain={chain}")
    log("[SIMULATION] → LOCK     (simulated) buyer's USDC escrow-locked on Base")
    log("[SIMULATION] → DELIVER  provider runs the analysis pipeline and delivers the Trust Report")

    fetch = report_fetcher or (lambda t, c: _default_local_fetch(t, c, app_base_url))
    report = _as_dict(fetch(token_address, chain))

    log("[SIMULATION] ✓ CLEAR    (simulated) escrow cleared; Trust Report received")
    _summarize_report(report, mode="SIMULATION", log=log)
    return {"mode": "simulation", "report": report}


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def run_consumer(
    token_address: str,
    chain: str = "ethereum",
    *,
    force_mode: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Auto-select live vs simulation (or force one) and hire the analyzer."""
    cfg = consumer_config()
    live = _is_live(cfg) if force_mode is None else (force_mode == "live")
    if live:
        if not CROO_AVAILABLE:
            raise CAPError("live mode requested but croo-sdk is not installed.")
        if not cfg.service_id:
            raise CAPError("CROO_SERVICE_ID is required for live mode.")
        return asyncio.run(run_live_requester(token_address, chain, cfg, log=log))
    return run_simulation(token_address, chain, log=log)


def main(argv: Optional[list] = None) -> dict:
    from dotenv import load_dotenv

    load_dotenv()
    # Keep the SDK's own INFO chatter quiet so the [LIVE CROO]/[SIMULATION] narration
    # reads cleanly in the demo video; raise LOG_LEVEL to see the raw SDK logs.
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"), format="%(message)s")

    parser = argparse.ArgumentParser(
        description="CROO consumer agent — hire the Token Trust Analyzer over CAP (A2A)."
    )
    parser.add_argument("address", nargs="?", default=DAI, help="ERC-20 contract address (default: DAI)")
    parser.add_argument("--chain", default="ethereum", help="ethereum | base (default: ethereum)")
    parser.add_argument(
        "--mode", choices=["auto", "live", "simulation"], default="auto",
        help="auto = live when CROO is configured, else simulation",
    )
    args = parser.parse_args(argv)

    force = None if args.mode == "auto" else args.mode
    try:
        return run_consumer(args.address, args.chain, force_mode=force, log=print)
    except CAPError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
