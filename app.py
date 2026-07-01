"""Token Trust Analyzer — FastAPI entry point.

Wires the pipeline together: collect -> features -> score -> report.

Phase 1 exposes ``/health``, ``/analyze`` (full on-chain pipeline) and ``/score``
(ML-only, needs no API keys — handy for offline testing/demos). The AI-content
detector (``/detect-ai``) and the CAP payment flow (``/cap/analyze``) are added in
Phase 2.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from cap.cap_wrapper import simulate_cap_cycle
from collectors.onchain_collector import OnChainCollector
from detectors.ai_content_detector import get_detector
from features.feature_extractor import FEATURE_ORDER, FeatureExtractor
from ml.anomaly_model import get_anomaly_model
from ml.scorer import TrustScorer
from models.request import (
    SUPPORTED_CHAINS,
    AnalyzeRequest,
    AnalyzeBatchRequest,
    BatchTokenItem,
    DetectAIRequest,
    ScoreRequest,
)
from models.response import (
    AIContentResult,
    AnalyzeBatchResponse,
    BatchResultItem,
    DataQuality,
    TokenInfo,
    TokenMetrics,
    TrustReport,
)
from utils.cache import TTLCache

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("token_trust.app")

# --- configuration ------------------------------------------------------------

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
WEB3_RPC_URL = os.getenv("WEB3_RPC_URL")
GOPLUS_API_KEY = os.getenv("GOPLUS_API_KEY")  # optional; public endpoint needs none
DEFAULT_CHAIN = os.getenv("CHAIN", "ethereum").lower()

# Shared, stateless singletons.
_extractor = FeatureExtractor()
_scorer = TrustScorer()


def _cache_ttl_seconds() -> float:
    """Cache TTL from env (default 600s). 0 (or invalid) disables caching."""
    try:
        return float(os.getenv("CACHE_TTL_SECONDS", "600"))
    except ValueError:
        return 600.0


# Caches the expensive on-chain collection (RawTokenData) keyed by (chain, address).
# Scoring + AI detection still run fresh on top, so project_text stays correct.
# TTL is read live from the env so CACHE_TTL_SECONDS=0 disables it at runtime.
raw_cache = TTLCache(_cache_ttl_seconds, max_entries=int(os.getenv("CACHE_MAX_ENTRIES", "512")))

# Batch endpoint: bounded concurrency protects the external APIs' rate limits.
_BATCH_WORKERS = max(1, int(os.getenv("BATCH_CONCURRENCY", "5")))

_BOOLEAN_METRIC_FIELDS = {
    "liquidity_locked", "source_verified", "has_mint",
    "ownership_renounced", "has_blacklist", "is_honeypot",
    "hidden_owner", "can_take_back_ownership", "is_anti_whale",
}
_INT_METRIC_FIELDS = {"holder_count", "recent_tx_count"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metrics_from_raw(raw: dict[str, Optional[float]]) -> TokenMetrics:
    """Turn the raw (float|None) feature dict into the typed TokenMetrics model."""
    payload: dict[str, object] = {}
    for name in FEATURE_ORDER:
        value = raw.get(name)
        if value is None:
            payload[name] = None
        elif name in _BOOLEAN_METRIC_FIELDS:
            payload[name] = bool(value >= 0.5)
        elif name in _INT_METRIC_FIELDS:
            payload[name] = int(round(value))
        else:
            payload[name] = float(value)
    return TokenMetrics(**payload)


# --- shared pipeline (also reused by the CAP worker in Phase 2) ---------------

def analyze_token(req: AnalyzeRequest) -> TrustReport:
    """Run the full analysis pipeline for one token and return a TrustReport.

    Raises ``ValueError`` for a structurally invalid address; all other data
    gaps degrade gracefully inside the collector.

    The expensive on-chain collection is cached by ``(chain, address)``; scoring and
    AI detection always run fresh on top (so ``project_text`` is honored per call).
    """
    cache_key = (req.chain.lower(), req.contract_address.lower())
    raw = raw_cache.get(cache_key)
    was_cached = raw is not None
    if not was_cached:
        collector = OnChainCollector(
            chain=req.chain,
            rpc_url=WEB3_RPC_URL,
            etherscan_api_key=ETHERSCAN_API_KEY,
            goplus_api_key=GOPLUS_API_KEY,
        )
        raw = collector.collect(req.contract_address)
        raw_cache.set(cache_key, raw)

    feature_set = _extractor.extract(raw)
    result = _scorer.score(feature_set)

    # Secondary, optional signal: only runs if project_text was supplied.
    ai_result = get_detector().detect(req.project_text)

    report = TrustReport(
        contract_address=req.contract_address,
        chain=req.chain,
        token=TokenInfo(**raw.get("token_info", {})),
        trust_score=result.trust_score,
        risk_level=result.risk_level,
        flags=result.flags,
        metrics=_metrics_from_raw(feature_set.raw),
        score_breakdown=result.breakdown,
        ai_generated_content=ai_result,
        data_quality=DataQuality(
            sources_used=raw.get("sources_used", []),
            missing_fields=raw.get("missing_fields", []),
            notes=raw.get("notes", []),
        ),
        explanation=result.explanation,
        generated_at=_now_iso(),
        cached=was_cached,
    )
    return report


def _run_batch(req: AnalyzeBatchRequest) -> AnalyzeBatchResponse:
    """Analyze a batch of tokens: dedupe, bounded concurrency, per-token isolation.

    Runs on a worker thread (see the endpoint) since the pipeline is sync/requests-
    based. Each token is validated + analyzed independently — one bad token yields an
    ``error`` entry without failing the batch. Duplicate ``(chain, address)`` pairs are
    computed once and reused.
    """
    # Preserve request order while collapsing duplicates.
    order: list[tuple[tuple[str, str], BatchTokenItem]] = []
    unique: dict[tuple[str, str], BatchTokenItem] = {}
    for item in req.tokens:
        key = (item.chain.strip().lower(), item.contract_address.strip().lower())
        order.append((key, item))
        unique.setdefault(key, item)

    def work(item: BatchTokenItem):
        try:
            single = AnalyzeRequest(
                contract_address=item.contract_address,
                chain=item.chain,
                project_text=req.project_text,
            )
        except ValidationError as exc:
            return ("error", _validation_message(exc))
        try:
            return ("report", analyze_token(single))
        except ValueError as exc:
            return ("error", str(exc))
        except Exception as exc:  # pragma: no cover - defensive per-token isolation
            logger.exception("Batch token %s failed", item.contract_address)
            return ("error", f"analysis failed: {exc}")

    computed: dict[tuple[str, str], tuple[str, object]] = {}
    with ThreadPoolExecutor(max_workers=_BATCH_WORKERS) as pool:
        futures = {pool.submit(work, item): key for key, item in unique.items()}
        for future in as_completed(futures):
            computed[futures[future]] = future.result()

    results: list[BatchResultItem] = []
    for key, item in order:
        kind, value = computed[key]
        results.append(
            BatchResultItem(
                contract_address=item.contract_address,
                chain=item.chain,
                report=value if kind == "report" else None,
                error=value if kind == "error" else None,
            )
        )
    return AnalyzeBatchResponse(results=results)


def _validation_message(exc: ValidationError) -> str:
    errors = exc.errors()
    if errors:
        msg = errors[0].get("msg", "invalid request")
        return msg.replace("Value error, ", "")
    return "invalid request"


def score_features(req: ScoreRequest) -> TrustReport:
    """Score a caller-supplied raw feature set (no network access needed)."""
    unknown = set(req.features) - set(FEATURE_ORDER)
    if unknown:
        raise ValueError(f"Unknown feature keys: {sorted(unknown)}")
    feature_set = _extractor.extract(dict(req.features))
    result = _scorer.score(feature_set)
    return TrustReport(
        contract_address=req.contract_address or "0x" + "0" * 40,
        chain=req.chain,
        token=TokenInfo(),
        trust_score=result.trust_score,
        risk_level=result.risk_level,
        flags=result.flags,
        metrics=_metrics_from_raw(feature_set.raw),
        score_breakdown=result.breakdown,
        ai_generated_content=AIContentResult(checked=False),
        data_quality=DataQuality(
            sources_used=["caller-supplied"],
            missing_fields=feature_set.imputed_features,
            notes=["Scored from caller-supplied features; no on-chain data fetched."],
        ),
        explanation=result.explanation,
        generated_at=_now_iso(),
    )


# --- FastAPI app --------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm (train-or-load) the anomaly model so the first request is fast and any
    # training error surfaces at startup rather than mid-request.
    try:
        get_anomaly_model()
        logger.info("Anomaly model ready.")
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to prepare anomaly model at startup: %s", exc)
    yield


app = FastAPI(
    title="Token Trust Analyzer",
    version="0.1.0",
    description="Explainable ERC-20 token trust scoring (rules + Isolation Forest).",
    lifespan=lifespan,
)


_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


@app.get("/ui", include_in_schema=False)
async def ui() -> FileResponse:
    """Serve the single-page demo frontend (talks to /analyze on the same origin)."""
    return FileResponse(os.path.join(_WEB_DIR, "index.html"), media_type="text/html")


@app.get("/health")
async def health() -> dict:
    model = None
    try:
        model = get_anomaly_model()
    except Exception:
        model = None
    return {
        "status": "ok",
        "supported_chains": list(SUPPORTED_CHAINS),
        "default_chain": DEFAULT_CHAIN,
        "anomaly_model_ready": bool(model and model.is_fitted),
        "goplus_enabled": True,  # public endpoint; primary raw-signal source
        "etherscan_configured": bool(ETHERSCAN_API_KEY),
        "rpc_configured": bool(WEB3_RPC_URL),
        "ai_detector_configured": get_detector().available,
        "cache_enabled": raw_cache.enabled,
        "cache_ttl_seconds": _cache_ttl_seconds(),
    }


@app.post("/analyze", response_model=TrustReport)
async def analyze(req: AnalyzeRequest) -> TrustReport:
    try:
        return analyze_token(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unexpected error in /analyze")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@app.post("/analyze/batch", response_model=AnalyzeBatchResponse)
async def analyze_batch(req: AnalyzeBatchRequest) -> AnalyzeBatchResponse:
    """Analyze up to 25 tokens in one request.

    Tokens are deduplicated and run with bounded concurrency; each is isolated, so a
    bad address / failing API for one token becomes an ``error`` entry (never a 500).
    Batches over the size cap are rejected with 422 by request validation.
    """
    # The pipeline is synchronous (requests + sklearn); run the whole batch off the
    # event loop so it doesn't block other requests.
    return await asyncio.to_thread(_run_batch, req)


@app.post("/score", response_model=TrustReport)
async def score(req: ScoreRequest) -> TrustReport:
    try:
        return score_features(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error in /score")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")


@app.post("/detect-ai", response_model=AIContentResult)
async def detect_ai(req: DetectAIRequest) -> AIContentResult:
    """AI-generated-content detection only (Claude). Needs ANTHROPIC_API_KEY."""
    try:
        return get_detector().detect(req.project_text)
    except Exception as exc:  # pragma: no cover - detector already degrades gracefully
        logger.exception("Unexpected error in /detect-ai")
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}")


@app.post("/cap/analyze")
async def cap_analyze(req: AnalyzeRequest) -> dict:
    """Run /analyze inside a (simulated) CROO Post→Lock→Deliver→Clear payment cycle.

    The real on-chain settlement is event-driven and served by the CAP worker
    (`python -m cap.cap_wrapper`); this endpoint is a local demo of the same
    lifecycle wrapped around one analysis.
    """
    try:
        return await simulate_cap_cycle(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error in /cap/analyze")
        raise HTTPException(status_code=500, detail=f"CAP analysis failed: {exc}")


@app.exception_handler(ValueError)
async def _value_error_handler(_request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
