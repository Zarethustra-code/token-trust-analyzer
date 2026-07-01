"""Token Trust Analyzer — FastAPI entry point.

Wires the pipeline together: collect -> features -> score -> report.

Phase 1 exposes ``/health``, ``/analyze`` (full on-chain pipeline) and ``/score``
(ML-only, needs no API keys — handy for offline testing/demos). The AI-content
detector (``/detect-ai``) and the CAP payment flow (``/cap/analyze``) are added in
Phase 2.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from cap.cap_wrapper import simulate_cap_cycle
from collectors.onchain_collector import OnChainCollector
from detectors.ai_content_detector import get_detector
from features.feature_extractor import FEATURE_ORDER, FeatureExtractor
from ml.anomaly_model import get_anomaly_model
from ml.scorer import TrustScorer
from models.request import (
    SUPPORTED_CHAINS,
    AnalyzeRequest,
    DetectAIRequest,
    ScoreRequest,
)
from models.response import (
    AIContentResult,
    DataQuality,
    TokenInfo,
    TokenMetrics,
    TrustReport,
)

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
    """
    collector = OnChainCollector(
        chain=req.chain,
        rpc_url=WEB3_RPC_URL,
        etherscan_api_key=ETHERSCAN_API_KEY,
        goplus_api_key=GOPLUS_API_KEY,
    )
    raw = collector.collect(req.contract_address)

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
    )
    return report


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
