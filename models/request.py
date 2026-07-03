"""Input schemas for the Token Trust Analyzer."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

# Chains the on-chain collector knows how to reach. Maps the human name to the
# EVM chain id used by the Etherscan V2 multichain API.
SUPPORTED_CHAINS: dict[str, int] = {
    "ethereum": 1,
    "base": 8453,
}

_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


class AnalyzeRequest(BaseModel):
    """Full-pipeline request: collect -> features -> score -> (AI detect) -> report."""

    model_config = {"extra": "forbid"}

    contract_address: str = Field(
        ...,
        description="ERC-20 token contract address (0x-prefixed, 40 hex chars).",
        examples=["0x6B175474E89094C44Da98b954EedeAC495271d0F"],
    )
    chain: str = Field(
        default="ethereum",
        description=f"Chain to analyze. One of: {', '.join(SUPPORTED_CHAINS)}.",
    )
    project_text: Optional[str] = Field(
        default=None,
        description=(
            "Optional project marketing text / whitepaper excerpt. If given, the "
            "AI-content detector assesses whether it looks AI-generated."
        ),
        max_length=20_000,
    )
    project_url: Optional[str] = Field(
        default=None,
        description=(
            "Optional http(s) URL to a project page. Used only when project_text "
            "is absent: the app fetches the page (SSRF-guarded), extracts its text, "
            "and runs the AI-content detector on that. Ignored if project_text is set."
        ),
        max_length=2048,
    )

    @field_validator("contract_address")
    @classmethod
    def _valid_address(cls, v: str) -> str:
        v = v.strip()
        if not _ADDRESS_RE.match(v):
            raise ValueError(
                "contract_address must be a 0x-prefixed 40-hex-character EVM address"
            )
        return v

    @field_validator("chain")
    @classmethod
    def _known_chain(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in SUPPORTED_CHAINS:
            raise ValueError(
                f"chain must be one of: {', '.join(SUPPORTED_CHAINS)} (got {v!r})"
            )
        return v

    @field_validator("project_url")
    @classmethod
    def _valid_url(cls, v: Optional[str]) -> Optional[str]:
        # Syntactic check only: must be a well-formed http(s) URL. The SSRF safety
        # check (resolving the host and rejecting private/internal IPs) happens at
        # fetch time, so a well-formed-but-internal URL degrades to
        # ``ai_generated_content.checked = False`` rather than a 400.
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        parts = urlsplit(v)
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
            raise ValueError("project_url must be a valid http(s) URL")
        return v


class ScoreRequest(BaseModel):
    """Score-only request: caller supplies a pre-built raw feature set.

    Keys are the canonical feature names (see features.feature_extractor.FEATURE_ORDER).
    Any missing key is imputed to the healthy-token prior; unknown keys are rejected.
    """

    model_config = {"extra": "forbid"}

    features: dict[str, Optional[float]] = Field(
        ...,
        description="Raw feature name -> value (None allowed; missing keys imputed).",
    )
    contract_address: Optional[str] = Field(
        default=None, description="Optional address, echoed into the report."
    )
    chain: str = Field(default="ethereum")


class DetectAIRequest(BaseModel):
    """AI-content-detection-only request."""

    model_config = {"extra": "forbid"}

    project_text: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="Marketing / whitepaper text to assess for AI generation.",
    )


# Max tokens accepted in a single batch request.
BATCH_MAX_TOKENS = 25


class BatchTokenItem(BaseModel):
    """One token in a batch request.

    Deliberately lenient (plain strings, no address/chain validators): a bad
    address or unknown chain is caught per-token in the pipeline and returned as an
    ``error`` entry, so it never fails the whole batch.
    """

    model_config = {"extra": "forbid"}

    contract_address: str
    chain: str = "ethereum"


class AnalyzeBatchRequest(BaseModel):
    """Batch request: 1..BATCH_MAX_TOKENS tokens, with optional shared project_text."""

    model_config = {"extra": "forbid"}

    tokens: list[BatchTokenItem] = Field(
        ...,
        min_length=1,
        max_length=BATCH_MAX_TOKENS,
        description=f"1 to {BATCH_MAX_TOKENS} tokens to analyze.",
    )
    project_text: Optional[str] = Field(
        default=None,
        max_length=20_000,
        description="Optional marketing text applied to every token in the batch.",
    )
