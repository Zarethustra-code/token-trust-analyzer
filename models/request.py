"""Input schemas for the Token Trust Analyzer."""

from __future__ import annotations

import re
from typing import Optional

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
