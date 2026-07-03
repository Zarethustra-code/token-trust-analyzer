"""AI-generated content detector (secondary / optional).

If a project's marketing text or whitepaper excerpt is supplied, this asks
Claude whether the text reads as AI-generated and returns a structured verdict.
It is deliberately isolated: the on-chain trust pipeline runs fully without it,
and this module degrades gracefully when no API key (or the ``anthropic`` package)
is present.

Per the build spec the model is instructed to emit JSON only, and the reply is
parsed defensively — we never assume well-formed output.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from detectors.url_fetch import fetch_project_text
from models.response import AIContentResult

logger = logging.getLogger("token_trust.ai_detector")

# The build spec pins Claude Sonnet 4.6 for this step.
DEFAULT_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = (
    "You are a forensic analyst that judges whether a crypto project's "
    "marketing / whitepaper text was written by an AI language model rather than "
    "a human team. Consider generic filler, hollow buzzwords, templated structure, "
    "uniform sentence rhythm, and lack of concrete specifics.\n\n"
    "Respond with a SINGLE JSON object and NOTHING else — no prose, no code fences. "
    "Schema:\n"
    '{"is_ai_generated": <true|false>, "confidence": <number 0..1>, '
    '"reason": "<one or two sentences>"}'
)


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from a model reply."""
    if not text:
        return None
    cleaned = text.strip()
    # Strip ```json ... ``` fences if present.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} span.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


class AIContentDetector:
    """Thin wrapper around the Anthropic Messages API for AI-text detection."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 500,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self._client = None  # created lazily so import never requires anthropic

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            import anthropic  # imported lazily; optional dependency

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def analyze(
        self,
        project_text: Optional[str] = None,
        project_url: Optional[str] = None,
    ) -> AIContentResult:
        """Resolve the text source, then run the existing detector on it.

        Precedence (matches the request contract):
          1. ``project_text`` — used as-is.
          2. else ``project_url`` — fetched (SSRF-guarded, see ``url_fetch``) and
             its extracted text is assessed.
          3. else — behaves exactly as before (``checked = False``).

        The returned result records which ``source`` was analyzed. Detection logic
        is never duplicated: this only chooses the input text and calls ``detect``.
        """
        if project_text and project_text.strip():
            return self.detect(project_text).model_copy(update={"source": "provided_text"})

        if project_url:
            fetched = fetch_project_text(project_url)
            if not fetched:
                return AIContentResult(
                    checked=False,
                    source="fetched_url",
                    reason=(
                        "Could not analyze project_url — it was blocked for safety "
                        "(non-public address or disallowed scheme), was unreachable, "
                        "or returned no readable text."
                    ),
                )
            return self.detect(fetched).model_copy(update={"source": "fetched_url"})

        return self.detect(None)

    def detect(self, project_text: Optional[str]) -> AIContentResult:
        """Assess ``project_text``; returns a graceful, never-raising result."""
        text = (project_text or "").strip()
        if not text:
            return AIContentResult(checked=False, reason="No project text was provided.")
        if not self.available:
            return AIContentResult(
                checked=False,
                reason="AI-content detection skipped (ANTHROPIC_API_KEY not set).",
            )

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Assess whether the following project text is AI-generated. "
                            "Return only the JSON object.\n\n"
                            f"<text>\n{text[:12000]}\n</text>"
                        ),
                    }
                ],
            )
            reply = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )
        except Exception as exc:  # includes anthropic.* errors and ImportError
            logger.warning("AI-content detection failed: %s", exc)
            return AIContentResult(
                checked=False, reason=f"AI-content detection unavailable: {exc}"
            )

        data = _extract_json(reply)
        if not isinstance(data, dict) or "is_ai_generated" not in data:
            return AIContentResult(
                checked=False,
                reason="AI-content detector returned an unparseable response.",
            )

        confidence = data.get("confidence")
        try:
            confidence = None if confidence is None else float(confidence)
            if confidence is not None:
                confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = None

        return AIContentResult(
            checked=True,
            is_ai_generated=bool(data.get("is_ai_generated")),
            confidence=confidence,
            reason=str(data.get("reason") or "").strip() or None,
        )


_DETECTOR: Optional[AIContentDetector] = None


def get_detector() -> AIContentDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = AIContentDetector()
    return _DETECTOR
