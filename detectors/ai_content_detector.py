"""AI-generated content detector (secondary / optional) — pluggable backends.

If a project's marketing text or whitepaper excerpt is supplied, the detector
judges whether the text reads as AI-generated and returns a structured verdict
(``AIContentResult``). It is deliberately isolated: the on-chain trust pipeline
runs fully without it, and every backend degrades gracefully — a missing
dependency, key, or model never crashes a request.

The backend is selected with ``AI_DETECTOR_BACKEND``:

* ``local`` (default) — a fully offline two-model pipeline (a RoBERTa-style
  AI-text classifier for the verdict + a small instruct SLM for the prose
  reason); see ``detectors/local_ai_detector.py``. Needs the optional deps in
  ``requirements-slm.txt``.
* ``anthropic`` — the original Claude path below (needs ``ANTHROPIC_API_KEY``).
* ``off`` — detection always returns ``checked=False``.

All backends share ``BaseAIDetector.analyze`` (text > url source precedence)
and the same result shape, so ``app.py`` is backend-agnostic.

For the Claude path the model is instructed to emit JSON only, and the reply is
parsed defensively — we never assume well-formed output.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
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


class BaseAIDetector:
    """Shared contract for detector backends: source resolution + result shape.

    Subclasses implement ``detect`` (and ``available``); ``analyze`` is the
    single place that decides WHICH text gets assessed, so the precedence rules
    are never duplicated across backends.
    """

    @property
    def available(self) -> bool:  # pragma: no cover - trivially overridden
        return False

    def detect(self, project_text: Optional[str]) -> AIContentResult:
        raise NotImplementedError

    def analyze(
        self,
        project_text: Optional[str] = None,
        project_url: Optional[str] = None,
    ) -> AIContentResult:
        """Resolve the text source, then run the backend's detector on it.

        Precedence (matches the request contract):
          1. ``project_text`` — used as-is.
          2. else ``project_url`` — fetched (SSRF-guarded, see ``url_fetch``) and
             its extracted text is assessed.
          3. else — behaves exactly as before (``checked = False``).

        ``source`` records the backend/models used when ``detect`` sets it (the
        local pipeline does, e.g. ``"local:<classifier>+<slm>"``); otherwise it
        falls back to the origin of the analyzed text. Detection logic is never
        duplicated: this only chooses the input text and calls ``detect``.
        """
        if project_text and project_text.strip():
            result = self.detect(project_text)
            if result.source:
                return result
            return result.model_copy(update={"source": "provided_text"})

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
            result = self.detect(fetched)
            if result.source:
                return result
            return result.model_copy(update={"source": "fetched_url"})

        return self.detect(None)


class OffDetector(BaseAIDetector):
    """``AI_DETECTOR_BACKEND=off`` — detection is disabled, always unchecked."""

    _REASON = "AI-content detection is disabled (AI_DETECTOR_BACKEND=off)."

    @property
    def available(self) -> bool:
        return False

    def analyze(
        self,
        project_text: Optional[str] = None,
        project_url: Optional[str] = None,
    ) -> AIContentResult:
        # Skip even the URL fetch: no point doing network work for a disabled check.
        return AIContentResult(checked=False, reason=self._REASON)

    def detect(self, project_text: Optional[str]) -> AIContentResult:
        return AIContentResult(checked=False, reason=self._REASON)


class AIContentDetector(BaseAIDetector):
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


VALID_BACKENDS = ("local", "anthropic", "off")
DEFAULT_BACKEND = "local"

# One cached instance per backend, so flipping AI_DETECTOR_BACKEND at runtime
# (tests do) picks the right detector without re-creating it on every request.
# The lock matters for the local backend: concurrent first requests (e.g. a
# /analyze/batch fan-out) must share ONE LocalAIDetector, or each would lazily
# load its own multi-GB model pipelines.
_DETECTORS: dict[str, BaseAIDetector] = {}
_DETECTORS_LOCK = threading.Lock()


def get_backend() -> str:
    """Resolve AI_DETECTOR_BACKEND, falling back to the default on junk values."""
    backend = (os.getenv("AI_DETECTOR_BACKEND") or DEFAULT_BACKEND).strip().lower()
    if backend not in VALID_BACKENDS:
        logger.warning(
            "Unknown AI_DETECTOR_BACKEND=%r; using %r. Valid values: %s",
            backend,
            DEFAULT_BACKEND,
            ", ".join(VALID_BACKENDS),
        )
        backend = DEFAULT_BACKEND
    return backend


def get_detector() -> BaseAIDetector:
    backend = get_backend()
    detector = _DETECTORS.get(backend)
    if detector is None:
        with _DETECTORS_LOCK:
            detector = _DETECTORS.get(backend)
            if detector is None:
                if backend == "anthropic":
                    detector = AIContentDetector()
                elif backend == "off":
                    detector = OffDetector()
                else:
                    # Imported lazily: the local backend's module stays out of the
                    # hot import path; heavy deps load only on first detect().
                    # Use the PROCESS-WIDE shared instance so the local SLM is
                    # loaded at most once across the detector and the risk-narrative
                    # writer (ml.narrative), never a second copy.
                    from detectors.local_ai_detector import get_shared_local_detector

                    detector = get_shared_local_detector()
                _DETECTORS[backend] = detector
    return detector
