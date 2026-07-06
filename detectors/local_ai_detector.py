"""Local (offline) AI-content detection — the ``AI_DETECTOR_BACKEND=local`` backend.

A fully local two-model pipeline, so detection runs offline and free:

1. **Classifier** (accuracy-critical): a RoBERTa-style AI-text detector run via a
   ``transformers`` text-classification pipeline. Its label maps to
   ``is_ai_generated`` and its score to ``confidence``. Default:
   ``Hello-SimpleAI/chatgpt-detector-roberta`` (labels ``ChatGPT`` / ``Human``);
   ``openai-community/roberta-base-openai-detector`` (labels ``Fake`` / ``Real``)
   also works. Override with ``AI_DETECTOR_CLASSIFIER_MODEL``.
2. **Reason generator** (best-effort): a small instruct SLM prompted for a one–two
   sentence justification of the classifier's verdict. Default:
   ``Qwen/Qwen2.5-1.5B-Instruct`` (override with ``AI_DETECTOR_SLM_MODEL``).
   Greedy decoding (``do_sample=False``) keeps the reason stable across runs.

The classifier is REQUIRED for a verdict: if it can't be loaded or fails,
``detect`` returns ``checked=False`` with a clear reason. The SLM is best-effort:
if only the reason generator is unavailable, the (accuracy-critical) verdict is
kept and a deterministic templated reason is returned instead — ``source`` then
names only the classifier, so the degradation is visible.

Heavy deps (``transformers``, ``torch``) live in ``requirements-slm.txt`` and are
imported lazily on first use — importing this module is cheap, the app boots
instantly, and a missing install degrades to ``checked=False`` instead of
crashing. Load failures are cached so a broken install doesn't re-pay download
timeouts on every request, but retried after a backoff so a transient Hub/network
blip doesn't disable detection until restart. Inference on the shared pipelines
is serialized with a lock: it is CPU-bound anyway, and the HF fast tokenizer's
truncation state is not safe under concurrent calls.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

from detectors.ai_content_detector import BaseAIDetector
from models.response import AIContentResult

logger = logging.getLogger("token_trust.local_ai_detector")

DEFAULT_CLASSIFIER_MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"
DEFAULT_SLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Same cap url_fetch applies to extracted page text before it reaches a detector.
_TEXT_CAP = 8000
# The reason SLM only needs a flavor of the writing style, not the whole document;
# a shorter excerpt keeps CPU latency tolerable for a ~1B-parameter model.
_REASON_EXCERPT_CAP = 2000
_MAX_REASON_TOKENS = 80
_MAX_REASON_CHARS = 400
# How long a failed model load stays cached before the next request may retry it.
_ERROR_RETRY_SECONDS = 300.0

# Classifier label -> verdict. Known human-side labels are matched exactly;
# AI-side labels by substring (covers "ChatGPT", "Fake", "AI-generated", ...).
# An unrecognized label yields checked=False rather than a guessed verdict.
_HUMAN_LABELS = {"human", "real", "human-written", "human_written"}
_AI_LABEL_HINTS = ("chatgpt", "gpt", "fake", "machine", "generated", "synthetic", "bot")

_INSTALL_HINT = (
    "Install the optional local-model deps (pip install -r requirements-slm.txt) "
    "or set AI_DETECTOR_BACKEND=anthropic or off."
)


def _import_transformers():
    """Import seam for ``transformers`` — tests monkeypatch this, no real import."""
    import transformers

    return transformers


def _short_name(model_id: str) -> str:
    """``Hello-SimpleAI/chatgpt-detector-roberta`` -> ``chatgpt-detector-roberta``."""
    return model_id.rsplit("/", 1)[-1].strip().lower() or model_id


def _label_means_ai(label: Any) -> Optional[bool]:
    """Map a classifier label to a verdict; None when the label is unrecognized."""
    name = str(label or "").strip().lower()
    if not name:
        return None
    if name in _HUMAN_LABELS:
        return False
    if any(hint in name for hint in _AI_LABEL_HINTS):
        return True
    return None


def _extract_generated_text(output: Any) -> Optional[str]:
    """Pull the generated string out of a text-generation pipeline result.

    Handles the shapes transformers has used across versions: a plain string,
    ``[{"generated_text": "<str>"}]``, and the chat form where ``generated_text``
    is the message list and the reply is the last message's ``content``.
    """
    try:
        item = output[0] if isinstance(output, (list, tuple)) else output
        text = item.get("generated_text") if isinstance(item, dict) else item
        if isinstance(text, list):
            last = text[-1] if text else None
            text = last.get("content") if isinstance(last, dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:  # never let a shape surprise escape the graceful path
        pass
    return None


class LocalAIDetector(BaseAIDetector):
    """Two-model local pipeline behind the standard detector API."""

    def __init__(
        self,
        classifier_model: Optional[str] = None,
        slm_model: Optional[str] = None,
    ) -> None:
        self.classifier_model = classifier_model or os.getenv(
            "AI_DETECTOR_CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL
        )
        self.slm_model = slm_model or os.getenv("AI_DETECTOR_SLM_MODEL", DEFAULT_SLM_MODEL)
        self._lock = threading.Lock()  # guards lazy loading of both pipelines
        self._infer_lock = threading.Lock()  # serializes inference on them
        # Lazily-created pipelines; a load failure is remembered in *_error (with
        # its timestamp, so it can be retried after _ERROR_RETRY_SECONDS) and
        # fails fast in the meantime instead of re-paying download timeouts.
        self._classifier = None
        self._classifier_error: Optional[str] = None
        self._classifier_error_at: Optional[float] = None
        self._slm = None
        self._slm_error: Optional[str] = None
        self._slm_error_at: Optional[float] = None

    @property
    def available(self) -> bool:
        """Cheap readiness probe: deps importable (or already loaded) — no weights."""
        if self._classifier is not None:
            return True
        if self._classifier_error is not None:
            return False
        try:
            import importlib.util

            return (
                importlib.util.find_spec("transformers") is not None
                and importlib.util.find_spec("torch") is not None
            )
        except Exception:
            return False

    # --- lazy model loading -------------------------------------------------- #
    def _load_classifier(self):
        transformers = _import_transformers()
        return transformers.pipeline("text-classification", model=self.classifier_model)

    def _load_slm(self):
        transformers = _import_transformers()
        return transformers.pipeline("text-generation", model=self.slm_model)

    def _retry_due(self, error_at: Optional[float]) -> bool:
        """A cached load failure may have been transient (Hub 429, DNS blip);
        allow a fresh attempt once the backoff has elapsed."""
        return error_at is not None and time.monotonic() - error_at >= _ERROR_RETRY_SECONDS

    def _get_classifier(self):
        if self._classifier is not None:
            return self._classifier
        with self._lock:
            if self._classifier is None and self._retry_due(self._classifier_error_at):
                self._classifier_error = None
                self._classifier_error_at = None
            if self._classifier is None and self._classifier_error is None:
                try:
                    logger.info(
                        "Loading local AI-text classifier %r (first use)...",
                        self.classifier_model,
                    )
                    self._classifier = self._load_classifier()
                except Exception as exc:
                    self._classifier_error = str(exc) or exc.__class__.__name__
                    self._classifier_error_at = time.monotonic()
                    logger.warning("Local AI-text classifier unavailable: %s", exc)
        return self._classifier

    def _get_slm(self):
        if self._slm is not None:
            return self._slm
        with self._lock:
            if self._slm is None and self._retry_due(self._slm_error_at):
                self._slm_error = None
                self._slm_error_at = None
            if self._slm is None and self._slm_error is None:
                try:
                    logger.info(
                        "Loading local reason SLM %r (first use)...", self.slm_model
                    )
                    self._slm = self._load_slm()
                except Exception as exc:
                    self._slm_error = str(exc) or exc.__class__.__name__
                    self._slm_error_at = time.monotonic()
                    logger.warning("Local reason SLM unavailable: %s", exc)
        return self._slm

    # --- detection ------------------------------------------------------------ #
    def detect(self, project_text: Optional[str]) -> AIContentResult:
        """Classify ``project_text`` locally; graceful, never-raising."""
        text = (project_text or "").strip()
        if not text:
            return AIContentResult(checked=False, reason="No project text was provided.")
        text = text[:_TEXT_CAP]

        classifier = self._get_classifier()
        if classifier is None:
            return AIContentResult(
                checked=False,
                reason=(
                    f"Local SLM pipeline unavailable ({self._classifier_error}). "
                    + _INSTALL_HINT
                ),
            )

        try:
            # truncation=True: RoBERTa-style models cap at 512 tokens; without it,
            # long marketing pages would raise instead of being truncated.
            # _infer_lock: the fast tokenizer's truncation state is not safe under
            # concurrent calls ("Already borrowed"), and CPU inference gains
            # nothing from running in parallel threads.
            with self._infer_lock:
                output = classifier(text, truncation=True)
            item = output[0] if isinstance(output, (list, tuple)) else output
            if isinstance(item, (list, tuple)):  # some versions nest: [[{...}]]
                item = item[0]
            label = item["label"]
            score = float(item["score"])
        except Exception as exc:
            logger.warning("Local classifier inference failed: %s", exc)
            return AIContentResult(
                checked=False,
                reason=f"Local SLM pipeline failed during classification ({exc}). "
                + _INSTALL_HINT,
            )

        is_ai = _label_means_ai(label)
        if is_ai is None:
            return AIContentResult(
                checked=False,
                reason=(
                    f"Local classifier {self.classifier_model!r} returned an "
                    f"unrecognized label {label!r}; cannot map it to a verdict."
                ),
            )
        confidence = max(0.0, min(1.0, score))

        reason, slm_used = self._build_reason(text, is_ai, confidence)
        source = f"local:{_short_name(self.classifier_model)}"
        if slm_used:
            source += f"+{_short_name(self.slm_model)}"
        return AIContentResult(
            checked=True,
            is_ai_generated=is_ai,
            confidence=confidence,
            reason=reason,
            source=source,
        )

    def _build_reason(self, text: str, is_ai: bool, confidence: float) -> tuple[str, bool]:
        """One–two sentence justification; (reason, whether the SLM produced it)."""
        verdict = "AI-generated" if is_ai else "human-written"
        slm = self._get_slm()
        if slm is not None:
            prompt = (
                "A specialized classifier judged the following crypto-project text "
                f"to be {verdict} (confidence {confidence:.2f}). In one or two short "
                "sentences, explain what about the writing style supports that "
                "verdict. Reply with the explanation only.\n\n"
                f"<text>\n{text[:_REASON_EXCERPT_CAP]}\n</text>"
            )
            try:
                with self._infer_lock:
                    output = slm(
                        [{"role": "user", "content": prompt}],
                        max_new_tokens=_MAX_REASON_TOKENS,
                        do_sample=False,  # greedy -> deterministic reason
                        return_full_text=False,
                    )
                reply = _extract_generated_text(output)
                if reply:
                    return reply[:_MAX_REASON_CHARS], True
                logger.warning("Local reason SLM returned no usable text.")
            except Exception as exc:
                logger.warning("Local reason generation failed: %s", exc)
        return (
            f"Local classifier judged the text {verdict} "
            f"(confidence {confidence:.2f}); no SLM explanation available.",
            False,
        )
