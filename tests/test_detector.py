"""Tests for detectors/ai_content_detector.py — no real Anthropic calls."""

from __future__ import annotations

import detectors.ai_content_detector as det_mod
from detectors.ai_content_detector import AIContentDetector, _extract_json
from models.response import AIContentResult


def _boom(*_a, **_k):
    raise AssertionError("this callable should not have been invoked")


# --- _extract_json ---------------------------------------------------------- #
def test_extract_json_plain():
    data = _extract_json('{"is_ai_generated": true, "confidence": 0.8, "reason": "x"}')
    assert data == {"is_ai_generated": True, "confidence": 0.8, "reason": "x"}


def test_extract_json_fenced():
    text = '```json\n{"is_ai_generated": false, "confidence": 0.2, "reason": "y"}\n```'
    assert _extract_json(text)["is_ai_generated"] is False


def test_extract_json_prose_embedded():
    text = 'Sure! Here is my verdict:\n{"is_ai_generated": true, "confidence": 1}\nHope that helps.'
    assert _extract_json(text)["confidence"] == 1


def test_extract_json_non_json_returns_none():
    assert _extract_json("not json at all") is None
    assert _extract_json("") is None


# --- AIContentDetector ------------------------------------------------------ #
def test_detect_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    det = AIContentDetector(api_key=None)
    assert det.available is False

    res = det.detect("Some marketing copy about our revolutionary token.")
    assert res.checked is False
    assert res.is_ai_generated is None
    assert res.reason and "ANTHROPIC_API_KEY" in res.reason


def test_detect_without_text():
    det = AIContentDetector(api_key="sk-test")  # available, but nothing to check
    res = det.detect(None)
    assert res.checked is False
    res2 = det.detect("   ")
    assert res2.checked is False


def test_detect_parses_mocked_reply(monkeypatch):
    class _Block:
        type = "text"
        text = '{"is_ai_generated": true, "confidence": 0.9, "reason": "generic buzzwords"}'

    class _Resp:
        content = [_Block()]

    class _Messages:
        def create(self, **kwargs):
            return _Resp()

    class _Client:
        def __init__(self):
            self.messages = _Messages()

    det = AIContentDetector(api_key="sk-test")
    monkeypatch.setattr(det, "_get_client", lambda: _Client())

    res = det.detect("We are the most decentralized synergistic web3 protocol ever.")
    assert res.checked is True
    assert res.is_ai_generated is True
    assert res.confidence == 0.9
    assert res.reason == "generic buzzwords"


# --- AIContentDetector.analyze (source precedence: text > url) --------------- #
def test_analyze_prefers_text_over_url(monkeypatch):
    det = AIContentDetector(api_key="sk-test")
    seen = {}
    monkeypatch.setattr(
        det, "detect",
        lambda text: (seen.__setitem__("text", text)
                      or AIContentResult(checked=True, is_ai_generated=False, reason="ok")),
    )
    # The URL path must NOT be taken when project_text is present.
    monkeypatch.setattr(det_mod, "fetch_project_text", _boom)

    res = det.analyze(project_text="a human wrote this", project_url="http://example.com/")
    assert seen["text"] == "a human wrote this"
    assert res.source == "provided_text"
    assert res.checked is True


def test_analyze_uses_url_when_no_text(monkeypatch):
    det = AIContentDetector(api_key="sk-test")
    seen = {}
    monkeypatch.setattr(
        det, "detect",
        lambda text: (seen.__setitem__("text", text)
                      or AIContentResult(checked=True, is_ai_generated=True, confidence=0.7, reason="ai")),
    )
    monkeypatch.setattr(det_mod, "fetch_project_text", lambda url: "extracted page text")

    res = det.analyze(project_url="http://example.com/whitepaper")
    assert seen["text"] == "extracted page text"  # the detector received the fetched text
    assert res.source == "fetched_url"
    assert res.checked is True
    assert res.is_ai_generated is True


def test_analyze_url_blocked_is_unchecked(monkeypatch):
    det = AIContentDetector(api_key="sk-test")
    monkeypatch.setattr(det_mod, "fetch_project_text", lambda url: None)  # blocked/failed/empty
    monkeypatch.setattr(det, "detect", _boom)  # detection must not run when there's no text

    res = det.analyze(project_url="http://169.254.169.254/")
    assert res.checked is False
    assert res.source == "fetched_url"
    assert res.reason and ("blocked" in res.reason.lower() or "could not" in res.reason.lower())


def test_analyze_neither_text_nor_url():
    det = AIContentDetector(api_key="sk-test")
    res = det.analyze()
    assert res.checked is False
    assert res.source is None
