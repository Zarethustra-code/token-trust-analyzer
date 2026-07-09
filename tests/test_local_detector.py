"""Tests for the local SLM detector backend + AI_DETECTOR_BACKEND switching.

Fully OFFLINE: the ``transformers`` pipelines are monkeypatched — no import of
transformers/torch, no model downloads, ever.
"""

from __future__ import annotations

import pytest

import detectors.ai_content_detector as det_mod
import detectors.local_ai_detector as local_mod
from detectors.ai_content_detector import AIContentDetector, OffDetector, get_detector
from detectors.local_ai_detector import (
    LocalAIDetector,
    _extract_generated_text,
    _label_means_ai,
)


def _boom(*_a, **_k):
    raise AssertionError("this callable should not have been invoked")


@pytest.fixture(autouse=True)
def _fresh_detector_cache():
    """Isolate the per-backend detector cache so env flips in one test don't leak.

    Also drops the process-wide shared LocalAIDetector: ``get_detector()`` for the
    ``local`` backend now returns that shared instance, so a load/error cached on
    it by one test must not leak into the next.
    """
    det_mod._DETECTORS.clear()
    local_mod.reset_shared_local_detector()
    yield
    det_mod._DETECTORS.clear()
    local_mod.reset_shared_local_detector()


def _fake_classifier(label: str, score: float, seen: dict | None = None):
    def classify(text, **kwargs):
        if seen is not None:
            seen["text"] = text
            seen["kwargs"] = kwargs
        return [{"label": label, "score": score}]

    return classify


def _fake_slm(reply: str = "Generic buzzwords and uniform sentence rhythm throughout."):
    def generate(messages, **kwargs):
        return [{"generated_text": reply}]

    return generate


def _local_det(**kwargs) -> LocalAIDetector:
    # Explicit model names so results don't depend on the host's env overrides.
    return LocalAIDetector(
        classifier_model=kwargs.pop("classifier_model", "acme/clf-roberta"),
        slm_model=kwargs.pop("slm_model", "acme/slm-qwen"),
    )


# --- label mapping ----------------------------------------------------------- #
def test_label_means_ai_known_labels():
    assert _label_means_ai("ChatGPT") is True       # Hello-SimpleAI/chatgpt-detector-roberta
    assert _label_means_ai("Fake") is True          # roberta-base-openai-detector
    assert _label_means_ai("AI-generated") is True
    assert _label_means_ai("Human") is False
    assert _label_means_ai("Real") is False


def test_label_means_ai_unknown_is_none():
    assert _label_means_ai("LABEL_1") is None
    assert _label_means_ai("") is None
    assert _label_means_ai(None) is None


# --- SLM output shape handling ----------------------------------------------- #
def test_extract_generated_text_shapes():
    assert _extract_generated_text([{"generated_text": " plain reply "}]) == "plain reply"
    chat = [{"generated_text": [{"role": "user", "content": "q"},
                                {"role": "assistant", "content": "chat reply"}]}]
    assert _extract_generated_text(chat) == "chat reply"
    assert _extract_generated_text("bare string") == "bare string"
    assert _extract_generated_text([]) is None
    assert _extract_generated_text([{"generated_text": ""}]) is None
    assert _extract_generated_text(None) is None


# --- classifier -> verdict/confidence mapping --------------------------------- #
def test_detect_maps_ai_label_and_score():
    det = _local_det()
    det._classifier = _fake_classifier("ChatGPT", 0.93)
    det._slm = _fake_slm()

    res = det.detect("We are the most synergistic decentralized web3 protocol ever.")
    assert res.checked is True
    assert res.is_ai_generated is True
    assert res.confidence == pytest.approx(0.93)
    assert res.reason == "Generic buzzwords and uniform sentence rhythm throughout."
    assert res.source == "local:clf-roberta+slm-qwen"


def test_detect_maps_human_label():
    det = _local_det()
    det._classifier = _fake_classifier("Human", 0.88)
    det._slm = _fake_slm("Specific numbers and an irregular, personal voice.")

    res = det.detect("Q3 audit done by Trail of Bits; 4 criticals fixed, 1 wontfix.")
    assert res.checked is True
    assert res.is_ai_generated is False
    assert res.confidence == pytest.approx(0.88)
    assert res.reason.startswith("Specific numbers")


def test_detect_clamps_confidence():
    det = _local_det()
    det._classifier = _fake_classifier("Fake", 1.7)  # out-of-range score
    det._slm = _fake_slm()
    res = det.detect("some text")
    assert res.confidence == 1.0


def test_detect_handles_nested_classifier_output():
    det = _local_det()
    det._classifier = lambda text, **kw: [[{"label": "ChatGPT", "score": 0.6}]]
    det._slm = _fake_slm()
    res = det.detect("some text")
    assert res.checked is True
    assert res.is_ai_generated is True


def test_detect_truncates_input_and_requests_truncation():
    seen: dict = {}
    det = _local_det()
    det._classifier = _fake_classifier("Human", 0.5, seen=seen)
    det._slm = _fake_slm()

    det.detect("x" * 20_000)
    assert len(seen["text"]) == 8000  # the existing ~8000-char cap
    assert seen["kwargs"].get("truncation") is True  # tokenizer-level 512-token cap


def test_detect_unrecognized_label_is_unchecked():
    det = _local_det()
    det._classifier = _fake_classifier("LABEL_1", 0.9)
    # Recording fake, not a raising one: detect()'s except would swallow a raise.
    slm_calls = []
    det._slm = lambda *a, **k: slm_calls.append(1) or [{"generated_text": "x"}]
    res = det.detect("some text")
    assert res.checked is False
    assert res.is_ai_generated is None
    assert "unrecognized label" in res.reason
    assert not slm_calls  # no verdict -> the SLM must not be consulted


def test_detect_without_text():
    det = _local_det()
    clf_calls = []
    det._classifier = lambda *a, **k: clf_calls.append(1) or [{"label": "Human", "score": 0.5}]
    assert det.detect(None).checked is False
    assert det.detect("   ").checked is False
    assert not clf_calls  # no text -> the classifier must not run


# --- reason: populated, short, deterministic fallback -------------------------- #
def test_reason_is_capped_short():
    det = _local_det()
    det._classifier = _fake_classifier("ChatGPT", 0.9)
    det._slm = _fake_slm("very long explanation " * 100)
    res = det.detect("some text")
    assert res.reason
    assert len(res.reason) <= 400


def test_slm_generation_is_greedy_and_capped():
    captured: dict = {}

    def generate(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return [{"generated_text": "ok"}]

    det = _local_det()
    det._classifier = _fake_classifier("ChatGPT", 0.9)
    det._slm = generate
    det.detect("some text")
    assert captured["kwargs"]["do_sample"] is False  # deterministic reason
    assert captured["kwargs"]["max_new_tokens"] <= 80
    assert captured["messages"][0]["role"] == "user"
    assert "AI-generated" in captured["messages"][0]["content"]


def test_slm_failure_keeps_classifier_verdict():
    det = _local_det()
    det._classifier = _fake_classifier("ChatGPT", 0.91)
    det._slm = _boom  # reason generation blows up -> templated fallback

    res = det.detect("some text")
    assert res.checked is True  # the accuracy-critical verdict survives
    assert res.is_ai_generated is True
    assert res.confidence == pytest.approx(0.91)
    assert "AI-generated" in res.reason and "0.91" in res.reason
    assert res.source == "local:clf-roberta"  # SLM visibly absent from source


def test_slm_unloadable_keeps_classifier_verdict(monkeypatch):
    det = _local_det()
    det._classifier = _fake_classifier("Human", 0.75)
    monkeypatch.setattr(det, "_load_slm", _boom)  # load fails -> _get_slm caches None

    res = det.detect("some text")
    assert res.checked is True
    assert res.is_ai_generated is False
    assert "no SLM explanation available" in res.reason


# --- lazy loading: wiring, caching, and the never-load-eagerly guarantee ------- #
def test_lazy_load_wires_models_and_caches(monkeypatch):
    """The real load path must build the RIGHT pipelines and build them ONCE."""
    calls = []

    class _FakeTransformers:
        @staticmethod
        def pipeline(task, model=None, **kwargs):
            calls.append((task, model))
            if task == "text-classification":
                return _fake_classifier("ChatGPT", 0.9)
            return _fake_slm()

    monkeypatch.setattr(local_mod, "_import_transformers", lambda: _FakeTransformers)
    det = _local_det()

    res = det.detect("some text")
    assert res.checked is True
    assert res.is_ai_generated is True
    assert ("text-classification", "acme/clf-roberta") in calls  # classifier wiring
    assert ("text-generation", "acme/slm-qwen") in calls  # SLM wiring
    assert len(calls) == 2

    det.detect("more text")
    assert len(calls) == 2  # loaded pipelines are cached, not rebuilt per call


def test_nothing_loads_eagerly(client, monkeypatch):
    """Spec: lazy-load on first detect() only — never at construction, backend
    selection, or /health. A recording fake (not a raising one) is required:
    the loaders swallow exceptions into *_error, so a raise can't fail a test."""
    import_calls = []

    def recording_import():
        import_calls.append(1)
        raise ImportError("transformers deliberately unavailable")

    monkeypatch.setattr(local_mod, "_import_transformers", recording_import)
    monkeypatch.delenv("AI_DETECTOR_BACKEND", raising=False)

    local_mod.LocalAIDetector()  # construction
    det = get_detector()  # default-backend selection
    assert isinstance(det, local_mod.LocalAIDetector)
    assert client.get("/health").status_code == 200  # probes .available only
    assert import_calls == []

    det.detect("some text")  # first real detection is what triggers the load
    assert import_calls == [1]


def test_load_failure_retried_after_backoff(monkeypatch):
    """A transient load failure must not disable detection until restart."""

    class _FakeTime:
        now = 1000.0

        @classmethod
        def monotonic(cls):
            return cls.now

    monkeypatch.setattr(local_mod, "time", _FakeTime)
    det = _local_det()
    det._slm = _fake_slm()

    attempts = []

    def load_flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("HF Hub briefly unreachable")
        return _fake_classifier("ChatGPT", 0.9)

    monkeypatch.setattr(det, "_load_classifier", load_flaky)

    assert det.detect("some text").checked is False  # first attempt fails
    assert det.detect("some text").checked is False  # within backoff: no retry
    assert len(attempts) == 1

    _FakeTime.now += 301.0  # past _ERROR_RETRY_SECONDS
    res = det.detect("some text")
    assert len(attempts) == 2  # retried once the backoff elapsed
    assert res.checked is True
    assert res.is_ai_generated is True


# --- graceful degradation: deps missing / load or inference failure ------------ #
def test_transformers_missing_is_unchecked(monkeypatch):
    monkeypatch.setattr(
        local_mod,
        "_import_transformers",
        lambda: (_ for _ in ()).throw(ImportError("No module named 'transformers'")),
    )
    det = _local_det()
    res = det.detect("some text")
    assert res.checked is False
    assert res.is_ai_generated is None
    assert "unavailable" in res.reason
    assert "requirements-slm.txt" in res.reason
    # The failure is cached: a second call must not retry the import.
    monkeypatch.setattr(local_mod, "_import_transformers", _boom)
    res2 = det.detect("more text")
    assert res2.checked is False


def test_classifier_inference_failure_is_unchecked():
    det = _local_det()
    det._classifier = _boom
    det._slm = _fake_slm()
    res = det.detect("some text")
    assert res.checked is False
    assert "failed during classification" in res.reason


def test_available_short_circuits(monkeypatch):
    # With a loaded classifier the probe must short-circuit to True; with a
    # cached load error, to False — no import probing in either case.
    det = _local_det()
    det._classifier = _fake_classifier("Human", 0.5)
    assert det.available is True
    det2 = _local_det()
    det2._classifier_error = "boom"
    assert det2.available is False


def test_available_probes_importability(monkeypatch):
    # A fresh detector's availability is exactly "are the deps importable" —
    # /health's ai_detector_configured field is driven by this branch.
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    assert _local_det().available is False

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    assert _local_det().available is True


# --- analyze(): source precedence + backend source preserved ------------------- #
def test_analyze_preserves_local_source(monkeypatch):
    det = _local_det()
    det._classifier = _fake_classifier("ChatGPT", 0.8)
    det._slm = _fake_slm()
    monkeypatch.setattr(det_mod, "fetch_project_text", _boom)  # text wins over url

    res = det.analyze(project_text="marketing copy", project_url="http://example.com/")
    assert res.checked is True
    assert res.source == "local:clf-roberta+slm-qwen"  # not overwritten by origin


def test_analyze_url_flow_with_local_backend(monkeypatch):
    det = _local_det()
    det._classifier = _fake_classifier("Human", 0.7)
    det._slm = _fake_slm()
    monkeypatch.setattr(det_mod, "fetch_project_text", lambda url: "extracted page text")

    res = det.analyze(project_url="http://example.com/whitepaper")
    assert res.checked is True
    assert res.source == "local:clf-roberta+slm-qwen"


def test_analyze_blocked_url_still_reports_fetched_url(monkeypatch):
    det = _local_det()
    clf_calls = []
    det._classifier = lambda *a, **k: clf_calls.append(1) or [{"label": "Human", "score": 0.5}]
    monkeypatch.setattr(det_mod, "fetch_project_text", lambda url: None)

    res = det.analyze(project_url="http://169.254.169.254/")
    assert res.checked is False
    assert res.source == "fetched_url"
    assert not clf_calls  # blocked fetch -> detection must not run


# --- AI_DETECTOR_BACKEND switching --------------------------------------------- #
def test_get_detector_default_is_local(monkeypatch):
    monkeypatch.delenv("AI_DETECTOR_BACKEND", raising=False)
    assert isinstance(get_detector(), LocalAIDetector)


def test_get_detector_anthropic(monkeypatch):
    monkeypatch.setenv("AI_DETECTOR_BACKEND", "anthropic")
    assert isinstance(get_detector(), AIContentDetector)


def test_get_detector_off(monkeypatch):
    monkeypatch.setenv("AI_DETECTOR_BACKEND", "off")
    det = get_detector()
    assert isinstance(det, OffDetector)
    assert det.available is False

    res = det.detect("any text at all")
    assert res.checked is False
    assert "off" in res.reason

    # Off must not even fetch the URL.
    monkeypatch.setattr(det_mod, "fetch_project_text", _boom)
    res2 = det.analyze(project_url="http://example.com/")
    assert res2.checked is False


def test_get_detector_unknown_value_falls_back_to_local(monkeypatch):
    monkeypatch.setenv("AI_DETECTOR_BACKEND", "quantum")
    assert isinstance(get_detector(), LocalAIDetector)


def test_get_detector_is_cached_per_backend(monkeypatch):
    monkeypatch.setenv("AI_DETECTOR_BACKEND", "off")
    first = get_detector()
    assert get_detector() is first
    monkeypatch.setenv("AI_DETECTOR_BACKEND", "anthropic")
    second = get_detector()
    assert isinstance(second, AIContentDetector)
    monkeypatch.setenv("AI_DETECTOR_BACKEND", "off")
    assert get_detector() is first


def test_env_model_overrides(monkeypatch):
    monkeypatch.setenv("AI_DETECTOR_CLASSIFIER_MODEL", "org/custom-clf")
    monkeypatch.setenv("AI_DETECTOR_SLM_MODEL", "org/custom-slm")
    det = LocalAIDetector()
    assert det.classifier_model == "org/custom-clf"
    assert det.slm_model == "org/custom-slm"


# --- API-level: /detect-ai honors the backend switch --------------------------- #
def test_detect_ai_endpoint_backend_off(client, monkeypatch):
    monkeypatch.setenv("AI_DETECTOR_BACKEND", "off")
    resp = client.post("/detect-ai", json={"project_text": "synergistic web3 protocol"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["checked"] is False
    assert body["is_ai_generated"] is None
