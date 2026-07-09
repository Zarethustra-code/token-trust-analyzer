"""SLM-written risk narrative — the analyst-style summary in a Trust Report.

The deterministic pipeline (rules + Isolation Forest) already produces the
``trust_score``, ``risk_level``, ``flags``, ``score_breakdown`` and the templated
``explanation``. This module adds *only prose*: a short (2–4 sentence) paragraph
that reads like a human analyst connecting those signals ("although liquidity is
locked and ownership renounced, the high top-10 concentration combined with the
contract's young age suggests …").

Design rules — the SLM **phrases, it never decides**:

* The narrative is built from a *facts block* extracted from the finished report
  (:func:`build_narrative_facts`). The model is told to use ONLY those facts and
  not to invent numbers, signals, or advice — grounded generation to curb
  hallucination.
* Decoding is deterministic-ish (greedy, ``max_new_tokens`` ≤ 200) via the ONE
  shared local SLM (see :mod:`detectors.local_ai_detector`) — never a second copy
  of the model.
* Output passes sanity bounds (:func:`_sanitize`): whitespace is collapsed and an
  empty / too-short / too-long result is dropped to ``None``. On any failure the
  deterministic ``explanation`` remains the source of truth.

Nothing here mutates the score or the explanation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from models.response import TrustReport

logger = logging.getLogger("token_trust.narrative")

# Deterministic-ish decoding: a couple of sentences fit well under 200 tokens.
_MAX_NARRATIVE_TOKENS = 200

# Sanity bounds on the generated prose (after whitespace collapse). Outside these
# the narrative is dropped to None and the templated explanation stands alone.
_MIN_CHARS = 30
_MAX_CHARS = 900

# A handful of the most narrative-worthy raw metrics: (report field, human label,
# kind). ``kind`` controls rendering — "bool" -> yes/no, "pct" -> "…%", else the
# value as-is. A ``None`` metric is surfaced as "unknown" (never invented).
_NOTABLE_METRICS: list[tuple[str, str, str]] = [
    ("top_holder_pct", "top holder share", "pct"),
    ("top10_holder_pct", "top-10 holder share", "pct"),
    ("holder_count", "holder count", "num"),
    ("liquidity_locked", "liquidity locked", "bool"),
    ("ownership_renounced", "ownership renounced", "bool"),
    ("contract_age_days", "contract age (days)", "num"),
    ("is_honeypot", "honeypot", "bool"),
    ("has_mint", "mintable", "bool"),
    ("buy_tax", "buy tax %", "num"),
    ("sell_tax", "sell tax %", "num"),
]


def _fmt_metric(value: Any, kind: str) -> str:
    """Render a raw metric for the facts block; None -> 'unknown' (never guessed)."""
    if value is None:
        return "unknown"
    if kind == "bool":
        return "yes" if bool(value) else "no"
    if kind == "pct":
        try:
            return f"{float(value):.1f}%"
        except (TypeError, ValueError):
            return "unknown"
    if kind == "num":
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return "unknown"
        # Integers render without a trailing ".0"; keep one decimal otherwise.
        return f"{int(fv)}" if fv.is_integer() else f"{fv:.1f}"
    return str(value)


def build_narrative_facts(report: TrustReport) -> dict:
    """Extract the grounded facts block the prompt is built from.

    Pulls *only* values the deterministic pipeline already produced — the score,
    risk level, confidence/completeness, every fired rule (flag + points), the
    anomaly signal, and a handful of notable metrics (nulls -> "unknown"). The SLM
    sees nothing else, so it has nothing to hallucinate from.
    """
    bd = report.score_breakdown
    fired = [
        {"flag": p.flag, "points": round(float(p.points), 2)}
        for p in bd.rule_penalties
    ]
    metrics = {
        label: _fmt_metric(getattr(report.metrics, field, None), kind)
        for field, label, kind in _NOTABLE_METRICS
    }
    return {
        "trust_score": int(report.trust_score),
        "risk_level": report.risk_level.value,
        "confidence": bd.confidence,
        "data_completeness_pct": round(float(bd.data_completeness) * 100),
        "anomaly_score": round(float(bd.anomaly_score), 1),
        "anomaly_contribution": round(float(bd.anomaly_contribution), 1),
        "fired_rules": fired,
        "metrics": metrics,
    }


def build_narrative_prompt(facts: dict) -> str:
    """Turn the facts block into a grounded, instruction-constrained SLM prompt.

    The prompt embeds the ``trust_score``, ``risk_level`` and every fired rule
    flag verbatim, and forbids inventing anything not listed — this is what the
    grounding tests assert on.
    """
    lines: list[str] = []
    lines.append(
        f"- Risk score: {facts['trust_score']} out of 100 "
        "(higher = riskier)."
    )
    lines.append(f"- Risk level: {facts['risk_level']}.")
    lines.append(
        f"- Confidence: {facts['confidence']} "
        f"(data completeness {facts['data_completeness_pct']}%)."
    )
    lines.append(
        f"- Isolation Forest anomaly score: {facts['anomaly_score']} out of 100, "
        f"contributing {facts['anomaly_contribution']} points."
    )

    fired = facts.get("fired_rules") or []
    if fired:
        lines.append("- Rules that fired:")
        for rule in fired:
            lines.append(f"    - {rule['flag']} (+{rule['points']:g} points)")
    else:
        lines.append("- Rules that fired: none.")

    lines.append("- Notable metrics:")
    for label, value in facts.get("metrics", {}).items():
        lines.append(f"    - {label}: {value}")

    facts_block = "\n".join(lines)

    return (
        "You are a blockchain security analyst writing a short risk summary for an "
        "ERC-20 token Trust Report.\n\n"
        "Use ONLY the facts listed below. Do NOT invent numbers, signals, or "
        "recommendations, and do NOT contradict the stated risk level. Write 2 to 4 "
        "sentences in a neutral, professional analyst tone that connect the signals "
        "the way a human analyst would (e.g. weighing mitigating factors against "
        "risk drivers). Output the summary text only — no preamble, no bullet "
        "points, no headings.\n\n"
        f"FACTS:\n{facts_block}\n\n"
        "SUMMARY:"
    )


def _sanitize(text: Optional[str]) -> Optional[str]:
    """Collapse whitespace and enforce the length bounds; None if out of range."""
    if not text:
        logger.info("Risk narrative empty; using deterministic explanation only.")
        return None
    collapsed = " ".join(str(text).split())
    length = len(collapsed)
    if length < _MIN_CHARS:
        logger.info("Risk narrative too short (%d chars); dropped.", length)
        return None
    if length > _MAX_CHARS:
        logger.info("Risk narrative too long (%d chars); dropped.", length)
        return None
    return collapsed


def _run_slm(prompt: str) -> Optional[str]:
    """Run ``prompt`` through the ONE shared local SLM; None if unavailable.

    Isolated as its own function so tests can mock generation without loading any
    model, and so the "shared instance, never a second copy" rule lives in exactly
    one place.
    """
    try:
        from detectors.local_ai_detector import get_shared_local_detector

        detector = get_shared_local_detector()
    except Exception as exc:  # import/deps unavailable while requested
        logger.warning("Risk-narrative SLM unavailable: %s", exc)
        return None
    return detector.generate_text(prompt, max_new_tokens=_MAX_NARRATIVE_TOKENS)


def generate_narrative(report: TrustReport) -> Optional[str]:
    """Produce the analyst-style narrative for ``report``; None if unavailable.

    Never raises: any failure (prompt build, SLM load/inference, sanity bounds)
    degrades to ``None`` so the caller keeps the deterministic report intact.
    """
    try:
        facts = build_narrative_facts(report)
        prompt = build_narrative_prompt(facts)
    except Exception as exc:  # pragma: no cover - defensive; facts come from a valid report
        logger.warning("Failed to build risk-narrative prompt: %s", exc)
        return None

    raw = _run_slm(prompt)
    return _sanitize(raw)
