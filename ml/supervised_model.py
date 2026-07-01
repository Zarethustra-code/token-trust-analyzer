"""Optional supervised classifier (XGBoost) — bonus layer.

This is a scaffold. Until a *labeled* scam/legit dataset is provided and a model
is trained + persisted, it is a graceful no-op: ``predict_proba`` returns ``None``
and the scorer treats its contribution as 0. Nothing here is required for the
Phase 1 pipeline to run end to end.

When a trained model is present at ``DEFAULT_MODEL_PATH`` (an XGBoost booster
saved via joblib alongside the fitted scaler), ``predict_proba`` returns the
model's P(scam) in [0, 1].
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

from features.feature_extractor import FEATURE_ORDER

logger = logging.getLogger("token_trust.supervised")

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(_HERE, "data", "supervised_model.joblib")


class SupervisedModel:
    """Thin wrapper around an optional XGBoost classifier."""

    def __init__(self, booster=None, scaler=None) -> None:
        self.booster = booster
        self.scaler = scaler
        self.feature_order = list(FEATURE_ORDER)

    @property
    def is_available(self) -> bool:
        return self.booster is not None

    def predict_proba(self, vector: list[float]) -> Optional[float]:
        """Return P(scam) in [0, 1], or ``None`` if no model is loaded."""
        if not self.is_available:
            return None
        try:
            x = np.asarray(vector, dtype=float).reshape(1, -1)
            if self.scaler is not None:
                x = self.scaler.transform(x)
            proba = self.booster.predict_proba(x)[0][1]
            return float(max(0.0, min(1.0, proba)))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Supervised predict failed: %s", exc)
            return None

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "SupervisedModel":
        """Load a persisted model if present; otherwise return a no-op instance."""
        if joblib is None or not os.path.exists(path):
            return cls()  # no-op
        try:
            bundle = joblib.load(path)
            return cls(booster=bundle.get("booster"), scaler=bundle.get("scaler"))
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load supervised model: %s", exc)
            return cls()


_MODEL: Optional[SupervisedModel] = None


def get_supervised_model() -> SupervisedModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = SupervisedModel.load()
    return _MODEL
