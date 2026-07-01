"""Isolation Forest anomaly detector — the ML centerpiece.

The model is trained on a seed set of *healthy* tokens (``data/training_tokens.json``),
so it learns what "normal" looks like. A new token that sits far from that manifold
gets a high anomaly score. The raw sklearn score is calibrated onto a 0-100 scale
using the training distribution: a median-normal token maps to ~0, and a token at
or beyond the 95th percentile of normal maps to ~100.

The fitted (scaler + forest + calibration) bundle is persisted with joblib and
reloaded on subsequent runs. If no persisted model exists it is trained lazily on
first use, so the service works out of the box with no separate training step.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except Exception:  # pragma: no cover - joblib ships with scikit-learn
    joblib = None  # type: ignore

from features.feature_extractor import FEATURE_ORDER, FeatureExtractor

logger = logging.getLogger("token_trust.anomaly")

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(_HERE, "data", "anomaly_model.joblib")
DEFAULT_TRAINING_PATH = os.path.join(_HERE, "data", "training_tokens.json")


class AnomalyModel:
    """Wraps a StandardScaler + IsolationForest with 0-100 calibration."""

    def __init__(
        self,
        scaler: Optional[StandardScaler] = None,
        forest: Optional[IsolationForest] = None,
        calib_lo: float = 0.0,
        calib_hi: float = 1.0,
    ) -> None:
        self.scaler = scaler
        self.forest = forest
        self.calib_lo = calib_lo
        self.calib_hi = calib_hi
        self.feature_order = list(FEATURE_ORDER)

    @property
    def is_fitted(self) -> bool:
        return self.scaler is not None and self.forest is not None

    # -- training --------------------------------------------------------------

    def train(self, vectors: list[list[float]]) -> "AnomalyModel":
        """Fit the scaler + forest on healthy-token vectors and calibrate."""
        if not vectors:
            raise ValueError("Cannot train AnomalyModel on an empty dataset.")
        matrix = np.asarray(vectors, dtype=float)
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(matrix)
        # contamination: the training set is a curated set of *healthy* tokens, so
        # we expect only a few genuine oddities (e.g. legit tokens with very high
        # single-holder concentration). A small fixed value keeps this assumption
        # explicit and conservative rather than letting 'auto' infer a larger
        # outlier fraction on the broader set. NB: our 0-100 output is calibrated
        # from `score_samples` percentiles below, which sklearn computes
        # independently of `contamination`, so this mainly documents intent and
        # only affects the (unused) predict()/decision_function() threshold.
        self.forest = IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=42,
            n_jobs=-1,
        )
        self.forest.fit(scaled)

        # Calibrate: raw = -score_samples (higher => more anomalous).
        raw = -self.forest.score_samples(scaled)
        lo = float(np.percentile(raw, 50))
        hi = float(np.percentile(raw, 95))
        if hi <= lo:  # degenerate spread -> fall back to a std-based band
            std = float(np.std(raw)) or 1.0
            lo = float(np.median(raw))
            hi = lo + 2.0 * std
        self.calib_lo, self.calib_hi = lo, hi
        return self

    # -- scoring ---------------------------------------------------------------

    def score(self, vector: list[float]) -> float:
        """Return a normalized 0-100 anomaly contribution for one feature vector."""
        if not self.is_fitted:
            raise RuntimeError("AnomalyModel is not fitted.")
        x = np.asarray(vector, dtype=float).reshape(1, -1)
        scaled = self.scaler.transform(x)
        raw = float(-self.forest.score_samples(scaled)[0])
        span = self.calib_hi - self.calib_lo
        if span <= 0:
            return 0.0
        normalized = 100.0 * (raw - self.calib_lo) / span
        return float(max(0.0, min(100.0, normalized)))

    # -- persistence -----------------------------------------------------------

    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        if joblib is None:
            logger.warning("joblib unavailable; skipping model save.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "forest": self.forest,
                "calib_lo": self.calib_lo,
                "calib_hi": self.calib_hi,
                "feature_order": self.feature_order,
            },
            path,
        )
        logger.info("Saved anomaly model to %s", path)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> Optional["AnomalyModel"]:
        if joblib is None or not os.path.exists(path):
            return None
        try:
            bundle = joblib.load(path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load anomaly model: %s", exc)
            return None
        # Guard against a schema change: retrain if the feature order differs.
        if bundle.get("feature_order") != list(FEATURE_ORDER):
            logger.info("Persisted model feature order is stale; will retrain.")
            return None
        model = cls(
            scaler=bundle["scaler"],
            forest=bundle["forest"],
            calib_lo=bundle["calib_lo"],
            calib_hi=bundle["calib_hi"],
        )
        return model

    @classmethod
    def load_or_train(
        cls,
        model_path: str = DEFAULT_MODEL_PATH,
        training_path: str = DEFAULT_TRAINING_PATH,
        persist: bool = True,
    ) -> "AnomalyModel":
        """Load a persisted model, or train one from the seed dataset and cache it."""
        model = cls.load(model_path)
        if model is not None and model.is_fitted:
            logger.info("Loaded anomaly model from %s", model_path)
            return model

        vectors = cls._load_training_vectors(training_path)
        model = cls().train(vectors)
        if persist:
            try:
                model.save(model_path)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not persist model: %s", exc)
        logger.info("Trained anomaly model on %d seed tokens.", len(vectors))
        return model

    @staticmethod
    def _load_training_vectors(training_path: str) -> list[list[float]]:
        with open(training_path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
        extractor = FeatureExtractor()
        vectors: list[list[float]] = []
        for row in rows:
            vectors.append(extractor.extract(row).vector)
        return vectors


# Process-wide singleton so we train/load once.
_MODEL: Optional[AnomalyModel] = None


def get_anomaly_model() -> AnomalyModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = AnomalyModel.load_or_train()
    return _MODEL
