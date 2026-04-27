"""
ML Classifier Module
=====================
Fuses feature scores from all analysis modules and produces an overall
authenticity classification:
  • Feature vector construction
  • Weighted ensemble scoring (heuristic mode)
  • XGBoost classification (when a trained model is available)
  • Risk-level assessment (Genuine / Suspicious / Likely Forged)
"""

import os
import json
import numpy as np
from typing import Dict, Optional, Tuple


class MLClassifier:
    """
    Combine per-module scores into an overall authenticity verdict.

    Operates in two modes:
        1. **Heuristic** (default) – weighted combination of module scores.
           Ready to use without training data.
        2. **XGBoost** – uses a trained model if one is found on disk.
    """

    # Weights for heuristic fusion (sum = 1.0)
    DEFAULT_WEIGHTS = {
        "ocr_score":       0.15,
        "layout_score":    0.15,
        "metadata_score":  0.15,
        "signature_score": 0.15,
        "seal_score":      0.10,
        "forensic_score":  0.30,
    }

    RISK_THRESHOLDS = {
        "genuine":    75,   # score >= 75
        "suspicious": 45,   # 45 <= score < 75
        # below 45 → "likely_forged"
    }

    MODEL_PATH = os.path.join("models", "xgb_classifier.json")

    def __init__(self):
        self._xgb_model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, scores: Dict[str, float]) -> Dict:
        """
        Accept a dict of module scores and return an authenticity verdict.

        Expected keys in `scores`:
            ocr_score, layout_score, metadata_score,
            signature_score, seal_score, forensic_score

        Returns dict with:
            authenticity_score  – 0-100
            risk_level          – 'Genuine' | 'Suspicious' | 'Likely Forged'
            classification      – same as risk_level (alias)
            confidence          – model confidence (0-1)
            feature_vector      – the raw feature vector used
            module_contributions – per-module weighted contribution
        """
        feature_vector = self._build_feature_vector(scores)

        if self._xgb_model is not None:
            return self._classify_xgb(feature_vector, scores)

        return self._classify_heuristic(feature_vector, scores)

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _build_feature_vector(self, scores: Dict[str, float]) -> np.ndarray:
        """Build a fixed-order feature vector from module scores."""
        keys = [
            "ocr_score", "layout_score", "metadata_score",
            "signature_score", "seal_score", "forensic_score",
        ]
        return np.array([float(scores.get(k, 50)) for k in keys])

    # ------------------------------------------------------------------
    # Heuristic classifier
    # ------------------------------------------------------------------

    def _classify_heuristic(
        self, features: np.ndarray, scores: Dict[str, float]
    ) -> Dict:
        """Weighted-average heuristic classifier."""
        keys = list(self.DEFAULT_WEIGHTS.keys())
        weights = np.array([self.DEFAULT_WEIGHTS[k] for k in keys])

        weighted_scores = features * weights
        raw_score = float(np.sum(weighted_scores))

        # Apply non-linear scaling – penalise any very low individual score
        min_score = float(np.min(features))
        if min_score < 30:
            raw_score -= (30 - min_score) * 0.3

        authenticity = max(0, min(100, raw_score))

        risk_level = self._score_to_risk(authenticity)
        confidence = self._estimate_confidence(features)

        contributions = {}
        for i, k in enumerate(keys):
            contributions[k] = {
                "raw": round(float(features[i]), 1),
                "weight": self.DEFAULT_WEIGHTS[k],
                "weighted": round(float(weighted_scores[i]), 1),
            }

        return {
            "authenticity_score": round(authenticity, 1),
            "risk_level": risk_level,
            "classification": risk_level,
            "confidence": round(confidence, 3),
            "method": "heuristic_ensemble",
            "feature_vector": [round(float(f), 1) for f in features],
            "module_contributions": contributions,
        }

    # ------------------------------------------------------------------
    # XGBoost classifier (if model available)
    # ------------------------------------------------------------------

    def _load_model(self):
        """Attempt to load a pre-trained XGBoost model."""
        if not os.path.exists(self.MODEL_PATH):
            return
        try:
            import xgboost as xgb
            self._xgb_model = xgb.XGBClassifier()
            self._xgb_model.load_model(self.MODEL_PATH)
        except Exception as e:
            print(f"[ML] Could not load XGBoost model: {e}")
            self._xgb_model = None

    def _classify_xgb(
        self, features: np.ndarray, scores: Dict[str, float]
    ) -> Dict:
        """Classify using the trained XGBoost model."""
        try:
            proba = self._xgb_model.predict_proba(features.reshape(1, -1))[0]
            pred_class = int(np.argmax(proba))
            class_map = {0: "Genuine", 1: "Suspicious", 2: "Likely Forged"}
            risk_level = class_map.get(pred_class, "Unknown")

            # Map probability to 0-100 score
            authenticity = float(proba[0]) * 100  # P(genuine)

            return {
                "authenticity_score": round(authenticity, 1),
                "risk_level": risk_level,
                "classification": risk_level,
                "confidence": round(float(np.max(proba)), 3),
                "method": "xgboost",
                "feature_vector": [round(float(f), 1) for f in features],
                "class_probabilities": {
                    "genuine": round(float(proba[0]), 3),
                    "suspicious": round(float(proba[1]), 3) if len(proba) > 1 else 0,
                    "likely_forged": round(float(proba[2]), 3) if len(proba) > 2 else 0,
                },
            }
        except Exception as e:
            print(f"[ML] XGBoost prediction failed, falling back: {e}")
            return self._classify_heuristic(features, scores)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _score_to_risk(self, score: float) -> str:
        if score >= self.RISK_THRESHOLDS["genuine"]:
            return "Genuine"
        if score >= self.RISK_THRESHOLDS["suspicious"]:
            return "Suspicious"
        return "Likely Forged"

    def _estimate_confidence(self, features: np.ndarray) -> float:
        """
        Estimate how confident we are in the verdict.
        Low variance among module scores → higher confidence.
        """
        std = float(np.std(features))
        # Normalise: std of 0 → confidence 1.0, std of 50 → ~0.5
        confidence = max(0.3, 1.0 - std / 50)
        return confidence
