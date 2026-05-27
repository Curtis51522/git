"""Anomaly Detector -- Isolation Forest + severity classifier for B1 Monitor.

No LLM calls.  Pure statistical model, < 100 ms per cycle.
Uses sklearn IsolationForest for unsupervised anomaly detection
on 6-dimensional feature vectors derived from cross_source_audit.

Model is persisted to disk and retrained daily to adapt to
changing store patterns.
"""

import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("s5.anomaly")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
)
MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_isolation_forest.pkl")
HISTORY_PATH = os.path.join(MODEL_DIR, "anomaly_history.json")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_feature_vector(audit_result: dict) -> Optional[np.ndarray]:
    """Convert a cross_source_audit result dict into a 6-dim feature vector.

    Features:
      0: deviation_pct     -- |forecast - actual| / forecast * 100  (or 0)
      1: stock_coverage    -- inventory / max(forecast, 1)
      2: headcount_gap     -- max(0, needed_hc - actual_hc)
      3: waste_rate         -- waste_units / max(total_produced, 1) * 100
      4: capacity_pressure  -- recommended_restock / max(capacity, 1)
      5: anomaly_count      -- number of issues found in audit (0-20)
    """
    try:
        forecast = float(audit_result.get("forecast", 0))
        inventory = float(audit_result.get("inventory", 0))
        capacity = float(audit_result.get("capacity", 0))
        issues = audit_result.get("issues", [])
        actual = audit_result.get("actual", forecast)  # fallback if no actual
        
        # Deviation
        deviation_pct = abs(actual - forecast) / max(abs(forecast), 1) * 100 if forecast != 0 else 0
        
        # Stock coverage (days of inventory at forecast rate)
        daily_demand = max(forecast, 1)
        stock_coverage = inventory / daily_demand
        
        # Headcount gap (aggregate from R7 anomalies)
        headcount_gap = 0
        for issue in issues:
            if issue.get("rule") == "R7":
                # Extract gap from message: "1 staff for 40 transactions"
                msg = issue.get("message", "")
                import re
                m = re.search(r"(\d+)\s+staff\s+for\s+(\d+)\s+transactions", msg)
                if m:
                    hc = int(m.group(1))
                    txn = int(m.group(2))
                    needed = max(1, txn // 8)  # heuristic: 8 txns/hr per person
                    headcount_gap += max(0, needed - hc)
        
        # Waste rate (if available)
        waste_rate = float(audit_result.get("waste_rate", 0))
        
        # Capacity pressure
        restock = float(audit_result.get("recommended_restock", 0))
        capacity_pressure = restock / max(capacity, 1) if capacity > 0 else 0
        
        # Issue count
        anomaly_count = min(float(len(issues)), 20.0)
        
        vec = np.array([
            deviation_pct,
            stock_coverage,
            headcount_gap,
            waste_rate,
            capacity_pressure,
            anomaly_count,
        ], dtype=np.float64)
        
        return vec
    except Exception as e:
        logger.warning("Feature vector build failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Isolation Forest model
# ---------------------------------------------------------------------------
class AnomalyDetector:
    """Unsupervised anomaly detector using Isolation Forest.

    - Trained on historical feature vectors (7-day rolling window).
    - Retrained daily.
    - Persisted to disk for server restarts.
    """

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.model: Optional[IsolationForest] = None
        self._last_trained: Optional[str] = None
        self._history: List[Dict] = []  # list of {ts, vector, audit_id}

    # ------------------------------------------------------------------
    def load_or_train(self, feature_vectors: List[np.ndarray]) -> bool:
        """Load saved model or train a new one if needed."""
        # Try loading
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                self._load_history()
                logger.info("Loaded anomaly model from %s (%d history points)",
                           MODEL_PATH, len(self._history))
                return True
            except Exception as e:
                logger.warning("Failed to load model: %s -- will train new", e)

        # Train new
        return self._train(feature_vectors)

    # ------------------------------------------------------------------
    def should_retrain(self) -> bool:
        """Check if model was last trained > 24 hours ago."""
        if self._last_trained is None:
            return True
        try:
            last = datetime.fromisoformat(self._last_trained)
            return datetime.now() - last > timedelta(hours=24)
        except Exception:
            return True

    # ------------------------------------------------------------------
    def retrain(self, feature_vectors: List[np.ndarray]) -> bool:
        """Retrain with new data (daily)."""
        return self._train(feature_vectors)

    # ------------------------------------------------------------------
    def predict(self, feature_vector: np.ndarray) -> Tuple[bool, float]:
        """Return (is_anomaly, anomaly_score).

        anomaly_score < -0.5 typically indicates an anomaly.
        Lower score = more anomalous.
        """
        if self.model is None:
            return False, 0.0

        vec = feature_vector.reshape(1, -1)
        pred = self.model.predict(vec)[0]       # 1 = inlier, -1 = outlier
        score = self.model.score_samples(vec)[0] # lower = more anomalous

        is_anomaly = pred == -1 and score < -0.5
        return is_anomaly, float(score)

    # ------------------------------------------------------------------
    def classify_severity(self, feature_vector: np.ndarray,
                          is_anomaly: bool, score: float) -> str:
        """Classify alert severity based on feature values + anomaly score."""
        if not is_anomaly:
            return "info"

        deviation, stock_cov, hc_gap, waste, cap_pres, issue_cnt = feature_vector

        # CRITICAL: stockout imminent OR severe understaffing OR extreme deviation
        if stock_cov < 0.2:          # < 0.2 days of stock
            return "critical"
        if hc_gap >= 2:              # 2+ missing staff
            return "critical"
        if deviation > 50:           # > 50% forecast error
            return "critical"
        if cap_pres > 1.5:           # > 150% capacity pressure
            return "critical"

        # WARNING: moderate issues
        if stock_cov < 0.5:
            return "warning"
        if hc_gap >= 1:
            return "warning"
        if deviation > 30:
            return "warning"
        if score < -0.7:             # very anomalous
            return "warning"

        return "info"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _train(self, vectors: List[np.ndarray]) -> bool:
        """Train Isolation Forest on the given vectors."""
        if len(vectors) < 10:
            logger.warning("Not enough data to train anomaly model (%d points)", len(vectors))
            # Bootstrap with synthetic normal points
            synthetic = np.random.normal(loc=0.3, scale=0.15, size=(20, 6))
            synthetic = np.clip(synthetic, 0, 2)
            vectors = [synthetic[i] for i in range(20)]

        X = np.array(vectors)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
        )
        self.model.fit(X)
        self._last_trained = datetime.now().isoformat()

        # Persist
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(self.model, f)
            self._save_history()
            logger.info("Anomaly model trained + saved (%d vectors)", len(vectors))
        except Exception as e:
            logger.error("Failed to save model: %s", e)

        return True

    def _load_history(self):
        try:
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH, "r") as f:
                    self._history = json.load(f)
        except Exception:
            self._history = []

    def _save_history(self):
        try:
            with open(HISTORY_PATH, "w") as f:
                json.dump(self._history[-500:], f, default=str)  # keep last 500
        except Exception as e:
                        logger.warning("Failed to persist anomaly history: %s", e)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_detector: Optional[AnomalyDetector] = None


def get_detector() -> AnomalyDetector:
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector
