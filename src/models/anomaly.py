"""Unsupervised anomaly detector.

``IsolationForest`` trained **only on healthy rows** of the training split, so it
learns the manifold of normal operation. It is then scored against a held-out
*mixed* set (healthy + failures). This is the classic "novelty detection"
framing used when labelled failures are scarce in production.

Reported metrics: ROC-AUC and PR-AUC of ``-score_samples`` (higher = more
anomalous) against the true failure label, plus the flag rate at a chosen
contamination quantile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .. import config


@dataclass
class AnomalyDetector:
    model: Any
    features: list
    threshold: float  # score above this => anomaly

    def score(self, X) -> np.ndarray:
        """Anomaly score, higher = more anomalous."""
        X = np.asarray(X, dtype=float)
        # score_samples: lower = more anomalous. Flip sign.
        return -self.model.score_samples(X)

    def predict(self, X) -> np.ndarray:
        return (self.score(X) >= self.threshold).astype(int)


def train_anomaly_detector(
    train_df, contamination: float = 0.05, seed: int = config.RANDOM_STATE
) -> AnomalyDetector:
    """Fit IsolationForest on healthy-only training rows."""
    from sklearn.ensemble import IsolationForest

    healthy = train_df[train_df[config.TARGET] == 0]
    X = healthy[config.FEATURE_COLS].to_numpy(dtype=float)
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X)

    # Threshold = (1-contamination) quantile of healthy training scores.
    healthy_scores = -model.score_samples(X)
    threshold = float(np.quantile(healthy_scores, 1.0 - contamination))
    return AnomalyDetector(model, list(config.FEATURE_COLS), threshold)


def evaluate_anomaly(detector: AnomalyDetector, mixed_df) -> Dict[str, Any]:
    """Score the detector on a held-out mixed set."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    X = mixed_df[config.FEATURE_COLS].to_numpy(dtype=float)
    y = mixed_df[config.TARGET].to_numpy(dtype=int)
    scores = detector.score(X)
    pred = (scores >= detector.threshold).astype(int)
    return {
        "anomaly_roc_auc": float(roc_auc_score(y, scores)),
        "anomaly_pr_auc": float(average_precision_score(y, scores)),
        "threshold": float(detector.threshold),
        "flag_rate": float(pred.mean()),
        "test_positives": int(y.sum()),
        "test_rows": int(len(y)),
    }
