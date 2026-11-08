"""Failure classifier.

Primary estimator: ``HistGradientBoostingClassifier`` (fast, native NaN
handling, strong on tabular imbalance). If ``xgboost`` is installed we expose an
identical interface on top of it and let the pipeline pick via ``backend=``.

The classifier is *calibrated* (isotonic, cross-validated) so the output
probability is meaningful enough for cost-based thresholding and for the API.
Primary metric is PR-AUC (average precision) because failures are ~3.4% of rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from .. import config


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa

        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass
class TrainedClassifier:
    model: Any
    features: list
    backend: str
    params: Dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def _base_estimator(backend: str, seed: int):
    if backend == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=-1,
        )
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=seed,
    )


def train_classifier(
    train_df,
    backend: str = "auto",
    calibrate: bool = True,
    seed: int = config.RANDOM_STATE,
) -> TrainedClassifier:
    """Fit a calibrated failure classifier on the training split."""
    from sklearn.calibration import CalibratedClassifierCV

    if backend == "auto":
        backend = "xgboost" if xgboost_available() else "hist_gbdt"
    if backend == "xgboost" and not xgboost_available():
        print("[classifier] xgboost requested but not installed -> hist_gbdt")
        backend = "hist_gbdt"

    X = train_df[config.FEATURE_COLS].to_numpy(dtype=float)
    y = train_df[config.TARGET].to_numpy(dtype=int)

    base = _base_estimator(backend, seed)
    if calibrate:
        # Hold out a slice for calibration via cross-validation.
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    else:
        model = base
    model.fit(X, y)

    params = {
        "backend": backend,
        "calibrated": calibrate,
        "max_iter": 400,
        "learning_rate": 0.06,
        "n_features": len(config.FEATURE_COLS),
    }
    return TrainedClassifier(model, list(config.FEATURE_COLS), backend, params)


def evaluate_classifier(model: TrainedClassifier, test_df) -> Dict[str, Any]:
    """Full evaluation dict: PR-AUC focus + ROC-AUC + Brier."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    X = test_df[config.FEATURE_COLS].to_numpy(dtype=float)
    y = test_df[config.TARGET].to_numpy(dtype=int)
    prob = model.predict_proba(X)
    return {
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "test_positives": int(y.sum()),
        "test_rows": int(len(y)),
        "baseline_pr_auc": float(y.mean()),
    }
