"""Cost-sensitive decision policy.

Choosing a decision threshold of 0.5 for an imbalanced failure classifier is a
mistake: the cost of a missed failure is ~27x the cost of a false alarm, so the
optimal operating point sits well below 0.5. This module searches the threshold
grid that minimises *expected business cost*.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import config


def expected_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_fn: float = config.COST_FN,
    cost_fp: float = config.COST_FP,
) -> float:
    """Expected cost (currency units) of operating at `threshold`.

    Predicted positive => schedule maintenance. We pay ``cost_fp`` per false
    alarm and ``cost_fn`` per missed failure.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return fp * cost_fp + fn * cost_fn


def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = config.COST_FN,
    cost_fp: float = config.COST_FP,
    grid: int = 199,
) -> Dict[str, float]:
    """Return the cost-minimising threshold and its associated costs.

    Also reports the naive-0.5 baseline so the report can show the *delta*.
    """
    thresholds = np.linspace(0.001, 0.999, grid)
    costs = np.array([expected_cost(y_true, y_prob, t, cost_fn, cost_fp)
                      for t in thresholds])
    best_idx = int(np.argmin(costs))
    best_t = float(thresholds[best_idx])
    baseline_cost = expected_cost(y_true, y_prob, 0.5, cost_fn, cost_fp)
    return {
        "threshold": best_t,
        "expected_cost": float(costs[best_idx]),
        "baseline_cost_at_0.5": float(baseline_cost),
        "savings_vs_0.5": float(baseline_cost - costs[best_idx]),
        "cost_fn": float(cost_fn),
        "cost_fp": float(cost_fp),
    }


def threshold_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_fn: float = config.COST_FN,
    cost_fp: float = config.COST_FP,
) -> Dict[str, float]:
    """Confusion-matrix-derived metrics at a fixed threshold."""
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "expected_cost": expected_cost(y_true, y_prob, threshold, cost_fn, cost_fp),
    }
