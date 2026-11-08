"""Cost-sensitive threshold logic."""
import numpy as np

from src import decision


def _toy():
    # 4 negatives, 2 positives. Probabilities chosen so the operating point
    # matters: one positive sits at 0.45 (below 0.5).
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.10, 0.20, 0.40, 0.50, 0.45, 0.60])
    return y, p


def test_expected_cost_counts_errors():
    y, p = _toy()
    # At threshold 0.5: catches 0.50(neg,FP) and 0.60(pos,TP); misses 0.45.
    cost = decision.expected_cost(y, p, 0.5, cost_fn=8000, cost_fp=300)
    assert cost == 8000 * 1 + 300 * 1  # 1 FN + 1 FP


def test_threshold_below_half_when_fn_expensive():
    y, p = _toy()
    res = decision.choose_threshold(y, p, cost_fn=8000, cost_fp=300)
    assert res["threshold"] < 0.5
    assert res["savings_vs_0.5"] > 0


def test_threshold_shifts_up_when_fp_expensive():
    """Cheap missed failures + costly false alarms => higher threshold."""
    y, p = _toy()
    fn_heavy = decision.choose_threshold(y, p, cost_fn=8000, cost_fp=300)
    fp_heavy = decision.choose_threshold(y, p, cost_fn=100, cost_fp=5000)
    assert fp_heavy["threshold"] > fn_heavy["threshold"]


def test_threshold_report_confusion_counts():
    y, p = _toy()
    rep = decision.threshold_report(y, p, 0.5, cost_fn=8000, cost_fp=300)
    assert rep["tp"] == 1 and rep["fp"] == 1 and rep["fn"] == 1 and rep["tn"] == 3
    assert 0.0 <= rep["f1"] <= 1.0
