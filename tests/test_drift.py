"""Drift detection: PSI on identical vs shifted data, KS, chi-square."""
import numpy as np
import pandas as pd

from src import config, drift


def _frame(n=1000, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Air temperature [K]": rng.normal(300, 2, n),
        "Process temperature [K]": rng.normal(310, 1.5, n),
        "Rotational speed [rpm]": rng.normal(1500, 100, n) + shift,
        "Torque [Nm]": rng.normal(40, 8, n) + shift,
        "Tool wear [min]": rng.integers(0, 250, n).astype(float),
        "Type": rng.choice(["L", "M", "H"], n),
    })


def test_psi_identical_is_near_zero():
    ref = _frame(seed=1)
    psi = drift.population_stability_index(ref["Torque [Nm]"], ref["Torque [Nm]"])
    assert psi < 0.01


def test_psi_shifted_is_large():
    ref = _frame(seed=1)
    cur = _frame(seed=2, shift=30.0)  # torque shifted by +30 Nm
    psi = drift.population_stability_index(ref["Torque [Nm]"], cur["Torque [Nm]"])
    assert psi > config.PSI_THRESHOLD


def test_ks_detects_shift():
    ref = _frame(seed=3)
    cur = _frame(seed=4, shift=100.0)
    out = drift.ks_test(ref["Rotational speed [rpm]"], cur["Rotational speed [rpm]"])
    assert out["ks_pvalue"] < config.KS_P_THRESHOLD
    assert out["ks_stat"] > 0.3


def test_chi_square_same_distribution():
    ref = _frame(seed=5)
    out = drift.chi_square_test(ref["Type"], ref["Type"].sample(frac=1, random_state=1))
    assert out["chi2_pvalue"] > 0.05


def test_detect_drift_nominal_vs_drifted():
    ref = _frame(seed=7)
    drifted = drift.simulate_drift(ref, seed=8)
    nominal = drift.detect_drift(ref, _frame(seed=9))
    real = drift.detect_drift(ref, drifted)
    # Nominal (same distribution) should not trigger; drifted should.
    assert not nominal["should_retrain"]
    assert real["should_retrain"]
    assert real["drift_score"] > nominal["drift_score"]
    trig = drift.retraining_trigger(real)
    assert trig["action"] == "retrain"


def test_drift_markdown_renders():
    ref = _frame(seed=11)
    rep = drift.detect_drift(ref, drift.simulate_drift(ref, seed=12))
    md = drift.drift_markdown(rep, rep)
    assert "# Drift Monitoring Report" in md
    assert "Retraining Decision" in md
