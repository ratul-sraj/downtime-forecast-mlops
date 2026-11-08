"""Drift monitoring.

Detects distribution shift between a *reference* batch (training data) and a
*current* batch (production) using three complementary tests:

  * **PSI** (Population Stability Index) for numeric features — binned.
  * **KS test** (two-sample Kolmogorov-Smirnov) for numeric features.
  * **Chi-square** for the categorical product ``Type``.

Thresholds live in ``config`` (PSI>0.2 significant, 0.1-0.2 moderate). An overall
``drift_score`` = fraction of features flagged feeds a retraining trigger.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------
def population_stability_index(
    reference: np.ndarray, current: np.ndarray, buckets: int = 10
) -> float:
    """PSI between two numeric samples using quantile bins from the reference.

    PSI = sum((cur_pct - ref_pct) * ln(cur_pct / ref_pct)).
    A small epsilon avoids log(0); empty/missing bins are smoothed.
    """
    reference = np.asarray(reference, float)
    current = np.asarray(current, float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Bin edges from reference quantiles, made unique.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    eps = 1e-6
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_test(reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    from scipy.stats import ks_2samp  # scipy ships with scikit-learn

    reference = np.asarray(reference, float)
    current = np.asarray(current, float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    stat, p = ks_2samp(reference, current)
    return {"ks_stat": float(stat), "ks_pvalue": float(p)}


def chi_square_test(reference: pd.Series, current: pd.Series) -> Dict[str, float]:
    from scipy.stats import chi2_contingency

    cats = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
    ref_counts = reference.value_counts().reindex(cats, fill_value=0).to_numpy()
    cur_counts = current.value_counts().reindex(cats, fill_value=0).to_numpy()
    table = np.vstack([ref_counts, cur_counts])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return {"chi2_stat": 0.0, "chi2_pvalue": 1.0}
    chi2, p, _, _ = chi2_contingency(table)
    return {"chi2_stat": float(chi2), "chi2_pvalue": float(p)}


def _psi_severity(psi: float) -> str:
    if psi > config.PSI_THRESHOLD:
        return "significant"
    if psi > config.PSI_WARN:
        return "moderate"
    return "stable"


# ---------------------------------------------------------------------------
# Full drift report
# ---------------------------------------------------------------------------
def detect_drift(reference: pd.DataFrame, current: pd.DataFrame) -> Dict:
    """Run all tests and return a structured drift report.

    ``drift_score`` = fraction of monitored features flagged by at least one
    test. ``should_retrain`` is True when drift_score exceeds
    ``config.DRIFT_RETRAIN_SCORE`` or any numeric feature has PSI>0.2.
    """
    numeric = config.RAW_NUMERIC_COLS
    features: List[Dict] = []

    n_flagged = 0
    any_significant_psi = False
    for col in numeric:
        psi = population_stability_index(reference[col].to_numpy(),
                                         current[col].to_numpy())
        ks = ks_test(reference[col].to_numpy(), current[col].to_numpy())
        sev = _psi_severity(psi)
        flagged = psi > config.PSI_THRESHOLD or ks["ks_pvalue"] < config.KS_P_THRESHOLD
        if flagged:
            n_flagged += 1
        if psi > config.PSI_THRESHOLD:
            any_significant_psi = True
        features.append({
            "feature": col,
            "kind": "numeric",
            "psi": round(psi, 4),
            "psi_severity": sev,
            "ks_stat": round(ks["ks_stat"], 4),
            "ks_pvalue": ks["ks_pvalue"],
            "flagged": bool(flagged),
        })

    # Categorical
    chi = chi_square_test(reference["Type"], current["Type"])
    cat_flagged = chi["chi2_pvalue"] < config.CHI2_P_THRESHOLD
    if cat_flagged:
        n_flagged += 1
    features.append({
        "feature": "Type",
        "kind": "categorical",
        "chi2_stat": round(chi["chi2_stat"], 4),
        "chi2_pvalue": chi["chi2_pvalue"],
        "flagged": bool(cat_flagged),
    })

    total = len(numeric) + 1
    drift_score = n_flagged / total
    return {
        "features": features,
        "n_features": total,
        "n_flagged": n_flagged,
        "drift_score": round(drift_score, 4),
        "any_significant_psi": any_significant_psi,
        "should_retrain": bool(drift_score > config.DRIFT_RETRAIN_SCORE
                               or any_significant_psi),
        "thresholds": {
            "psi": config.PSI_THRESHOLD,
            "psi_warn": config.PSI_WARN,
            "ks_p": config.KS_P_THRESHOLD,
            "chi2_p": config.CHI2_P_THRESHOLD,
            "retrain_score": config.DRIFT_RETRAIN_SCORE,
        },
    }


def simulate_drift(df: pd.DataFrame, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Create a drifted production batch: torque up, speed down, hotter.

    This models tool-load creep + a cooling degradation — the kind of slow
    physical shift that invalidates a model trained on nominal conditions.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["Torque [Nm]"] = out["Torque [Nm]"] * 1.35 + rng.normal(0, 1.5, len(out))
    out["Rotational speed [rpm]"] = (
        out["Rotational speed [rpm]"] * 0.82 + rng.normal(0, 30, len(out))
    ).round().astype(int)
    out["Process temperature [K]"] = out["Process temperature [K]"] + 2.5 + rng.normal(0, 0.4, len(out))
    out["Air temperature [K]"] = out["Air temperature [K]"] + 3.0 + rng.normal(0, 0.4, len(out))
    return out


def retraining_trigger(drift_report: Dict) -> Dict:
    """Return a machine-readable retraining decision."""
    should = bool(drift_report["should_retrain"])
    return {
        "action": "retrain" if should else "monitor",
        "reason": (
            f"drift_score={drift_report['drift_score']} "
            f"(threshold {drift_report['thresholds']['retrain_score']}), "
            f"significant_psi={drift_report['any_significant_psi']}"
        ),
        "urgent": should,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def drift_markdown(report: Dict, drifted: Dict | None = None) -> str:
    lines: List[str] = []
    lines.append("# Drift Monitoring Report\n")
    lines.append("_Auto-generated by `src/drift.py`. Do not edit by hand._\n")

    def _table(rep: Dict, title: str):
        out = [f"## {title}\n"]
        out.append(f"- Overall drift score: **{rep['drift_score']}** "
                   f"({rep['n_flagged']}/{rep['n_features']} features flagged)")
        out.append(f"- Retraining trigger: **{'YES' if rep['should_retrain'] else 'no'}**\n")
        out.append("| Feature | Kind | PSI | PSI severity | KS stat | KS p | Chi2 p | Flagged |")
        out.append("|---|---|---|---|---|---|---|---|")
        for f in rep["features"]:
            out.append(
                "| {feature} | {kind} | {psi} | {sev} | {ks} | {ksp} | {cp} | {flag} |".format(
                    feature=f["feature"], kind=f["kind"],
                    psi=f.get("psi", "-"), sev=f.get("psi_severity", "-"),
                    ks=f.get("ks_stat", "-"),
                    ksp=(f"{f['ks_pvalue']:.2e}" if "ks_pvalue" in f else "-"),
                    cp=(f"{f['chi2_pvalue']:.2e}" if "chi2_pvalue" in f else "-"),
                    flag="YES" if f["flagged"] else "",
                )
            )
        return "\n".join(out) + "\n"

    lines.append(_table(report, "Reference vs Current (nominal)"))
    if drifted:
        lines.append(_table(drifted, "Reference vs Simulated Drifted Batch"))
        trig = retraining_trigger(drifted)
        lines.append("## Retraining Decision\n")
        lines.append(f"- Action: **{trig['action']}**")
        lines.append(f"- Reason: {trig['reason']}\n")

    lines.append("## Thresholds\n")
    t = report["thresholds"]
    lines.append(f"- PSI significant: >{t['psi']} (moderate >{t['psi_warn']})")
    lines.append(f"- KS p-value: <{t['ks_p']}")
    lines.append(f"- Chi-square p-value: <{t['chi2_p']}")
    lines.append(f"- Retrain drift score: >{t['retrain_score']}\n")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    """CLI: compare reference vs a simulated drifted batch and write reports."""
    import json

    from . import data

    raw = data.load_raw()
    featured = data.build_features(raw)
    train, test = data.stratified_splits(featured)
    drifted = simulate_drift(test)

    nominal = detect_drift(train, test)
    drifted_rep = detect_drift(train, drifted)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "drift.md").write_text(drift_markdown(nominal, drifted_rep))
    (config.REPORTS_DIR / "drift_report.json").write_text(
        json.dumps({"nominal": nominal, "drifted": drifted_rep}, indent=2, default=str))

    trig = retraining_trigger(drifted_rep)
    print(f"[drift] nominal score={nominal['drift_score']} "
          f"drifted score={drifted_rep['drift_score']} -> {trig['action']}")


if __name__ == "__main__":  # pragma: no cover
    main()
