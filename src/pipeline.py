"""End-to-end pipeline: train -> evaluate -> register champion -> report.

One command runs everything::

    python -m src.pipeline

Steps:
  1. Prepare data (validate, features, stratified splits, synthetic daily series).
  2. Train + evaluate the failure classifier; select a cost-sensitive threshold.
  3. Train + evaluate the IsolationForest anomaly detector.
  4. Compare four forecasters on a time-based holdout.
  5. Log all three model runs to the tracker (MLflow or JSON fallback).
  6. Register the champion in models/registry.json and persist artifacts.
  7. Render reports/evaluation.md (with figures) and reports/drift.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config, data, decision, drift, registry, tracking
from .models import anomaly as anomaly_mod
from .models import classifier as clf_mod
from .models import forecast as fc_mod

# Matplotlib: headless-safe.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _fig_confusion_matrix(cm, threshold, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(f"Confusion matrix @ threshold={threshold:.3f}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_pr_curve(y_true, y_prob, ap, path: Path) -> None:
    from sklearn.metrics import precision_recall_curve

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(rec, prec, label=f"PR-AUC = {ap:.3f}")
    ax.axhline(y_true.mean(), ls="--", color="grey",
               label=f"baseline = {y_true.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test set)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_forecast(holdout, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(holdout["date"], holdout["actual"], label="actual", color="black", lw=1.6)
    for name in ("seasonal_naive", "ses", "gb_lags"):
        ax.plot(holdout["date"], holdout[name], label=name, lw=1.0, alpha=0.8)
    ax.set_title("Downtime forecast — last 90 days holdout")
    ax.set_ylabel("downtime minutes/day")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_drift(ref, cur, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    axes[0].hist(ref["Torque [Nm]"], bins=40, alpha=0.6, label="reference", density=True)
    axes[0].hist(cur["Torque [Nm]"], bins=40, alpha=0.6, label="drifted", density=True)
    axes[0].set_title("Torque distribution")
    axes[0].legend(fontsize=8)
    axes[1].hist(ref["Rotational speed [rpm]"], bins=40, alpha=0.6, label="reference", density=True)
    axes[1].hist(cur["Rotational speed [rpm]"], bins=40, alpha=0.6, label="drifted", density=True)
    axes[1].set_title("Rotational speed distribution")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _md_table(rows, headers) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def render_evaluation_md(ctx: dict) -> str:
    clf = ctx["clf_metrics"]
    thr = ctx["threshold_report"]
    best_thr = ctx["threshold_choice"]
    anom = ctx["anom_metrics"]
    fc = ctx["forecast"]
    lines = []
    lines.append("# Evaluation Report\n")
    lines.append(f"_Generated {ctx['timestamp']} · tracker backend: "
                 f"**{ctx['tracker_backend']}** · git: `{ctx['git_commit']}`_\n")

    lines.append("## 1. Failure classifier (imbalanced, PR-AUC focus)\n")
    lines.append(_md_table(
        [["PR-AUC (average precision)", f"{clf['pr_auc']:.4f}"],
         ["ROC-AUC", f"{clf['roc_auc']:.4f}"],
         ["Brier score", f"{clf['brier']:.4f}"],
         ["Baseline PR-AUC (failure rate)", f"{clf['baseline_pr_auc']:.4f}"],
         ["Test rows / positives", f"{clf['test_rows']} / {clf['test_positives']}"]],
        ["Metric", "Value"]))
    lines.append("")

    lines.append("## 2. Cost-sensitive decision policy\n")
    lines.append(f"Assumed costs: missed failure = **${best_thr['cost_fn']:,.0f}**, "
                 f"false alarm = **${best_thr['cost_fp']:,.0f}** "
                 f"(ratio {best_thr['cost_fn']/best_thr['cost_fp']:.0f}:1).\n")
    lines.append(_md_table(
        [["Chosen threshold", f"{best_thr['threshold']:.3f}"],
         ["Expected cost @ chosen", f"${best_thr['expected_cost']:,.0f}"],
         ["Expected cost @ 0.5", f"${best_thr['baseline_cost_at_0.5']:,.0f}"],
         ["Savings vs 0.5", f"${best_thr['savings_vs_0.5']:,.0f}"],
         ["Precision @ chosen", f"{thr['precision']:.4f}"],
         ["Recall @ chosen", f"{thr['recall']:.4f}"],
         ["F1 @ chosen", f"{thr['f1']:.4f}"],
         ["TP / FP / FN / TN", f"{thr['tp']} / {thr['fp']} / {thr['fn']} / {thr['tn']}"]],
        ["Metric", "Value"]))
    lines.append("")

    lines.append("## 3. Anomaly detector (IsolationForest, healthy-only train)\n")
    lines.append(_md_table(
        [["ROC-AUC", f"{anom['anomaly_roc_auc']:.4f}"],
         ["PR-AUC", f"{anom['anomaly_pr_auc']:.4f}"],
         ["Flag rate", f"{anom['flag_rate']:.4f}"],
         ["Flag threshold", f"{anom['threshold']:.4f}"]],
        ["Metric", "Value"]))
    lines.append("")

    lines.append("## 4. Downtime forecasting (time-based holdout)\n")
    lines.append(f"Holdout horizon: **{fc['horizon']} days**. "
                 f"SES alpha = {fc['ses_alpha']:.3f}.\n")
    rows = [[name, f"{m['mae']:.2f}", f"{m['rmse']:.2f}", f"{m['mape']:.2f}%"]
            for name, m in fc["models"].items()]
    header = ["Model", "MAE (min)", "RMSE (min)", "MAPE (%)"]
    if "gb_lags" in fc["models"]:
        pass
    lines.append(_md_table(rows, header))
    lines.append(f"\n**Best by MAE:** `{fc['best']}`\n")

    lines.append("## 5. Figures\n")
    lines.append("![Confusion matrix](figures/confusion_matrix.png)\n")
    lines.append("![PR curve](figures/pr_curve.png)\n")
    lines.append("![Forecast](figures/forecast.png)\n")
    lines.append("![Drift](figures/drift.png)\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(backend: str = "auto", tracker=None, save: bool = True) -> dict:
    config.ensure_dirs()
    tracker = tracker or tracking.get_tracker()
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. data
    prepared = data.prepare(save=save)
    train, test, daily = prepared.train, prepared.test, prepared.daily

    # 2. classifier
    model = clf_mod.train_classifier(train, backend=backend)
    clf_metrics = clf_mod.evaluate_classifier(model, test)

    Xte = test[config.FEATURE_COLS].to_numpy(float)
    yte = test[config.TARGET].to_numpy(int)
    prob = model.predict_proba(Xte)
    threshold_choice = decision.choose_threshold(yte, prob)
    thr = threshold_choice["threshold"]
    threshold_report = decision.threshold_report(yte, prob, thr)

    # 3. anomaly
    detector = anomaly_mod.train_anomaly_detector(train)
    anom_metrics = anomaly_mod.evaluate_anomaly(detector, test)

    # 4. forecast
    fc = fc_mod.forecast_all(daily)

    # 5. tracking
    with tracker.start_run("classifier", tags={"model": "failure-classifier"}) as r:
        r.log_params(model.params)
        r.log_metrics({**clf_metrics, "threshold": thr,
                       "precision": threshold_report["precision"],
                       "recall": threshold_report["recall"],
                       "f1": threshold_report["f1"],
                       "expected_cost": threshold_report["expected_cost"]})
        r.log_dict(threshold_choice, "threshold_choice.json")

    with tracker.start_run("anomaly", tags={"model": "isolation-forest"}) as r:
        r.log_params({"n_estimators": 300, "contamination": 0.05})
        r.log_metrics(anom_metrics)

    with tracker.start_run("forecast", tags={"model": "downtime-forecast"}) as r:
        r.log_params({"horizon": fc["horizon"], "ses_alpha": fc["ses_alpha"]})
        r.log_metrics({f"{name}_{m}": v
                       for name, mm in fc["models"].items()
                       for m, v in mm.items()})
        r.log_dict({k: v for k, v in fc["models"].items()}, "forecast_metrics.json")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # 6. persist champion artifacts
    import joblib

    config.CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.CHAMPION_DIR / "classifier.joblib")
    joblib.dump(detector, config.CHAMPION_DIR / "anomaly.joblib")
    (config.CHAMPION_DIR / "forecast_meta.json").write_text(
        json.dumps({"models": fc["models"], "best": fc["best"],
                    "ses_alpha": fc["ses_alpha"], "horizon": fc["horizon"]}, indent=2))

    metrics_for_registry = {
        "pr_auc": clf_metrics["pr_auc"],
        "roc_auc": clf_metrics["roc_auc"],
        "f1": threshold_report["f1"],
        "precision": threshold_report["precision"],
        "recall": threshold_report["recall"],
        "threshold": thr,
        "anomaly_roc_auc": anom_metrics["anomaly_roc_auc"],
        "forecast_mae": fc["models"][fc["best"]]["mae"],
        "forecast_mape": fc["models"][fc["best"]]["mape"],
    }
    version = "v1.0.0"
    entry = registry.save_champion(metrics_for_registry, run_id, version,
                                   threshold_choice,
                                   extra={"backend": model.backend,
                                          "forecast_best": fc["best"]})

    # 7. drift
    drifted_df = drift.simulate_drift(test)
    nominal_report = drift.detect_drift(train, test)          # same dist -> stable
    drifted_report = drift.detect_drift(train, drifted_df)    # shifted -> flagged
    (config.REPORTS_DIR / "drift.md").write_text(
        drift.drift_markdown(nominal_report, drifted_report))
    (config.REPORTS_DIR / "drift_report.json").write_text(
        json.dumps({"nominal": nominal_report, "drifted": drifted_report},
                   indent=2, default=str))

    # figures
    from sklearn.metrics import confusion_matrix

    img_dir = config.FIGURES_DIR
    img_dir.mkdir(parents=True, exist_ok=True)
    pred = (prob >= thr).astype(int)
    _fig_confusion_matrix(confusion_matrix(yte, pred), thr,
                          img_dir / "confusion_matrix.png")
    _fig_pr_curve(yte, prob, clf_metrics["pr_auc"], img_dir / "pr_curve.png")
    _fig_forecast(fc["holdout"], img_dir / "forecast.png")
    _fig_drift(train, drifted_df, img_dir / "drift.png")

    ctx = {
        "timestamp": timestamp,
        "tracker_backend": tracker.name,
        "git_commit": entry["git_commit"],
        "clf_metrics": clf_metrics,
        "threshold_choice": threshold_choice,
        "threshold_report": threshold_report,
        "anom_metrics": anom_metrics,
        "forecast": fc,
        "nominal_drift": nominal_report,
        "drifted_drift": drifted_report,
    }
    (config.REPORTS_DIR / "evaluation.md").write_text(render_evaluation_md(ctx))
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description="Downtime-forecast MLOps pipeline")
    ap.add_argument("--backend", default="auto", choices=["auto", "xgboost", "hist_gbdt"])
    ap.add_argument("--no-save", action="store_true", help="skip writing processed data")
    args = ap.parse_args(argv)

    ctx = run(backend=args.backend, save=not args.no_save)
    print("\n=== PIPELINE COMPLETE ===")
    print(f"tracker      : {ctx['tracker_backend']}")
    print(f"PR-AUC       : {ctx['clf_metrics']['pr_auc']:.4f}")
    print(f"ROC-AUC      : {ctx['clf_metrics']['roc_auc']:.4f}")
    print(f"F1 @thr      : {ctx['threshold_report']['f1']:.4f} "
          f"(threshold={ctx['threshold_report']['threshold']:.3f})")
    print(f"forecast best: {ctx['forecast']['best']} "
          f"MAE={ctx['forecast']['models'][ctx['forecast']['best']]['mae']:.2f}")
    print(f"drift score  : {ctx['drifted_drift']['drift_score']}")


if __name__ == "__main__":  # pragma: no cover
    main()
