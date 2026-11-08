# Downtime-Forecast MLOps

An end-to-end, **genuinely runnable** predictive-maintenance MLOps pipeline for a
manufacturing fleet: a calibrated failure classifier, an unsupervised anomaly
detector, a downtime forecaster, a cost-sensitive decision policy, drift
monitoring with a retraining trigger, experiment tracking, a champion registry,
a FastAPI service and an Airflow DAG — all wired to the AI4I 2020 dataset.

Everything below is backed by an actual run on the data in `data/raw/`.
No stubs, no fabricated numbers.

---

## 1. Problem framing

Unplanned machine downtime is expensive: a single line stop costs throughput,
emergency labour and expedited parts. The value of a predictive-maintenance
model is **not** accuracy — it is *dollars saved by deciding when to act*.

Two coupled questions:

1. **Classification** — given a live reading, what is the probability this
   machine fails soon? Only **3.4%** of readings are failures, so the metric
   that matters is **PR-AUC**, not accuracy or ROC-AUC alone.
2. **Forecasting** — how many downtime *minutes* should we expect per day, so
   maintenance can be scheduled around load?

A third, unsupervised model (IsolationForest) flags novel behaviour when
labelled failures are scarce in production.

Because a **missed failure** costs ~27× a **false alarm** in our assumptions, the
decision threshold is chosen by minimising expected business cost — *not* 0.5.

---

## 2. Architecture

```
              ┌────────────────────┐
              │  AI4I 2020  (UCI)  │  10,000 rows, 3.4% failures
              └─────────┬──────────┘
                        │  src/data.py: validate (pydantic) → features → splits
                        │            + synthetic daily series (730 days)
                        ▼
   ┌──────────────┬──────────────┬──────────────────┐
   │  Failure      │  Anomaly      │  Downtime        │
   │  Classifier   │  Detector     │  Forecaster      │
   │  HistGBDT /   │  Isolation    │  naive / seasonal│
   │  XGBoost,     │  Forest on    │  / SES (hand-    │
   │  calibrated   │  healthy-only │  rolled) / GBDT  │
   └──────┬────────┴──────┬────────┴────────┬─────────┘
          │               │                 │
          ▼               ▼                 ▼
   ┌──────────────────────────────────────────────┐
   │  src/decision.py  cost-sensitive threshold    │
   │  src/tracking.py  MLflow  OR  JSON-file runs/ │
   └───────────────────┬──────────────────────────┘
                       ▼
            ┌───────────────────────┐
            │ models/champion/  +   │  registry.json
            │ registry.json         │  (version, metrics, git sha, ts)
            └─────┬─────────────┬───┘
                  ▼             ▼
        ┌──────────────┐   ┌──────────────────┐
        │ FastAPI      │   │ reports/         │
        │ /predict     │   │  evaluation.md   │
        │ /anomaly     │   │  drift.md        │
        │ /forecast    │   │  figures/*.png   │
        │ /model-info  │   └──────────────────┘
        └──────────────┘
                  ▲
                  │  drift_score > threshold  ⇒  retrain
        ┌─────────┴──────────┐
        │ src/drift.py       │  PSI + KS + chi-square
        └─────────┬──────────┘
                  ▼
        ┌──────────────────────┐        ┌──────────────────────┐
        │ Airflow DAG (daily)  │ ─────▶ │ docker-compose:      │
        │ airflow/dags/*.py    │        │ airflow + mlflow     │
        └──────────────────────┘        └──────────────────────┘
```

**Data → features → 3 models → cost policy → registry → serving → drift → retrain.**

---

## 3. Results (real, from `python -m src.pipeline`)

### 3.1 Failure classifier

`HistGradientBoostingClassifier`, isotonic-calibrated (3-fold), 8,000 train /
2,000 test rows (stratified), 68 test positives.

| Metric | Value |
|---|---|
| **PR-AUC (average precision)** | **0.8997** |
| ROC-AUC | 0.9772 |
| Brier score | 0.0074 |
| Baseline PR-AUC (= failure rate) | 0.0340 |
| Test rows / positives | 2000 / 68 |

PR-AUC of 0.90 vs a 0.034 baseline is a **26× lift** over random.

### 3.2 Cost-sensitive decision policy

Assumed costs: missed failure = **$8,000**, false alarm = **$300** (27:1).

| Metric | Value |
|---|---|
| **Chosen threshold (cost-min)** | **0.087** |
| Expected cost @ chosen | **$63,900** |
| Expected cost @ 0.5 | $112,600 |
| **Savings vs naive 0.5** | **$48,700** |
| Precision @ chosen | 0.5391 |
| Recall @ chosen | 0.9118 |
| **F1 @ chosen** | **0.6776** |
| TP / FP / FN / TN | 62 / 53 / 6 / 1879 |

Choosing 0.5 instead of the cost-optimal 0.087 would leave **$48,700 on the
table** on a 2,000-row test set by missing far more failures.

### 3.3 Anomaly detector

IsolationForest (300 trees) trained on **healthy-only** rows, scored on the
held-out mixed set.

| Metric | Value |
|---|---|
| ROC-AUC | 0.8333 |
| PR-AUC | 0.1871 |
| Flag rate | 0.0565 |
| Flag threshold | 0.5730 |

Unsupervised detection is intentionally the weakest signal — it is a **safety
net** for unlabelled novel behaviour, not a replacement for the classifier.

### 3.4 Downtime forecasting (90-day time-based holdout)

| Model | MAE (min) | RMSE (min) | MAPE (%) |
|---|---|---|---|
| **naive** | **159.79** | 266.16 | 100.00 |
| seasonal_naive | 175.62 | 263.56 | 87.73 |
| ses | 174.17 | **213.09** | 52.45 |
| gb_lags | 185.19 | 250.90 | 79.63 |

**Best by MAE:** `naive` (SES alpha = 0.041). The synthetic series is
noise-dominated with zero-downtime days, so a strong baseline wins on MAE and
**MAPE is unreliable** (division by near-zero). SES wins on RMSE. This is an
honest, unflattering result and it is reported as-is — see Limitations.

### 3.5 Drift detection

Reference = train split; "current" = (a) nominal test split, (b) simulated
drifted batch (torque ×1.35, speed ×0.82, temperatures +2.5/+3.0 K).

| Feature | PSI (nominal) | PSI (drifted) | Severity | KS p (drifted) |
|---|---|---|---|---|
| Air temperature [K] | 0.0024 | **2.7645** | significant | 6.0e-321 |
| Process temperature [K] | 0.0066 | **2.2648** | significant | 3.6e-321 |
| Rotational speed [rpm] | 0.0045 | **2.8620** | significant | 1.7e-321 |
| Torque [Nm] | 0.0079 | **1.2398** | significant | 7.3e-321 |
| Tool wear [min] | 0.0134 | 0.0134 | stable | 0.21 |
| Type (chi-square) | p=0.27 | p=0.27 | stable | — |

| Batch | drift_score | flagged | retrain? |
|---|---|---|---|
| Nominal test split | **0.0000** | 0 / 6 | no |
| Simulated drift | **0.6667** | 4 / 6 | **YES** |

The monitor correctly stays quiet on same-distribution data and fires the
**retraining trigger** on the shifted batch.

### 3.6 Tests

```
$ python -m pytest -q
33 passed, 1 warning in 11.93s
```

Fast and training-free — the API tests monkeypatch the models.

---

## 4. Business framing (cost model)

| Assumption | Value | Rationale |
|---|---|---|
| Cost of a **missed failure** (`COST_FN`) | $8,000 | unplanned stop: lost throughput + emergency labour + expedited parts |
| Cost of a **false alarm** (`COST_FP`) | $300 | ~45 min planned inspection, no lost production |
| Cost ratio | 27:1 | drives the threshold well below 0.5 |

Both are **assumptions**, overridable via env vars `DT_COST_FN` / `DT_COST_FP`.
The threshold is *recomputed* from whatever costs are supplied, so an ops team
can plug in their own numbers and the pipeline stays correct.

---

## 5. Repository layout

```
downtime-forecast-mlops/
├── src/
│   ├── config.py            paths, cost model, drift thresholds
│   ├── data.py              load, pydantic schema, features, splits, daily series
│   ├── decision.py          cost-sensitive threshold search
│   ├── drift.py             PSI + KS + chi-square + retraining trigger + CLI
│   ├── registry.py          champion registry (models/registry.json)
│   ├── tracking.py          MLflow OR JSON-file tracker (same API)
│   ├── pipeline.py          end-to-end train→evaluate→register→report
│   ├── models/
│   │   ├── classifier.py    calibrated HistGBDT / XGBoost
│   │   ├── anomaly.py       IsolationForest (healthy-only)
│   │   └── forecast.py      naive / seasonal-naive / SES / GBDT-lags
│   └── serving/app.py       FastAPI service
├── tests/                   33 fast tests (schema, features, cost, drift, registry, API)
├── airflow/dags/            Airflow-2.x DAG (importable without airflow)
├── data/raw/ai4i2020.csv    raw dataset (+ data/README.md provenance)
├── models/                  champion/ artifacts + registry.json (generated)
├── reports/                 evaluation.md, drift.md, figures/ (generated)
├── runs/                    JSON-tracker fallback runs (generated)
├── Dockerfile               API image
├── docker-compose.yml       OPTIONAL airflow + mlflow stack
├── Makefile
├── requirements.txt / requirements-extras.txt
└── LICENSE (MIT, Ratul S Raj, 2026)
```

---

## 6. Reproduction commands

```bash
cd downtime-forecast-mlops

# install core deps (only what the pipeline/tests/API need)
pip install -r requirements.txt

# (optional) heavier extras: mlflow + xgboost
pip install -r requirements-extras.txt

# prepare data (validate, features, splits, synthetic daily series)
make data          # == python -c "from src import data; data.prepare()"

# FULL end-to-end: train -> evaluate -> register champion -> write reports
make pipeline      # == python -m src.pipeline

# drift monitoring only
make drift         # == python -m src.drift

# fast tests (no training)
make test          # == python -m pytest -q

# run the API
make serve         # == uvicorn src.serving.app:app --port 8000
curl -s localhost:8000/health
curl -s localhost:8000/model-info
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"air_temp":298,"process_temp":308,"rotational_speed":1500,
       "torque":40,"tool_wear":10,"type":"M"}'
curl -s "localhost:8000/forecast?days=14"
```

### API surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + models_loaded |
| GET | `/model-info` | champion version + metrics + git sha |
| POST | `/predict` | single-reading failure probability + decision |
| POST | `/predict/batch` | up to 1000 readings |
| POST | `/anomaly` | IsolationForest score + flag |
| GET | `/forecast?days=n` | forward downtime-minutes forecast |

All requests log **latency** (`X-Process-Time-ms` header + log line); requests
are validated by pydantic (`422` on bad `type` / out-of-range values).

---

## 7. Tracking backends

`src/tracking.py` exposes one API (`start_run` → `log_params/log_metrics/
log_artifact/log_dict`) with two interchangeable backends:

* **MLflow** — used automatically when `mlflow` is importable. This repo's
  reference run used **MLflow 3.18.0** (sqlite backend; the plain-filesystem
  backend is deprecated in MLflow 3.x).
* **JSON-file fallback** — writes `runs/<run_id>/{params,metrics,meta}.json`
  plus `artifacts/`. Used automatically when `mlflow` is absent, so the pipeline
  never depends on a heavy install.

Both log all three model runs. You can see real fallback artifacts in `runs/`.

---

## 8. Orchestration

* **Local**: `make pipeline` (the same command the DAG shells out to).
* **Airflow**: `airflow/dags/downtime_pipeline.py` is a valid Airflow-2.x DAG
  (`BashOperator`/`PythonOperator`) over the identical CLI entrypoints
  (`python -m src.pipeline`, `python -m src.drift`). It is **importable without
  Airflow installed** (guarded imports), so CI and `pytest` are unaffected.
* **docker-compose.yml**: an optional full stack (Airflow standalone + an
  MLflow tracking server). **Not used in CI** — it exists for anyone who wants
  the real scheduler/server locally.

---

## 9. Limitations (honest)

1. **Synthetic time index.** AI4I has no timestamps. The daily downtime series
   is manufactured (trend + weekly/annual seasonality + Poisson/Gamma noise)
   with a fixed seed. Forecasting results here demonstrate the *machinery*, not
   a real plant signal — and indeed a naive baseline wins on MAE.
   `data/README.md` documents this in full.
2. **No real sensor stream.** The API is fed by request JSON, not a live
   PLC/SCADA feed. There is no streaming ingestion, no message queue.
3. **MAPE is unreliable** on this series (zero-downtime days ⇒ division by
   near-zero). Trust MAE/RMSE.
4. **Cost figures are assumptions**, not measured plant economics.
5. **Drift batch is simulated** by perturbing feature distributions; real drift
   would be detected from production telemetry.
6. **No automated rollback** of a bad champion — the registry records history
   but promotion is unconditional in the pipeline run.
7. The IsolationForest is deliberately weak (ROC-AUC 0.83) and is a safety net,
   not a primary detector.

---

## 10. License

MIT © 2026 Ratul S Raj — see [`LICENSE`](LICENSE).

Dataset: AI4I 2020 Predictive Maintenance Dataset, UCI ML Repository id **601**
(DOI 10.24432/C5HS5C) — see [`data/README.md`](data/README.md).
