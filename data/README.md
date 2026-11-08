# Data

## Raw dataset

`raw/ai4i2020.csv` — **AI4I 2020 Predictive Maintenance Dataset** (10,000 rows).

| Field | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Dataset ID | **601** |
| URL | https://archive.ics.uci.edu/dataset/601/ai4i2020 |
| DOI | 10.24432/C5HS5C |
| Rows / columns | 10,000 × 14 |
| Positive class | `Machine failure` = 1 in **339** rows (**3.39%**) |
| File size | 522,048 bytes |

### Citation

> Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset* [Data set].
> UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

### Columns

`UDI, Product ID, Type (L/M/H), Air temperature [K], Process temperature [K],
Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure,`
plus the five failure-mode flags `TWF, HDF, PWF, OSF, RNF`.

The five modes are *sub-labels* of `Machine failure`; the pipeline predicts the
overall `Machine failure` label and does **not** leak the mode flags into the
feature set.

### Provenance / licence

The dataset is distributed via the UCI ML Repository for research use. See the
source page for the dataset's own terms. This repo redistributes the CSV for
reproducibility of the portfolio project only.

---

## Derived features (`src/data.py::build_features`)

| Feature | Definition |
|---|---|
| `temp_delta` | `Process temperature − Air temperature` (heat removed by the process) |
| `power` | `Torque × (rpm × 2π / 60)` — mechanical power in watts |
| `wear_rate` | `Tool wear / (|power| + ε)` — wear accrued per unit of work |
| `torque_per_speed` | `Torque / rpm` — a load ratio that spikes near HDF/PWF |
| `type_L/M/H` | one-hot of the product quality variant |

---

## Synthetic time index (IMPORTANT — read this)

The AI4I dataset is **tabular and has no timestamps**, so it cannot support a
forecasting task on its own. To exercise a genuine, ordered forecasting problem
we synthesise a daily series over **730 days** in
`_simulate_daily_downtime()`:

1. The observed failure rate (~3.4%) fixes a base of ~0.47 failures/day, scaled
   by the fleet the 10k rows represent.
2. A slow **upward trend** (+15% across the horizon, "equipment ageing"), a
   **weekly** seasonality (weekday load) and an **annual** seasonality are
   multiplied in, then gaussian noise.
3. Daily failure counts are drawn as **Poisson(λ_t)** and converted to downtime
   minutes via a per-failure **Gamma(shape=4, scale=60)** stop (mean ≈ 240 min =
   4 h, consistent with the cost model in `src/config.py`).

Output: `processed/daily_downtime.csv` with columns `date, failures,
downtime_minutes`. Generation is **deterministic** (fixed RNG seed = 42).

> ⚠️ **This time index is synthetic.** It is grounded in the real failure rate
> but the trend/seasonality/noise are manufactured to create a learnable
> signal. No claim is made that it reflects real plant downtime. A production
> deployment would replace it with a real sensor/event stream (see README
> "Limitations"). Because the series is noise-dominated with zero-downtime
> days, **MAPE is unreliable here** (division by near-zero) — MAE/RMSE are the
> metrics to trust, and this is stated in the report.

---

## Reproduce

```bash
python -c "from src import data; data.prepare()"
# or
make data
```
