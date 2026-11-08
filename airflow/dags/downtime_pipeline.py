"""Airflow DAG: end-to-end downtime-forecast MLOps pipeline.

The DAG simply shells out to the *same CLI entrypoints* a developer runs by hand
(``python -m src.pipeline``, ``python -m src.drift``), so there is zero logic
duplication between local runs and scheduled runs.

This module is importable **without Airflow installed**: if the airflow imports
fail, ``dag`` is set to ``None`` and importing the file is a no-op. That keeps it
safe for CI, where Airflow is intentionally not installed (see README).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

# --- guarded Airflow imports -------------------------------------------------
try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import PythonOperator
    from airflow.utils.dates import days_ago  # noqa: F401

    AIRFLOW_AVAILABLE = True
except Exception:  # noqa: BLE001  (ModuleNotFoundError when airflow isn't installed)
    AIRFLOW_AVAILABLE = False
    DAG = None  # type: ignore
    BashOperator = None  # type: ignore
    PythonOperator = None  # type: ignore


# Repo root: assumes this DAG lives at <repo>/airflow/dags/downtime_pipeline.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable


def _log_summary():
    """Emit the champion metrics into the Airflow task log."""
    sys.path.insert(0, REPO_ROOT)
    from src import registry  # local import so the module imports w/o airflow too

    champ = registry.get_champion()
    if champ:
        print("Champion version:", champ["version"])
        print("Metrics:", champ["metrics"])
    else:
        print("No champion registered yet.")


default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "depends_on_past": False,
}

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="downtime_forecast_pipeline",
        description="Train, evaluate, register and monitor the downtime models",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["mlops", "predictive-maintenance"],
    ) as dag:

        prepare = BashOperator(
            task_id="prepare_data",
            bash_command=f"cd {REPO_ROOT} && {PY} -c 'from src import data; data.prepare()'",
        )

        train = BashOperator(
            task_id="train_and_register",
            bash_command=f"cd {REPO_ROOT} && {PY} -m src.pipeline",
        )

        drift = BashOperator(
            task_id="drift_check",
            bash_command=f"cd {REPO_ROOT} && {PY} -m src.drift",
        )

        summarize = PythonOperator(
            task_id="summarize",
            python_callable=_log_summary,
        )

        prepare >> train >> drift >> summarize
else:  # pragma: no cover
    dag = None
