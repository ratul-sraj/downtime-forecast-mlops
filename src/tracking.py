"""Unified experiment-tracking interface.

Two interchangeable backends behind one API:

  * ``MLflowTracker``  - used automatically when ``mlflow`` is importable.
  * ``JsonFileTracker``- the always-available fallback that writes
    ``runs/<run_id>/{params,metrics}.json`` plus an ``artifacts/`` folder.

Both expose::

    with tracker.start_run("name", tags=...) as run:
        run.log_params({...})
        run.log_metrics({...})
        run.log_artifact(path)
        run.log_dict(obj, "file.json")

so the pipeline never branches on which backend is active.
"""
from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from . import config

# Silence MLflow's agent hint banner before mlflow is ever imported.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Run:
    """Backend-agnostic run handle."""

    def __init__(self, run_id: str, name: str, backend: "BaseTracker"):
        self.run_id = run_id
        self.name = name
        self.backend = backend

    def log_params(self, params: Dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError

    def log_metrics(self, metrics: Dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError

    def log_artifact(self, path) -> None:  # pragma: no cover
        raise NotImplementedError

    def log_dict(self, obj: Any, filename: str) -> None:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# JSON fallback
# ---------------------------------------------------------------------------
class _JsonRun(Run):
    def __init__(self, run_id, name, backend, tags):
        super().__init__(run_id, name, backend)
        self.dir = config.RUNS_DIR / run_id
        (self.dir / "artifacts").mkdir(parents=True, exist_ok=True)
        self._meta = {"run_id": run_id, "name": name, "start": _now(),
                      "tags": tags, "backend": "json"}

    def _write_meta(self):
        (self.dir / "meta.json").write_text(json.dumps(self._meta, indent=2))

    def log_params(self, params):
        (self.dir / "params.json").write_text(json.dumps(params, indent=2, default=str))

    def log_metrics(self, metrics):
        (self.dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    def log_artifact(self, path):
        p = Path(path)
        if p.exists():
            (self.dir / "artifacts" / p.name).write_bytes(p.read_bytes())

    def log_dict(self, obj, filename):
        (self.dir / "artifacts" / filename).write_text(
            json.dumps(obj, indent=2, default=str)
        )

    def _close(self):
        self._meta["end"] = _now()
        self._write_meta()


class JsonFileTracker:
    name = "json"

    @contextmanager
    def start_run(self, run_name: str, tags: Dict[str, Any] | None = None):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        run = _JsonRun(run_id, run_name, self, tags or {})
        config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            yield run
        finally:
            run._close()
        print(f"[tracking:json] run '{run_name}' -> {run.dir}")


# ---------------------------------------------------------------------------
# MLflow backend (used only if mlflow imports)
# ---------------------------------------------------------------------------
class _MlflowRun(Run):
    def __init__(self, run_id, name, backend, mlflow):
        super().__init__(run_id, name, backend)
        self._mlflow = mlflow

    def log_params(self, params):
        self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics):
        clean = {k: float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float))}
        self._mlflow.log_metrics(clean)

    def log_artifact(self, path):
        p = Path(path)
        if p.exists():
            self._mlflow.log_artifact(str(p))

    def log_dict(self, obj, filename):
        self._mlflow.log_dict(obj, filename)


class MLflowTracker:
    name = "mlflow"

    def __init__(self, experiment: str = "downtime-forecast"):
        import mlflow  # noqa

        self._mlflow = mlflow
        # MLflow 3.x deprecated the plain-filesystem backend, so point at a
        # local sqlite DB (still zero-infrastructure, single file).
        db_path = (config.ROOT / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment(experiment)

    @contextmanager
    def start_run(self, run_name: str, tags: Dict[str, Any] | None = None):
        active = self._mlflow.start_run(run_name=run_name, tags=tags or {})
        run = _MlflowRun(active.info.run_id, run_name, self, self._mlflow)
        try:
            yield run
        finally:
            self._mlflow.end_run()
        print(f"[tracking:mlflow] run '{run_name}' id={active.info.run_id}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_tracker(experiment: str = "downtime-forecast"):
    """Return the MLflow tracker if importable, else the JSON fallback."""
    try:
        import mlflow  # noqa

        return MLflowTracker(experiment)
    except Exception as exc:  # noqa: BLE001
        print(f"[tracking] mlflow unavailable ({exc.__class__.__name__}); using JSON backend")
        return JsonFileTracker()


def backend_name() -> str:
    """Cheap check that avoids importing the heavy mlflow package."""
    import importlib.util

    return "mlflow" if importlib.util.find_spec("mlflow") else "json"
