"""Champion registry.

``models/registry.json`` records which model version is the current *champion*
plus the full promotion history. The serving layer and the API read the champion
through here so there is a single source of truth.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from . import config


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(config.ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return "uncommitted"


def load_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or config.REGISTRY_PATH
    if not path.exists():
        return {"champion": None, "history": []}
    return json.loads(path.read_text())


def save_champion(
    metrics: Dict[str, float],
    run_id: str,
    version: str,
    decision: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Promote a model to champion, appending to history."""
    path = path or config.REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    reg = load_registry(path)

    entry = {
        "version": version,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "metrics": {k: float(v) for k, v in metrics.items()
                    if isinstance(v, (int, float))},
        "decision": decision,
    }
    if extra:
        entry.update(extra)

    reg["champion"] = entry
    reg.setdefault("history", []).append(entry)
    path.write_text(json.dumps(reg, indent=2))
    return entry


def get_champion(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return the champion entry, or None if no model has been registered."""
    return load_registry(path).get("champion")
