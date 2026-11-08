"""Champion registry resolution."""
import json

from src import registry


def test_load_registry_empty(tmp_path):
    p = tmp_path / "registry.json"
    reg = registry.load_registry(p)
    assert reg["champion"] is None
    assert reg["history"] == []


def test_save_and_get_champion(tmp_path):
    p = tmp_path / "registry.json"
    entry = registry.save_champion(
        metrics={"pr_auc": 0.9, "roc_auc": 0.97},
        run_id="run-1",
        version="v1.0.0",
        decision={"threshold": 0.087},
        path=p,
    )
    assert entry["version"] == "v1.0.0"
    champ = registry.get_champion(p)
    assert champ["run_id"] == "run-1"
    assert champ["metrics"]["pr_auc"] == 0.9
    assert champ["decision"]["threshold"] == 0.087


def test_history_appends(tmp_path):
    p = tmp_path / "registry.json"
    registry.save_champion({"pr_auc": 0.8}, "r1", "v1", {"threshold": 0.5}, path=p)
    registry.save_champion({"pr_auc": 0.9}, "r2", "v2", {"threshold": 0.1}, path=p)
    reg = json.loads(p.read_text())
    assert len(reg["history"]) == 2
    # champion is the latest
    assert registry.get_champion(p)["version"] == "v2"
