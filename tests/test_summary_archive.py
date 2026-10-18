import csv
import json

import pytest

from scripts.archive_experiment_summaries import run_integrity


def make_run(path, *, steps=(0, 50, 100), success=0.0, incident=""):
    (path / "flags.json").write_text(json.dumps({"online_steps": 100, "eval_interval": 50}))
    (path / "run_status.json").write_text(json.dumps({
        "status": "complete", "completed_step": 100, "run_id": "task/method/seed1",
    }))
    (path / "incidents.jsonl").write_text(incident)
    with (path / "eval.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "success", "episode.return"])
        writer.writeheader()
        writer.writerows({"step": step, "success": success, "episode.return": 0}
                         for step in steps)


def test_complete_zero_success_is_valid_evidence(tmp_path):
    make_run(tmp_path)
    result = run_integrity(tmp_path)
    assert result["compact_integrity_passed"]
    assert result["scientific_review"] == "not inferred by this exporter"


@pytest.mark.parametrize("steps", [(50, 100), (0, 50, 50, 100), (0, 100)])
def test_missing_or_duplicate_evaluations_are_disclosed(tmp_path, steps):
    make_run(tmp_path, steps=steps)
    assert "evaluation grid mismatch" in run_integrity(tmp_path)["issues"]


def test_nonfinite_metric_is_not_archived_as_clean(tmp_path):
    make_run(tmp_path, success=float("nan"))
    assert not run_integrity(tmp_path)["compact_integrity_passed"]


def test_incident_is_preserved_as_review_required(tmp_path):
    make_run(tmp_path, incident='{"event": "error"}\n')
    assert not run_integrity(tmp_path)["compact_integrity_passed"]
