import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from research.posthoc_screen_comparisons import analyze, contract, digest, markdown_report, paired_summary


@pytest.fixture
def cohort():
    common = dict(online_steps=300, eval_interval=100, eval_seed=42,
                  eval_episodes=50, eval_at_start=True)
    tasks = dict(common=common, seeds=[10, 11, 12, 13], tasks=[
        dict(name="native", env_name="env", ds_mode="none"),
        dict(name="posthoc", env_name="env", ds_mode="adapter")])
    prereg = dict(training_seeds=tasks["seeds"], evaluation_steps=[0, 100, 200, 300],
                  budget_per_run=300, evaluation_seed=42, evaluation_episodes=50,
                  expected_runs=8, methods=["native", "legacy_latent"],
                  primary_comparisons=["legacy_latent - native"])
    plan = dict(seeds=tasks["seeds"], evaluation_steps=prereg["evaluation_steps"],
                expected_runs=8, source_commit="frozen-source", tasks=[
                    dict(task="task", env_name="env", methods=dict(native="native", legacy_latent="posthoc"))],
                contrasts=[dict(left="legacy_latent", right="native", family="original_primary", scope="full recipe"),
                           dict(left="native", right="legacy_latent", family="secondary_mechanism", scope="test reversal")])
    rows = []
    for task in tasks["tasks"]:
        for seed in tasks["seeds"]:
            success = np.array([0., .1, .2, .3]) + (.2 if task["name"] == "posthoc" else 0.)
            rows.append(dict(name=task["name"], seed=seed, source_commit="frozen-source", status="complete",
                returncode=0, budget=300, eval_steps=[0, 100, 200, 300], success=success.tolist(),
                auc=float(np.trapezoid(success, [0, 100, 200, 300]) / 300), endpoint=float(success[-1]),
                tail3=float(success[-3:].mean()), time_to_50=300 if task["name"] == "posthoc" else None,
                time_to_50_censored=task["name"] != "posthoc", started=10., ended=20.,
                effective_config=dict(action_dim=6 if task["name"] == "posthoc" else 5),
                checkpoint_sha256="a" * 64, effective_config_sha256="b" * 64,
                observed_eval_episode_counts="not_recorded_in_frozen_driver; flags_only"))
    report = dict(status="complete", expected_runs=8, complete_runs=8, source_commit="frozen-source", records=rows)
    return plan, tasks, prereg, report


def test_all_seeds_zero_axis_censoring_and_family_preserved(cohort):
    result = analyze(*cohort)
    assert result["status"] == "complete" and result["validated_curve_runs"] == 8
    row, reverse = result["comparisons"]
    assert row["family"] == "original_primary" and reverse["family"] == "secondary_mechanism"
    assert row["metrics"]["auc"]["mean"] == pytest.approx(.2)
    assert row["metrics"]["auc"]["positive_pairs"] == 4
    assert reverse["metrics"]["auc"]["mean"] == pytest.approx(-.2)
    assert row["mean_curve_difference"] == pytest.approx([.2] * 4)
    assert all(r["right_censored"] and r["right_event"] is None for r in row["threshold_pairs"])
    assert all(r["left_event"] == 300 and not r["left_censored"] for r in row["threshold_pairs"])
    assert row["effective_config_differences"][0]["differences"]["action_dim"] == dict(left=6, right=5)
    assert "+20.000" in markdown_report(result)


def test_missing_pair_does_not_create_partial_group_estimate(cohort):
    plan, tasks, prereg, report = cohort
    report["records"].pop(); report.update(status="needs_review", complete_runs=7)
    result = analyze(plan, tasks, prereg, report)
    assert result["status"] == "needs_review"
    for row in result["comparisons"]:
        assert row["available_seeds"] == [10, 11, 12]
        assert "metrics" not in row and "mean_curve_difference" not in row
    assert "incomplete" in markdown_report(result)


@pytest.mark.parametrize("problem", ["source", "exit", "axis", "success", "auc", "censor", "duration"])
def test_invalid_complete_receipt_is_retained_not_averaged(cohort, problem):
    row = cohort[3]["records"][0]
    if problem == "source": row["source_commit"] = "other"
    elif problem == "exit": row["returncode"] = 7
    elif problem == "axis": row["eval_steps"] = [100, 200, 300]
    elif problem == "success": row["success"][0] = float("nan")
    elif problem == "auc": row["auc"] = .8
    elif problem == "censor": row["time_to_50"] = 300
    elif problem == "duration": row["ended"] = 1.
    result = analyze(*cohort)
    assert result["status"] == "needs_review"
    assert result["records"][0]["status"] == "needs_review"
    assert "metrics" not in result["comparisons"][0]


def test_failed_seed_preserved_without_replacement(cohort):
    report = cohort[3]; report["records"][0].update(status="failed", returncode=7)
    report.update(status="needs_review", complete_runs=7)
    result = analyze(*cohort)
    assert result["records"][0]["upstream_status"] == "failed"
    assert result["comparisons"][0]["unavailable_pairs"][0]["seed"] == 10


@pytest.mark.parametrize("problem", ["seed", "grid", "task", "primary", "duplicate", "running", "count"])
def test_frozen_contract_rejects_scope_changes(cohort, problem):
    plan, tasks, prereg, report = cohort
    if problem == "seed": plan["seeds"] = [20, 21, 22, 23]
    elif problem == "grid": plan["evaluation_steps"] = [100, 200, 300]
    elif problem == "task": plan["tasks"][0]["methods"]["native"] = "other"
    elif problem == "primary": plan["contrasts"][0]["family"] = "secondary_mechanism"
    elif problem == "duplicate": report["records"].append(copy.deepcopy(report["records"][0]))
    elif problem == "running": report["records"][0]["status"] = "running"
    elif problem == "count": report["complete_runs"] = 7
    with pytest.raises(ValueError): contract(plan, tasks, prereg, report)


def test_all_zero_results_and_enumerated_descriptive_interval(cohort):
    for row in cohort[3]["records"]:
        row.update(success=[0.] * 4, auc=0., endpoint=0., tail3=0., time_to_50=None, time_to_50_censored=True)
    result = analyze(*cohort)
    summary = result["comparisons"][0]["metrics"]["auc"]
    assert summary["zero_pairs"] == 4 and summary["positive_pairs"] == 0
    assert summary["descriptive_paired_bootstrap_percentile_95"] == [0., 0.]
    assert summary["enumerated_resamples"] == 256
    assert paired_summary([-1, 0, 0, 1])["median"] == 0


def test_upstream_needs_review_is_not_promoted_to_complete(cohort):
    cohort[3]["status"] = "needs_review"
    assert analyze(*cohort)["status"] == "needs_review"


def test_cli_binds_inputs_and_refuses_overwrite(cohort, tmp_path):
    plan, tasks, prereg, report = cohort
    inputs = {}
    for name, value in [("tasks", tasks), ("prereg", prereg), ("report", report)]:
        inputs[name] = tmp_path / (name + ".json"); inputs[name].write_text(json.dumps(value))
    plan["preregistration_sha256"] = digest(inputs["prereg"])
    manifest = dict(source_commit=plan["source_commit"], tasks_sha256=digest(inputs["tasks"]))
    for name, value in [("plan", plan), ("manifest", manifest)]:
        inputs[name] = tmp_path / (name + ".json"); inputs[name].write_text(json.dumps(value))
    output = tmp_path / "result.json"
    command = [sys.executable, "-m", "research.posthoc_screen_comparisons"]
    for flag, name in [("plan", "plan"), ("task-config", "tasks"), ("preregistration", "prereg"),
                       ("manifest", "manifest"), ("cohort-report", "report")]:
        command += ["--" + flag, str(inputs[name])]
    command += ["--output", str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    result = json.loads(output.read_text())
    assert result["input_hashes"][str(inputs["report"])] == digest(inputs["report"])
    before = output.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0 and "Preserve existing" in second.stderr
    assert output.read_bytes() == before
    inputs["tasks"].write_text(json.dumps(dict(tasks, seeds=[20, 21, 22, 23])))
    modified = [str(tmp_path / "modified.json") if word == str(output) else word for word in command]
    third = subprocess.run(modified, capture_output=True, text=True)
    assert third.returncode != 0 and "binding differs" in third.stderr
    assert not (tmp_path / "modified.json").exists()
