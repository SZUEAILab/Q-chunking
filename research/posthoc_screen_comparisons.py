"""Describe registered posthoc contrasts from terminal, validated run receipts.

This does not repeat the upstream lifecycle validation. It binds its inputs and
independently recomputes curve metrics. Four-seed intervals are descriptive.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(plan, tasks, preregistration, report):
    seeds = plan["seeds"]
    if (len(seeds) != len(set(seeds)) or not 1 <= len(seeds) <= 6
            or seeds != tasks["seeds"] or seeds != preregistration["training_seeds"]):
        raise ValueError("Registered seed identities differ")
    steps = plan["evaluation_steps"]
    if (steps != preregistration["evaluation_steps"] or steps[0] != 0
            or len(steps) < 3 or np.any(np.diff(steps) <= 0)
            or steps[-1] != tasks["common"]["online_steps"]
            or steps[-1] != preregistration["budget_per_run"]
            or steps != list(range(0, steps[-1] + 1, tasks["common"]["eval_interval"]))):
        raise ValueError("Registered evaluation grid differs")
    if (tasks["common"]["eval_seed"] != preregistration["evaluation_seed"]
            or tasks["common"]["eval_episodes"] != preregistration["evaluation_episodes"]
            or not tasks["common"]["eval_at_start"]):
        raise ValueError("Registered evaluation protocol differs")
    expected = {}
    for task in tasks["tasks"]:
        cfg = dict(tasks["common"], **task)
        if cfg["name"] in expected:
            raise ValueError("Duplicate task identity")
        for key in ("online_steps", "eval_interval", "eval_seed", "eval_episodes", "eval_at_start"):
            if cfg[key] != tasks["common"][key]:
                raise ValueError("Task changes common evaluation protocol")
        expected[cfg["name"]] = cfg
    registered = []
    for task in plan["tasks"]:
        if set(task["methods"]) != set(preregistration["methods"]):
            raise ValueError("Method set differs from frozen screening")
        for name in task["methods"].values():
            if name not in expected or expected[name]["env_name"] != task["env_name"]:
                raise ValueError("Registered task/environment identity differs")
            registered.append(name)
    if len(registered) != len(set(registered)) or set(registered) != set(expected):
        raise ValueError("Missing or duplicate registered method identities")
    contrasts = plan["contrasts"]
    pairs = [(r["left"], r["right"]) for r in contrasts]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate contrast")
    primary = {f"{r['left']} - {r['right']}" for r in contrasts if r["family"] == "original_primary"}
    if primary != set(preregistration["primary_comparisons"]):
        raise ValueError("Original primary comparison family changed")
    for row in contrasts:
        if (row["left"] == row["right"] or row["left"] not in preregistration["methods"]
                or row["right"] not in preregistration["methods"]
                or row["family"] not in ("original_primary", "secondary_mechanism", "secondary_recipe")):
            raise ValueError("Invalid registered contrast")
    total = len(expected) * len(seeds)
    if (total != plan["expected_runs"] or total != preregistration["expected_runs"]
            or report["expected_runs"] != total or report["source_commit"] != plan["source_commit"]):
        raise ValueError("Cohort/source identity differs")
    if report["status"] not in ("complete", "needs_review"):
        raise ValueError("Only terminal cohort reports may be analyzed")
    records = {}
    for row in report["records"]:
        key = (row["name"], row["seed"])
        if key in records or key[0] not in expected or key[1] not in seeds:
            raise ValueError("Unexpected or duplicate run identity")
        if row["status"] == "running":
            raise ValueError("Receipt is still running")
        records[key] = row
    complete = sum(r["status"] == "complete" for r in records.values())
    if complete != report["complete_runs"] or (report["status"] == "complete" and complete != total):
        raise ValueError("Terminal completion count differs")
    return expected, records


def validate_curve(row, plan):
    if row.get("source_commit") != plan["source_commit"] or row.get("returncode") != 0:
        raise ValueError("Receipt source or real exit code differs")
    if row["budget"] != plan["evaluation_steps"][-1] or row["eval_steps"] != plan["evaluation_steps"]:
        raise ValueError("Run budget or evaluation axis differs")
    success = np.asarray(row["success"], dtype=float)
    if (success.shape != (len(plan["evaluation_steps"]),) or not np.isfinite(success).all()
            or np.any((success < 0) | (success > 1))):
        raise ValueError("Invalid success curve")
    steps = np.asarray(plan["evaluation_steps"])
    metrics = dict(auc=float(np.trapezoid(success, steps) / steps[-1]),
                   endpoint=float(success[-1]), tail3=float(success[-3:].mean()))
    for name, value in metrics.items():
        if not np.isclose(row[name], value, atol=1e-12, rtol=0):
            raise ValueError("Receipt metric differs from curve: " + name)
    reached = steps[success >= 0.5]
    event = int(reached[0]) if len(reached) else None
    if row["time_to_50"] != event or row["time_to_50_censored"] != (event is None):
        raise ValueError("Threshold event/censoring differs from curve")
    elapsed = row["ended"] - row["started"]
    if not np.isfinite([row["started"], row["ended"], elapsed]).all() or elapsed < 0:
        raise ValueError("Invalid run duration")
    return dict(status="validated_curve", name=row["name"], seed=row["seed"],
                source_commit=row["source_commit"], metrics=metrics, success=success.tolist(),
                time_to_50=event, time_to_50_censored=event is None, elapsed_seconds=elapsed,
                effective_config=row["effective_config"], checkpoint_sha256=row["checkpoint_sha256"],
                effective_config_sha256=row["effective_config_sha256"],
                observed_eval_episode_counts=row["observed_eval_episode_counts"])


def paired_summary(differences):
    values = np.asarray(differences, dtype=float)
    # Enumerate all n**n resamples at n=4; this removes Monte Carlo variation,
    # not finite-sample uncertainty or the limitations of percentile intervals.
    indices = np.asarray(list(itertools.product(range(len(values)), repeat=len(values))))
    interval = np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()
    return dict(paired_differences=values.tolist(), mean=float(values.mean()),
                median=float(np.median(values)), positive_pairs=int((values > 0).sum()),
                zero_pairs=int((values == 0).sum()), negative_pairs=int((values < 0).sum()),
                descriptive_paired_bootstrap_percentile_95=interval,
                enumerated_resamples=len(indices))


def analyze(plan, tasks, preregistration, report):
    expected, upstream = contract(plan, tasks, preregistration, report)
    rows = []
    for name in expected:
        for seed in plan["seeds"]:
            row = upstream.get((name, seed))
            value = dict(name=name, seed=seed, status="unavailable",
                         upstream_status=row["status"] if row else "missing")
            if row and row["status"] == "complete":
                try:
                    value = validate_curve(row, plan)
                except (ValueError, KeyError, TypeError) as error:
                    value.update(status="needs_review", error=repr(error))
            rows.append(value)
    indexed = {(r["name"], r["seed"]): r for r in rows}
    comparisons = []
    for task in plan["tasks"]:
        for contrast in plan["contrasts"]:
            left = [indexed[task["methods"][contrast["left"]], s] for s in plan["seeds"]]
            right = [indexed[task["methods"][contrast["right"]], s] for s in plan["seeds"]]
            available = [s for s, a, b in zip(plan["seeds"], left, right, strict=True)
                         if a["status"] == b["status"] == "validated_curve"]
            result = dict(task=task["task"], env_name=task["env_name"], **contrast,
                          expected_seeds=plan["seeds"], available_seeds=available,
                          status="complete" if available == plan["seeds"] else "incomplete")
            if result["status"] == "complete":
                result["metrics"] = {}
                for metric in ("auc", "endpoint", "tail3"):
                    a, b = [r["metrics"][metric] for r in left], [r["metrics"][metric] for r in right]
                    result["metrics"][metric] = dict(left_values=a, right_values=b,
                        left_mean=float(np.mean(a)), right_mean=float(np.mean(b)),
                        **paired_summary(np.asarray(a) - b))
                result["mean_curve_difference"] = (np.mean([r["success"] for r in left], axis=0)
                                                   - np.mean([r["success"] for r in right], axis=0)).tolist()
                result["threshold_pairs"] = [dict(seed=s, left_event=a["time_to_50"],
                    left_censored=a["time_to_50_censored"], right_event=b["time_to_50"],
                    right_censored=b["time_to_50_censored"])
                    for s, a, b in zip(plan["seeds"], left, right, strict=True)]
                result["effective_config_differences"] = [dict(seed=s, differences={
                    k: dict(left=a["effective_config"].get(k), right=b["effective_config"].get(k))
                    for k in sorted(set(a["effective_config"]) | set(b["effective_config"]))
                    if a["effective_config"].get(k) != b["effective_config"].get(k)})
                    for s, a, b in zip(plan["seeds"], left, right, strict=True)]
            else:
                result["unavailable_pairs"] = [dict(seed=s, left_status=a["status"], right_status=b["status"])
                    for s, a, b in zip(plan["seeds"], left, right, strict=True) if s not in available]
            comparisons.append(result)
    count = sum(r["status"] == "validated_curve" for r in rows)
    return dict(status="complete" if count == len(rows) and report["status"] == "complete" else "needs_review",
        source_commit=plan["source_commit"], expected_runs=len(rows), validated_curve_runs=count,
        records=rows, comparisons=comparisons, evaluation_steps=plan["evaluation_steps"],
        interpretation="Four-seed descriptive screening, not confirmatory inference. Secondary contrasts were registered after training started, before interim outcome inspection. No causal mechanism is established by a single contrast.",
        limitations=["Trusts the terminal upstream lifecycle validator; does not independently restore checkpoints or reread raw CSVs.",
                     "Evaluation episode counts were not recorded by this frozen driver; configured counts are not observed counts.",
                     "Intervals use the empirical paired bootstrap, with no simultaneous coverage or confirmatory significance claim.",
                     "Censored threshold times remain separate; no average over only successful runs.",
                     "Run durations on shared GPUs do not measure exclusive GPU compute hours."])


def markdown_report(report):
    lines = ["# Posthoc terminal screening comparisons", "", report["interpretation"], "",
             "All values below are percentage points. Incomplete pairs have no group aggregate.", "",
             "| Task | Family | Left − right | AUC difference | Positive pairs | Endpoint difference |",
             "|---|---|---|---:|---:|---:|"]
    for row in report["comparisons"]:
        if row["status"] == "complete":
            auc = row["metrics"]["auc"]
            values = (f"{100*auc['mean']:+.3f}", f"{auc['positive_pairs']}/{len(row['expected_seeds'])}",
                      f"{100*row['metrics']['endpoint']['mean']:+.3f}")
        else:
            values = ("incomplete", f"{len(row['available_seeds'])} available", "incomplete")
        lines.append(f"| {row['task']} | {row['family']} | {row['left']} − {row['right']} | {' | '.join(values)} |")
    lines += ["", "Limitations:", "", *["- " + text for text in report["limitations"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "task-config", "preregistration", "manifest", "cohort-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".md").exists():
        raise FileExistsError("Preserve existing analysis outputs before reanalysis")
    paths = (args.plan, args.task_config, args.preregistration, args.manifest, args.cohort_report)
    hashes = {str(p): digest(p) for p in paths}
    plan, tasks, prereg, manifest, report = [json.loads(p.read_text()) for p in paths]
    if (digest(args.task_config) != manifest["tasks_sha256"] or manifest["source_commit"] != plan["source_commit"]
            or digest(args.preregistration) != plan["preregistration_sha256"]):
        raise ValueError("Frozen source, task config or preregistration binding differs")
    result = analyze(plan, tasks, prereg, report)
    if any(digest(Path(p)) != value for p, value in hashes.items()):
        raise ValueError("Analysis inputs changed during execution")
    result.update(tool_sha256=digest(Path(__file__)), input_hashes=hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".md").open("x") as handle:
        handle.write(markdown_report(result))
    print(json.dumps({k: result[k] for k in ("status", "expected_runs", "validated_curve_runs")}))
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
