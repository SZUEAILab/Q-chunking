"""Recompute a hash-bound historical adapter cohort without inventing step zero.

This is retrospective descriptive analysis, not a lifecycle validator. The input
index fixes the original files and expected identities; a new Git checkout is
recorded separately from the older index's data revision.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


STEPS = np.arange(100_000, 1_000_001, 100_000)
SEEDS = tuple(range(10))
EXPECTED = {(f"task{task}", h, method, seed)
            for task in range(1, 6) for h in (1, 5)
            for method in ("none", "adapter") for seed in SEEDS}
PAIR_DIFFERENCES = {"ds_mode", "allow_posthoc_direction_speed_rlpd", "save_dir"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve_metrics(steps, success):
    steps, success = np.asarray(steps), np.asarray(success, dtype=float)
    if not np.array_equal(steps, STEPS):
        raise ValueError("Legacy cohort requires the observed 100k--1M grid")
    if not np.isfinite(success).all() or np.any((success < 0) | (success > 1)):
        raise ValueError("Invalid success rate")
    return dict(endpoint=float(success[-1]), tail3=float(success[-3:].mean()),
                observed_auc=float(np.trapezoid(success, steps) / (steps[-1] - steps[0])))


def analyze(index, data_root, *, bootstrap_seed=20261019, bootstrap_samples=10000):
    """Verify all 200 runs, reconstruct raw outcomes and pair all ten seeds."""
    records, grouped = [], {}
    rows = index["records"]
    keys = [(r["task"], r["horizon"], r["method"], r["seed"]) for r in rows]
    if len(keys) != len(EXPECTED) or set(keys) != EXPECTED:
        raise ValueError("Expected exactly 200 unique historical identities")
    expected_paths = {r["path"] for r in rows}
    discovered = set()
    for path in data_root.glob("task*/h*/rlpd/*/seed*/flags.json"):
        relative = path.parent.relative_to(data_root)
        task, horizon, _, method, seed = relative.parts
        if (task in {f"task{i}" for i in range(1, 6)} and horizon in ("h1", "h5")
                and method in ("none", "posthoc", "adapter") and seed in {f"seed{s}" for s in SEEDS}):
            discovered.add(str(relative))
    if discovered != expected_paths:
        raise ValueError("Historical file set changed or duplicate aliases exist")
    for old in sorted(rows, key=lambda r: (r["task"], r["horizon"], r["method"], r["seed"])):
        if old["issues"]:
            raise ValueError("Input index has unresolved run issues")
        relative = Path(old["path"])
        expected_prefix = (old["task"], f"h{old['horizon']}", "rlpd")
        if relative.is_absolute() or relative.parts[:3] != expected_prefix or len(relative.parts) != 5:
            raise ValueError("Unexpected historical run path")
        allowed = ("none",) if old["method"] == "none" else ("posthoc", "adapter")
        if relative.parts[3] not in allowed or relative.parts[4] != f"seed{old['seed']}":
            raise ValueError("Path disagrees with indexed method/seed")
        directory = data_root / relative
        flags_file, curve_file = directory / "flags.json", directory / "eval.csv"
        for name, path in (("flags", flags_file), ("eval", curve_file)):
            if digest(path) != old[f"{name}_sha256"]:
                raise ValueError(f"Historical {name} file changed: {relative}")
        flags = json.loads(flags_file.read_text())
        if flags != old["flags"]:
            raise ValueError("Indexed flags differ from actual file")
        for name, expected in (("seed", old["seed"]), ("horizon_length", old["horizon"]),
                               ("online_steps", 1_000_000), ("eval_episodes", 100),
                               ("env_name", f"cube-triple-play-singletask-{old['task']}-v0")):
            if flags[name] != expected:
                raise ValueError(f"Historical identity mismatch: {name}")
        if flags["ds_mode"] not in allowed:
            raise ValueError("Historical method flag mismatch")
        with curve_file.open() as stream:
            curve = list(csv.DictReader(stream))
        steps = [int(r["step"]) for r in curve]
        success = [float(r["success"]) for r in curve]
        metrics = curve_metrics(steps, success)
        for name, value in metrics.items():
            if not np.isclose(value, old["metrics"][name], rtol=0, atol=1e-12):
                raise ValueError(f"Independent recomputation disagrees with prior index: {name}")
        row = dict(task=old["task"], horizon=old["horizon"], method=old["method"], seed=old["seed"],
                   path=str(relative), flags_sha256=digest(flags_file), eval_sha256=digest(curve_file),
                   observed_steps=steps, success=success, metrics=metrics,
                   status="evaluation_complete_lifecycle_unverified")
        records.append(row)
        grouped[old["task"], old["horizon"], old["method"], old["seed"]] = (row, flags)
    rng = np.random.default_rng(bootstrap_seed)
    comparisons = []
    for task in range(1, 6):
        for horizon in (1, 5):
            left, right = [], []
            for seed in SEEDS:
                base, base_flags = grouped[f"task{task}", horizon, "none", seed]
                adapter, adapter_flags = grouped[f"task{task}", horizon, "adapter", seed]
                unexpected = [k for k in set(base_flags) | set(adapter_flags)
                              if k not in PAIR_DIFFERENCES and base_flags.get(k) != adapter_flags.get(k)]
                if unexpected:
                    raise ValueError(f"Unmatched paired settings: {unexpected}")
                left.append(base)
                right.append(adapter)
            metrics = {}
            for name in ("endpoint", "tail3", "observed_auc"):
                baseline = np.asarray([r["metrics"][name] for r in left])
                adapter = np.asarray([r["metrics"][name] for r in right])
                difference = adapter - baseline
                indices = rng.integers(0, len(SEEDS), (bootstrap_samples, len(SEEDS)))
                metrics[name] = dict(baseline_mean=float(baseline.mean()), adapter_mean=float(adapter.mean()),
                                     mean_paired_difference=float(difference.mean()),
                                     paired_differences=difference.tolist(), positive_pairs=int((difference > 0).sum()),
                                     zero_pairs=int((difference == 0).sum()),
                                     descriptive_paired_bootstrap_95=np.quantile(difference[indices].mean(axis=1), [.025, .975]).tolist())
            comparisons.append(dict(task=f"task{task}", horizon=horizon, expected_pairs=10,
                                    complete_pairs=10, metrics=metrics))
    return dict(status="evaluation_evidence_reverified", analysis_kind="retrospective_descriptive",
                prior_index_data_git_commit=index["data_git_commit"], verified_runs=len(records),
                verified_files=2*len(records), observed_auc_window=[100000,1000000],
                has_step_zero=False, bootstrap=dict(seed=bootstrap_seed, samples=bootstrap_samples),
                limitations=index["limitations"], records=records, comparisons=comparisons)


def plot_report(report, path):
    """Display all task/horizon groups, including zeros and every paired seed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["comparisons"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, layout="constrained")
    labels = [f"{r['task']}  H={r['horizon']}" for r in rows]
    positions = np.arange(len(rows))
    for y, row in zip(positions, rows, strict=True):
        auc = row["metrics"]["observed_auc"]
        differences = np.asarray(auc["paired_differences"]) * 100
        axes[0].scatter(differences, y + np.linspace(-.12,.12,len(differences)), color="#5b8aad", s=15, alpha=.7)
        mean = 100 * auc["mean_paired_difference"]
        low, high = np.asarray(auc["descriptive_paired_bootstrap_95"]) * 100
        axes[0].plot([low,high],[y,y],color="#163b5c",lw=2)
        axes[0].scatter(mean,y,marker="D",color="#163b5c",s=32)
        endpoint = row["metrics"]["endpoint"]
        base, adapter = 100*endpoint["baseline_mean"], 100*endpoint["adapter_mean"]
        axes[1].plot([base,adapter],[y,y],color="#b2bcc6",lw=2)
        axes[1].scatter(base,y,color="#d27b36",s=32,label="Cartesian" if y==0 else None)
        axes[1].scatter(adapter,y,color="#246d99",s=32,label="Legacy posthoc" if y==0 else None)
    axes[0].axvline(0,color="#888888",lw=1,ls="--")
    axes[0].set_yticks(positions,labels);axes[0].invert_yaxis()
    axes[0].set_xlabel("Paired observed-AUC difference (percentage points)")
    axes[0].set_title("Posthoc − Cartesian: mean, paired seeds, descriptive 95% interval")
    axes[1].set_xlabel("1M endpoint success (%)");axes[1].set_xlim(-3,103)
    axes[1].set_title("Same endpoint definition for both methods");axes[1].legend(loc="lower right")
    for axis in axes:
        axis.grid(axis="x",alpha=.15)
        axis.spines[["top","right"]].set_visible(False)
    figure.suptitle("Historical Cube-Triple: 200 runs, 10 paired seeds per task/horizon",fontsize=14)
    figure.supxlabel("AUC uses observed 100k–1M only; no step-zero imputation. Retrospective, one task family; lifecycle unverified.",fontsize=9)
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path,dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    report = analyze(json.loads(args.index.read_text()), args.data_root)
    report.update(index_sha256=digest(args.index), analyzer_sha256=digest(Path(__file__)),
                  current_data_git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=args.data_root,text=True).strip())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    if args.figure:
        plot_report(report,args.figure)
    print(json.dumps({k:report[k] for k in ("status","verified_runs","verified_files","current_data_git_commit")}))


if __name__ == "__main__":
    main()
