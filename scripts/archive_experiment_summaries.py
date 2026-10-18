#!/usr/bin/env python3
"""Export small, byte-verified research evidence into the Hugging Face submodule.

The export preserves negative results. File integrity is not a scientific pass.
Weights, trajectories, env/agent logs and operational cloud metadata stay local.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


COHORTS = (
    "rlpd_semantic_subspace_uncertainty_screen_100k_20261018",
    "rlpd_critic_flat_semantic_exploration_screen_100k_20261018",
    "rlpd_state_conditional_controllability_screen_20261018",
)
RUN_FILES = (
    "eval.csv", "flags.json", "run_status.json", "run_manifest.json",
    "runtime_metrics.json", "incidents.jsonl",
)
SUFFIXES = {".csv", ".png", ".pdf", ".json", ".sha256", ".md"}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(directory: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def run_integrity(directory: Path) -> dict:
    """Check a compact run without opening reward-bearing env/agent logs."""
    flags = json.loads((directory / "flags.json").read_text())
    status = json.loads((directory / "run_status.json").read_text())
    budget = flags.get("online_steps")
    interval = flags.get("eval_interval")
    issues = []
    if type(budget) is not int or type(interval) is not int or interval <= 0:
        issues.append("invalid budget/evaluation interval")
    with (directory / "eval.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    steps = []
    for row in rows:
        raw_step = float(row["step"])
        if not math.isfinite(raw_step) or raw_step != int(raw_step):
            issues.append("noninteger/nonfinite evaluation step")
            continue
        steps.append(int(raw_step))
        for name in ("success", "episode.return"):
            if name in row and not math.isfinite(float(row[name])):
                issues.append(f"nonfinite {name}")
    if not issues:
        expected = list(range(0, budget + 1, interval))
        if not expected or expected[-1] != budget:
            expected.append(budget)
        if steps != expected:
            issues.append("evaluation grid mismatch")
    if status.get("status") != "complete" or status.get("completed_step") != budget:
        issues.append("lifecycle/budget incomplete")
    incident = directory / "incidents.jsonl"
    if not incident.exists() or incident.stat().st_size:
        issues.append("incident file missing or nonempty; review required")
    return {
        "run_id": status.get("run_id"), "status": status.get("status"),
        "budget": budget, "evaluation_rows": len(rows), "issues": issues,
        "compact_integrity_passed": not issues,
        "scientific_review": "not inferred by this exporter",
        "checkpoint_validation": "not performed; weights remain on data disk",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-worktree", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records, omitted, runs = [], [], []

    def copy(source: Path, relative: Path, evidence: str) -> None:
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"regular file required: {source}")
        target = destination / relative
        if not target.resolve().is_relative_to(destination):
            raise ValueError(f"archive path escapes destination: {relative}")
        source_hash = sha(source)
        if target.exists() and sha(target) != source_hash:
            raise ValueError(f"refusing to replace different archive bytes: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        if sha(target) != source_hash or sha(source) != source_hash:
            raise ValueError(f"copy/source changed during export: {source}")
        records.append({"path": relative.as_posix(), "source": str(source),
                        "bytes": source.stat().st_size, "sha256": source_hash,
                        "evidence_scope": evidence})

    # Historical analysis tables and plots: retain provenance, not a blanket pass.
    imported = root / "reports" / "repository-import-20261019"
    for source in sorted((root / "reports" / "analysis").glob("*")):
        if source.suffix not in SUFFIXES or source.stat().st_size > 1024 * 1024:
            omitted.append({"source": str(source), "sha256": sha(source),
                            "reason": "long-form/large output retained on data disk"})
            continue
        copy(source, Path("legacy-analysis/images") / source.name,
             "historical analysis; interpretation limited by paired protocol")
    for source in sorted((imported / "results").rglob("*")):
        relative = source.relative_to(imported / "results")
        if not source.is_file() or source.suffix not in SUFFIXES:
            continue
        if relative.parts[0] != "cross_family_noise_screen_20261016" and source.name not in {
            "eight-gpu-interim-summary.json", "cube_double_noise_target_20261016.json",
            "scene_noise_target_20261016.json",
        }:
            continue
        copy(source, Path("legacy-analysis/results") / relative,
             "historical report; preserves pilot, failed and censored boundaries")
    docs_root = args.repo / "docs"
    for relative in (
        "current/research.md", "current/experiments.md", "current/results.md",
        "archive/action_space.md", "archive/noise_shaping.md", "archive/synthetic.md",
        "archive/robotics.md", "archive/results.md", "archive/research_history.md",
    ):
        copy(docs_root / relative, Path("protocols") / relative,
             "documentation snapshot at archival time; historical boundaries retained")

    for source in sorted((root / "reports").rglob("*")):
        relative = source.relative_to(root / "reports")
        if len(relative.parts) > 2 or "layout-v2" in relative.parts[0]:
            continue
        if not source.is_file() or source.suffix not in SUFFIXES:
            continue
        if "20261018" not in relative.parts[0] or source.stat().st_size > 1024 * 1024:
            continue
        copy(source, Path("reports") / relative,
             "reported evidence; consult decision, scientific_role and verification report")

    for cohort in COHORTS:
        statuses = sorted((root / "results" / cohort).rglob("run_status.json"))
        if not statuses:
            raise ValueError(f"no runs for registered cohort: {cohort}")
        for status_path in statuses:
            run = status_path.parent
            record = run_integrity(run)
            record["cohort"] = cohort
            record["archive_directory"] = str(Path("cohorts") / cohort / run.parent.name / run.name)
            runs.append(record)
            for name in RUN_FILES:
                source = run / name
                if source.is_file():
                    copy(source, Path(record["archive_directory"]) / name,
                         "compact raw evaluation/configuration; see run_checks.json")
        for name in (cohort + ".json", cohort + ".compiled.json"):
            source = args.source_worktree / "experiment_configs" / name
            if source.is_file():
                copy(source, Path("cohorts") / cohort / name, "configuration/scheduler snapshot")

    # The main branch and the execution worktree differ. Record both explicitly.
    sources = {}
    for label, directory in (("main", args.repo), ("execution-worktree", args.source_worktree)):
        status = git(directory, "status", "--porcelain=v1", "-uno")
        sources[label] = {"head": git(directory, "rev-parse", "HEAD"),
                          "branch": git(directory, "branch", "--show-current"),
                          "tracked_status": status,
                          "snapshot_role": "current working source, not retroactive run provenance"}
    code_paths = []
    for pattern in ("*.py", "agents/*.py", "direction_speed/**/*.py", "experiment_analysis/*.py",
                    "envs/*.py", "utils/*.py", "rlpd_distributions/*.py", "rlpd_networks/*.py"):
        code_paths.extend(args.source_worktree.glob(pattern))
    for source in sorted(set(code_paths)):
        if "__pycache__" not in source.parts and not source.is_symlink():
            copy(source, Path("source-snapshot") / source.relative_to(args.source_worktree),
                 "execution-worktree source snapshot; original run manifests remain authoritative")
    for source in sorted((args.source_worktree / "docs").glob("*20261018.md")):
        copy(source, Path("protocols/execution-worktree") / source.name, "execution protocol snapshot")

    def write(name: str, value: object) -> None:
        path = destination / name
        content = (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"immutable metadata changed; select a new destination: {path}")
        path.write_bytes(content)

    write("run_checks.json", runs)
    write("source_index.json", sources)
    write("manifest.json", {"schema_version": 1, "archive_date": "2026-10-19",
          "selection_policy": "scientifically informative evidence, including negative results; no outcome selection",
          "not_a_claim": "integrity verification does not imply scientific validity or performance improvement",
          "files": records, "omitted": omitted})
    checksums = destination / "SHA256SUMS"
    checksums.write_text("".join(f"{sha(path)}  {path.relative_to(destination).as_posix()}\n"
                                 for path in sorted(destination.rglob("*"))
                                 if path.is_file() and path != checksums))
    print(json.dumps({"files": len(records), "bytes": sum(x["bytes"] for x in records),
                      "runs": len(runs), "compact_integrity_passed": sum(x["compact_integrity_passed"] for x in runs),
                      "omitted_large_files": len(omitted), "destination": str(destination)}))


if __name__ == "__main__":
    main()
