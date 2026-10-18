"""Stream executed-action secondary diagnostics after a cohort reaches terminal.

No learner or GPU is loaded. Covariance is marginal over visited states and time;
boundary occupancy does not identify how often a pre-clipped action was clipped.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np


TOLERANCE = 1e-6


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


class Moments:
    """Merge centered block moments; memory does not grow with trajectory length."""

    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.count = 0
        self.mean = np.zeros(dimensions + 1)
        self.centered = np.zeros((dimensions + 1, dimensions + 1))
        self.bound_counts = np.zeros(dimensions, dtype=np.int64)
        self.xyz_cube_count = self.sphere_count = self.outside_sphere_count = 0
        self.norm_min = self.norm_max = None

    def update(self, actions):
        if len(actions) == 0:
            return
        norms = np.linalg.norm(actions[:, :3], axis=1)
        values = np.column_stack([actions, norms])
        count = len(values)
        mean = values.mean(axis=0)
        centered = values - mean
        delta = mean - self.mean
        total = self.count + count
        self.centered += centered.T @ centered + np.outer(delta, delta)*(self.count*count/total)
        self.mean += delta*(count/total)
        self.count = total
        near_cube = np.abs(actions) >= 1 - TOLERANCE
        self.bound_counts += near_cube.sum(axis=0)
        self.xyz_cube_count += int(near_cube[:, :3].any(axis=1).sum())
        self.sphere_count += int((np.abs(norms - 1) <= TOLERANCE).sum())
        self.outside_sphere_count += int((norms > 1 + TOLERANCE).sum())
        self.norm_min = float(norms.min()) if self.norm_min is None else min(self.norm_min, float(norms.min()))
        self.norm_max = float(norms.max()) if self.norm_max is None else max(self.norm_max, float(norms.max()))

    def result(self):
        covariance = self.centered/(self.count-1) if self.count >= 2 else None
        return dict(count=self.count,
                    action_mean=self.mean[:-1].tolist() if self.count else None,
                    action_sample_covariance=covariance[:-1, :-1].tolist() if covariance is not None else None,
                    xyz_covariance_eigenvalues=np.linalg.eigvalsh(covariance[:3,:3]).tolist() if covariance is not None else None,
                    xyz_norm_mean=float(self.mean[-1]) if self.count else None,
                    xyz_norm_sample_variance=float(covariance[-1,-1]) if covariance is not None else None,
                    xyz_norm_min=self.norm_min, xyz_norm_max=self.norm_max,
                    component_cube_boundary_fraction=(self.bound_counts/self.count).tolist() if self.count else None,
                    xyz_cube_boundary_fraction=self.xyz_cube_count/self.count if self.count else None,
                    unit_sphere_boundary_fraction=self.sphere_count/self.count if self.count else None,
                    outside_unit_sphere_fraction=self.outside_sphere_count/self.count if self.count else None)


def stream_actions(path, *, dimensions, budget, warmup, window=100000, block_size=8192):
    """Require every environment step; split random warmup from policy collection."""
    if dimensions < 3 or budget < 1 or warmup < 0 or window < 1 or block_size < 1:
        raise ValueError("Invalid action-stream contract")
    path = Path(path)
    before = path.stat()
    checksum = hashlib.sha256()
    all_actions, random_actions, policy_actions = [Moments(dimensions) for _ in range(3)]
    windows = [Moments(dimensions) for _ in range((budget+window-1)//window)]
    expected_step, processed, block = 1, 0, []

    def consume():
        nonlocal processed
        if not block:
            return
        values = np.asarray(block, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite action or logged norm")
        actions = values[:, :dimensions]
        if np.any(np.abs(actions) > 1 + 1e-5):
            raise ValueError("Executed action exceeds cube bound")
        norms = np.linalg.norm(actions[:, :3], axis=1)
        linf = np.max(np.abs(actions[:, :3]), axis=1)
        if not np.allclose(norms, values[:, -2], atol=2e-6, rtol=2e-6):
            raise ValueError("Logged translation norm disagrees with action coordinates")
        if not np.allclose(linf, values[:, -1], atol=2e-6, rtol=2e-6):
            raise ValueError("Logged translation Linf disagrees with action coordinates")
        steps = np.arange(processed+1, processed+1+len(actions))
        all_actions.update(actions)
        random_actions.update(actions[steps <= warmup])
        policy = steps > warmup
        policy_actions.update(actions[policy])
        if policy.any():
            indices = (steps-1)//window
            for index in np.unique(indices[policy]):
                windows[index].update(actions[policy & (indices == index)])
        processed += len(actions)
        block.clear()

    with path.open("rb") as raw:
        def lines():
            for line in raw:
                checksum.update(line)
                yield line.decode("utf-8")
        reader = csv.DictReader(lines())
        fields = [f"action_{i}" for i in range(dimensions)] + ["action_xyz_norm", "action_xyz_linf"]
        if not reader.fieldnames or not set(["step", *fields]).issubset(reader.fieldnames):
            raise ValueError("Missing per-component executed-action columns")
        for row in reader:
            if None in row or int(row["step"]) != expected_step or expected_step > budget:
                raise ValueError("Missing, repeated, unordered or extra environment step")
            block.append([float(row[name]) for name in fields])
            expected_step += 1
            if len(block) >= block_size:
                consume()
        consume()
    if processed != budget:
        raise ValueError(f"Incomplete environment stream: {processed}/{budget}")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Environment log changed during analysis")
    if policy_actions.count != max(0, budget-warmup):
        raise ValueError("Wrong post-warmup sample count")
    return dict(env_csv_sha256=checksum.hexdigest(), env_csv_bytes=after.st_size,
                rows=processed, dimensions=dimensions, warmup_last_step=warmup,
                threshold_tolerance=TOLERANCE, covariance_ddof=1,
                all_actions=all_actions.result(), random_warmup=random_actions.result(),
                learned_policy=policy_actions.result(),
                learning_windows=[dict(first_step=i*window+1,last_step=min(budget,(i+1)*window),
                                       policy_first_step=max(warmup+1,i*window+1) if value.count else None, **value.result())
                                  for i,value in enumerate(windows)])


def analyze_run(record, expected, source_commit):
    if record.get("status") != "complete" or record.get("returncode") != 0:
        raise ValueError("Run has no validated successful training exit")
    if record.get("source_commit") != source_commit:
        raise ValueError("Wrong training source identity")
    directory = Path(record["run_dir"])
    identity_hashes = {name:digest(directory/name) for name in ("flags.json","effective_agent_config.json")}
    flags = json.loads((directory/"flags.json").read_text())
    config = json.loads((directory/"effective_agent_config.json").read_text())
    if flags["seed"] != record["seed"] or config != record["effective_config"]:
        raise ValueError("Receipt/config identity mismatch")
    for name in ("env_name","run_group","online_steps","start_training","horizon_length","ds_mode",
                 "ds_adapter_critic_space","ds_adapter_speed_squash","ds_adapter_speed_interpretation",
                 "ds_speed_bound","eval_seed","eval_episodes","actor_update_mode","sparse"):
        if name in expected and flags.get(name) != expected[name]:
            raise ValueError(f"Frozen task/flag mismatch: {name}")
    if record.get("budget") != flags["online_steps"]:
        raise ValueError("Receipt budget mismatch")
    elapsed = float(record["ended"]) - float(record["started"])
    if not np.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Invalid receipt wall duration")
    geometry = stream_actions(directory/"env.csv", dimensions=int(config["env_action_dim"]),
                              budget=flags["online_steps"], warmup=flags["start_training"])
    if any(digest(directory/name)!=value for name,value in identity_hashes.items()):
        raise ValueError("Run identity files changed during analysis")
    return dict(name=record["name"],seed=record["seed"],status="geometry_complete",run_dir=str(directory),
                source_commit=source_commit,flags_sha256=identity_hashes["flags.json"],
                effective_config_sha256=identity_hashes["effective_agent_config.json"],
                run_wall_seconds=elapsed, env_steps_per_run_wall_second=flags["online_steps"]/elapsed,
                cuda_visible_devices=record.get("cuda_visible_devices"), geometry=geometry)


def analyze_cohort(report, tasks, source_commit, *, workers=1):
    if workers not in range(1,5):
        raise ValueError("Use one to four CPU workers")
    expected = {}
    for task in tasks["tasks"]:
        config = dict(tasks["common"]); config.update(task)
        for seed in tasks["seeds"]:
            key = (task["name"], seed)
            if key in expected:
                raise ValueError("Duplicate expected run identity")
            expected[key] = config
    if not expected or report["source_commit"] != source_commit or report["expected_runs"] != len(expected):
        raise ValueError("Upstream source or cohort size mismatch")
    if sum(r.get("status")=="complete" for r in report["records"]) != report["complete_runs"]:
        raise ValueError("Upstream completion count disagrees with its records")
    actual = {}
    for record in report["records"]:
        key = (record["name"],record["seed"])
        if key not in expected or key in actual:
            raise ValueError("Unexpected or duplicate upstream run")
        actual[key] = record
    output, jobs = {}, []
    for key, config in expected.items():
        row = actual.get(key)
        if not row or row.get("status") != "complete":
            output[key] = dict(name=key[0],seed=key[1],status="upstream_unavailable",
                               upstream_status=row.get("status") if row else "missing")
        else:
            jobs.append((key,row,config))
    def collect(key, operation):
        try:
            output[key] = operation()
        except Exception as error:
            output[key] = dict(name=key[0],seed=key[1],status="needs_review",error=repr(error))
    if workers == 1:
        for key,row,config in jobs:
            collect(key,lambda row=row,config=config:analyze_run(row,config,source_commit))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            pending={pool.submit(analyze_run,row,config,source_commit):key for key,row,config in jobs}
            for future in as_completed(pending):
                collect(pending[future],future.result)
    records = [output[key] for key in expected]
    groups = []
    for task in tasks["tasks"]:
        rows = [output[task["name"],seed] for seed in tasks["seeds"]]
        valid = [r for r in rows if r["status"] == "geometry_complete"]
        group = dict(name=task["name"],expected_seeds=tasks["seeds"],
                     available_seeds=[r["seed"] for r in valid],
                     status="complete" if len(valid)==len(rows) else "incomplete")
        if len(valid)==len(rows):
            group["per_seed_run_wall_seconds"] = [r["run_wall_seconds"] for r in valid]
            for metric in ("xyz_norm_mean","xyz_cube_boundary_fraction","unit_sphere_boundary_fraction","outside_unit_sphere_fraction"):
                values = [r["geometry"]["learned_policy"][metric] for r in valid]
                group[f"per_seed_{metric}"] = values
                group[f"mean_{metric}"] = float(np.mean(values)) if all(v is not None for v in values) else None
            covariances = [r["geometry"]["learned_policy"]["action_sample_covariance"] for r in valid]
            group["mean_within_run_action_covariance"] = np.mean(covariances,axis=0).tolist() if all(v is not None for v in covariances) else None
        groups.append(group)
    complete = sum(r["status"]=="geometry_complete" for r in records)
    return dict(source_commit=source_commit,expected_runs=len(expected),geometry_complete_runs=complete,
                status="complete" if complete==len(expected) and report["status"]=="complete" else "needs_review",
                records=records,groups=groups,
                interpretation="Descriptive executed-action marginals across visited states and time; not conditional exploration covariance or a causal mechanism test.",
                wall_time_scope="Receipt elapsed includes source check, process startup, JIT, data loading, training, evaluation and saving; excludes subsequent validation. Shared GPU wall durations are not exclusive GPU compute hours.",
                boundary_scope="Near-boundary occupancy at fixed tolerance; not direct observation of pre-clipping events.")


def verify_source(root, manifest):
    for row in manifest["python_sources"]:
        if digest(root/row["path"]) != row["sha256"]:
            raise ValueError(f"Frozen training file changed: {row['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-report",required=True,type=Path)
    parser.add_argument("--task-config",required=True,type=Path)
    parser.add_argument("--release-manifest",required=True,type=Path)
    parser.add_argument("--source-root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--workers",default=1,type=int,choices=range(1,5))
    args=parser.parse_args()
    inputs = {name:digest(path) for name,path in (("cohort_report",args.cohort_report),
              ("task_config",args.task_config),("release_manifest",args.release_manifest))}
    manifest=json.loads(args.release_manifest.read_text())
    expected_tasks=manifest.get("tasks_sha256",manifest.get("runtime_files",{}).get("remote_tasks.json"))
    if expected_tasks is None or inputs["task_config"]!=expected_tasks:
        raise ValueError("Task configuration is not bound to this frozen release")
    verify_source(args.source_root,manifest)
    report=analyze_cohort(json.loads(args.cohort_report.read_text()),json.loads(args.task_config.read_text()),
                          manifest["source_commit"],workers=args.workers)
    verify_source(args.source_root,manifest)
    if inputs!={name:digest(path) for name,path in (("cohort_report",args.cohort_report),
                ("task_config",args.task_config),("release_manifest",args.release_manifest))}:
        raise ValueError("Analysis contract changed during execution")
    report.update(tool_sha256=digest(Path(__file__)),cohort_report_sha256=inputs["cohort_report"],
                  task_config_sha256=inputs["task_config"],release_manifest_sha256=inputs["release_manifest"],
                  analysis_workers=args.workers)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix(args.output.suffix+".tmp")
    temp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");temp.replace(args.output)
    print(json.dumps({k:report[k] for k in ("status","expected_runs","geometry_complete_runs")}))
    if report["status"]!="complete":
        raise SystemExit(2)


if __name__=="__main__":
    main()
