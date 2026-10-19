"""Verify native preservation and original posthoc latent updates against upstream."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import flax
import jax

from agents.rlpd_posthoc_reference import RLPDPosthocReferenceAgent, get_config
from agents.rlpd_reference import UPSTREAM_COMMIT
from research.check_rlpd_reference import load_upstream, run_case


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_definitions():
    native = [dict(family="native", utd=u, backup_entropy=b, subset=s)
              for u, b, s in ((1, True, 2), (20, True, 2), (1, False, 1),
                              (20, False, 2), (1, True, None))]
    native.append(dict(family="native", utd=1, backup_entropy=True, subset=2, full_width=True))
    posthoc = [dict(family="posthoc", utd=u, backup_entropy=b, subset=s)
               for u, b, s in ((1, True, 2), (20, True, 2), (1, False, 1), (1, True, None))]
    posthoc.append(dict(family="posthoc", utd=20, backup_entropy=True, subset=2, full_width=True))
    posthoc.append(dict(family="posthoc", utd=1, backup_entropy=True, subset=2, common_entropy=True))
    return native + posthoc


def run_worker(args, hashes):
    definition = dict(case_definitions()[args.case_index])
    family = definition.pop("family")
    common_entropy = definition.pop("common_entropy", False)
    learner, upstream_files = load_upstream(args.upstream_root)
    if family == "posthoc":
        config = get_config().to_dict()
        semantic = {key: config[key] for key in ("agent_name", "adapter_representation", "adapter_speed_squash",
                    "adapter_speed_interpretation", "adapter_critic_space", "ds_speed_bound", "ds_regularizer")}
        if common_entropy:
            semantic["target_entropy"] = -2.5
        definition.update(action_dim=6, agent_class=RLPDPosthocReferenceAgent, config_overrides=semantic)
    case = run_case(learner, **definition)
    return dict(status="passed", case_index=args.case_index, family=family, case=case,
                local_source_hashes=hashes, upstream_files=upstream_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-index", type=int, choices=range(12))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous numerical verification before rerunning")
    if jax.default_backend() != "cpu":
        raise ValueError("Run numerical verification with JAX_PLATFORMS=cpu")
    source = Path(__file__).resolve().parents[1]
    inputs = [source / name for name in ("agents/rlpd_reference.py", "agents/rlpd_posthoc_reference.py",
              "research/check_rlpd_reference.py", "research/check_rlpd_posthoc_reference.py")]
    hashes = {str(p.relative_to(source)): digest(p) for p in inputs}
    if args.case_index is not None:
        report = run_worker(args, hashes)
        if any(digest(p) != hashes[str(p.relative_to(source))] for p in inputs):
            raise ValueError("Worker source changed during execution")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(case_index=args.case_index, status=report["status"])))
        return
    # Each case has the same fixed seed and three updates as the combined check.
    # Isolated processes release compiled code and mappings before the next case.
    case_root = args.output.with_suffix(".cases")
    case_root.mkdir(parents=True, exist_ok=False)
    receipts = []
    environment = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for index, definition in enumerate(case_definitions()):
        output = case_root / f"case-{index:02d}.json"
        command = [sys.executable, "-m", "research.check_rlpd_posthoc_reference", "--upstream-root",
                   str(args.upstream_root.resolve()), "--output", str(output.resolve()), "--case-index", str(index)]
        print(json.dumps(dict(case_index=index, status="started", definition=definition)), flush=True)
        with (case_root / f"case-{index:02d}.log").open("x") as handle:
            result = subprocess.run(command, cwd=source, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        (case_root / f"case-{index:02d}.exit.json").write_text(json.dumps(dict(returncode=result.returncode)) + "\n")
        if result.returncode:
            raise RuntimeError(f"Numerical case {index} failed; preserve its log and exit receipt")
        receipt = json.loads(output.read_text())
        if (receipt["case_index"] != index or receipt["family"] != definition["family"]
                or receipt["status"] != "passed" or receipt["local_source_hashes"] != hashes):
            raise ValueError("Worker result identity differs")
        receipts.append(receipt)
        print(json.dumps(dict(case_index=index, status="passed", maximum_state_error=receipt["case"]["max_absolute_state_error"])), flush=True)
    upstream_files = receipts[0]["upstream_files"]
    if any(r["upstream_files"] != upstream_files for r in receipts):
        raise ValueError("Upstream source changed across workers")
    native = [r["case"] for r in receipts if r["family"] == "native"]
    cases = [r["case"] for r in receipts if r["family"] == "posthoc"]
    if any(digest(p) != hashes[str(p.relative_to(source))] for p in inputs):
        raise ValueError("Verification source changed during execution")
    report = dict(status="passed", upstream_commit=UPSTREAM_COMMIT, upstream_files=upstream_files,
        local_source_hashes=hashes, jax_version=jax.__version__, flax_version=flax.__version__,
        native_cases=native, posthoc_cases=cases, execution="one independent CPU process per case",
        scope="Native five-dimensional learner preserved. Posthoc six-dimensional latent learner states/updates match unmodified upstream instantiated in six dimensions; the environment execution map is verified separately. Not an upstream posthoc method or task-performance comparison.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], native_cases=len(native), posthoc_cases=len(cases),
                         maximum_state_error=max(c["max_absolute_state_error"] for c in native + cases))))


if __name__ == "__main__":
    main()
