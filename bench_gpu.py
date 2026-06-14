#!/usr/bin/env python3
"""GPU concurrency stress test — uses schedule.py for task orchestration.

Tests concurrency 1→10 on a single GPU. Each concurrency level launches N
identical tasks (10k online steps, stereographic DS, no eval) and measures
wall-clock throughput. Uses schedule.py's compiled JSON + status tracking
for clean launch/recovery.
"""

import subprocess, sys, os, json, time, re, tempfile, shutil, argparse
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
GPU_ID          = 0
GPU_TASKS_BASE  = 1       # --gpu_tasks for each test (set to N)
TOTAL_TASKS     = 10      # max concurrency to test
ENV_NAME        = "cube-triple-play-singletask-task2-v0"
DS_MODE         = "stereographic"
ONLINE_STEPS    = 10000
HORIZON         = 1        # H=1, no chunking overhead
EVAL_EPISODES   = 0        # skip eval entirely

OUTPUT_PLOT     = "docs/images/gpu_bench_4090.png"

TASKS_JSON      = "tasks_bench.json"
COMPILED_JSON   = "tasks_bench.compiled.json"


def write_tasks_json(n_tasks: int):
    """Write a tasks.json with N identical tasks for the given concurrency level.

    Uses a single task definition × N seeds = N commands.
    """
    config = {
        "common": {
            "entry": "main_online.py",
            "env_name": ENV_NAME,
            "horizon_length": HORIZON,
            "ds_mode": DS_MODE,
            "online_steps": ONLINE_STEPS,
            "eval_episodes": EVAL_EPISODES,
            "eval_interval": 0,
            "save_interval": -1,
            "log_interval": 100000,
            "run_group": "Bench",
        },
        "tasks": [{"name": f"bench_c{n_tasks}"}],
        "seeds": list(range(n_tasks)),
    }
    with open(TASKS_JSON, "w") as f:
        json.dump(config, f, indent=2)


def reset_compiled_status():
    """Reset all command statuses to 'pending' so schedule.py re-runs them."""
    with open(COMPILED_JSON) as f:
        compiled = json.load(f)
    for c in compiled["commands"]:
        c["status"] = "pending"
        c["pid"] = None
        c["log"] = None
        c["gpu"] = None
    with open(COMPILED_JSON, "w") as f:
        json.dump(compiled, f, indent=2)


def run_scheduler(gpu_tasks: int) -> dict:
    """Run schedule.py with given gpu_tasks, parse output for timing."""
    cmd = [
        sys.executable, "-u", "schedule.py",
        "--run", COMPILED_JSON,
        "--gpus", str(GPU_ID),
        "--gpu_tasks", str(gpu_tasks),
        "--stagger", "0",  # no stagger for benchmark
    ]

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    t1 = time.perf_counter()
    total_wall = t1 - t0

    stdout = proc.stdout
    stderr = proc.stderr

    # Parse schedule.py log for per-task timing
    # Pattern: [HH:MM:SS] ▶ GPU=0  <name>  seed=<s>  pid=<p>  [N/total]
    start_pat = re.compile(
        r"\[(\d{2}:\d{2}:\d{2})\]\s*▶\s*GPU=\d+\s+(\S+)\s+seed=(\d+)"
    )
    # Pattern: [HH:MM:SS] ✓ GPU=0  <name>  seed=<s>  ...
    end_pat = re.compile(
        r"\[(\d{2}:\d{2}:\d{2})\]\s*✓\s*GPU=\d+\s+(\S+)\s+seed=(\d+)"
    )

    starts = {}  # (name, seed) → datetime
    ends = {}    # (name, seed) → datetime

    today = datetime.now().strftime("%Y-%m-%d ")
    for line in stdout.splitlines():
        m = start_pat.search(line)
        if m:
            t = datetime.strptime(today + m.group(1), "%Y-%m-%d %H:%M:%S")
            key = (m.group(2), int(m.group(3)))
            starts[key] = t

        m = end_pat.search(line)
        if m:
            t = datetime.strptime(today + m.group(1), "%Y-%m-%d %H:%M:%S")
            key = (m.group(2), int(m.group(3)))
            ends[key] = t

    # Compute wall clock from first start to last finish
    if starts and ends:
        first_start = min(starts.values())
        last_finish = max(ends.values())
        measured_wall = (last_finish - first_start).total_seconds()
    else:
        measured_wall = total_wall  # fallback

    n_tasks = len(ends)  # successfully completed tasks

    total_its = (n_tasks * ONLINE_STEPS) / measured_wall if measured_wall > 0 else 0
    avg_its = total_its / n_tasks if n_tasks > 0 else 0

    # Check for failures
    failed = 0
    for line in stdout.splitlines():
        if "✗" in line or "FAIL" in line:
            failed += 1

    return {
        "concurrency": gpu_tasks,
        "wall_clock": measured_wall,
        "total_its": total_its,
        "avg_its": avg_its,
        "n_completed": n_tasks,
        "failed": failed,
    }


def main():
    parser = argparse.ArgumentParser(description="GPU concurrency stress test via schedule.py")
    parser.add_argument("--max", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default=OUTPUT_PLOT)
    parser.add_argument("--keep-logs", action="store_true",
                        help="Keep per-task log files")
    args = parser.parse_args()

    global GPU_ID
    GPU_ID = args.gpu

    print(f"GPU {GPU_ID} | {DS_MODE} | {ONLINE_STEPS} steps | {ENV_NAME} | H={HORIZON}")
    print(f"No warmup, no eval, no checkpoints")
    print(f"Using schedule.py for task orchestration\n")

    results = []

    for n in range(1, args.max + 1):
        print(f"\n{'='*60}")
        print(f"  Concurrency = {n:2d}  ({n} tasks, --gpu_tasks={n})")
        print(f"{'='*60}")

        # 1. Write tasks.json
        write_tasks_json(n)

        # 2. Compile
        subprocess.run(
            [sys.executable, "schedule.py", "--tasks_config", TASKS_JSON, "--compile"],
            check=True, capture_output=True,
        )

        # 3. Reset and run
        reset_compiled_status()
        r = run_scheduler(gpu_tasks=n)
        results.append(r)

        print(f"  wall={r['wall_clock']:.1f}s  "
              f"total={r['total_its']:.0f} it/s  "
              f"avg={r['avg_its']:.0f} it/s  "
              f"completed={r['n_completed']}/{n}"
              + (f"  failed={r['failed']}" if r['failed'] else ""))

    # ── Summary table ──
    print(f"\n{'─'*60}")
    print(f"{'N':>3s}  {'Wall(s)':>8s}  {'Total it/s':>10s}  {'Avg it/s':>10s}  {'Done':>5s}  {'Failed':>6s}")
    print(f"{'─'*60}")
    for r in results:
        print(f"{r['concurrency']:3d}  {r['wall_clock']:8.1f}  {r['total_its']:10.0f}  "
              f"{r['avg_its']:10.0f}  {r['n_completed']:5d}  {r['failed']:6d}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n⚠ matplotlib not available, skipping plot")
        # Cleanup
        for f in [TASKS_JSON, COMPILED_JSON]:
            Path(f).unlink(missing_ok=True)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    xs = np.array([r["concurrency"] for r in results])
    total_its = np.array([r["total_its"] for r in results])
    avg_its = np.array([r["avg_its"] for r in results])
    wall = np.array([r["wall_clock"] for r in results])

    # Left: total throughput
    color_total = "#2166ac"
    ax1.plot(xs, total_its, "o-", color=color_total, linewidth=2, markersize=8,
             label="Total it/s")
    ax1.set_xlabel("Concurrency (N tasks)", fontsize=12)
    ax1.set_ylabel("Total it/s", fontsize=12, color=color_total)
    ax1.tick_params(axis="y", labelcolor=color_total)
    ax1.set_title("Total Throughput", fontsize=13, fontweight="bold")

    # Mark peak
    best_idx = np.argmax(total_its)
    best_n = xs[best_idx]
    best_total = total_its[best_idx]
    ax1.annotate(f"Peak: N={best_n}\n{best_total:.0f} it/s",
                 xy=(best_n, best_total),
                 xytext=(best_n + 0.8, best_total * 0.88),
                 arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.5),
                 fontsize=10, color="#b2182b", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Right: per-task + wall clock
    color_avg = "#4dac26"
    ax2.plot(xs, avg_its, "s-", color=color_avg, linewidth=2, markersize=8,
             label="Avg it/s per task")
    ax2.set_xlabel("Concurrency (N tasks)", fontsize=12)
    ax2.set_ylabel("Avg it/s per task", fontsize=12, color=color_avg)
    ax2.tick_params(axis="y", labelcolor=color_avg)

    ax2b = ax2.twinx()
    color_wall = "#d73027"
    ax2b.plot(xs, wall, "D--", color=color_wall, linewidth=1.5, markersize=6,
              alpha=0.6, label="Wall clock")
    ax2b.set_ylabel("Wall clock (s)", fontsize=12, color=color_wall)
    ax2b.tick_params(axis="y", labelcolor=color_wall)
    ax2.set_title("Per-Task Speed & Wall Time", fontsize=13, fontweight="bold")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2b, labels2b = ax2b.get_legend_handles_labels()
    fig.legend(lines1 + lines2 + lines2b, labels1 + labels2 + labels2b,
               loc="upper center", ncol=4, framealpha=0.85, fontsize=9,
               bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(f"RTX 4090 — GPU Concurrency Stress Test\n"
                 f"{DS_MODE} DS, H={HORIZON}, {ONLINE_STEPS} steps, no warmup — schedule.py",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Plot saved to {out_path}")

    print(f"\n  Recommended concurrency: {best_n} tasks "
          f"(peak total throughput {best_total:.0f} it/s)")

    # Cleanup temp files
    for f in [TASKS_JSON, COMPILED_JSON]:
        Path(f).unlink(missing_ok=True)

    if not args.keep_logs:
        import shutil
        log_dir = Path("exp_logs")
        if log_dir.exists():
            shutil.rmtree(log_dir)
        exp_dir = Path("exp/bench") if Path("exp/bench").exists() else None
        # keep exp dir for reference


if __name__ == "__main__":
    main()
