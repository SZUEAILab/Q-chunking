#!/usr/bin/env python3
"""Generate comparison plots for parallel_benchmark.md.

Reads data from docs/data/parallel_benchmark/ and produces:
  - parallel_bench_total.png    — total it/s comparison
  - parallel_bench_per_task.png — per-task it/s comparison
  - parallel_bench_speedup.png  — speedup ratio comparison
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_DIR = Path("docs/data/parallel_benchmark")
IMG_DIR  = Path("docs/images")

# ── Load data ──────────────────────────────────────────────────────────────

def load_results(path):
    with open(path) as f:
        return json.load(f)

r6000    = load_results(DATA_DIR / "RTX6000Ada_48GB/results.json")
r4090_tn = load_results(DATA_DIR / "RTX4090_24GB/results.json")
r4090_ds = load_results(DATA_DIR / "RTX4090_24GB/results_ds_stereographic.json")

datasets = {
    "6000 Ada — TanhNormal": {
        "data": r6000, "color": "#2166ac", "marker": "o", "linestyle": "-",
        "status_ok": lambda r: r["status"] == "OK",
    },
    "4090 — TanhNormal": {
        "data": r4090_tn, "color": "#4dac26", "marker": "s", "linestyle": "-",
        "status_ok": lambda r: r["status"] == "OK",
    },
    "4090 — DS stereographic": {
        "data": r4090_ds, "color": "#d73027", "marker": "D", "linestyle": "-",
        "status_ok": lambda r: r["status"] == "OK",
    },
}

# ── Plot 1: Total throughput ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5.5))
for label, ds in datasets.items():
    results = ds["data"]["results"]
    xs = [r["concurrency"] for r in results if ds["status_ok"](r)]
    ys = [r["total_it_s"] for r in results if ds["status_ok"](r)]
    ax.plot(xs, ys, marker=ds["marker"], color=ds["color"],
            linestyle=ds["linestyle"], linewidth=2, markersize=7,
            label=label)

ax.set_xlabel("Concurrency (N tasks)", fontsize=12)
ax.set_ylabel("Total it/s", fontsize=12)
ax.set_title("Total Throughput Comparison", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", framealpha=0.85, fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 12.5)

out = IMG_DIR / "parallel_bench_total.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {out}")


# ── Plot 2: Per-task throughput ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5.5))
for label, ds in datasets.items():
    results = ds["data"]["results"]
    xs = [r["concurrency"] for r in results if ds["status_ok"](r)]
    ys = [r["per_proc"] for r in results if ds["status_ok"](r)]
    ax.plot(xs, ys, marker=ds["marker"], color=ds["color"],
            linestyle=ds["linestyle"], linewidth=2, markersize=7,
            label=label)

    # Annotate single-task baseline
    if ys:
        ax.annotate(f"{ys[0]}", xy=(xs[0], ys[0]),
                    xytext=(xs[0] + 0.4, ys[0] + 5),
                    fontsize=8, color=ds["color"], fontweight="bold")

ax.set_xlabel("Concurrency (N tasks)", fontsize=12)
ax.set_ylabel("Avg it/s per task", fontsize=12)
ax.set_title("Per-Task Throughput (Efficiency Drop)", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", framealpha=0.85, fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 12.5)

out = IMG_DIR / "parallel_bench_per_task.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {out}")


# ── Plot 3: Speedup ratio ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5.5))
for label, ds in datasets.items():
    results = ds["data"]["results"]
    xs = [r["concurrency"] for r in results if ds["status_ok"](r)]
    ys = [r["scale"] for r in results if ds["status_ok"](r)]
    ax.plot(xs, ys, marker=ds["marker"], color=ds["color"],
            linestyle=ds["linestyle"], linewidth=2, markersize=7,
            label=label)

# Ideal linear scaling line
max_x = max(max(r["concurrency"] for r in ds["data"]["results"] if ds["status_ok"](r))
            for ds in datasets.values())
ideal_x = np.arange(1, max_x + 1)
ax.plot(ideal_x, ideal_x, "--", color="gray", alpha=0.5, linewidth=1, label="Ideal linear")

ax.set_xlabel("Concurrency (N tasks)", fontsize=12)
ax.set_ylabel("Speedup (× vs single task)", fontsize=12)
ax.set_title("Parallel Speedup Ratio", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", framealpha=0.85, fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 12.5)

out = IMG_DIR / "parallel_bench_speedup.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {out}")


# ── Print summary table ────────────────────────────────────────────────────

print("\n## Comparison Summary\n")
print("| GPU | Method | Single it/s | Max N | Max total it/s | Speedup |")
print("|-----|--------|:----------:|:-----:|:-------------:|:-------:|")
for label, ds in datasets.items():
    results = [r for r in ds["data"]["results"] if ds["status_ok"](r)]
    if not results:
        continue
    s = results[0]   # single task
    m = results[-1]  # max concurrency
    print(f"| {label} | {s['total_it_s']} | {m['concurrency']} | "
          f"{m['total_it_s']} | {m['scale']:.1f}× |")

print("\n✅ Done!")
