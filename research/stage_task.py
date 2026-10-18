#!/usr/bin/env python3
"""Controlled two-stage diagnostic; run with ``python -m research.stage_task``.

The experiment is deliberately policy-learning-free: it isolates the action
sampling mechanism by asking whether a fixed stochastic policy discovers a
directional target while remaining on a narrow radial shell, then executes a
precision action after the mean has been recentered on the discovered target.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


BASE_SIGMA = 0.08
APPROACH_RADIUS = 0.75
PRECISION_RADIUS = 0.15
DS_TANGENTIAL_STD = 0.25
PRECISION_STD = 0.01
ANGLE_TOLERANCE = np.deg2rad(7.5)
APPROACH_BUDGET = 100
PRECISION_BUDGET = 20
OFFSETS_DEG = (15, 30, 45)
SHELL_TOLERANCES = (0.015, 0.025, 0.05)
FORMAL_SEEDS = tuple(range(100, 116))


@dataclass(frozen=True)
class Method:
    name: str
    family: str
    stage0_scale: float
    stage0_radial: float
    stage1_scale: float
    stage1_radial: float
    project: bool = False


def matching_radial_sigma(base_sigma, tangential_std):
    """Match the square root of the local action-covariance determinant."""
    return base_sigma**3 / tangential_std**2


def local_covariance_volume(tangential_std, radial_std):
    """Return sqrt(det(covariance)) for two tangent and one radial axis."""
    return tangential_std**2 * radial_std


def within_budget_probability(one_step_probability, budget):
    return 1.0 - (1.0 - one_step_probability) ** budget


def target_directions(rng, count, offset_radians):
    azimuth = rng.uniform(0.0, 2.0 * np.pi, size=count)
    return np.column_stack(
        (
            np.full(count, np.cos(offset_radians)),
            np.sin(offset_radians) * np.cos(azimuth),
            np.sin(offset_radians) * np.sin(azimuth),
        )
    )


def cartesian_actions(direction, radius, standard_normal, sigma, project=False):
    actions = radius * direction + sigma * standard_normal
    if project:
        norms = np.linalg.norm(actions, axis=-1, keepdims=True)
        actions = actions / np.maximum(norms, 1.0)
    return actions


def direction_speed_actions(
    direction,
    radius,
    standard_normal,
    direction_scale,
    radial_sigma,
):
    """Sample around e1 or a batch of target directions in tangent coordinates."""
    count = len(standard_normal)
    if direction.ndim == 1:
        direction = np.broadcast_to(direction, (count, 3))
    reference = np.broadcast_to(np.asarray([0.0, 0.0, 1.0]), direction.shape).copy()
    near_pole = np.abs(direction[:, 2]) > 0.9
    reference[near_pole] = np.asarray([0.0, 1.0, 0.0])
    tangent1 = np.cross(direction, reference)
    tangent1 /= np.linalg.norm(tangent1, axis=-1, keepdims=True)
    tangent2 = np.cross(direction, tangent1)
    sampled_direction = (
        direction
        + direction_scale * standard_normal[:, :1] * tangent1
        + direction_scale * standard_normal[:, 1:2] * tangent2
    )
    sampled_direction /= np.linalg.norm(
        sampled_direction, axis=-1, keepdims=True
    )
    sampled_radius = np.clip(
        radius + radial_sigma * standard_normal[:, 2], 0.0, 1.0
    )
    return sampled_direction * sampled_radius[:, None]


def hit_probability(actions, targets, target_radius, angular_tolerance, radial_tolerance):
    radii = np.linalg.norm(actions, axis=-1)
    directions = actions / np.maximum(radii[:, None], 1e-12)
    angles = np.arccos(np.clip(np.sum(directions * targets, axis=-1), -1.0, 1.0))
    return float(
        np.mean(
            (angles < angular_tolerance)
            & (np.abs(radii - target_radius) < radial_tolerance)
        )
    )


def calibrate_cartesian_sigma(offset_deg, shell_tolerance, sample_count):
    rng = np.random.default_rng(20260925 + int(offset_deg * 10 + shell_tolerance * 1000))
    targets = target_directions(rng, sample_count, np.deg2rad(offset_deg))
    noise = rng.normal(size=(sample_count, 3))
    center = np.broadcast_to(np.asarray([1.0, 0.0, 0.0]), targets.shape)
    candidates = np.linspace(0.04, 0.35, 32)
    probabilities = []
    for sigma in candidates:
        actions = cartesian_actions(center, APPROACH_RADIUS, noise, sigma)
        probabilities.append(
            hit_probability(
                actions,
                targets,
                APPROACH_RADIUS,
                ANGLE_TOLERANCE,
                shell_tolerance,
            )
        )
    best = int(np.argmax(probabilities))
    return float(candidates[best]), float(probabilities[best])


def methods(oracle_sigma):
    matched_radial = matching_radial_sigma(BASE_SIGMA, DS_TANGENTIAL_STD)
    return (
        Method("Cartesian global", "cartesian", BASE_SIGMA, 0.0, BASE_SIGMA, 0.0),
        Method("Cartesian stage", "cartesian", BASE_SIGMA, 0.0, PRECISION_STD, 0.0),
        Method(
            "Cartesian stage + L2",
            "cartesian",
            BASE_SIGMA,
            0.0,
            PRECISION_STD,
            0.0,
            project=True,
        ),
        Method(
            "Cartesian oracle stage",
            "cartesian",
            oracle_sigma,
            0.0,
            PRECISION_STD,
            0.0,
        ),
        Method(
            "DS global",
            "direction_speed",
            DS_TANGENTIAL_STD / APPROACH_RADIUS,
            matched_radial,
            DS_TANGENTIAL_STD / APPROACH_RADIUS,
            matched_radial,
        ),
        Method(
            "DS stage",
            "direction_speed",
            DS_TANGENTIAL_STD / APPROACH_RADIUS,
            matched_radial,
            PRECISION_STD / PRECISION_RADIUS,
            PRECISION_STD,
        ),
        Method(
            "DS stage, coupled radius",
            "direction_speed",
            DS_TANGENTIAL_STD / APPROACH_RADIUS,
            BASE_SIGMA,
            PRECISION_STD / PRECISION_RADIUS,
            PRECISION_STD,
        ),
    )


def evaluate_method(method, targets, approach_noise, precision_noise, shell_tolerance):
    center = np.broadcast_to(np.asarray([1.0, 0.0, 0.0]), targets.shape)
    if method.family == "cartesian":
        approach_actions = cartesian_actions(
            center,
            APPROACH_RADIUS,
            approach_noise,
            method.stage0_scale,
            project=method.project,
        )
        precision_actions = cartesian_actions(
            targets,
            PRECISION_RADIUS,
            precision_noise,
            method.stage1_scale,
            project=method.project,
        )
    else:
        approach_actions = direction_speed_actions(
            np.asarray([1.0, 0.0, 0.0]),
            APPROACH_RADIUS,
            approach_noise,
            method.stage0_scale,
            method.stage0_radial,
        )
        precision_actions = direction_speed_actions(
            targets,
            PRECISION_RADIUS,
            precision_noise,
            method.stage1_scale,
            method.stage1_radial,
        )
    approach_one_step = hit_probability(
        approach_actions,
        targets,
        APPROACH_RADIUS,
        ANGLE_TOLERANCE,
        shell_tolerance,
    )
    precision_one_step = hit_probability(
        precision_actions,
        targets,
        PRECISION_RADIUS,
        ANGLE_TOLERANCE,
        0.015,
    )
    approach = within_budget_probability(approach_one_step, APPROACH_BUDGET)
    precision = within_budget_probability(precision_one_step, PRECISION_BUDGET)
    return approach_one_step, approach, precision_one_step, precision, approach * precision


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(differences, seed=20260925):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(20000, len(differences)))
    means = differences[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-seed", type=int, default=100_000)
    parser.add_argument("--calibration-samples", type=int, default=500_000)
    parser.add_argument("--output-dir", type=Path, default=reports_dir())
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    oracle_sigmas = {}
    for offset_deg in OFFSETS_DEG:
        for shell_tolerance in SHELL_TOLERANCES:
            oracle_sigma, _ = calibrate_cartesian_sigma(
                offset_deg, shell_tolerance, args.calibration_samples
            )
            oracle_sigmas[(offset_deg, shell_tolerance)] = oracle_sigma
            for seed in FORMAL_SEEDS:
                rng = np.random.default_rng(seed)
                targets = target_directions(
                    rng, args.samples_per_seed, np.deg2rad(offset_deg)
                )
                approach_noise = rng.normal(size=(args.samples_per_seed, 3))
                precision_noise = rng.normal(size=(args.samples_per_seed, 3))
                for method in methods(oracle_sigma):
                    values = evaluate_method(
                        method,
                        targets,
                        approach_noise,
                        precision_noise,
                        shell_tolerance,
                    )
                    rows.append(
                        {
                            "method": method.name,
                            "seed": seed,
                            "offset_deg": offset_deg,
                            "shell_tolerance": shell_tolerance,
                            "oracle_cartesian_sigma": oracle_sigma,
                            "approach_one_step": values[0],
                            "approach_discovery": values[1],
                            "precision_one_step": values[2],
                            "precision_completion": values[3],
                            "episode_success": values[4],
                        }
                    )
    write_csv(args.output_dir / "synthetic_stage_task_per_seed.csv", rows)

    summary = []
    for offset_deg in OFFSETS_DEG:
        for shell_tolerance in SHELL_TOLERANCES:
            for method in methods(oracle_sigmas[(offset_deg, shell_tolerance)]):
                selected = [
                    row
                    for row in rows
                    if row["method"] == method.name
                    and row["offset_deg"] == offset_deg
                    and row["shell_tolerance"] == shell_tolerance
                ]
                item = {
                    "method": method.name,
                    "offset_deg": offset_deg,
                    "shell_tolerance": shell_tolerance,
                    "num_seeds": len(selected),
                    "oracle_cartesian_sigma": oracle_sigmas[
                        (offset_deg, shell_tolerance)
                    ],
                }
                for metric in (
                    "approach_discovery",
                    "precision_completion",
                    "episode_success",
                ):
                    values = np.asarray([row[metric] for row in selected])
                    item[f"{metric}_mean"] = float(values.mean())
                    item[f"{metric}_std"] = float(values.std(ddof=1))
                summary.append(item)
    write_csv(args.output_dir / "synthetic_stage_task_summary.csv", summary)

    main_rows = [
        row
        for row in rows
        if row["offset_deg"] == 30 and row["shell_tolerance"] == 0.025
    ]
    ds = np.asarray(
        [row["episode_success"] for row in main_rows if row["method"] == "DS stage"]
    )
    comparator = np.asarray(
        [
            row["episode_success"]
            for row in main_rows
            if row["method"] == "Cartesian oracle stage"
        ]
    )
    differences = ds - comparator
    ci_low, ci_high = paired_bootstrap(differences)
    paired = [
        {
            "contrast": "DS stage - Cartesian oracle stage",
            "offset_deg": 30,
            "shell_tolerance": 0.025,
            "mean_delta": float(differences.mean()),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "num_positive": int(np.sum(differences > 0)),
            "paired_differences": ";".join(f"{value:.8f}" for value in differences),
        }
    ]
    write_csv(args.output_dir / "synthetic_stage_task_paired.csv", paired)

    # Constant-volume allocation curve for the preregistered main setting.
    rng = np.random.default_rng(20260926)
    count = args.calibration_samples
    targets = target_directions(rng, count, np.deg2rad(30))
    noise = rng.normal(size=(count, 3))
    allocation = []
    for tangential_std in np.linspace(0.04, 0.35, 32):
        radial_std = matching_radial_sigma(BASE_SIGMA, tangential_std)
        actions = direction_speed_actions(
            np.asarray([1.0, 0.0, 0.0]),
            APPROACH_RADIUS,
            noise,
            tangential_std / APPROACH_RADIUS,
            radial_std,
        )
        one_step = hit_probability(
            actions,
            targets,
            APPROACH_RADIUS,
            ANGLE_TOLERANCE,
            0.025,
        )
        allocation.append(
            {
                "tangential_std": tangential_std,
                "radial_std": radial_std,
                "local_covariance_volume": local_covariance_volume(
                    tangential_std, radial_std
                ),
                "approach_discovery": within_budget_probability(
                    one_step, APPROACH_BUDGET
                ),
            }
        )
    write_csv(args.output_dir / "synthetic_stage_allocation_curve.csv", allocation)

    method_order = [method.name for method in methods(oracle_sigmas[(30, 0.025)])]
    main_summary = {
        row["method"]: row
        for row in summary
        if row["offset_deg"] == 30 and row["shell_tolerance"] == 0.025
    }
    robustness_deltas = []
    for offset_deg in OFFSETS_DEG:
        for shell_tolerance in SHELL_TOLERANCES:
            selected = {
                row["method"]: row
                for row in summary
                if row["offset_deg"] == offset_deg
                and row["shell_tolerance"] == shell_tolerance
            }
            robustness_deltas.append(
                selected["DS stage"]["episode_success_mean"]
                - selected["Cartesian oracle stage"]["episode_success_mean"]
            )
    decision = [
        {
            "criterion": "main_delta_at_least_20pp",
            "value": float(differences.mean()),
            "threshold": 0.20,
            "passed": bool(differences.mean() >= 0.20),
        },
        {
            "criterion": "all_16_main_pairs_positive",
            "value": int(np.sum(differences > 0)),
            "threshold": 16,
            "passed": bool(np.all(differences > 0)),
        },
        {
            "criterion": "phase_ablation_at_least_15pp",
            "value": main_summary["DS stage"]["episode_success_mean"]
            - main_summary["DS global"]["episode_success_mean"],
            "threshold": 0.15,
            "passed": bool(
                main_summary["DS stage"]["episode_success_mean"]
                - main_summary["DS global"]["episode_success_mean"]
                >= 0.15
            ),
        },
        {
            "criterion": "radial_ablation_at_least_20pp",
            "value": main_summary["DS stage"]["episode_success_mean"]
            - main_summary["DS stage, coupled radius"]["episode_success_mean"],
            "threshold": 0.20,
            "passed": bool(
                main_summary["DS stage"]["episode_success_mean"]
                - main_summary["DS stage, coupled radius"]["episode_success_mean"]
                >= 0.20
            ),
        },
        {
            "criterion": "all_9_robustness_cells_positive",
            "value": int(np.sum(np.asarray(robustness_deltas) > 0)),
            "threshold": 9,
            "passed": bool(np.all(np.asarray(robustness_deltas) > 0)),
        },
    ]
    write_csv(args.output_dir / "synthetic_stage_task_decision.csv", decision)
    colors = ["#d62728"] * 4 + ["#1f77b4", "#2ca02c", "#9467bd"]
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))
    x = np.arange(len(method_order))
    means = [100 * main_summary[name]["episode_success_mean"] for name in method_order]
    stds = [100 * main_summary[name]["episode_success_std"] for name in method_order]
    axes[0, 0].bar(x, means, yerr=stds, color=colors, capsize=3)
    axes[0, 0].set_xticks(x, method_order, rotation=30, ha="right")
    axes[0, 0].set_ylabel("End-to-end success within budget (%)")
    axes[0, 0].set_title("Main setting: 30° offset, ±0.025 shell")

    width = 0.36
    approach = [
        100 * main_summary[name]["approach_discovery_mean"] for name in method_order
    ]
    precision = [
        100 * main_summary[name]["precision_completion_mean"] for name in method_order
    ]
    axes[0, 1].bar(x - width / 2, approach, width, label="Discover shell target")
    axes[0, 1].bar(x + width / 2, precision, width, label="Precision completion")
    axes[0, 1].set_xticks(x, method_order, rotation=30, ha="right")
    axes[0, 1].set_ylabel("Stage success within budget (%)")
    axes[0, 1].legend(fontsize=8)

    heatmap = np.zeros((len(SHELL_TOLERANCES), len(OFFSETS_DEG)))
    for row in summary:
        if row["method"] not in ("DS stage", "Cartesian oracle stage"):
            continue
        shell_index = SHELL_TOLERANCES.index(row["shell_tolerance"])
        offset_index = OFFSETS_DEG.index(row["offset_deg"])
        sign = 1.0 if row["method"] == "DS stage" else -1.0
        heatmap[shell_index, offset_index] += sign * 100 * row["episode_success_mean"]
    image = axes[1, 0].imshow(heatmap, cmap="RdYlGn", vmin=-100, vmax=100)
    axes[1, 0].set_xticks(range(len(OFFSETS_DEG)), [f"{x}°" for x in OFFSETS_DEG])
    axes[1, 0].set_yticks(
        range(len(SHELL_TOLERANCES)), [f"±{x:.3f}" for x in SHELL_TOLERANCES]
    )
    axes[1, 0].set_xlabel("Target angular offset")
    axes[1, 0].set_ylabel("Radial shell tolerance")
    axes[1, 0].set_title("DS stage − tuned Cartesian (percentage points)")
    for row_index in range(heatmap.shape[0]):
        for column_index in range(heatmap.shape[1]):
            axes[1, 0].text(
                column_index,
                row_index,
                f"{heatmap[row_index, column_index]:+.1f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].plot(
        [row["tangential_std"] for row in allocation],
        [100 * row["approach_discovery"] for row in allocation],
        marker="o",
        markersize=3,
        label="DS, constant local covariance volume",
    )
    axes[1, 1].axvline(
        DS_TANGENTIAL_STD, color="#2ca02c", linestyle="--", label="Frozen DS allocation"
    )
    axes[1, 1].axhline(
        100 * main_summary["Cartesian oracle stage"]["approach_discovery_mean"],
        color="#d62728",
        linestyle=":",
        label="Tuned Cartesian upper comparator",
    )
    axes[1, 1].set_xlabel("Tangential action std")
    axes[1, 1].set_ylabel("Discovery within 100 samples (%)")
    axes[1, 1].set_title("Angular/radial allocation at matched local volume")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.suptitle("Controlled staged direction--speed action-sampling diagnostic")
    figure.tight_layout()
    figure.savefig(args.output_dir / "synthetic_stage_task.png", dpi=220)
    plt.close(figure)

    print(f"matched radial sigma: {matching_radial_sigma(BASE_SIGMA, DS_TANGENTIAL_STD):.8f}")
    print(f"Cartesian local volume: {BASE_SIGMA**3:.8f}")
    print(
        "DS local volume: "
        f"{local_covariance_volume(DS_TANGENTIAL_STD, matching_radial_sigma(BASE_SIGMA, DS_TANGENTIAL_STD)):.8f}"
    )
    for name in method_order:
        row = main_summary[name]
        print(
            f"{name}: discovery={100 * row['approach_discovery_mean']:.2f}% "
            f"precision={100 * row['precision_completion_mean']:.2f}% "
            f"end_to_end={100 * row['episode_success_mean']:.2f}%"
        )
    print(paired[0])


if __name__ == "__main__":
    main()
