#!/usr/bin/env python3
"""Monte Carlo diagnostic for Cartesian direction-noise amplification."""

import csv
from pathlib import Path
from utils.artifact_paths import reports_dir

import matplotlib.pyplot as plt
import numpy as np


def angular_error(samples):
    unit = samples / np.linalg.norm(samples, axis=-1, keepdims=True)
    cosine = np.clip(unit[:, 0], -1.0, 1.0)
    sine = np.linalg.norm(unit[:, 1:], axis=-1)
    return np.arctan2(sine, cosine)


def main():
    rng = np.random.default_rng(20260920)
    radii = np.asarray([0.10, 0.15, 0.25, 0.40, 0.65, 1.00])
    sigma = 0.03
    angular_sigma = 0.10
    num_samples = 200_000
    rows = []
    for radius in radii:
        cartesian = rng.normal(size=(num_samples, 3)) * sigma
        cartesian[:, 0] += radius
        cartesian_mse = float(np.mean(angular_error(cartesian) ** 2))

        factorized = np.ones((num_samples, 3))
        factorized[:, 1:] = rng.normal(
            scale=angular_sigma, size=(num_samples, 2)
        )
        factorized_mse = float(np.mean(angular_error(factorized) ** 2))
        rows.append(
            {
                "radius": radius,
                "cartesian_angular_mse": cartesian_mse,
                "small_noise_prediction": 2 * sigma**2 / radius**2,
                "radius_sq_times_mse": radius**2 * cartesian_mse,
                "factorized_angular_mse": factorized_mse,
            }
        )

    output_dir = reports_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "direction_noise_geometry.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fitted = np.polyfit(
        np.log(radii[2:]),
        np.log([row["cartesian_angular_mse"] for row in rows[2:]]),
        deg=1,
    )
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.loglog(
        radii,
        [row["cartesian_angular_mse"] for row in rows],
        "o-",
        label=f"Cartesian Gaussian (slope={fitted[0]:.2f})",
    )
    axis.loglog(
        radii,
        [row["small_noise_prediction"] for row in rows],
        "--",
        label=r"Small-noise prediction $2\sigma^2/r^2$",
    )
    axis.loglog(
        radii,
        [row["factorized_angular_mse"] for row in rows],
        "s-",
        label="Factorized fixed angular noise",
    )
    axis.set(xlabel="Mean action radius r", ylabel="Mean squared angular error (rad²)")
    axis.grid(which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    plot_path = output_dir / "direction_noise_geometry.png"
    fig.savefig(plot_path, dpi=200)
    print(f"log-log slope (r >= 0.25): {fitted[0]:.4f}")
    print(f"mean r^2*MSE (r >= 0.25): {np.mean([r['radius_sq_times_mse'] for r in rows[2:]]):.6f}")
    print(f"theory 2*sigma^2: {2 * sigma**2:.6f}")
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
