#!/usr/bin/env python3
"""Measure retained state-dependent raw-speed capacity at final checkpoints."""

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from analyze_robot_manifold_policy_geometry import create_agent, latest_checkpoint
from envs.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent_with_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-observations", type=int, default=512)
    args = parser.parse_args()

    candidates = []
    for flags_path in sorted(args.root.rglob("flags.json")):
        flags = json.loads(flags_path.read_text())
        if flags.get("run_group") != args.run_group:
            continue
        if not flags["agent"].get("ds_moment_controlled", False):
            continue
        try:
            checkpoint = latest_checkpoint(flags_path.parent)
        except FileNotFoundError:
            continue
        if checkpoint.name != "params_300000.pkl":
            continue
        candidates.append((flags_path, flags, checkpoint))
    if not candidates:
        raise SystemExit("no complete moment-controlled runs found")

    env, eval_env, dataset, _ = make_env_and_datasets(
        candidates[0][1]["env_name"]
    )
    try:
        all_observations = np.asarray(dataset["observations"])
        indices = np.linspace(
            0, len(all_observations) - 1, args.num_observations, dtype=np.int64
        )
        observations = all_observations[indices]
        rows = []
        for flags_path, flags, checkpoint in candidates:
            agent, _, _ = create_agent(
                flags, observations.shape[-1], dataset["actions"].shape[-1]
            )
            agent = restore_agent_with_file(agent, str(checkpoint))
            distribution = agent.network.select("actor")(
                jnp.asarray(observations)
            )
            raw_speed_location = np.asarray(
                jax.device_get(distribution.raw_distribution.loc[..., 2])
            )
            rows.append(
                {
                    "seed": int(flags["seed"]),
                    "speed_residual_scale": float(
                        flags["agent"].get("ds_moment_speed_residual_scale", 0.0)
                    ),
                    "raw_speed_location_mean": float(raw_speed_location.mean()),
                    "raw_speed_location_std_across_states": float(
                        raw_speed_location.std()
                    ),
                    "raw_speed_location_p10": float(
                        np.quantile(raw_speed_location, 0.10)
                    ),
                    "raw_speed_location_p90": float(
                        np.quantile(raw_speed_location, 0.90)
                    ),
                }
            )
            del agent
            jax.clear_caches()
    finally:
        env.close()
        eval_env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
