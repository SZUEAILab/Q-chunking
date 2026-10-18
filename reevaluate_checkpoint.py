#!/usr/bin/env python3
"""Reevaluate one saved run with fixed environment and policy RNG seeds."""

import argparse
import copy
import csv
import json
import re
from pathlib import Path

import numpy as np
import ogbench

from agents import agents
from direction_speed import ActionExecutionPipeline, configure_direction_speed
from envs.env_utils import EpisodeMonitor, SafeMujocoStep
from evaluation import evaluate
from utils.flax_utils import restore_agent_with_file


def latest_checkpoint(run_dir):
    checkpoints = sorted(
        run_dir.glob("params_*.pkl"),
        key=lambda path: int(re.search(r"params_(\d+)\.pkl", path.name).group(1)),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {run_dir}")
    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=20260925)
    parser.add_argument("--output-name", default="eval_fixed_seed.csv")
    args = parser.parse_args()

    flags = json.loads((args.run_dir / "flags.json").read_text())
    config = copy.deepcopy(flags["agent"])
    config["horizon_length"] = flags["horizon_length"]
    ds_config = configure_direction_speed(
        ds_mode=flags["ds_mode"],
        ds_speed_bound=flags["ds_speed_bound"],
        ds_integrated_type=flags["ds_integrated_type"],
        ds_adapter_representation=flags["ds_adapter_representation"],
        ds_adapter_speed_squash=flags["ds_adapter_speed_squash"],
        ds_adapter_speed_interpretation=flags["ds_adapter_speed_interpretation"],
        ds_adapter_critic_space=flags["ds_adapter_critic_space"],
        ds_regularizer=flags["ds_regularizer"],
        ds_separate_alpha=flags["ds_separate_alpha"],
        config=config,
        agent_name=config["agent_name"],
        ds_speed_bound_default="cube",
    )

    eval_env = ogbench.make_env_and_datasets(flags["env_name"], env_only=True)
    eval_env = SafeMujocoStep(eval_env)
    eval_env = EpisodeMonitor(
        eval_env, filter_regexes=[".*privileged.*", ".*proprio.*"]
    )
    observation, _ = eval_env.reset(seed=args.eval_seed)
    obs_dim = observation.shape[-1]
    env_action_dim = eval_env.action_space.shape[-1]
    policy_action_dim = (
        ds_config.adapter.policy_action_dim(env_action_dim)
        if ds_config.use_adapter
        else env_action_dim
    )
    critic_action_dim = (
        env_action_dim if ds_config.use_env_action_critic else policy_action_dim
    )
    config["env_action_dim"] = env_action_dim
    config["policy_action_dim"] = policy_action_dim
    config["critic_action_dim"] = critic_action_dim

    dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
    dummy_action = np.zeros((1, critic_action_dim), dtype=np.float32)
    agent = agents[config["agent_name"]].create(
        flags["seed"], dummy_obs, dummy_action, config
    )
    checkpoint = latest_checkpoint(args.run_dir)
    agent = restore_agent_with_file(agent, str(checkpoint))
    eval_agent = ActionExecutionPipeline(
        agent, ds_config.postprocessor, policy_action_dim
    )

    info, _, _ = evaluate(
        agent=eval_agent,
        env=eval_env,
        action_dim=env_action_dim,
        num_eval_episodes=args.episodes,
        num_video_episodes=0,
        eval_seed=args.eval_seed,
    )
    eval_env.close()

    step = int(re.search(r"params_(\d+)\.pkl", checkpoint.name).group(1))
    row = {
        "seed": flags["seed"],
        "ds_regularizer": flags["ds_regularizer"],
        "ds_separate_alpha": flags["ds_separate_alpha"],
        "checkpoint_step": step,
        "eval_seed": args.eval_seed,
        "episodes": args.episodes,
        "success": float(info["success"]),
        "episode_return": float(info["episode.return"]),
        "episode_length": float(info["episode.length"]),
    }
    output = args.run_dir / args.output_name
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
