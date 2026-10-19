from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
from functools import partial


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def observed_evaluation_counts(stats, trajectories, *, expected_episodes):
    """Audit completed evaluation trajectories, excluding rendered episodes.

    Count terminal binary success only when the environment exposes it. Missing
    or partial labels cannot be silently interpreted as failed episodes.
    """
    count = len(trajectories)
    if count != expected_episodes:
        raise ValueError(f"Observed {count} evaluation episodes; expected {expected_episodes}")
    successes = []
    for trajectory in trajectories:
        done, infos = trajectory.get("done", []), trajectory.get("info", [])
        if not done or len(done) != len(infos) or not bool(done[-1]) or any(done[:-1]):
            raise ValueError("Evaluation trajectory is incomplete or contains multiple episodes")
        terminal = flatten(infos[-1])
        if "success" in terminal:
            value = np.asarray(terminal["success"])
            if value.ndim != 0 or not np.isfinite(value) or float(value) not in (0., 1.):
                raise ValueError("Terminal evaluation success must be a finite binary scalar")
            successes.append(int(value))
    observed = {"evaluation_episodes_observed": count}
    if successes or "success" in stats:
        if len(successes) != count or not count or "success" not in stats:
            raise ValueError("Evaluation success labels or aggregate are missing")
        success_count = sum(successes)
        aggregate = np.asarray(stats["success"])
        if aggregate.ndim != 0 or not np.isfinite(aggregate) or not np.isclose(
            float(aggregate), success_count/count, rtol=0., atol=1e-7
        ):
            raise ValueError("Aggregate success differs from observed terminal successes")
        observed["evaluation_successes_observed"] = success_count
    return observed


def evaluate(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    action_shape=None,
    observation_shape=None,
    action_dim=None,
    eval_seed=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.
        eval_seed: Optional fixed base seed for both policy sampling and episode
            resets.  Episode ``i`` uses ``eval_seed + i``.  ``None`` preserves
            the historical behavior for backward compatibility.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_seed = (
        np.random.randint(0, 2**32) if eval_seed is None else int(eval_seed)
    )
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(actor_seed))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        if eval_seed is None:
            observation, info = env.reset()
        else:
            observation, info = env.reset(seed=int(eval_seed) + i)
            
        observation_history = []
        action_history = []
        
        done = False
        step = 0
        render = []
        action_chunk_lens = defaultdict(lambda: 0)

        action_queue = []

        gripper_contact_lengths = []
        gripper_contact_length = 0
        while not done:
            
            action = actor_fn(observations=observation)

            if len(action_queue) == 0:
                have_new_action = True
                action = np.array(action).reshape(-1, action_dim)
                action_chunk_len = action.shape[0]
                for a in action:
                    action_queue.append(a)
            else:
                have_new_action = False
            
            action = action_queue.pop(0)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)

            next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            
            observation = next_observation
            # print(info)
            if "proprio" in info and "gripper_contact" in info["proprio"]:
                # print(info["gripper_contact"])
                gripper_contact = info["proprio"]["gripper_contact"]
            elif "gripper_contact" in info:
                gripper_contact = info["gripper_contact"]
            else:
                gripper_contact = None

            if gripper_contact is not None:
                if info["gripper_contact"] > 0.1:
                    gripper_contact_length += 1
                else:
                    if gripper_contact_length > 0:
                        gripper_contact_lengths.append(gripper_contact_length)
                    gripper_contact_length = 0

        if gripper_contact_length > 0:
            gripper_contact_lengths.append(gripper_contact_length)
        
        num_gripper_contacts = len(gripper_contact_lengths)

        if num_gripper_contacts > 0:
            avg_gripper_contact_length = np.mean(np.array(gripper_contact_lengths))
        else:
            avg_gripper_contact_length = 0
            
        add_to(stats, {"avg_gripper_contact_length": avg_gripper_contact_length, "num_gripper_contacts": num_gripper_contacts})

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean([x for x in v if x is not None])

    return stats, trajs, renders
