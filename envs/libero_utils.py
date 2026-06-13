"""LIBERO benchmark utilities — env creation, dataset loading, lowdim wrapper.

LIBERO is a robot manipulation benchmark built on robosuite with 7D action space
(same structure as RoboMimic: dx,dy,dz, droll,dpitch,dyaw, grip).

We use a 120D low-dim observation:
    robot0_proprio-state(50) + object-state(70) = 120
This includes robot proprioception AND all object positions — essential for
the agent to know where objects are relative to the gripper.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import h5py

from utils.datasets import Dataset

# ── robosuite 1.5 compatibility monkey-patches ───────────────────
import robosuite
from robosuite import load_composite_controller_config
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

# 1. Add old `load_controller_config` entry point
robosuite.load_controller_config = (
    lambda default_controller="BASIC", **kw: load_composite_controller_config(
        controller="BASIC"
    )
)

# 2. Register LIBERO's custom robot models so Robot.__init__ can find them
from libero.libero.envs.robots import MountedPanda, OnTheGroundPanda  # noqa: E402

MountedPanda.arms = ["right"]
OnTheGroundPanda.arms = ["right"]
REGISTERED_ROBOTS["MountedPanda"] = MountedPanda
REGISTERED_ROBOTS["OnTheGroundPanda"] = OnTheGroundPanda

# ── Observation: proprio-state(50) + object-state(70) = 120 ──────
OBS_DIM = 50 + 70  # 120

# Cache for preprocessed datasets (avoid recomputing from MuJoCo states)
_PREPROCESSED_CACHE: dict[str, dict] = {}


def _extract_lowdim_obs(env_obs: dict) -> np.ndarray:
    """Extract 120D rich observation from LIBERO env observation dict."""
    proprio = env_obs["robot0_proprio-state"].astype(np.float32)
    obj_state = env_obs["object-state"].astype(np.float32)
    return np.concatenate([proprio, obj_state])


def is_libero_env(env_name: str) -> bool:
    """Determine if an env name refers to a LIBERO task."""
    return env_name.startswith("libero_")


# ── Task name to HDF5 path mapping ───────────────────────────────
_SUITE_TASK_MAP = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_bowl_on_top_of_the_cabinet",
        "put_the_cream_cheese_in_the_bowl",
        "put_the_wine_bottle_on_the_rack",
        "turn_on_the_stove",
    ],
}

# datasets path from LIBERO config
from libero.libero import get_libero_path  # noqa: E402

LIBERO_DATASET_DIR = get_libero_path("datasets")


def _get_max_episode_length(task_name: str) -> int:
    """Return max episode length for a LIBERO task."""
    if "turn_on_the_stove" in task_name:
        return 520  # longest
    elif "push_the_plate" in task_name:
        return 320
    elif "open_the_middle_drawer" in task_name:
        return 370
    elif "open_the_top_drawer" in task_name:
        return 370
    else:
        return 280  # most tasks


def _find_dataset_path(task_name: str) -> str:
    """Locate the HDF5 dataset for a LIBERO task.

    task_name can be either:
      - Full task name like 'pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate'
      - Shorthand like 'libero_spatial/pick_up_the_...'
      - Suite index like 'libero_spatial/0' (first task in suite)
    """
    if "/" in task_name:
        suite, task_ref = task_name.split("/", 1)
        if task_ref.isdigit():
            idx = int(task_ref)
            if suite in _SUITE_TASK_MAP and idx < len(_SUITE_TASK_MAP[suite]):
                task_ref = _SUITE_TASK_MAP[suite][idx]
            else:
                raise ValueError(f"Invalid task index {idx} for suite {suite}")
    else:
        # Try to find which suite this task belongs to
        suite = None
        task_ref = task_name
        for s, tasks in _SUITE_TASK_MAP.items():
            if task_name in tasks:
                suite = s
                break
        if suite is None:
            # Check if task_name itself is a suite name
            if task_name in _SUITE_TASK_MAP:
                suite = task_name
                task_ref = _SUITE_TASK_MAP[suite][0]  # default to first task
            else:
                raise ValueError(f"Unknown LIBERO task: {task_name}")

    hdf5_path = os.path.join(LIBERO_DATASET_DIR, suite, f"{task_ref}_demo.hdf5")
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"LIBERO dataset not found: {hdf5_path}")
    return hdf5_path, suite, task_ref


def make_env(task_name: str, seed: int = 0, render_offscreen: bool = False):
    """Create a LIBERO low-dim environment.

    Args:
        task_name: Task identifier (see _find_dataset_path for formats).
        seed: Random seed.
        render_offscreen: Enable offscreen rendering for video recording.
    """
    from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

    hdf5_path, suite, task_ref = _find_dataset_path(task_name)

    # Build BDDL file path
    from libero.libero.benchmark import get_benchmark  # noqa: E402

    bm = get_benchmark(suite)(task_order_index=0)
    bddl_path = None
    for i in range(bm.n_tasks):
        if bm.tasks[i].name == task_ref:
            bddl_path = bm.get_task_bddl_file_path(i)
            break
    if bddl_path is None:
        raise ValueError(f"BDDL file not found for task {task_ref} in suite {suite}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        robots=["Panda"],
        controller="OSC_POSE",
        gripper_types="default",
        has_renderer=False,
        has_offscreen_renderer=render_offscreen,
        control_freq=20,
        horizon=_get_max_episode_length(task_ref),
        camera_names=["agentview"],
        camera_heights=128,
        camera_widths=128,
    )

    env = LiberoLowdimWrapper(env, max_episode_length=_get_max_episode_length(task_ref))
    env.seed(seed)
    return env


def _get_preprocessed_obs(task_name: str, hdf5_path: str) -> np.ndarray:
    """Reconstruct rich 120D observations from HDF5 MuJoCo states.

    The HDF5 only stores basic obs keys but has the full MuJoCo states(92D).
    We load the env, set each state, and read robot0_proprio-state + object-state.
    Results are cached per task.
    """
    cache_key = f"{task_name}"
    if cache_key in _PREPROCESSED_CACHE:
        return _PREPROCESSED_CACHE[cache_key]

    from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

    suite = task_name.split("/")[0] if "/" in task_name else task_name
    from libero.libero.benchmark import get_benchmark  # noqa: E402

    bm = get_benchmark(suite)(task_order_index=0)

    # Find the correct task index matching the HDF5 file
    hdf5_task_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    hdf5_task_name = hdf5_task_name.replace("_demo", "")
    task_idx = None
    for i in range(bm.n_tasks):
        if bm.tasks[i].name == hdf5_task_name:
            task_idx = i
            break
    if task_idx is None:
        task_idx = 0  # fallback

    env = OffScreenRenderEnv(
        bddl_file_name=bm.get_task_bddl_file_path(task_idx),
        robots=["Panda"],
        has_renderer=False,
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=1000,
    )
    # Must reset before state-setting works
    env.reset()

    f = h5py.File(hdf5_path, "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))

    all_obs = []
    for ep in demos:
        group = f[f"data/{ep}"]
        states = np.array(group["states"], dtype=np.float64)
        ep_obs = []
        for i in range(len(states)):
            env.env.sim.set_state_from_flattened(states[i])
            env.env.sim.forward()
            raw = env.env._get_observations()
            ep_obs.append(_extract_lowdim_obs(raw))
        all_obs.append(np.array(ep_obs, dtype=np.float32))

    f.close()

    # Cache
    _PREPROCESSED_CACHE[cache_key] = all_obs
    return all_obs


def get_dataset(task_name: str):
    """Load LIBERO HDF5 demonstrations as a Dataset with rich 120D observations.

    Returns:
        Dataset with observations(120D), actions(7D), rewards, terminals, masks,
        next_observations.
    """
    hdf5_path, suite, task_ref = _find_dataset_path(task_name)

    # Get preprocessed rich observations
    all_obs = _get_preprocessed_obs(task_name, hdf5_path)

    f = h5py.File(hdf5_path, "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))

    observations = []
    actions = []
    next_observations = []
    terminals = []
    rewards = []
    masks = []

    for i, ep in enumerate(demos):
        group = f[f"data/{ep}"]
        a = np.array(group["actions"], dtype=np.float32)
        obs = all_obs[i]

        # No next_obs in HDF5 — shift obs
        next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)

        dones = np.array(group["dones"], dtype=np.float32)
        r = np.array(group["rewards"], dtype=np.float32)

        observations.append(obs)
        actions.append(a)
        rewards.append(r)
        terminals.append(dones)
        masks.append(1.0 - dones)
        next_observations.append(next_obs)

    f.close()

    return Dataset.create(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
        next_observations=np.concatenate(next_observations, axis=0),
    )


class LiberoLowdimWrapper(gym.Env):
    """Gymnasium wrapper for LIBERO envs that provides lowdim observations.

    Converts LIBERO's dict observations into a 120D vector (proprio-state + object-state)
    and handles the standard (obs, reward, terminated, truncated, info) step interface.
    """

    def __init__(
        self,
        env,
        max_episode_length: int = 280,
    ):
        self.env = env
        self.max_episode_length = max_episode_length
        self.env_step = 0
        self.n_episodes = 0
        self.t = 0

        # Action space: [-1, 1]^7
        act_low = np.full(7, -1.0, dtype=np.float32)
        act_high = np.full(7, 1.0, dtype=np.float32)
        self.action_space = Box(low=act_low, high=act_high, dtype=np.float32)

        # Observation space: 15D
        obs_low = np.full(OBS_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def reset(self, *, seed=None, options=None):
        self.t = 0
        self.n_episodes += 1
        if seed is not None:
            self.seed(seed)
        raw_obs = self.env.reset()
        return _extract_lowdim_obs(raw_obs), {}

    def step(self, action):
        # action in [-1, 1]^7, LIBERO env expects same range
        raw_obs, reward, done, info = self.env.step(action)
        obs = _extract_lowdim_obs(raw_obs)

        self.t += 1
        self.env_step += 1

        terminated = bool(done)
        truncated = self.t >= self.max_episode_length

        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        h, w = 480, 480
        return self.env.env.sim.render(
            camera_name="agentview", width=w, height=h
        )

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
