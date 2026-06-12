"""LIBERO benchmark utilities — env creation, dataset loading, lowdim wrapper.

LIBERO is a robot manipulation benchmark built on robosuite with 7D action space
(same structure as RoboMimic: dx,dy,dz, droll,dpitch,dyaw, grip).

We use a 15D low-dim observation:
    ee_pos(3) + ee_ori(3) + gripper_qpos(2) + joint_pos(7) = 15
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


def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) → axis-angle (3D)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(1.0 - w * w)
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([x, y, z]) / sin_half
    return (axis * angle).astype(np.float32)


# ── Low-dim observation keys ─────────────────────────────────────
# Environment → observation feature mapping
ENV_OBS_KEYS = [
    "robot0_eef_pos",      # 3
    "robot0_eef_quat",     # 4 → converted to axis-angle (3)
    "robot0_gripper_qpos", # 2
    "robot0_joint_pos",    # 7
]
OBS_DIM = 3 + 3 + 2 + 7  # 15


def _extract_lowdim_obs(env_obs: dict) -> np.ndarray:
    """Extract 15D low-dim observation from LIBERO env observation dict."""
    ee_pos = env_obs["robot0_eef_pos"].astype(np.float32)
    ee_ori = _quat_to_axis_angle(env_obs["robot0_eef_quat"])
    gripper = env_obs["robot0_gripper_qpos"].astype(np.float32)
    joint = env_obs["robot0_joint_pos"].astype(np.float32)
    return np.concatenate([ee_pos, ee_ori, gripper, joint])


# ── Dataset key names (HDF5) → observation feature mapping ───────
HDF5_OBS_KEYS = [
    "ee_pos",         # 3
    "ee_ori",         # 3 (axis-angle)
    "gripper_states", # 2
    "joint_states",   # 7
]


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


def get_dataset(task_name: str):
    """Load LIBERO HDF5 demonstrations as a Dataset.

    Returns:
        Dataset with observations(15D), actions(7D), rewards, terminals, masks,
        next_observations.
    """
    hdf5_path, suite, task_ref = _find_dataset_path(task_name)

    f = h5py.File(hdf5_path, "r")
    demos = list(f["data"].keys())
    # Sort numerically: demo_0, demo_1, ...
    demos = sorted(demos, key=lambda x: int(x.split("_")[-1]))

    observations = []
    actions = []
    next_observations = []
    terminals = []
    rewards = []
    masks = []

    for ep in demos:
        group = f[f"data/{ep}"]
        a = np.array(group["actions"], dtype=np.float32)

        # Build lowdim observation matching env format
        obs_parts = []
        for key in HDF5_OBS_KEYS:
            obs_parts.append(np.array(group[f"obs/{key}"], dtype=np.float32))
        obs = np.concatenate(obs_parts, axis=-1)

        # LIBERO HDF5 has no next_obs — compute from shifted obs
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

    Converts LIBERO's dict observations into a 15D vector and handles
    the standard (obs, reward, terminated, truncated, info) step interface.
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
