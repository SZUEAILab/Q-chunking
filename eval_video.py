"""
load checkpoint, run eval, and save rollout videos.
supports both OGBench (MuJoCo) and RoboMimic (robosuite) environments.

usage: MUJOCO_GL=egl python eval_video.py <exp_dir>
example: MUJOCO_GL=egl python eval_video.py /data/chainpray/qc/results/<cohort>/<run>
"""

import os, sys, json, re
import jax, numpy as np, imageio
from evaluation import evaluate
from agents import agents
from direction_speed import ActionExecutionPipeline, configure_direction_speed
from envs.libero_utils import is_libero_env
from envs.robomimic_utils import is_robomimic_env
from utils.flax_utils import restore_agent_with_file

NUM_VIDEO = 5
FRAME_SKIP = 1
HEIGHT, WIDTH = 480, 480


def main():
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if exp_dir is None:
        raise SystemExit("Usage: python eval_video.py <explicit-run-directory>")

    print(f"exp dir: {exp_dir}")
    with open(os.path.join(exp_dir, "flags.json")) as f:
        flags = json.load(f)

    config = flags.get("agent", flags.get("config", {}))
    env_name = flags.get("env_name", flags.get("env", "?"))
    seed = flags.get("seed", 0)
    horizon = flags.get("horizon_length", 1)
    config["horizon_length"] = horizon

    # ---- ds_mode config ----
    ds_config = configure_direction_speed(
        ds_mode=flags.get("ds_mode", "none"),
        ds_speed_bound=flags.get("ds_speed_bound", "config"),
        ds_integrated_type=flags.get("ds_integrated_type", "spherical"),
        ds_adapter_representation=flags.get("ds_adapter_representation", "cartesian"),
        ds_adapter_speed_squash=flags.get("ds_adapter_speed_squash", "tanh"),
        ds_adapter_speed_interpretation=flags.get(
            "ds_adapter_speed_interpretation", "log"
        ),
        ds_adapter_critic_space=flags.get("ds_adapter_critic_space", "config"),
        ds_adapter_execution_map=flags.get("ds_adapter_execution_map", "config"),
        ds_regularizer=flags.get("ds_regularizer", "config"),
        ds_separate_alpha=flags.get("ds_separate_alpha", False),
        config=config,
        agent_name=config.get("agent_name", ""),
        ds_speed_bound_default="sphere",
    )
    use_adapter = ds_config.use_adapter
    adapter = ds_config.adapter
    action_postprocessor = ds_config.postprocessor

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"]
        os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"]

    # ---- env ----
    is_robo = is_robomimic_env(env_name)
    is_libero = is_libero_env(env_name)
    if is_robo:
        if not hasattr(np.dtypes, "StringDType"):
            np.dtypes.StringDType = np.dtypes.StrDType
        from envs.robomimic_utils import make_env, _check_dataset_exists
        from robomimic.utils import env_utils as EnvUtils, file_utils as FileUtils
        from envs.robomimic_utils import (
            RobomimicLowdimWrapper,
            low_dim_keys,
            _get_max_episode_length,
        )

        # Create env WITH offscreen rendering
        dataset_path = _check_dataset_exists(env_name)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        max_ep_len = _get_max_episode_length(env_name)
        eval_env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=True
        )
        eval_env = RobomimicLowdimWrapper(
            eval_env,
            low_dim_keys=low_dim_keys["low_dim"],
            max_episode_length=max_ep_len,
        )
        eval_env.seed(seed)

        raw_action_dim = eval_env.action_space.shape[-1]
        obs_sample = eval_env.reset()
        obs_dim = (
            obs_sample[0].shape[-1]
            if isinstance(obs_sample, tuple)
            else obs_sample.shape[-1]
        )
    elif is_libero:
        from envs.libero_utils import make_env as libero_make_env

        eval_env = libero_make_env(env_name, seed=seed, render_offscreen=True)

        raw_action_dim = eval_env.action_space.shape[-1]
        obs_sample = eval_env.reset()
        obs_dim = (
            obs_sample[0].shape[-1]
            if isinstance(obs_sample, tuple)
            else obs_sample.shape[-1]
        )
    else:
        ogbench_dataset_dir = flags.get("ogbench_dataset_dir")
        if ogbench_dataset_dir is not None:
            from envs.ogbench_utils import make_ogbench_env_and_datasets

            env, eval_env, _, _ = make_ogbench_env_and_datasets(
                env_name, dataset_path=None, compact_dataset=False
            )
        else:
            from envs.env_utils import make_env_and_datasets

            env, eval_env, _, _ = make_env_and_datasets(env_name)

        raw_action_dim = eval_env.action_space.shape[-1]
        obs_dim = eval_env.observation_space.shape[-1]
        render_env = eval_env  # OGBench renders from eval_env

        # MuJoCo render setup
        for env_obj in [env, eval_env]:
            uw = env_obj.unwrapped
            uw.model.vis.global_.offwidth = WIDTH
            uw.model.vis.global_.offheight = HEIGHT
            uw._render_height = HEIGHT
            uw._render_width = WIDTH
            uw._renderer = None

    if use_adapter:
        agent_action_dim = adapter.policy_action_dim(raw_action_dim)
    else:
        agent_action_dim = raw_action_dim
    critic_action_dim = (
        raw_action_dim if ds_config.use_env_action_critic else agent_action_dim
    )
    config["env_action_dim"] = raw_action_dim
    config["policy_action_dim"] = agent_action_dim
    config["critic_action_dim"] = critic_action_dim

    # ---- agent ----
    dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
    dummy_act = np.zeros((1, critic_action_dim), dtype=np.float32)
    agent = agents[config["agent_name"]].create(seed, dummy_obs, dummy_act, config)

    ckpts = sorted(
        [
            f
            for f in os.listdir(exp_dir)
            if f.startswith("params_") and f.endswith(".pkl")
        ],
        key=lambda x: int(re.search(r"params_(\d+)\.pkl", x).group(1)),
    )
    if not ckpts:
        print("No checkpoint found")
        return
    ckpt_path = os.path.join(exp_dir, ckpts[-1])
    epoch = int(ckpts[-1].replace("params_", "").replace(".pkl", ""))
    print(f"checkpoint: params_{epoch}.pkl ({epoch/1e6:.1f}M steps)")
    agent = restore_agent_with_file(agent, ckpt_path)

    # ---- action_pi -> action_env execution boundary ----
    eval_agent = ActionExecutionPipeline(agent, action_postprocessor, agent_action_dim)

    # ---- eval + video ----
    if is_robo or is_libero:
        video_dir = os.path.join(exp_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        print(f"\nRecording {NUM_VIDEO} rollout videos...")
        for ep in range(NUM_VIDEO):
            frames = []
            action_queue = []
            obs = eval_env.reset(seed=seed + ep)
            step, done = 0, False
            while not done and step < 500:
                if is_libero:
                    # LiberoLowdimWrapper -> OffScreenRenderEnv.sim
                    img = eval_env.env.sim.render(
                        camera_name="agentview", width=WIDTH, height=HEIGHT
                    )
                else:
                    # RobomimicLowdimWrapper -> EnvRobosuite -> robosuite env -> MjSim
                    img = eval_env.env.env.sim.render(
                        camera_name="agentview", width=WIDTH, height=HEIGHT
                    )
                if img is not None:
                    frames.append(img[::-1])

                if isinstance(obs, tuple):
                    obs_in = obs[0][np.newaxis, :]
                else:
                    obs_in = obs[np.newaxis, :]
                if not action_queue:
                    chunk = np.asarray(
                        eval_agent.sample_actions(
                            obs_in,
                            rng=jax.random.PRNGKey(ep * 1000 + step),
                        )
                    ).reshape(-1, raw_action_dim)
                    action_queue.extend(chunk)
                action = action_queue.pop(0)
                result = eval_env.step(action)
                obs, done = result[0], bool(result[2] or result[3])
                step += 1

            path = os.path.join(video_dir, f"rollout_{ep}.mp4")
            if frames:
                imageio.mimsave(path, frames, fps=20)
                print(f"  ep {ep}: {len(frames)} frames -> {path}")
            else:
                print(f"  ep {ep}: no frames")

        eval_env.close()

    else:
        # OGBench: evaluate() handles rendering natively
        eval_info, _, renders = evaluate(
            agent=eval_agent,
            env=eval_env,
            action_dim=raw_action_dim,
            num_eval_episodes=50,
            num_video_episodes=NUM_VIDEO,
            video_frame_skip=FRAME_SKIP,
        )

        print("\n=== Eval Results ===")
        for k, v in sorted(eval_info.items()):
            print(f"  {k}: {v:.4f}")

        video_dir = os.path.join(exp_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        for i, render in enumerate(renders):
            if render is not None and len(render) > 0:
                path = os.path.join(video_dir, f"rollout_{i}.mp4")
                imageio.mimsave(path, render, fps=20)
                print(f"  video {i}: {len(render)} frames -> {path}")

        env.close()
        eval_env.close()

    print(f"\nDone! videos saved to {video_dir}/")


if __name__ == "__main__":
    main()
