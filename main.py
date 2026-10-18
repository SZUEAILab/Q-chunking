import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env
from envs.libero_utils import is_libero_env

from utils.flax_utils import save_agent
from utils.datasets import Dataset, ReplayBuffer
from utils.artifact_paths import results_dir

from evaluation import evaluate
from agents import agents
from direction_speed import ActionExecutionPipeline, configure_direction_speed
import numpy as np

if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"]

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "env_name", "cube-triple-play-singletask-task2-v0", "Environment (dataset) name."
)
flags.DEFINE_string("save_dir", str(results_dir()), "Experiment output root (QC_DATA_ROOT/results by default).")

flags.DEFINE_integer("offline_steps", 1000000, "Number of online steps.")
flags.DEFINE_integer("online_steps", 1000000, "Number of online steps.")
flags.DEFINE_integer("buffer_size", 2000000, "Replay buffer size.")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Evaluation interval.")
flags.DEFINE_integer("save_interval", -1, "Save interval.")
flags.DEFINE_integer("start_training", 5000, "when does training start")

flags.DEFINE_integer("utd_ratio", 1, "update to data ratio")

flags.DEFINE_float("discount", 0.99, "discount factor")

flags.DEFINE_integer("eval_episodes", 50, "Number of evaluation episodes.")
flags.DEFINE_integer(
    "eval_seed",
    20260925,
    "Fixed base seed for evaluation episode resets and policy sampling.",
)
flags.DEFINE_integer("video_episodes", 0, "Number of video episodes for each task.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")

config_flags.DEFINE_config_file("agent", "agents/acfql.py", lock_config=False)

flags.DEFINE_float("dataset_proportion", 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer(
    "dataset_replace_interval",
    1000,
    "Dataset replace interval, used for large datasets because of memory constraints",
)
flags.DEFINE_string("ogbench_dataset_dir", None, "OGBench dataset directory")

flags.DEFINE_integer("horizon_length", 5, "action chunking length.")
flags.DEFINE_bool("sparse", False, "make the task sparse reward")

flags.DEFINE_bool("save_all_online_states", False, "save all trajectories to npy")

# Direction+speed decomposition (first 3 dims = spatial, rest = yaw/grip scalars).
flags.DEFINE_enum(
    "ds_mode",
    "none",
    ["none", "projection", "adapter", "integrated"],
    "Action architecture: none, Cartesian L2 projection, adapter, or integrated DS.",
)
flags.DEFINE_enum(
    "ds_integrated_type",
    "spherical",
    ["spherical", "stereographic", "tangent", "projected_tangent"],
    "Direction distribution used when --ds_mode=integrated.",
)
flags.DEFINE_enum(
    "ds_speed_bound",
    "config",
    ["config", "sphere", "cube"],
    "Speed/radius bound: config, unit sphere, or cube radial bound.",
)
flags.DEFINE_enum(
    "ds_adapter_representation",
    "cartesian",
    ["cartesian", "spherical", "stereographic"],
    "Policy-space direction coordinates used by --ds_mode=adapter.",
)
flags.DEFINE_enum(
    "ds_adapter_speed_squash",
    "sigmoid",
    ["tanh", "sigmoid"],
    "Actor squash used for the adapter speed coordinate.",
)
flags.DEFINE_enum(
    "ds_adapter_speed_interpretation",
    "direct",
    ["log", "direct"],
    "Adapter interpretation: exponentiate log-speed or use speed directly.",
)
flags.DEFINE_enum(
    "ds_adapter_critic_space",
    "latent",
    ["latent", "env", "canonical"],
    "Adapter critic input: latent, executed environment action, or canonical embedding.",
)
flags.DEFINE_enum(
    "ds_regularizer",
    "config",
    ["config", "policy_coordinate", "action_volume", "semantic_product"],
    "Reference measure used by the DS actor regularizer.",
)
flags.DEFINE_bool(
    "ds_separate_alpha",
    False,
    "Use separate direction, speed, and scalar temperatures for semantic DS.",
)


class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f"{prefix}/{k}": v for k, v in data.items()}, step=step)


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project="qc", group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    ds_config = configure_direction_speed(
        ds_mode=FLAGS.ds_mode,
        ds_speed_bound=FLAGS.ds_speed_bound,
        ds_integrated_type=FLAGS.ds_integrated_type,
        ds_adapter_representation=FLAGS.ds_adapter_representation,
        ds_adapter_speed_squash=FLAGS.ds_adapter_speed_squash,
        ds_adapter_speed_interpretation=FLAGS.ds_adapter_speed_interpretation,
        ds_adapter_critic_space=FLAGS.ds_adapter_critic_space,
        ds_regularizer=FLAGS.ds_regularizer,
        ds_separate_alpha=FLAGS.ds_separate_alpha,
        config=config,
        agent_name=config.get("agent_name", ""),
        ds_speed_bound_default="cube",
    )
    use_adapter_ds = ds_config.use_adapter
    use_env_action_critic = ds_config.use_env_action_critic
    use_latent_adapter_critic = use_adapter_ds and not use_env_action_critic
    adapter = ds_config.adapter
    action_postprocessor = ds_config.postprocessor

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file
            for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz"))
            if "-val.npz" not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
            FLAGS.env_name
        )

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0

    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds["masks"]) * FLAGS.dataset_proportion)
            ds = Dataset.create(**{k: v[:new_size] for k, v in ds.items()})

        if is_robomimic_env(FLAGS.env_name) or is_libero_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)

        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds

    train_dataset = process_train_dataset(train_dataset)

    # Direction+speed: preprocess dataset actions (D -> D+1).
    raw_action_dim = train_dataset["actions"].shape[-1]
    if use_latent_adapter_critic:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=adapter.decompose(train_dataset["actions"]))
        )
    if use_adapter_ds:
        agent_action_dim = adapter.policy_action_dim(raw_action_dim)
    else:
        agent_action_dim = raw_action_dim
    config["env_action_dim"] = raw_action_dim
    config["policy_action_dim"] = agent_action_dim
    config["critic_action_dim"] = (
        raw_action_dim if use_env_action_critic else agent_action_dim
    )

    def make_eval_agent(base_agent):
        return ActionExecutionPipeline(
            base_agent, action_postprocessor, agent_action_dim
        )

    example_batch = train_dataset.sample(())

    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
    )

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={
            prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv"))
            for prefix in prefixes
        },
        wandb_logger=wandb,
    )

    offline_init_time = time.time()
    # Offline RL
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step += 1

        if (
            FLAGS.ogbench_dataset_dir is not None
            and FLAGS.dataset_replace_interval != 0
            and i % FLAGS.dataset_replace_interval == 0
        ):
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
            )
            train_dataset = process_train_dataset(train_dataset)
            if use_latent_adapter_critic:
                train_dataset = train_dataset.copy(
                    add_or_replace=dict(
                        actions=adapter.decompose(train_dataset["actions"])
                    )
                )

        batch = train_dataset.sample_sequence(
            config["batch_size"],
            sequence_length=FLAGS.horizon_length,
            discount=discount,
        )

        agent, offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

        # eval
        if i == FLAGS.offline_steps - 1 or (
            FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0
        ):
            _eval_agent = make_eval_agent(agent)

            eval_info, _, _ = evaluate(
                agent=_eval_agent,
                env=eval_env,
                action_dim=raw_action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                eval_seed=FLAGS.eval_seed,
            )
            logger.log(eval_info, "eval", step=log_step)

    # transition from offline to online
    if use_latent_adapter_critic:
        _raw = dict(train_dataset)
        _raw["actions"] = adapter.compose(train_dataset["actions"])
        replay_buffer = ReplayBuffer.create_from_initial_dataset(
            _raw, size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
    else:
        replay_buffer = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )

    ob, _ = env.reset(seed=FLAGS.seed)

    action_queue = []
    action_dim = raw_action_dim  # env action dim

    # Online RL
    update_info = {}

    from collections import defaultdict

    data = defaultdict(list)
    online_init_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            action_pi = agent.sample_actions(observations=ob, rng=key)
            action_chunk = action_postprocessor.compose_policy_output(
                action_pi, agent_action_dim
            )

            for a in action_chunk:
                action_queue.append(a)
        action = action_queue.pop(0)

        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))

        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        xyz_norm = float(np.linalg.norm(np.asarray(action)[:3]))
        env_info["action_xyz_norm"] = xyz_norm
        env_info["action_xyz_at_unit_bound"] = float(xyz_norm >= 1.0 - 1e-6)
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if "antmaze" in FLAGS.env_name and (
            "diverse" in FLAGS.env_name
            or "play" in FLAGS.env_name
            or "umaze" in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            # Adjust online (0, 1) reward for robomimic
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)

        # done
        if done:
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(
                config["batch_size"] * FLAGS.utd_ratio,
                sequence_length=FLAGS.horizon_length,
                discount=discount,
            )
            if use_latent_adapter_critic:
                batch = dict(batch)
                for k in ("actions", "next_actions"):
                    if k in batch:
                        batch[k] = adapter.decompose_chunked(
                            batch[k], FLAGS.horizon_length
                        ).reshape(batch[k].shape[0], FLAGS.horizon_length, -1)
            batch = jax.tree.map(
                lambda x: x.reshape(
                    (FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]
                ),
                batch,
            )

            agent, update_info["online_agent"] = agent.batch_update(batch)

        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or (
            FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0
        ):
            _eval_agent = make_eval_agent(agent)

            eval_info, _, _ = evaluate(
                agent=_eval_agent,
                env=eval_env,
                action_dim=raw_action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                eval_seed=FLAGS.eval_seed,
            )
            logger.log(eval_info, "eval", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

    end_time = time.time()

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {
            "steps": np.array(data["steps"]),
            "qpos": np.stack(data["qpos"], axis=0),
            "qvel": np.stack(data["qvel"], axis=0),
            "obs": np.stack(data["obs"], axis=0),
            "offline_time": online_init_time - offline_init_time,
            "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, "token.tk"), "w") as f:
        f.write(str(run.url or ""))


if __name__ == "__main__":
    app.run(main)
