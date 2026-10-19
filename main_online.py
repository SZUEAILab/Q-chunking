import glob, tqdm, wandb, os, json, random, time, jax

# Fix numpy 2.x compatibility for robosuite.
import numpy as _np

if not hasattr(_np.dtypes, "StringDType"):
    _np.dtypes.StringDType = _np.dtypes.StrDType

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
from direction_speed.moment_control import (
    MomentControllerConfig,
    MomentTargets,
    read_moment_controller_params,
    replace_moment_controller_params,
    sample_executed_translation_geometry,
    update_moment_controller,
)
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

flags.DEFINE_integer("online_steps", 1000000, "Number of online steps.")
flags.DEFINE_integer("buffer_size", 1000000, "Replay buffer size.")
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Evaluation interval.")
flags.DEFINE_bool("eval_at_start", False, "Evaluate the initialized policy at step zero.")
flags.DEFINE_integer("save_interval", -1, "Save interval.")
flags.DEFINE_integer("start_training", 5000, "when does training start")

flags.DEFINE_integer("utd_ratio", 1, "update to data ratio")
flags.DEFINE_enum(
    "actor_update_mode",
    "sac",
    ["sac", "replay_awr", "sac_direction_awr"],
    "Actor update: SAC, replay AWR, or SAC with direction-only AWR support.",
)
flags.DEFINE_float("awr_temperature", 1.0, "AWR advantage temperature.")
flags.DEFINE_float("awr_max_weight", 20.0, "Maximum unnormalized AWR weight.")
flags.DEFINE_float(
    "direction_awr_coefficient",
    1.0,
    "Coefficient on direction-only replay support under sac_direction_awr.",
)
flags.DEFINE_bool(
    "ds_separate_actor_trunks",
    False,
    "Use independent direction, speed, and scalar actor trunks for integrated DS.",
)
flags.DEFINE_float(
    "awr_entropy_scale",
    1.0,
    "Multiplier on the existing actor regularizer under AWR.",
)
flags.DEFINE_bool(
    "stage_balanced_sampling",
    False,
    "Balance prior and online sequence starts across observed reward stages.",
)

flags.DEFINE_float("discount", 0.99, "discount factor")

flags.DEFINE_integer("eval_episodes", 50, "Number of evaluation episodes.")
flags.DEFINE_integer(
    "eval_seed",
    20260925,
    "Fixed base seed for evaluation episode resets and policy sampling.",
)
flags.DEFINE_integer("video_episodes", 0, "Number of video episodes for each task.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")

config_flags.DEFINE_config_file("agent", "agents/acrlpd.py", lock_config=False)

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
    "ds_adapter_execution_map",
    "config",
    ["config", "direction_speed", "cartesian_aux"],
    "Adapter execution: direction-speed mapping or six-dimensional Cartesian auxiliary control.",
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
flags.DEFINE_enum(
    "ds_phase_regularization",
    "none",
    ["none", "reward_stage_translation"],
    "Optionally downweight direction/speed regularization after task progress.",
)
flags.DEFINE_float(
    "ds_phase_reward_floor",
    -3.0,
    "Reward corresponding to zero completed stages for phase regularization.",
)
flags.DEFINE_float(
    "ds_phase_decay",
    0.5,
    "Multiplicative direction/speed regularization decay per completed stage.",
)
flags.DEFINE_float(
    "ds_phase_max_progress",
    2.0,
    "Maximum reward-derived progress used by phase regularization.",
)
flags.DEFINE_enum(
    "ds_stage_speed_constraint",
    "none",
    ["none", "reward_stage_cap"],
    "Optionally penalize translation norms above a reward-stage-dependent cap.",
)
flags.DEFINE_float(
    "ds_stage_speed_base_cap",
    1.0,
    "Translation-norm cap before any cube-triple subtask is complete.",
)
flags.DEFINE_float(
    "ds_stage_speed_decay",
    0.5,
    "Multiplicative translation-norm cap decay per completed stage.",
)
flags.DEFINE_float(
    "ds_stage_speed_coefficient",
    100.0,
    "Actor-loss coefficient for squared translation speed-cap violations.",
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
    if FLAGS.eval_at_start and FLAGS.eval_seed is None:
        raise ValueError("Step-zero evaluation requires an independent fixed eval_seed.")
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
        ds_adapter_execution_map=FLAGS.ds_adapter_execution_map,
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
    config["ds_phase_regularization"] = FLAGS.ds_phase_regularization
    config["ds_phase_reward_floor"] = FLAGS.ds_phase_reward_floor
    config["ds_phase_decay"] = FLAGS.ds_phase_decay
    config["ds_phase_max_progress"] = FLAGS.ds_phase_max_progress
    config["ds_stage_speed_constraint"] = FLAGS.ds_stage_speed_constraint
    config["ds_stage_speed_base_cap"] = FLAGS.ds_stage_speed_base_cap
    config["ds_stage_speed_decay"] = FLAGS.ds_stage_speed_decay
    config["ds_stage_speed_coefficient"] = FLAGS.ds_stage_speed_coefficient
    config["actor_update_mode"] = FLAGS.actor_update_mode
    config["awr_temperature"] = FLAGS.awr_temperature
    config["awr_max_weight"] = FLAGS.awr_max_weight
    config["awr_entropy_scale"] = FLAGS.awr_entropy_scale
    config["direction_awr_coefficient"] = FLAGS.direction_awr_coefficient
    config["ds_separate_actor_trunks"] = FLAGS.ds_separate_actor_trunks
    if FLAGS.ds_separate_actor_trunks and not ds_config.use_integrated:
        raise ValueError("Separate actor trunks require integrated direction-speed.")
    if FLAGS.actor_update_mode in ("replay_awr", "sac_direction_awr"):
        if FLAGS.awr_temperature <= 0.0:
            raise ValueError("awr_temperature must be positive.")
        if FLAGS.awr_max_weight < 1.0:
            raise ValueError("awr_max_weight must be at least one.")
        if FLAGS.awr_entropy_scale < 0.0:
            raise ValueError("awr_entropy_scale must be nonnegative.")
    if FLAGS.actor_update_mode == "sac_direction_awr":
        if not ds_config.use_integrated:
            raise ValueError("sac_direction_awr requires integrated direction-speed.")
        if FLAGS.direction_awr_coefficient < 0.0:
            raise ValueError("direction_awr_coefficient must be nonnegative.")
    if FLAGS.ds_phase_regularization != "none":
        if not (
            ds_config.use_integrated
            and ds_config.action_reference.value == "semantic_product"
        ):
            raise ValueError(
                "Phase regularization requires integrated semantic_product DS."
            )
        if not 0.0 < FLAGS.ds_phase_decay <= 1.0:
            raise ValueError("ds_phase_decay must be in (0, 1].")
        if FLAGS.ds_phase_max_progress < 0.0:
            raise ValueError("ds_phase_max_progress must be nonnegative.")
    if FLAGS.ds_stage_speed_constraint != "none":
        if not (
            ds_config.use_integrated
            and ds_config.action_reference.value == "semantic_product"
        ):
            raise ValueError(
                "Stage speed constraint requires integrated semantic_product DS."
            )
        if FLAGS.ds_stage_speed_base_cap <= 0.0:
            raise ValueError("ds_stage_speed_base_cap must be positive.")
        if not 0.0 < FLAGS.ds_stage_speed_decay <= 1.0:
            raise ValueError("ds_stage_speed_decay must be in (0, 1].")
        if FLAGS.ds_stage_speed_coefficient < 0.0:
            raise ValueError("ds_stage_speed_coefficient must be nonnegative.")
        if FLAGS.ds_phase_max_progress < 0.0:
            raise ValueError("ds_phase_max_progress must be nonnegative.")

    uses_stage_labels = (
        FLAGS.ds_phase_regularization != "none"
        or FLAGS.ds_stage_speed_constraint != "none"
        or FLAGS.stage_balanced_sampling
    )

    # Seed global sources before environment construction.  Gym's own RNG is
    # seeded explicitly at the first training reset below.
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

        stage_rewards = np.asarray(ds["rewards"]).copy()

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

        if uses_stage_labels:
            ds = ds.copy(add_or_replace=dict(stage_rewards=stage_rewards))

        return ds

    train_dataset = process_train_dataset(train_dataset)

    # The fixed observation panel makes controller measurements comparable
    # across training.  It is selected before any online replay is added and
    # the controller never receives rewards, successes, or critic values.
    moment_controlled = bool(config.get("ds_moment_controlled", False))
    if moment_controlled:
        if not (
            ds_config.use_integrated
            and FLAGS.ds_integrated_type in ("tangent", "projected_tangent")
        ):
            raise ValueError(
                "Moment control requires integrated tangent or projected_tangent DS."
            )
        if FLAGS.horizon_length != 1:
            raise ValueError(
                "The first moment-control experiment requires horizon_length=1."
            )
        moment_interval = int(config["ds_moment_control_interval"])
        moment_panel_size = int(config["ds_moment_panel_size"])
        moment_samples = int(config["ds_moment_samples_per_state"])
        if moment_interval <= 0 or moment_panel_size <= 0 or moment_samples <= 1:
            raise ValueError(
                "Moment-control interval/panel/samples must be positive."
            )
        available_states = len(train_dataset["observations"])
        panel_indices = np.linspace(
            0,
            available_states - 1,
            min(moment_panel_size, available_states),
            dtype=np.int64,
        )
        moment_observations = np.asarray(train_dataset["observations"])[
            panel_indices
        ]
        moment_targets = MomentTargets(
            radius_mean=float(config["ds_moment_target_radius_mean"]),
            radius_std_within_state=float(config["ds_moment_target_radius_std"]),
            angular_deviation_rad=float(config["ds_moment_target_angle"]),
        )
        moment_config = MomentControllerConfig(
            control_radius_mean=bool(
                config.get("ds_moment_control_speed_mean", True)
            ),
            angular_gain=float(config["ds_moment_angular_gain"]),
            radial_std_gain=float(config["ds_moment_radial_std_gain"]),
            radius_mean_gain=float(config["ds_moment_radius_mean_gain"]),
        )

    # Direction+speed: preprocess dataset actions (D -> D+1).
    raw_action_dim = train_dataset["actions"].shape[-1]
    if FLAGS.ds_stage_speed_constraint != "none" and raw_action_dim < 3:
        raise ValueError(
            "Stage speed constraint requires at least three environment action dimensions."
        )
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
    with open(os.path.join(FLAGS.save_dir, "effective_agent_config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2, sort_keys=True)

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")
    if moment_controlled:
        prefixes.append("moment_controller")

    logger = LoggingHelper(
        csv_loggers={
            prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv"))
            for prefix in prefixes
        },
        wandb_logger=wandb,
    )

    def evaluate_and_log(step):
        eval_info, _, _ = evaluate(
            agent=make_eval_agent(agent),
            env=eval_env,
            action_dim=raw_action_dim,
            num_eval_episodes=FLAGS.eval_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
            eval_seed=FLAGS.eval_seed,
        )
        logger.log(eval_info, "eval", step=step)

    if FLAGS.eval_at_start:
        evaluate_and_log(0)

    # transition from offline to online
    # Replay buffer stores raw actions (D), not decomposed (D+1).
    if use_latent_adapter_critic:
        _raw_ex = dict(example_batch)
        _raw_ex["actions"] = adapter.compose(example_batch["actions"][np.newaxis])[0]
        replay_buffer = ReplayBuffer.create(_raw_ex, size=FLAGS.buffer_size)
    else:
        replay_buffer = ReplayBuffer.create(example_batch, size=FLAGS.buffer_size)

    ob, _ = env.reset(seed=FLAGS.seed)
    stage_reward = FLAGS.ds_phase_reward_floor

    action_queue = []

    from collections import defaultdict

    data = defaultdict(list)
    online_init_time = time.time()

    # Online RL
    update_info = {}
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            if i <= FLAGS.start_training:
                action = jax.random.uniform(
                    key, shape=(raw_action_dim,), minval=-1, maxval=1
                )
                action_chunk = np.array(action).reshape(-1, raw_action_dim)
            else:
                action_pi = agent.sample_actions(observations=ob, rng=key)
                action_chunk = action_postprocessor.compose_policy_output(
                    action_pi, agent_action_dim
                )

            for a in action_chunk:
                action_queue.append(a)
        action = action_queue.pop(0)

        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_stage_reward = float(int_reward)

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
        env_info["action_xyz_linf"] = float(np.max(np.abs(np.asarray(action)[:3])))
        for action_index, action_value in enumerate(np.asarray(action).reshape(-1)):
            env_info[f"action_{action_index}"] = float(action_value)
        env_info["action_xyz_at_unit_bound"] = float(xyz_norm >= 1.0 - 1e-6)
        env_info["stage_reward"] = float(stage_reward)
        env_info["next_stage_reward"] = next_stage_reward
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if "antmaze" in FLAGS.env_name and (
            "diverse" in FLAGS.env_name
            or "play" in FLAGS.env_name
            or "umaze" in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name) or is_libero_env(FLAGS.env_name):
            # Adjust online (0, 1) reward for robomimic / LIBERO
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
        if uses_stage_labels:
            transition["stage_rewards"] = float(stage_reward)
        replay_buffer.add_transition(transition)

        # done
        if done:
            ob, _ = env.reset()
            stage_reward = FLAGS.ds_phase_reward_floor
            action_queue = []  # reset the action queue
        else:
            ob = next_ob
            stage_reward = next_stage_reward

        if i >= FLAGS.start_training:
            dataset_batch = train_dataset.sample_sequence(
                config["batch_size"] // 2 * FLAGS.utd_ratio,
                sequence_length=FLAGS.horizon_length,
                discount=discount,
                balance_key=(
                    "stage_rewards" if FLAGS.stage_balanced_sampling else None
                ),
            )
            replay_batch = replay_buffer.sample_sequence(
                FLAGS.utd_ratio * config["batch_size"] // 2,
                sequence_length=FLAGS.horizon_length,
                discount=discount,
                balance_key=(
                    "stage_rewards" if FLAGS.stage_balanced_sampling else None
                ),
            )

            if use_latent_adapter_critic:
                replay_batch = dict(replay_batch)
                for k in ("actions", "next_actions"):
                    if k in replay_batch:
                        replay_batch[k] = adapter.decompose_chunked(
                            replay_batch[k], FLAGS.horizon_length
                        ).reshape(replay_batch[k].shape[0], FLAGS.horizon_length, -1)

            batch = {
                k: np.concatenate(
                    [
                        dataset_batch[k].reshape(
                            (FLAGS.utd_ratio, config["batch_size"] // 2)
                            + dataset_batch[k].shape[1:]
                        ),
                        replay_batch[k].reshape(
                            (FLAGS.utd_ratio, config["batch_size"] // 2)
                            + replay_batch[k].shape[1:]
                        ),
                    ],
                    axis=1,
                )
                for k in dataset_batch
            }

            agent, update_info["online_agent"] = agent.batch_update(batch)

        if moment_controlled and i % moment_interval == 0:
            online_rng, diagnostic_key, verification_key = jax.random.split(
                online_rng, 3
            )
            before_metrics = sample_executed_translation_geometry(
                agent,
                moment_observations,
                action_postprocessor,
                agent_action_dim,
                moment_samples,
                diagnostic_key,
            )
            controller_state = read_moment_controller_params(
                agent.network.params
            )
            controller_state = update_moment_controller(
                controller_state,
                before_metrics,
                moment_targets,
                moment_config,
            )
            controlled_params = replace_moment_controller_params(
                agent.network.params, controller_state
            )
            agent = agent.replace(
                network=agent.network.replace(params=controlled_params)
            )
            after_metrics = sample_executed_translation_geometry(
                agent,
                moment_observations,
                action_postprocessor,
                agent_action_dim,
                moment_samples,
                verification_key,
            )
            moment_info = {
                **after_metrics,
                "radius_mean_before": before_metrics["radius_mean"],
                "radius_std_before": before_metrics[
                    "radius_std_within_state"
                ],
                "angle_before": before_metrics["angular_deviation_rad"],
                "radius_mean_target": moment_targets.radius_mean,
                "radius_std_target": moment_targets.radius_std_within_state,
                "angle_target": moment_targets.angular_deviation_rad,
                "radius_mean_abs_error": abs(
                    after_metrics["radius_mean"] - moment_targets.radius_mean
                ),
                "radius_std_abs_error": abs(
                    after_metrics["radius_std_within_state"]
                    - moment_targets.radius_std_within_state
                ),
                "angle_abs_error": abs(
                    after_metrics["angular_deviation_rad"]
                    - moment_targets.angular_deviation_rad
                ),
                "angular_log_std": controller_state.angular_log_std,
                "speed_log_std": controller_state.speed_log_std,
                "speed_mean": controller_state.speed_mean,
                "control_radius_mean": float(
                    moment_config.control_radius_mean
                ),
                "num_updates": controller_state.num_updates,
            }
            logger.log(moment_info, "moment_controller", step=log_step)

        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0:
            evaluate_and_log(log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

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

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    end_time = time.time()

    if FLAGS.save_all_online_states:
        c_data = {
            "steps": np.array(data["steps"]),
            "qpos": np.stack(data["qpos"], axis=0),
            "qvel": np.stack(data["qvel"], axis=0),
            "obs": np.stack(data["obs"], axis=0),
            "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, "token.tk"), "w") as f:
        f.write(str(run.url or ""))


if __name__ == "__main__":
    app.run(main)
