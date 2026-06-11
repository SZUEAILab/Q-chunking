import glob, tqdm, wandb, os, json, random, time, jax

# Fix numpy 2.x compatibility for robosuite.
import numpy as _np
if not hasattr(_np.dtypes, 'StringDType'):
    _np.dtypes.StringDType = _np.dtypes.StrDType

from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
from posthoc_direction_speed import compose, decompose, decompose_chunked
import numpy as np

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']


FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 1000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acrlpd.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

# Direction+speed decomposition (first 3 dims = spatial, rest = yaw/grip scalars).
flags.DEFINE_enum(
    'ds_mode',
    'none',
    ['none', 'posthoc', 'stereographic', 'spherical'],
    'Direction-speed implementation: none, posthoc D+1, or Jacobian-corrected bijector.',
)
flags.DEFINE_bool('direction_speed', False, 'Deprecated alias for --ds_mode=posthoc.')
flags.DEFINE_bool(
    'allow_posthoc_direction_speed_rlpd',
    False,
    'Allow non-invertible D+1 post-hoc direction_speed with ACRLPD for ablations.',
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
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    ds_mode = FLAGS.ds_mode
    if FLAGS.direction_speed:
        if ds_mode != 'none':
            raise ValueError("Use either --ds_mode or deprecated --direction_speed, not both.")
        ds_mode = 'posthoc'

    legacy_agent_ds = bool(config.get("use_ds_bijector", False) or config.get("use_direction_speed", False))
    if config.get("use_direction_speed", False):
        config["use_ds_bijector"] = True
    if ds_mode in ('stereographic', 'spherical'):
        if legacy_agent_ds and config.get("ds_bijector_type", ds_mode) != ds_mode:
            raise ValueError("Conflicting DS settings between --ds_mode and agent config.")
        config["use_ds_bijector"] = True
        config["ds_bijector_type"] = ds_mode
    elif legacy_agent_ds:
        ds_mode = config.get("ds_bijector_type", "spherical")

    use_posthoc_ds = ds_mode == 'posthoc'
    use_agent_ds_bijector = ds_mode in ('stereographic', 'spherical')
    if use_agent_ds_bijector and config["agent_name"] != "acrlpd":
        raise ValueError("--ds_mode=stereographic/spherical is only implemented for ACRLPD.")
    if use_posthoc_ds and use_agent_ds_bijector:
        raise ValueError(
            "Use either post-hoc D+1 DS or a raw-action DS bijector, not both."
        )
    if (
        use_posthoc_ds
        and config["agent_name"] == "acrlpd"
        and not FLAGS.allow_posthoc_direction_speed_rlpd
    ):
        raise ValueError(
            "Post-hoc DS uses a non-invertible D+1 action representation, "
            "so ACRLPD/SAC log_prob cannot be Jacobian-corrected. For RLPD DS experiments, "
            "use --ds_mode=spherical or --ds_mode=stereographic. "
            "For an approximate ablation, also pass "
            "--allow_posthoc_direction_speed_rlpd=True."
        )
    
    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

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
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if is_robomimic_env(FLAGS.env_name):
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
    raw_action_dim = train_dataset['actions'].shape[-1]
    if use_posthoc_ds:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=decompose(train_dataset['actions'])))
        agent_action_dim = raw_action_dim + 1
    else:
        agent_action_dim = raw_action_dim

    def compose_posthoc_ds(sampled):
        sampled = np.asarray(sampled).reshape(-1, agent_action_dim)
        return compose(sampled)

    def make_eval_agent(base_agent):
        if not use_posthoc_ds:
            return base_agent

        class _PosthocDSEval:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def sample_actions(self, observations, rng=None):
                sampled = self._wrapped.sample_actions(observations, rng=rng)
                return compose_posthoc_ds(sampled)

        return _PosthocDSEval(base_agent)

    example_batch = train_dataset.sample(())

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    # transition from offline to online
    # Replay buffer stores raw actions (D), not decomposed (D+1).
    if use_posthoc_ds:
        _raw_ex = dict(example_batch)
        _raw_ex['actions'] = compose(example_batch['actions'][np.newaxis])[0]
        replay_buffer = ReplayBuffer.create(_raw_ex, size=FLAGS.buffer_size)
    else:
        replay_buffer = ReplayBuffer.create(example_batch, size=FLAGS.buffer_size)
        
    ob, _ = env.reset()
    
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
                action = jax.random.uniform(key, shape=(raw_action_dim,), minval=-1, maxval=1)
                action_chunk = np.array(action).reshape(-1, raw_action_dim)
            else:
                sampled = agent.sample_actions(observations=ob, rng=key)
                if use_posthoc_ds:
                    action_chunk = compose_posthoc_ds(sampled)
                else:
                    action_chunk = np.array(sampled).reshape(-1, raw_action_dim)

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
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
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
            dataset_batch = train_dataset.sample_sequence(config['batch_size'] // 2 * FLAGS.utd_ratio,
                        sequence_length=FLAGS.horizon_length, discount=discount)
            replay_batch = replay_buffer.sample_sequence(FLAGS.utd_ratio * config['batch_size'] // 2,
                sequence_length=FLAGS.horizon_length, discount=discount)

            if use_posthoc_ds:
                replay_batch = dict(replay_batch)
                for k in ('actions', 'next_actions'):
                    if k in replay_batch:
                        replay_batch[k] = decompose_chunked(
                            replay_batch[k], FLAGS.horizon_length
                        ).reshape(replay_batch[k].shape[0], FLAGS.horizon_length, -1)
                replay_batch = jax.tree_util.tree_map(lambda x: x, replay_batch)

            batch = {k: np.concatenate([
                dataset_batch[k].reshape((FLAGS.utd_ratio, config["batch_size"] // 2) + dataset_batch[k].shape[1:]),
                replay_batch[k].reshape((FLAGS.utd_ratio, config["batch_size"] // 2) + replay_batch[k].shape[1:])], axis=1) for k in dataset_batch}

            agent, update_info["online_agent"] = agent.batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            _eval_agent = make_eval_agent(agent)

            eval_info, _, _ = evaluate(
                agent=_eval_agent,
                env=eval_env,
                action_dim=raw_action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
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
            if use_posthoc_ds:
                train_dataset = train_dataset.copy(
                    add_or_replace=dict(actions=decompose(train_dataset['actions'])))


    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    end_time = time.time()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0), 
                 "qvel": np.stack(data["qpos"], axis=0), 
                 "obs": np.stack(data["obs"], axis=0), 
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

if __name__ == '__main__':
    app.run(main)
