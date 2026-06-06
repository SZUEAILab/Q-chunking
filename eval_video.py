"""
用保存的 checkpoint 跑评估并生成 rollout 视频
用法: MUJOCO_GL=egl python eval_video.py <实验目录>
示例: MUJOCO_GL=egl python eval_video.py exp/qc/reproduce/.../sd00020260603_213306
"""
import os, sys, json, pickle
import jax
import numpy as np
import imageio
from evaluation import evaluate
from agents import agents
from utils.flax_utils import restore_agent_with_file
from envs.env_utils import make_env_and_datasets
import tqdm

NUM_VIDEO_EPISODES = 5
VIDEO_FRAME_SKIP = 1
RENDER_HEIGHT = 480
RENDER_WIDTH = 480

def main():
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if exp_dir is None:
        # find latest RLPD reproduce run
        import glob
        candidates = sorted(glob.glob('exp/qc/reproduce/*/sd*/'))
        exp_dir = candidates[-1] if candidates else None
        if exp_dir is None:
            print("No experiment directory found")
            return

    print(f"实验目录: {exp_dir}")

    # 读取配置
    with open(os.path.join(exp_dir, 'flags.json')) as f:
        flags = json.load(f)

    config = flags['agent']

    # 还原训练时的环境变量
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
        os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

    # 设置 horizon_length
    horizon_length = flags.get('horizon_length', 1)
    config['horizon_length'] = horizon_length

    # 创建环境
    env_name = flags['env_name']
    ogbench_dataset_dir = flags.get('ogbench_dataset_dir')

    if ogbench_dataset_dir is not None:
        from envs.ogbench_utils import make_ogbench_env_and_datasets
        env, eval_env, _, _ = make_ogbench_env_and_datasets(
            env_name, dataset_path=None, compact_dataset=False)
    else:
        env, eval_env, _, _ = make_env_and_datasets(env_name)

    # 获取 action_dim
    action_dim = eval_env.action_space.shape[-1]

    # 提高渲染分辨率 (需要修改 MuJoCo offscreen framebuffer)
    for env_obj in [env, eval_env]:
        uw = env_obj.unwrapped
        uw.model.vis.global_.offwidth = RENDER_WIDTH
        uw.model.vis.global_.offheight = RENDER_HEIGHT
        uw._render_height = RENDER_HEIGHT
        uw._render_width = RENDER_WIDTH
        uw._renderer = None  # 强制重建 renderer
    print(f"渲染分辨率: {RENDER_WIDTH}x{RENDER_HEIGHT}")

    # 创建 agent (需要 example batch 来初始化网络形状)
    # 用零 batch 模拟
    dummy_obs = np.zeros((1, eval_env.observation_space.shape[-1]), dtype=np.float32)
    dummy_act = np.zeros((1, action_dim), dtype=np.float32)

    seed = flags.get('seed', 0)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        seed,
        dummy_obs,
        dummy_act,
        config,
    )

    # 自动找最新 checkpoint (按数值排序)
    import glob as _glob, re as _re
    ckpt_candidates = sorted(_glob.glob(os.path.join(exp_dir, 'params_*.pkl')),
                            key=lambda x: int(_re.search(r'params_(\d+)\.pkl', x).group(1)))
    if not ckpt_candidates:
        print(f"No checkpoint found in {exp_dir}")
        return
    ckpt_path = ckpt_candidates[-1]  # 最新的 (步数最大)
    ckpt_epoch = int(os.path.basename(ckpt_path).replace('params_', '').replace('.pkl', ''))
    print(f"Checkpoint: params_{ckpt_epoch}.pkl ({ckpt_epoch/1e6:.1f}M步)")

    agent = restore_agent_with_file(agent, ckpt_path)

    # 跑评估 + 录视频
    print(f"\n开始评估，录制 {NUM_VIDEO_EPISODES} 条视频...")

    eval_info, _, renders = evaluate(
        agent=agent,
        env=eval_env,
        action_dim=action_dim,
        num_eval_episodes=10,  # 统计用
        num_video_episodes=NUM_VIDEO_EPISODES,
        video_frame_skip=VIDEO_FRAME_SKIP,
    )

    # 保存统计
    print(f"\n评估结果:")
    for k, v in sorted(eval_info.items()):
        print(f"  {k}: {v:.4f}")

    # 保存视频
    video_dir = os.path.join(exp_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)

    for i, render in enumerate(renders):
        if len(render) > 0:
            video_path = os.path.join(video_dir, f'rollout_{i}.mp4')
            imageio.mimsave(video_path, render, fps=20)  # control_timestep=0.05s → 20fps = 1x 真实速度
            print(f"  视频 {i}: {len(render)} 帧 → {video_path}")

    env.close()
    eval_env.close()
    print(f"\n视频保存到: {video_dir}/")

if __name__ == '__main__':
    main()
