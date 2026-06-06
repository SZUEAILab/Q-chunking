"""
数据集重放脚本 — 在仿真器中回放演示轨迹并保存视频
用法: MUJOCO_GL=egl python replay_datasets.py [--task all|cube-double|cube-triple|scene|puzzle|robomimic] [--episodes 3]
"""
import os, sys, argparse
import numpy as np
import imageio
import gymnasium

os.makedirs('replay_videos', exist_ok=True)

# ============================================================
# OGBench 重放
# ============================================================
def replay_ogbench(task_name, label, num_episodes=3):
    """重放 OGBench 数据集中的轨迹"""
    import ogbench

    print(f"\n{'='*60}")
    print(f"重放: {label} ({task_name})")
    print(f"{'='*60}")

    # 加载数据和环境
    data = np.load(f'/home/ym/.ogbench/data/{task_name}.npz')
    env, _, _ = ogbench.make_env_and_datasets(task_name)

    # 获取底层 MuJoCo 环境
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    if hasattr(base_env, 'disable_render_order_enforcing'):
        base_env.disable_render_order_enforcing = True

    # 提高渲染分辨率
    base_env.model.vis.global_.offwidth = 480
    base_env.model.vis.global_.offheight = 480
    base_env._render_height = 480
    base_env._render_width = 480

    # 按 terminals 分割轨迹
    terminals = data['terminals']
    term_idx = np.where(terminals)[0]
    traj_starts = np.concatenate([[0], term_idx[:-1] + 1])
    traj_ends = term_idx + 1

    print(f"  总轨迹数: {len(traj_starts)}, 每条约 {traj_ends[0] - traj_starts[0]} 步")

    # 回放前 num_episodes 条轨迹
    for ep in range(min(num_episodes, len(traj_starts))):
        start, end = traj_starts[ep], traj_ends[ep]
        frames = []

        # 先 reset 一次绕过 render 检查
        base_env.reset()
        # Scene/Puzzle 环境需要 button_states
        qp, qv = data['qpos'][start], data['qvel'][start]
        bs = data.get('button_states')
        if bs is not None:
            base_env.set_state(qp, qv, bs[start])
        else:
            base_env.set_state(qp, qv)

        for t in range(start, min(end, start + args.max_steps)):
            qp, qv = data['qpos'][t], data['qvel'][t]
            if bs is not None:
                base_env.set_state(qp, qv, bs[t])
            else:
                base_env.set_state(qp, qv)
            img = base_env.render()
            frames.append(img)

            # 模拟执行动作看下一步
            # (仅用于视觉连续性)

        # 保存视频 (20fps = 1x 真实速度, control_timestep=0.05s)
        os.makedirs('replay_videos', exist_ok=True)
        video_path = f'replay_videos/{task_name}_ep{ep}.mp4'
        imageio.mimsave(video_path, frames, fps=20)
        print(f"  Episode {ep}: {len(frames)}帧 → {video_path}")

    env.close()
    return len(traj_starts)


# ============================================================
# RoboMimic 重放
# ============================================================
def replay_robomimic(task, label, num_episodes=3):
    """重放 RoboMimic 数据集中的演示"""
    import h5py
    import robomimic
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    print(f"\n{'='*60}")
    print(f"重放: {label} ({task})")
    print(f"{'='*60}")

    # 初始化 robomimic
    dataset_path = f'/home/ym/.robomimic/{task}/mh/low_dim_v15.hdf5'

    # 加载数据集获取环境元信息
    f = h5py.File(dataset_path, 'r')
    env_meta = dict(f['data'].attrs)
    # 获取第一条 demo 信息来确定环境
    demo0 = f['data/demo_0']
    env_name = env_meta.get('env_name', task)
    f.close()

    # 创建环境
    env_meta_to_pass = {
        'env_name': env_meta.get('env_name', task),
        'type': env_meta.get('env_type', 1),
        'env_kwargs': env_meta.get('env_kwargs', {}),
    }

    try:
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta_to_pass,
            render=False,
            render_offscreen=True,
            use_image_obs=False,
        )
        env.reset()
    except Exception as e:
        print(f"  ⚠ robomimic 环境创建失败: {e}")
        print(f"  改为直接读取数据做统计可视化...")
        replay_robomimic_from_data(task, label, num_episodes)
        return

    # 重放
    f = h5py.File(dataset_path, 'r')
    for ep in range(min(num_episodes, 300)):
        demo = f[f'data/demo_{ep}']
        obs_dict = {k: demo['obs'][k][:] for k in demo['obs'].keys()}
        actions = demo['actions'][:]

        frames = []
        env.reset()

        for t in range(len(actions)):
            # 尝试设置 mujoco 状态
            if hasattr(env, 'sim'):
                # 从 obs 提取状态信息来设置
                pass  # robosuite 状态设置较复杂
            obs, reward, done, info = env.step(actions[t])
            img = env.render(mode='rgb_array', height=240, width=240)
            frames.append(img)
            if done:
                break

        video_path = f'replay_videos/robomimic_{task}_ep{ep}.mp4'
        imageio.mimsave(video_path, frames, fps=20)
        print(f"  Demo {ep}: {len(frames)}帧 → {video_path}")

    f.close()
    env.close()


def replay_robomimic_from_data(task, label, num_episodes=3):
    """当无法创建 robomimic 环境时，从数据生成轨迹动画"""
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f = h5py.File(f'/home/ym/.robomimic/{task}/mh/low_dim_v15.hdf5', 'r')

    for ep in range(min(num_episodes, 300)):
        demo = f[f'data/demo_{ep}']
        steps = demo['actions'].shape[0]

        # 提取关键观测用于可视化
        eef_pos = demo['obs/robot0_eef_pos'][:]  # 末端执行器位置
        gripper = demo['obs/robot0_gripper_qpos'][:]  # 夹爪
        obj = demo['obs/object'][:]  # 物体状态

        frames = []
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        for t in range(steps):
            for ax in axes.flat:
                ax.clear()

            # 末端执行器轨迹 (3D)
            ax = axes[0, 0]
            ax.plot(eef_pos[:t+1, 0], eef_pos[:t+1, 1], 'b-', alpha=0.5, linewidth=1)
            ax.scatter([eef_pos[t, 0]], [eef_pos[t, 1]], c='red', s=50, zorder=5)
            ax.set_xlabel('X'); ax.set_ylabel('Y')
            ax.set_title(f'末端执行器 XY 轨迹 (t={t}/{steps})')
            ax.set_xlim(eef_pos[:, 0].min()-0.05, eef_pos[:, 0].max()+0.05)
            ax.set_ylim(eef_pos[:, 1].min()-0.05, eef_pos[:, 1].max()+0.05)

            # Z 轴高度
            ax = axes[0, 1]
            ax.plot(range(t+1), eef_pos[:t+1, 2], 'b-', linewidth=1.5)
            ax.axhline(y=eef_pos[:, 2].max(), color='green', linestyle='--', alpha=0.3)
            ax.set_xlabel('步数'); ax.set_ylabel('Z 高度')
            ax.set_title('末端执行器 Z 高度')
            ax.set_xlim(0, steps)

            # 夹爪状态
            ax = axes[1, 0]
            ax.plot(range(t+1), gripper[:t+1, 0], 'r-', label='左指', linewidth=1.5)
            ax.plot(range(t+1), gripper[:t+1, 1], 'b-', label='右指', linewidth=1.5)
            ax.set_xlabel('步数'); ax.set_ylabel('夹爪位置')
            ax.set_title('夹爪状态')
            ax.legend(fontsize=8)
            ax.set_xlim(0, steps)

            # 物体状态 (前3维)
            ax = axes[1, 1]
            for d in range(min(3, obj.shape[1])):
                ax.plot(range(t+1), obj[:t+1, d], label=f'obj dim {d}', linewidth=1.5)
            ax.set_xlabel('步数'); ax.set_ylabel('物体状态')
            ax.set_title('物体状态变化')
            ax.legend(fontsize=8)
            ax.set_xlim(0, steps)

            fig.suptitle(f'{label} — Demo {ep}', fontsize=14)
            fig.tight_layout()

            # Render frame
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(img)

        plt.close(fig)

        video_path = f'replay_videos/robomimic_{task}_ep{ep}.mp4'
        imageio.mimsave(video_path, frames, fps=20)
        print(f"  Demo {ep}: {len(frames)}帧 → {video_path}")

    f.close()


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据集重放脚本')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'cube-double', 'cube-triple', 'scene', 'puzzle', 'robomimic'])
    parser.add_argument('--env', type=str, default=None, help='自定义环境名, 如 cube-triple-play-singletask-task2-v0')
    parser.add_argument('--episodes', type=int, default=3, help='每个任务重放的轨迹数')
    parser.add_argument('--max-steps', type=int, default=2000, help='每条轨迹最大帧数')
    args = parser.parse_args()

    total_episodes = 0

    if args.env:
        # 自定义环境: 从 env_name 推导 dataset 名
        import re
        dataset_name = re.sub(r'-singletask-task\d+', '', args.env)
        total_episodes += replay_ogbench(dataset_name, args.env, args.episodes)
    else:
        if args.task in ('all', 'cube-double'):
            total_episodes += replay_ogbench('cube-double-play-v0', 'Cube-Double (双方块)', args.episodes)

        if args.task in ('all', 'cube-triple'):
            total_episodes += replay_ogbench('cube-triple-play-v0', 'Cube-Triple (三方块)', args.episodes)

        if args.task in ('all', 'scene'):
            total_episodes += replay_ogbench('scene-play-v0', 'Scene (场景操作)', args.episodes)

        if args.task in ('all', 'puzzle'):
            total_episodes += replay_ogbench('puzzle-3x3-play-v0', 'Puzzle-3×3 (按钮谜题)', args.episodes)

        if args.task in ('all', 'robomimic'):
            for t, l in [('lift', 'Lift'), ('can', 'Can'), ('square', 'Square')]:
                replay_robomimic(t, l, args.episodes)

    print(f"\n{'='*60}")
    print(f"全部完成! 视频保存在 replay_videos/")
    print(f"文件列表:")
    for f in sorted(os.listdir('replay_videos')):
        size = os.path.getsize(f'replay_videos/{f}') / 1024
        print(f"  {f}  ({size:.0f} KB)")
    print(f"{'='*60}")
