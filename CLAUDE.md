# QC — RLPD with Direction-Speed Action Head

OGBench / RoboMimic 离线+在线强化学习实验仓库。

## 环境

```bash
source .venv/bin/activate
export MUJOCO_GL=egl
```

GPU: 2× RTX 6000 Ada (48GB)。GPU 1 常被 RoboTwin 占用，默认用 GPU 0。

## 并行训练（单卡多进程）

**单卡只能跑满 ~33% GPU，因为 CPU 数据管线是瓶颈。并行化在数据管线等 CPU 时让其他进程用 GPU。**

```bash
# 8 并发，每进程 4 核，显存预分配 5%
for seed in 0 1 2 3 4 5 6 7; do
    taskset -c $((seed*4))-$((seed*4+3)) env \
        CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python main_online.py --seed=$seed ... &
done
wait
```

关键参数：
- `taskset -c <start>-<end>` — 绑定 CPU 核，避免进程争抢
- `XLA_PYTHON_CLIENT_MEM_FRACTION` — 限制每进程显存预分配，按显卡总显存/任务数计算（如 48GB 卡跑 8 任务约 5%，24GB 卡跑 6 任务约 15%）；每个任务实际需约 1.5GB，分配 4GB 安全
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — 禁用 JAX 显存预分配，动态分配（多进程必需）
- 8 并发 × 4 核 = 32 核全用，总吞吐 ~535 it/s（单进程 113 it/s 的 4.7×）
- 4 并发 × 8 核 = 32 核全用，总吞吐 ~480 it/s，每进程更快（适合少 seed）

详见 `docs/parallel_benchmark.md`。

## 主要实验命令

### RLPD baseline（TanhNormal, chunk=true）
```bash
python main_online.py --env_name=cube-triple-play-singletask-task2-v0 \
    --sparse=False --horizon_length=1
```

### DS-RLPD（DirectionSpeed, chunk=false）★ 最佳
```bash
python main_online.py --env_name=cube-triple-play-singletask-task2-v0 \
    --sparse=False --horizon_length=1 \
    --agent.use_direction_speed=true --agent.action_chunking=false
```

### RLPD-AC / DS-RLPD-AC（H=5, n-step returns）
```bash
python main_online.py --env_name=cube-triple-play-singletask-task2-v0 \
    --sparse=False --horizon_length=5 \
    --agent.use_direction_speed=true --agent.action_chunking=false
```

### 其他常用 flag
- `--seed=0`（默认）
- `--online_steps=1000000`
- `--save_interval=500000`
- `--eval_episodes=50 --video_episodes=0`（训练时关闭视频渲染，省显存）
- `--agent.use_direction_speed=true`（启用 DS head）
- `--agent.action_chunking=false`（DS 必须关 chunk）

详见 `docs/experiment.md`。
