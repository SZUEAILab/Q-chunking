# 并行训练压力测试

## 硬件环境

| 项目 | 规格 |
|------|------|
| GPU | 1× NVIDIA RTX 6000 Ada Generation (48GB)，单卡测试 |
| CPU | 32 逻辑核，单 NUMA 节点 |
| OS | Linux (X11) |
| CUDA | 12.9 |
| Driver | 575.57.08 |

> 注意：实验室有两张 RTX 6000 Ada，但压力测试仅使用单卡（`CUDA_VISIBLE_DEVICES=0`），另一张卡被 RoboTwin 占用。

## 方法

每个实验跑 20,000 步（JAX 编译后取稳定速度），通过 `taskset` 绑定不同 CPU 核，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.10`（RLPD）/ `0.05`（DS-RLPD）限制显存预分配。命令：

```bash
taskset -c <start>-<end> env CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 \
    python main_online.py --env_name=cube-triple-play-singletask-task2-v0 \
    --sparse=False --horizon_length=1 \
    --online_steps=20000 --eval_episodes=2 &
```

DS-RLPD 加 `--agent.use_direction_speed=true --agent.action_chunking=false`。

## 结果

### RLPD (TanhNormal)

| 并发数 | 核/进程 | 总核数 | 总 it/s | 倍率 | 每进程 it/s | GPU 利用率 |
|:---:|:---:|:---:|:------:|:---:|:---:|:---:|
| 1 | 8 | 8 | 113 | 1.0× | 113 | ~33% |
| 2 | 8 | 16 | 214 | 1.9× | 107 | ~45% |
| 3 | 8 | 24 | 281 | 2.5× | 94 | ~55% |
| 4 | 4 | 16 | 336 | 3.0× | 84 | ~65% |
| 5 | 6 | 30 | 458 | 4.1× | 92 | ~75% |
| 6 | 5 | 30 | 497 | 4.4× | 83 | ~80% |
| **8** | **4** | **32** | **558** | **4.9×** | 70 | ~90% |
| 10 | 2 | 20 | ❌ | — | — | — |

### DS-RLPD (DirectionSpeedNormal)

| 并发数 | 核/进程 | 存活 | 总 it/s | 倍率 |
|:---:|:---:|:---:|:------:|:---:|
| **8** | **4** | **8/8** | **535** | **4.7×** |

> DS-RLPD 单进程 ~113 it/s，和 RLPD 一致。8 并发全部存活，总吞吐 535 it/s，与 RLPD 的 558 it/s 基本持平。DS distribution 的 GPU 计算更重（球坐标 bijector），但 CPU 瓶颈仍然占主导，并行吞吐几乎不受影响。

## 结论

- **最佳配置：8 并发 × 4 核，单卡总吞吐 535–558 it/s，是单进程的 4.7–4.9 倍**
- 10 并发崩溃，6/10 静默失败，推测 CUDA context 或 JAX 编译并发冲突
- RLPD 和 DS-RLPD 在 8 并发下表现一致，配置可通用
- 每实验实际显存 ~1.5GB，预分配 5–10% 足够，48GB 单卡显存非瓶颈
- 单进程 GPU 利用率仅 33%，CPU 数据管线是唯一瓶颈；并行化通过 GPU 分时复用解决，8 并发 GPU 利用率 ~90%

---

## RTX 4090 压力测试

### 硬件环境

| 项目 | 规格 |
|------|------|
| GPU | 1× NVIDIA GeForce RTX 4090 (24GB) |
| CPU | 32 逻辑核，单 NUMA 节点 |
| CUDA | 12.8 |
| Driver | 570.181 |

### 方法

每个实验跑 5,000 步（JAX 预热后取稳定速度），通过 `taskset` 绑定 CPU 核，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.05` 限制显存，`XLA_PYTHON_CLIENT_PREALLOCATE=false` 动态分配：

```bash
taskset -c <start>-<end> env CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python main_online.py --env_name=cube-triple-play-singletask-task2-v0 \
    --online_steps=5000 --eval_episodes=2 --eval_interval=0 --horizon_length=1 \
    --start_training=1000 --action_decompose=False &
```

> 与 6000 Ada 测试差异：`main.py` 路径为 `.venv/bin/python`（不用 `uv run`），`MEM_FRACTION=0.05` 而非 `0.10`，步数 5K 而非 20K，核数固定每进程 32/N。

### 结果

| 并发数 | 核/进程 | 总 it/s | 倍率 | 每进程 | GPU util | 显存 | 状态 |
|:--:|:--:|------|:--:|:--:|:--:|:--:|:--:|
| 1 | 32 | 228 | 1.0× | 228 | 33% | 2GB | ✅ |
| 2 | 16 | 349 | 1.5× | 175 | 68% | 2GB | ✅ |
| 4 | 8 | 411 | 1.8× | 103 | 96% | 4GB | ✅ |
| 6 | 5 | 453 | 2.0× | 75 | 100% | 6GB | ✅ |
| 8 | 4 | 505 | 2.2× | 63 | 100% | 8GB | ✅ |
| 10 | 3 | 506 | 2.2× | 51 | 100% | 10GB | ✅ |
| **12** | **2** | **554** | **2.4×** | 46 | 100% | 12GB | ✅ |

> N=12 全部存活，总吞吐 554 it/s。N=8 时 GPU 已达 100% 利用率，继续增加并发通过更细粒度的 GPU 分时复用仍可小幅提升吞吐。12 并发每进程仅 2 核，CPU 严重受限但 GPU 端仍有重叠空间。

### 双卡对比

| | RTX 4090 (24GB) | RTX 6000 Ada (48GB) |
|--|:--:|:--:|
| 单进程基线 | **228 it/s** | 113 it/s |
| 最佳并发 | **12** (全存活) | 8 (N=10 崩溃) |
| 最佳吞吐 | 554 it/s | **558 it/s** |
| 倍率 | 2.4× | **4.9×** |
| 瓶颈 | GPU 100% + CPU | CPU 数据管线 |
| 最大显存 | 12GB / 24GB | 12GB / 48GB |

### 结论

- **RTX 4090 最佳配置：8–12 并发，总吞吐 505–554 it/s**
- 4090 单进程 228 it/s 是 6000 Ada 113 it/s 的 **2×**，但并行倍率低（2.4× vs 4.9×）——因为 4090 GPU 更快，单进程已用 33% GPU，并行提升空间更小
- 12 并发全部存活（6000 Ada 在 N=10 已崩溃），显存仅用 12GB/24GB——24GB 显存非瓶颈
- 核数分配：`32/N` 自动均分，N=12 时每进程仅 2.7 核，CPU 严重受限但 JAX 编译 + MuJoCo 仿真仍可运行
- 与 6000 Ada 一致：每实验实际显存 ~1.5GB，`MEM_FRACTION=0.05` 足够
