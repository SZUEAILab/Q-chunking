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
