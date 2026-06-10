# 实验数据

## 目录结构

```
data/
├── README.md
├── DirSpeed_FQL_H5_2M/            # FQL + H=5 + 离线1M→在线1M（cube-triple, seed=0）
│   └── {env}/{seed}/              #   eval.csv + flags.json
├── DirSpeed_RLPD_H5_500K/         # RLPD-AC + H=5 + 纯在线500K（cube-triple + cube-double, 3 seeds）
│   └── {env}/{seed}/              #   eval.csv + flags.json
└── parallel_benchmark/            # 并行训练压力测试
    ├── README.md
    ├── RTX4090_24GB/results.json
    └── RTX6000Ada_48GB/results.json
```

## 复制内容

原始实验目录位于 `exp/qc/`，每个实验目录包含以下文件：

| 文件 | 大小 | 内容 | 是否复制 |
|------|------|------|:--:|
| `eval.csv` | ~2KB | 每 100K 步的 eval 结果（success, return, episode length 等） | ✅ |
| `flags.json` | ~2KB | 完整运行参数（agent 配置、horizon_length、decompose 等） | ✅ |
| `env.csv` | ~8MB | 每步环境交互记录（qpos, qvel, reward, distance 等） | ✅ |
| `online_agent.csv` | ~22KB | 在线阶段训练 loss（critic_loss, actor_loss, alpha 等） | ✅ |
| `offline_agent.csv` | ~22KB | 离线阶段训练 loss（仅离线预训练实验有） | — |
| `token.tk` | ~52B | wandb run URL | ❌ |

> 以上文件均已从 `exp/qc/` 复制到各实验目录。`env.csv` 按步记录环境状态，日常分析一般不需要；`online_agent.csv` 可用于检查训练是否收敛、loss 是否异常。

## 实验分组说明

> ⚠️ **DirSpeed_FQL_H5_2M 和 DirSpeed_RLPD_H5_500K 使用的是错误的 post-hoc 实现**：
> 方向归一化时将所有 5 维（dx, dy, dz, dyaw, gripper）一起做了归一化，而不是只对前 3 维空间位移做。这导致 yaw 和 gripper 标量被错误地参与方向计算。
> 正确的做法见 [approach.md](../approach.md)：只对 3D 空间位移做方向-速度分解，yaw/gripper 保持 tanh 标量。
> 这两个目录的实验结果仍有参考价值（D+1 表示的方向-速度概念验证），但不能作为严格的概率方法对照。

### DirSpeed_FQL_H5_2M（2 个实验）

| ID | Agent | H | Decomp | 步数 | 成功率 |
|----|-------|:--:|------|------|:--:|
| `164019` | FQL | 5 | raw | 2M | 88% |
| `180505` | FQL | 5 | dir+speed | 2M | 98% |

> 入口：`main.py`，离线 1M + 在线 1M。使用错误的 5D 归一化。

### DirSpeed_RLPD_H5_500K（18 个实验）

| 环境 | 配置 | Seeds | 步数 |
|------|------|:--:|------|
| cube-triple | raw, dir+speed | 0,1,2 | 500K 在线 |
| cube-double | raw, dir+speed | 0,1,2 | 500K 在线 |

> 入口：`main_online.py`，纯在线 RLPD-AC。使用错误的 5D 归一化。

### ds_experiments（32 个实验）★ 正确实现

> 使用正确的 3D 方向-速度分解（只对 dx,dy,dz 做），通过 `--ds_mode` 控制实现方式。
> 详见 [experiments.md](../experiments.md#三种-ds-实现--baseline-对比实验-1m-步)。

| 批次 | H | chunk | 组数 × seed | 总实验 |
|------|:--:|:---:|:---:|:---:|
| ds_h5_1m | 5 | true | 4 组 × 4 seed | 16 |
| ds_h1_1m | 1 | false | 4 组 × 4 seed | 16 |

### FullMatrix（7 个有效实验）

| ID | Agent | H | Decomp | 步数 | 成功率 |
|----|-------|:--:|------|------|:--:|
| `212440` | RLPD-AC | 1 | raw | 2M | 2% |
| `212445` | RLPD-AC | 5 | raw | 2M | 2% |
| `213220` | RLPD-AC | 5 | dir+speed | 2M | 8% |
| `231752` | RLPD-AC | 5 | raw | 2M | 0% |
| `231753` | RLPD-AC | 5 | dir+speed | 2M | 2% |
| `022018` | FQL | 1 | raw | 2M | — |
| `022019` | FQL | 1 | dir+speed | 2M | — |

> 入口：`main.py`，离线 1M + 在线 1M。H=1 实验在线 eval 未写入 CSV。

## eval.csv 字段

| 列 | 含义 |
|----|------|
| `step` | 总训练步数 |
| `success` | 50 个 eval episode 的成功比例 |
| `episode.return` | 平均 episode return |
| `episode.length` | 平均 episode 长度 |
| `episode.duration` | 平均 episode 时长（秒） |

## flags.json

每个实验的完整运行参数，包括 agent 配置、horizon_length、action_decompose 等。
