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
| `env.csv` | ~8MB | 每步环境交互记录（qpos, qvel, reward, distance 等） | ❌ |
| `online_agent.csv` | ~22KB | 在线阶段训练 loss（critic_loss, actor_loss, alpha 等） | ❌ |
| `offline_agent.csv` | ~22KB | 离线阶段训练 loss | ❌ |
| `token.tk` | ~52B | wandb run URL | ❌ |

> 仅保留实验结果和分析所需的最小数据集。`env.csv` 26 个实验合计 ~200MB，按步记录环境状态，日常分析不需要。如需完整数据可在 `exp/qc/` 找到。

## 实验分组说明

### DirSpeed_FQL_H5_2M（2 个实验）

| ID | Agent | H | Decomp | 步数 | 成功率 |
|----|-------|:--:|------|------|:--:|
| `164019` | FQL | 5 | raw | 2M | 88% |
| `180505` | FQL | 5 | dir+speed | 2M | 98% |

> 入口：`main.py`，离线 1M + 在线 1M

### DirSpeed_RLPD_H5_500K（18 个实验）

| 环境 | 配置 | Seeds | 步数 |
|------|------|:--:|------|
| cube-triple | raw, dir+speed | 0,1,2 | 500K 在线 |
| cube-double | raw, dir+speed | 0,1,2 | 500K 在线 |

> 入口：`main_online.py`，纯在线 RLPD-AC

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
