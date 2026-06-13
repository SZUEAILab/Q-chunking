# DS — RLPD with Direction-Speed Action Head

OGBench / RoboMimic 离线+在线强化学习实验仓库。

## 架构

| 入口               | 范式              | Agent                      |
| ------------------ | ----------------- | -------------------------- |
| `main_online.py` | 纯在线 RL（RLPD） | `agents/acrlpd.py` (SAC) |
| `main.py`        | 离线→在线（QC）  | `agents/acfql.py` (FQL)  |

```
agents/               — ACRLPD, ACFQL
rlpd_distributions/   — TanhNormal, DirectionSpeed 动作分布
rlpd_networks/        — MLP, StateActionValue, Ensemble
envs/                 — OGBench, RoboMimic 环境封装
utils/                — ReplayBuffer, flax 工具, encoders
```

DS 模式：`none` (TanhNormal) → `posthoc` (消融) → `stereographic` / `spherical` (Jacobian)。

Agent 配置（`bc_alpha`, `num_qs` 等）在 `--agent` 指向的 config 文件里，不在 CLI flag。

## 环境

```bash
source .venv/bin/activate
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.05
```

`CUDA_VISIBLE_DEVICES` 由调度器自动设置。

## 实验配置

任务矩阵通过 JSON 定义，模板见 [tasks.example.json](tasks.example.json)：

```jsonc
{
  "common": { "online_steps": 1000000, "eval_episodes": 50, "video_episodes": 0 },
  "tasks": [
    {
      "name": "ds_rlpd_task2",
      "entry": "main_online.py",
      "env_name": "cube-triple-play-singletask-task2-v0",
      "horizon_length": 1,
      "ds_mode": "stereographic"
    }
  ],
  "seeds": [0]
}
```

**task 字段**（可覆写 `common`）：

| 字段                                   | 说明                                                                                                   |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `name`                               | **必填**，log 文件名和调度器标签                                                                 |
| `entry`                              | `main_online.py` / `main.py`                                                                       |
| `env_name`                           | 见下方环境表                                                                                           |
| `horizon_length`                     | 1 或 5                                                                                                 |
| `ds_mode`                            | `none` / `posthoc` / `stereographic` / `spherical`                                             |
| `action_chunking`                    | bool，覆写 agent 默认值                                                                                |
| `offline_steps`                      | int，仅 `main.py` 需要                                                                               |
| `save_interval` | int，ckpt 保存间隔（步数） |
| `run_group` | wandb run group，默认由入口脚本设定 |
| `allow_posthoc_direction_speed_rlpd` | bool，`ds_mode=posthoc` 且 `entry=main_online.py` 时 **必须设为 `true`**，否则代码主动拒绝 |

其他可透传 flag：`sparse`, `discount`, `utd_ratio`, `buffer_size`, `save_interval`, `dataset_proportion` 等，完整列表见 `schedule.py` 中 `FIELD_TO_FLAG`。

**工作流：Claude 根据用户需求生成 `tasks.json` → 用户审查确认 → 启动实验。**

### 支持的环境

| 来源                | 格式                             | 示例                                                                                                               |
| ------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **OGBench**   | `{env}-singletask{ -taskN}-v0` | `cube-triple-play-singletask-task2-v0`, `cube-double-play-singletask-v0`                                       |
| **RoboMimic** | `{task}-{type}-low_dim`        | `lift-ph-low_dim`, `can-mh-low_dim`, `square-ph-low_dim`, `transport-ph-low_dim`, `tool_hang-ph-low_dim` |

RoboMimic `{type}`: `ph` = proficient human, `mh` = multi human。

## GPU 与并行

### GPU 检测

```bash
nvidia-smi --query-gpu=index --format=csv,noheader | wc -l          # GPU 数量
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader  # 剩余显存
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader       # 占用进程
```

### 每卡并发数

默认 `--gpu_tasks=1`。已知测试结果：

| GPU                 | `--gpu_tasks`   |
| ------------------- | ----------------- |
| RTX 6000 Ada (48GB) | 6                 |
| RTX 4090 (24GB)     | 4                 |
| 其他                | 1，按显存公式估算 |

公式（参考）：`并发 ≈ 可用显存 / (1.5GB × H因子)`，H=1 因子 1.0，H=5 因子 1.7。

CPU 不用手动分配——Python GIL + JAX GPU dispatch 意味着 CPU 不是瓶颈。

### JIT 编译错开

多进程同时启动 → JAX 同时编译 → 显存峰值叠加 → OOM。
调度器在同一 GPU 两次启动间强制间隔 `--stagger=30`s。已验证 H=5 4 并发加 stagger 后全过。

## 启动实验

命名约定：`xxx.json` → `xxx.compiled.json`。当 compiled 文件**已存在**时，`--tasks_config=xxx.json` 默认读取编译文件执行，不会重新编译。

```bash
# 首次：编译并审查
python schedule.py --tasks_config=tasks.json --compile
# → 生成 tasks.compiled.json，逐条打印命令供审查

# 审查通过后，以下命令等价——都默认走 compiled 文件：
python schedule.py --tasks_config=tasks.json --gpus=0,1 --gpu_tasks=4
python schedule.py --run=tasks.compiled.json --gpus=0,1 --gpu_tasks=4
python schedule.py --gpus=0,1 --gpu_tasks=4                     # 自动找 ./tasks.compiled.json

# 强制重编译（修改了 tasks.json 后）
python schedule.py --tasks_config=tasks.json --compile

# 后台执行
nohup python -u schedule.py --gpus=0,1 --gpu_tasks=4 &> schedule.log &
tail -f schedule.log
```

> `python -u` 必须加，否则 stdout redirect 后缓冲导致日志空白。

调度逻辑：tasks × seeds 展开为命令队列 → 按 GPU slot 空闲事件驱动：

1. **进程结束** → 立即从队列取下一个分配到该 GPU，保持满载
2. 同 GPU 两次启动间 `--stagger=30s` 冷却
3. 启动/完成时**原地更新** `tasks.compiled.json` 中对应命令的 `status`
4. 队列空 + 全部完成 → 打印汇总

### 中断恢复

Ctrl+C 或意外中断后，**进度已保存在 `tasks.compiled.json`**。直接重跑相同命令即可自动续跑：

```bash
python -u schedule.py --run=tasks.compiled.json --gpus=0,1 --gpu_tasks=4
```

- `status: "done"` → 跳过
- `status: "running"` + PID 存活 → 重连跟踪
- `status: "running"` + PID 已死 / `"pending"` → 重新排队

## 实验速查

| 实验          | 关键字段                                                                              |
| ------------- | ------------------------------------------------------------------------------------- |
| RLPD baseline | `entry: main_online.py`, `H: 1`, `ds: none`                                     |
| DS-RLPD ★    | `entry: main_online.py`, `H: 1`, `ds: stereographic`                            |
| RLPD-AC       | `entry: main_online.py`, `H: 5`, `action_chunking: true`                        |
| DS-RLPD-AC    | `entry: main_online.py`, `H: 5`, `ds: stereographic`, `action_chunking: true` |
| QC ★         | `entry: main.py`, `H: 5`, `offline_steps: 1000000`                              |
| DS 消融       | `ds: posthoc`, `allow_posthoc_direction_speed_rlpd: true`（RLPD 必须）            |

详见 `python schedule.py --helpfull` 和 `docs/experiments.md`。
