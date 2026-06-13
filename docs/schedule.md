# Schedule.py — GPU 实验调度器

## 概述

`schedule.py` 将 `tasks.json` 编译为命令列表，按 GPU slot 空闲事件驱动执行，支持断点续跑。

## 快速开始

```bash
# 1. 编写 tasks.json（模板见 tasks.example.json）
# 2. 编译 + 审查
python schedule.py --compile --dry_run

# 3. 确认无误后，启动实验
python schedule.py --gpus=0,1 --gpu_tasks=4
```

## 工作流

```
tasks.json ──compile──▶ tasks.compiled.json ──run──▶ 实验
                │                        │
                │  人工审查               │  原地更新 status
                │                        │  中断重跑自动续
```

### 第一步：编译

```bash
python schedule.py --compile              # 编译 tasks.json → tasks.compiled.json
python schedule.py --compile --dry_run    # 编译 + 打印完整命令列表（不执行）
```

### 第二步：执行

```bash
python schedule.py --gpus=0,1 --gpu_tasks=4                    # 执行（compiled 存在时直接跑）
python schedule.py --run=tasks.compiled.json --gpus=0,1        # 显式指定编译文件
python schedule.py --run=tasks.compiled.json --dry_run         # 预览状态
```

编译文件已存在时，`python schedule.py` 默认直接执行，不会重新编译。修改 `tasks.json` 后需显式 `--compile` 重新生成。

## 中断恢复

Ctrl+C 或意外中断后，进度已保存在 `tasks.compiled.json`。直接重跑：

```bash
python schedule.py --gpus=0,1 --gpu_tasks=4
```

| compiled.json 中 status | 恢复行为 |
|------------------------|---------|
| `"done"` | 跳过 |
| `"running"` + PID 存活 | 重连跟踪 |
| `"running"` + PID 已死 | 重排队 |
| `"pending"` | 重排队 |

Ctrl+C 时调度器会 SIGTERM 所有子进程，等 2s 后 SIGKILL 残留，不留孤儿进程。

## 调度逻辑

命令队列 = tasks × seeds 展开，按 GPU slot 空闲事件驱动：

1. 首次填充：每个 GPU 逐个启动直至满员，间隔 `--stagger` 秒错开 JIT 编译
2. 进程结束 → 立即从队列取下一个分配到该 GPU，保持满载
3. 同 GPU 启动间隔 ≥ `--stagger` 秒
4. 启动/完成时原地更新 `tasks.compiled.json`

## CLI 参数

| Flag | 默认值 | 说明 |
|------|-------|------|
| `--tasks_config` | `tasks.json` | 任务配置文件 |
| `--run` | 自动检测 | 编译后的命令文件 |
| `--compile` | `False` | 仅编译，不执行 |
| `--gpus` | `0` | GPU ID 列表，逗号分隔 |
| `--gpu_tasks` | `1` | 每卡并行进程数 |
| `--stagger` | `30` | 同 GPU 启动间隔秒数 |
| `--dry_run` | `False` | 只打印，不执行 |

## tasks.json 格式

```jsonc
{
  "common": {                          // 所有 task 共享的默认值
    "online_steps": 1000000,
    "eval_episodes": 50,
    "video_episodes": 0,
    "sparse": false
  },
  "tasks": [
    {
      "name": "ds_rlpd_task2",         // 必填，log 文件名
      "entry": "main_online.py",       // main_online.py 或 main.py
      "env_name": "cube-triple-play-singletask-task2-v0",
      "horizon_length": 1,
      "ds_mode": "stereographic"
    }
  ],
  "seeds": [0]
}
```

task 字段可覆写 `common`。完整字段列表见 `schedule.py` 中 `FIELD_TO_FLAG`。

## 支持的环境

| 来源 | 格式 | 示例 |
|------|------|------|
| OGBench | `{env}-singletask{ -taskN}-v0` | `cube-triple-play-singletask-task2-v0` |
| RoboMimic | `{task}-{type}-low_dim` | `lift-ph-low_dim`, `square-mh-low_dim` |

## 每卡并发数参考

| GPU | `--gpu_tasks` |
|-----|--------------|
| RTX 6000 Ada (48GB) | 6 |
| RTX 4090 (24GB) | 4 |
| 其他 | 1（保守） |

公式：`并发 ≈ 可用显存 / (1.5GB × H因子)`

## 运行时输出

```
────────────────────────────────────────────────────────────
  [0] [✓] ds_rlpd_task2  seed=0    ← ✓=已完成 ·=待执行
  [1] [·] ds_rlpd_task2  seed=1
────────────────────────────────────────────────────────────

[14:32:05] ▶ GPU=0  ds_rlpd_task2  seed=1  pid=12345  [2/4]  GPU0[██░░] slot_free=2
[14:35:12] ✓ GPU=0  ds_rlpd_task2  seed=0  pid=11111  OK  [1/4]
[14:35:12] GPU0:[█░░░]  GPU1:[███░]  queue=2  done=2/4  ETA: 12.3m  elapsed=3.1m

============================================================
  结束时间: 2026-06-13 14:45:00
  总耗时:   31.2m (0.52h)
  完成:     4/4
  成功:     4
============================================================
```

- `▶` 启动，`✓` 完成
- GPU 负载条 `[██░░]` = 已用/空闲 slot
- 每 60s 心跳输出进度 + ETA
- 中断 Ctrl+C 保存进度后退出
