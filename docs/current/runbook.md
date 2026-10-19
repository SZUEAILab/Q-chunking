# 运行与存储：日常手册

[文档入口](../index.md)

## 执行位置和入口

使用批次登记的代码目录与 Python 环境；独立工作树和 `main` 的实现可能不同。
启动前核对实际命令、source/config/data 身份，不能拿当前 `main` 代替旧运行源码。

| 入口 | 用途 |
|---|---|
| `main_online.py` | RLPD，在线交互并混合先验数据 |
| `main_rlpd_reference.py` | 原论文 RLPD learner 配方的 H1 OGBench 适配；默认 UTD=20 |
| `main.py` | FQL/QC 离线到在线训练 |
| `main_fql_reference.py` | 独立 H1 官方 FQL learner 的本地 native/posthoc driver，使用来源绑定的 JSON plan |
| `main_gac_reference.py` | GAC 单环境入口，JSON plan 明确区分纯在线与 50/50 prior 协议，核对真实更新和采样次数 |
| `schedule.py` | 编译任务矩阵、分配 GPU 槽位、回填队列 |

可复用的合成诊断在 `research/`：`shell`、`shell_offpolicy`、`stage_task`、
`contextual_rotation`、`joint_contextual_policy`，用 `python -m research.<模块> --help`。
统计与旧数据加载器统一在 `research.metrics`，调用时显式提供批次身份；这些加载器不是
reward-blind 验证器。根目录不再保留按单个批次命名的分析、训练或绘图入口。
新批次配置和临时执行工具放数据盘 `control/<批次>/`，只有可复用工具进入 `scripts/`。

独立 RLPD 入口的来源、数值验证与适配边界见[基线参考](../reference/rlpd_baselines.md)。
它没有接入已冻结的 action-adapter 首轮筛选，使用自己的 TrainState checkpoint 格式；
旧的 `network.step` 验收器不能直接用于该入口。

数据集安装细节见[数据集参考](../reference/datasets.md)。常用环境变量：

```bash
source .venv/bin/activate
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export QC_DATA_ROOT=/data/chainpray/qc
```

任务配置包含 `common`、`tasks`、`seeds`，模板见 [tasks.example.json](../../tasks.example.json)。
`tasks × seeds` 展开为命令。先编译、再检查命令；修改配置后要明确重新编译。

```bash
python schedule.py --tasks_config=<批次配置.json> --compile
python schedule.py --run=<批次配置.compiled.json> --dry_run
# 检查冻结配置、身份和输出目录后，再执行同一 compiled 文件。
```

平台 SSH 连接需按当前控制台命令构造：该平台使用集群 SOCKS 代理下的 `root@jobId`，
`debug-url` 的 `sshUrl` 可能只是容器 IP。核对 `sshState` 和集群 `sshProxyPath`，不能假定
API 返回完整 SSH 命令，也不能把容器 IP 直接当作代理目标别名。

## GPU 调度和异常

本地两张 RTX 6000 Ada 使用每卡 6 并发；云端 RTX 4090 首先显式使用每卡 8 并发。
用 `--gpus`、`--gpu_tasks`、`--stagger` 配置，错开 JIT 并在槽位释放后回填。
显存、CPU 环境仿真、GPU 更新与磁盘 I/O 都可能限制吞吐，以实测为准。

进程结束后检查真实终态和验收材料；不要仅凭 GPU 空闲判断实验完成。
恢复前核对父 attempt、checkpoint、预算和身份。调度器具有重排队行为，直接重跑命令可能重新
启动死亡的任务，因此异常队列先诊断、登记恢复 attempt，再按批次授权恢复。
保留失败和 incident，不自动改 seed、覆盖目录或因科学结果不佳重跑。

## 文件放在哪里

| 内容 | 位置 |
|---|---|
| 原始运行、评估、权重 | `/data/chainpray/qc/results/<批次>/` |
| 冻结配置、来源与验收 | `/data/chainpray/qc/control/<批次>/` |
| 调度和训练日志 | `/data/chainpray/qc/logs/` |
| 生成的分析表、图和报告 | `/data/chainpray/qc/reports/` |
| 历史证据和迁移清单 | `/data/chainpray/qc/archive/` |
| 正文直接展示的图片 | 主仓库 `docs/images/`，必须保留并跟踪 |
| 精简评估、配置与来源哈希 | `docs/data` HF submodule |

不重建 `exp`/`exp_logs` 兼容入口。旧路径绑定证据保留原位置，通过清单关联。
必要 checkpoint 按批次冻结策略保留；清理先列出目标和引用关系，未解决 incident 或论文依赖的
模型保持保留。空间不足时停止新队列，不自动删模型。单块 HDD 加 HF 摘要不构成完整模型备份。

## 分析和发布

1. 核对原始运行和评估，记录缺失、失败和排除原因；
2. 按冻结指标做 seed/block 配对分析，输出表格与图；
3. 将正文使用的图收录到 `docs/images/`，更新[结果](results.md)；
4. 发布精简包：配置、逐 seed 评估、状态、统计、图和 SHA-256；
5. 核对远端 revision，再在主仓库固定 submodule 指针。

默认不上传权重、轨迹、逐步日志或运维凭据；有用的负结果同样归档。
本机 Git 上传使用其已配置代理；下载抽查需采用可达路径。
只读取已固定的子模块版本时使用 `git submodule update --init docs/data`，不自动追随远端 latest。

目前 HF 新归档在 `summaries/2026-10-19/`，专用导出器是
[`archive_experiment_summaries.py`](../../scripts/archive_experiment_summaries.py)。
未来按批次发布、统一 catalog、自动封存/保留规划尚未实现，不能把这些设计当作现有功能。
需要查旧命令、压力测试和详细设计时见[历史运维记录](../archive/operations.md)。

## 最小维护检查

```bash
python scripts/check_docs.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_docs.py tests/test_summary_archive.py -q
```

正文更新这四份日常文档；单次实验进度写批次记录，避免每轮增加一份项目级说明。
