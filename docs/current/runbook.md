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

本地观察控制器退出不等于云端训练退出。网络或 DNS 观察失败时，先核对准确实例的
远端 `started.json`、调度器 PID 与命令、终态文件和已有 launch lock，再检查所有本地
owner 的实际进程与脚本哈希。恢复观察前保存旧脚本、状态、PID 和日志；按依赖顺序恢复
owner，更新进程绑定，并确认新 owner 识别已有训练。不能用观察失败触发重新部署或重跑。

GAC 环境准备使用同一实例中的独立远端 owner，入口及绑定保存在数据盘
`control/gac_reference_comparison_20261019/environment-owner-binding.json`。它等待 FQL 完整终态
和四卡空闲，只执行一次冻结环境入口，实际等待子进程后记录退出码。已有 intent 或准备
失败均需检查，不能重新运行 staging 脚本；该 owner 不保存镜像或启动 GAC 科学矩阵。

[`research.observation_retry`](../../research/observation_retry.py) 为只读查询提供有界重试，
保留最终原始错误；认证、文件身份、schema 和平台业务错误立即上报。当前主观察器只对
GET 和已核实的 cluster-list 查询采用它，镜像保存、任务提交等写操作不进入重试。
SSH 查询 launch 状态的退出码必须明确区分 0、1 和连接错误，不能把 255 当作未启动。
2026-10-19 DNS 故障及七个 owner 的恢复材料保存在数据盘
`control/action_adapter_validation_20261019/monitor-recovery-dns-20261019/`；原科学训练未重试。

只读 health 探针复用平台查询时，重试事件必须写独立的
`health-observation-events.jsonl`，不能复用主 owner 的阶段文件回调。2026-10-20 的隔离修正
通过合成认证/客户端检查：GET 短暂失败可重试且不覆盖权威阶段，写请求失败只发出一次。
记录在数据盘 `control/action_adapter_validation_20261019/health-observation-isolation-20261020/`，
没有对训练、既有 owner 或科学配置执行恢复/重跑。

### 当前 UTD1 主线的共享槽位

已启用[并发部署修订](experiments.md#utd1-主线的并发部署修订2026-10-21)：Q40/M60
通过外置操作适配器使用每卡最多8个总槽（为C/F/GAC原队列保留1槽），两个协调器各自同卡错峰30秒；共用槽位锁约束总任务数。
原科学发布包与plan不改；不得重启原训练或直接重跑已有arm命令。原等待owner已经保留
快照并被操作owner接替，当前 started.json 包含 operational_policy_sha256 与 supersedes。
Q40 的真实 owner/spec/policy 绑定在数据盘 `control/posthoc_concurrency_20261021/owner-adoption.json`；
M60 最新绑定为同目录 `mapping-owner-adoption-r6.json`，各次退役的协调记录均保留。
完整只读查询：`python3 /data/chainpray/qc/control/posthoc_concurrency_20261021/observe.py`。
只检查旧控制器的 gate 会仍显示旧串行依赖；实际接纳门槛以已哈希的外置policy为准。
共同映射核对Q40原四项工程及科学接纳绑定后即可独立补充空槽，不等待Q40科学终态或整卡
空闲；原4/6项工程与独立恢复必须通过。M60 使用本机接纳互斥及共用槽位锁，以避开共享
存储的持续阻塞式互斥等待；不修改既有Q40进程或其锁文件。M60仅复核已完成工程，余下
身份首次执行。两个协调器之间不承诺严格30秒错峰；资源测量应报告这一部署条件。
显式操作修订不等于改变科学身份，不允许删除原intent或重新提交已有case。

Q40 的存活 owner 不能单独证明剩余身份继续接纳。定时器同时核对真实 science receipt
与活跃 driver；若已启动项全部真实验收退出、仍有未启动身份且无活跃 driver，触发技术
诊断。此时保留原退出链与锁，先审查再版本化仅接纳从未启动身份的控制器，不重跑已完成
或失败项；不得因观察到 PID 消失直接重启。

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

新 `main_online.py` 的评估日志记录从实际返回的完整非视频轨迹计算的
`evaluation_episodes_observed`；环境提供二值 success 时，另记
`evaluation_successes_observed` 并校验它与聚合 success 一致。配置 episode 数、视频数和
观测计数分别保留。未完成轨迹、部分 success 标签缺失或不一致聚合会显式报错；
环境完全不提供 success 时，只记录 episode 数，不伪造失败次数。
这项记录不回填旧日志；当前冻结首批仍按其 `flags_only` 限制解释。

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
首轮终态筛选图表由数据盘 `control/action_adapter_validation_20261019/watch_screen_figures.py`
接续既有配对汇总，真实产物在 `reports/action_adapter_validation_20261019/figures/`，
同时写入输入/输出 SHA 回执。检查生成结果后再把正文使用的图收录到 `docs/images/`；
控制目录中带 synthetic 水印的版式检查图不能当作实验结果。
需要查旧命令、压力测试和详细设计时见[历史运维记录](../archive/operations.md)。

## GAC 接续观察

本机持续 owner 的绑定、PID 与状态分别位于数据盘
`control/gac_image_continuation_20261020/owner-binding.json`、`owner-process.json` 和
`owner-r2/status.json`。查看状态前同时核对实际 `/proc/<pid>/cmdline`；状态文件单独存在
不能证明进程存活。源绑定的只读检查为：

```bash
cd /data/chainpray/qc/control/gac_image_continuation_20261020/release-0bc033b312a2
/home/ym/code/chainpray/qc/.venv/bin/python -m research.gac_continuation_owner inspect --spec continuation-spec.json
```

它只查询已绑定的远端控制器和 owner，不保存镜像或启动训练。`observation_unavailable`
表示未取得观察结果，继续检查同一任务/进程；不能据此判断停止或重建实例。
`needs_attention` 时保留全部 intent、失败记录及源绑定，先定位实际故障。未知提交只对账，
不删除 intent、重发保存请求或重跑科学命令。环境、可用镜像及正式启动都有独立门槛，
详见[接续范围](experiments.md#gac-镜像与正式启动的持续接续2026-10-20)。

## 共同六维映射接续观察

新 60 运行科学计划位于数据盘 `control/posthoc_matched_map_study_20261020/plan.json`，
它保留注册时的 `runtime_armed=False`；当前独立运行状态以
`control/posthoc_matched_map_runtime_20261020/watch-process.json`、`watch-status.json` 和
`remote-progress.json`，以及并发修订目录的 `owner-adoption.json` 为准。原 `remote-started.json`
保留首次 arm 的历史身份；当前远端 owner 以远端 `continuation/started.json` 的替换链为准。
远端 owner 和本机只读收集进程仍需核对 `/proc/<pid>/cmdline`，不能只看锁或旧状态。

远端纯观察入口为：

```bash
cd /workspace/releases/posthoc-matched-map-controller-9cd028fa0bf4/payload
/workspace/venv/bin/python -m research.execution_map_runtime inspect --spec runtime-spec.json
```

通过已绑定当前 job 的 SSH 执行，不复制私钥或认证到远端。
`waiting_for_existing_actor_q_queue` 表示等待既有 40 运行的完整技术终态和实际 supervisor
退出；首次串行控制器随后等待四卡空闲。现行并发操作适配器改为共享槽位接纳，
仍先六个 4090 工程、再 60 个科学 case，没有成绩方向门槛。
`arm` 已执行一次，勿因连接超时重发或删除 intent；先只读核对原 PID 和记录。等待阶段
不初始化 CUDA。本地收集进程只在完整终态后下载绑定的汇总和退出回执，不下载权重。

运行器、科学计划及冻结源码分别绑定，不覆盖旧发布目录。日志和失败原样保留；源或
数据身份变化、未知退出、恢复失败时先诊断。只读观察可重试，不自动重跑科学 case。
实时状态和日志会变化，归档哈希使用明确时间点的快照，不能用陈旧状态哈希认证当前状态。

## 最小维护检查

```bash
python scripts/check_docs.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_docs.py tests/test_summary_archive.py -q
```

正文更新这四份日常文档；单次实验进度写批次记录，避免每轮增加一份项目级说明。

### 独立 RLPD 配方因子设计的状态

`control/rlpd_recipe_factorial_20261020/` 已冻结 320 运行协议、106 文件新源码、精确命令和
终态分析契约；[设计与统计规则](experiments.md#rlpd-配方失败的独立全因子配对消融2026-10-20)
另行维护。该批当前没有启动 owner，没有 arm 或提交正式训练，也未上传训练快照；
已纳入本机条件接续安排，但没有训练 owner 或 GPU arm。登记的最早接续位置是既有
60 运行共同映射队列之后。2026-10-21 用户将 Q-chunking UTD1 配方下的 posthoc 对比及
消融设为主线，UTD20 为补充；M60 之后先完成主线结果、配对消融及证据缺口审查，再按
原技术门槛接续未改动的 320 项补充研究，不以有利成绩作为门槛。主线/参考配方区别见
[实验记录](experiments.md#主线与补充诊断的优先级2026-10-21)。本机调度入口为数据盘
`control/action_adapter_continuation_schedule_20261020/tick.py`，当前指令及后续优先级以同目录
`continuation-prompt.txt` 为准；不得将调度指令登记当成训练已执行。
两任务各 16 单元的 32 份 4090 工程命令已经准备，仍须实际 GPU 验收、初始化捕获、
恢复验证和独立运行控制器验收后才能 arm。已有 posthoc/FQL/GAC/Q40/M60 的控制器继续使用
各自原发布包；不能把新设计的登记回执当作云端已运行的证据。
