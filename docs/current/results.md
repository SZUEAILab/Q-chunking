# 结果：已知结论和待解释实验

[文档入口](../index.md)

更新：2026-10-19。本页汇总已有记录，不作为实时训练监控。性能解释主要来自此前已分析报告，
新归档中的运行完整性检查不自动升级为科学结论。

## 原版 posthoc 的统一重算（2026-10-19）

从当前固定 HF revision `a16bfe8` 重新读取 200 次 Cube-Triple 原版运行：五任务、H1/H5、
两个方法、各十个 seed。全部 400 个 flags/eval 文件与此前保存索引的 SHA-256 相同；
独立重算逐运行指标也与旧索引一致。索引原有 revision `e3785e2` 单独保留，没有改写其来源。
每个任务内按训练 seed 配对，未混入四 seed 小批次、FQL 或后来的 direct-speed 配方。

表中两个成功率列均为 baseline / posthoc，单位是百分比。Endpoint 固定为 1M 单点；
tail3 固定为 800k/900k/1M 三点均值。AUC 只使用真实观测的 100k–1M 区间并除以 900k，
不补造 step 0，不能与新批次的 0–1M AUC 直接混算。区间是十个配对 seed 的回顾性描述
bootstrap，未作确认性多比较结论。

此处配对指文件中相同的训练 seed 标签；旧文件未证明所有环境与评估随机源都按相同规则
隔离，因此不把它描述为完整的 common-random-number 控制。

| 任务 | H | Endpoint baseline / posthoc | Tail3 baseline / posthoc | Observed-AUC 配对差及 95% 描述区间（百分点） | AUC 正配对 |
|---|---:|---:|---:|---:|---:|
| task1 | 1 | 91.90 / 99.90 | 89.40 / 99.73 | +10.97 [−4.32, +26.67] | 5/10 |
| task1 | 5 | 54.10 / 84.90 | 49.13 / 81.30 | +29.29 [+4.70, +51.30] | 7/10 |
| task2 | 1 | 49.60 / 94.90 | 37.30 / 90.97 | +32.28 [+24.84, +39.51] | 10/10 |
| task2 | 5 | 2.30 / 31.00 | 2.00 / 22.03 | +7.93 [+4.15, +11.95] | 9/10 |
| task3 | 1 | 1.10 / 28.40 | 1.10 / 22.41 | +8.49 [+4.16, +13.16] | 10/10 |
| task3 | 5 | 0.70 / 31.70 | 0.60 / 27.49 | +13.65 [+10.70, +16.55] | 10/10 |
| task4 | 1 | 49.40 / 66.50 | 44.43 / 61.40 | +8.37 [+1.71, +15.34] | 8/10 |
| task4 | 5 | 1.60 / 17.60 | 1.13 / 14.57 | +6.36 [+4.22, +8.51] | 10/10 |
| task5 | 1 | 0.00 / 0.00 | 0.00 / 0.00 | +0.00 [+0.00, +0.00] | 0/10 |
| task5 | 5 | 0.00 / 0.00 | 0.00 / 0.00 | +0.00 [+0.00, +0.00] | 0/10 |

![原版 posthoc 的全部任务、配对 AUC 与统一终点](../images/action_adapter_legacy_paired_auc_20261019.png)

该重算保留了原方法值得优先复现的正向信号，同时保留 task1 H1 的跨零描述区间和 task5 的
两组全零。旧文件不能认证进程退出码、执行源码身份或独立评估 RNG；100 episodes 是 flags
中的配置值，尚无逐 episode 原始记录的独立核验。这是同一任务族的历史证据，不是新的确认实验。
复算工具为 [`research.action_adapter_legacy`](../../research/action_adapter_legacy.py)，10 项测试通过；
原始曲线、逐 seed 差值及文件身份记录在数据盘
`reports/action_adapter_validation_20261019/legacy-outcome-reverification.json`。

2026-09-25 的 [critic-space 协议](../data/summaries/2026-10-19/protocols/envq_counterfactual_protocol.md)
复用 [projection 协议](../data/summaries/2026-10-19/protocols/projection_counterfactual_protocol.md)
中的 sigmoid/direct latent 运行，属于另一配方。其 env-Q 降幅不能解释六月 tanh/log 原版
优势的来源，也不构成放弃完整原方法的证据。新筛选因而分别保留 legacy、direct 及各自的
latent/env/canonical 对照，完整方法复现和机制解释分开判断。

## Action-adapter 当前执行状态（2026-10-19）

本分支已冻结 action-adapter 对比与消融的首轮筛选：九方法、两个任务、四配对 seed、
每运行 1M 步，共 72 个运行。具体范围见[实验](experiments.md#action-adapter-冻结筛选)。
执行源码的精确 commit 与逐文件 SHA-256 固定在批次的 `source-commit.txt` 和
`release-manifest.json`；后续日常文档更新不改变运行中的发布包。

29 项相关测试和 3 个参数化子项通过；本地真实 GPU 的 legacy-canonical H1 512 步、
direct-canonical H5 128 步，以及 native 64 步完整运行验收均退出 0。检查了真实 step 0、
训练更新、精确终点评估、解析配置、动作边界和可读有限 checkpoint。
配置序列化与本地缓存路径的两次工程失败已保留，修复后的运行使用独立 attempt。
入场前额外发现零方向处的范数梯度为 NaN；现已保留原前向值并定义有限零点次梯度，
非零方向导数与 env/canonical、log/direct 四种零点回归均通过。所有长程运行使用修复后的冻结源码。

用户因八卡等待过久改为 4×4090。原任务 `a21561187422171136807711` 已确认 Stopped，
停止前处于 Queuing、startedTime=0。替换任务 `a21566001929515008267857` 已由平台接受，
最近查询仍为 Queuing、子任务 Pending，尚未开始训练；当前只有这一个活动实验环境。
平台返回配置为 4×4090、56 CPU、448 GiB 内存、标价 4 算力点/小时，不限时长。
该页面不是实时监控。部署、critic、基线和动作几何四个监控进程已恢复，分配后先做
SSH、GPU、挂载与远程工程验收，通过才启动筛选。当前尚无这 72 个长程运行的新性能结果。

配置、原始工程尝试、源码/数据 SHA-256、等待进程状态及后续验收记录统一放在数据盘
`control/action_adapter_validation_20261019/`；生成报告放在 `reports/action_adapter_validation_20261019/`。

部署控制包 revision 3 在首次 JAX 导入之前关闭 GPU 显存预分配，避免验收父进程占用
子进程需要的显存。此次资源切换更新至 revision 4：GPU 验收和调度设备改为 0–3，
监控改指向四卡替换任务。旧部署包、计划和进程记录保存在 `before-4gpu-switch/`；
训练源码、数据、72/16 运行任务配置及首轮科学预注册保持不变，资源修订单独登记。
两份发布包的 84/90 个 Python 源文件和外层包成员均已逐一核验哈希，控制器语法检查通过。

### Critic 等效动作检查

独立离线诊断新增 9 项测试通过。真实 checkpoint 的配对工程检查使用相同训练 seed 9004、
64 个环境步（33 次 optimizer 更新，计数器从 1 变为 34）、相同源码和同一批
64 个 validation-prior 状态；
解析配置唯一差异为 latent/canonical critic。共同动作探针不依赖已学 actor，方向缩放 0.5
后两组的最大实际执行动作差均为 0。

| 工程检查 | Latent critic | Canonical critic |
|---|---:|---:|
| 等效动作 Q 绝对差均值 | 0.0072749 | 0 |
| `u · ∇u Q` 的 RMS | 0.0183553 | 1.04e-9 |
| 方向坐标 Q 梯度的平均绝对径向投影占比 | 0.524853 | 3.00e-8 |

该结果证明当前诊断能检测到等效动作的价值差异，且 canonical 约束在训练后 checkpoint 上
按定义工作。它没有证明成功率、样本效率或总 actor 梯度改善；此处不包含熵梯度，64 步结果
不用于筛选优胜方法。原始报告与配对身份核对保存在数据盘
`reports/action_adapter_validation_20261019/engineering-critic-invariance-v2.json` 和
`engineering-invariance-comparison.json`。72 次长程筛选的性能结果仍待平台执行。

### 原论文 RLPD 配方的独立验收

新增原论文 RLPD learner 配方的 H1 OGBench 入口，与首轮冻结筛选分开；
来源和不同 RLPD 实现的区别见[基线参考](../reference/rlpd_baselines.md)。
固定上游 commit 的六类数值对照全部通过，参数、优化器状态与 RNG 最大差为 0；
另外 10 项更新顺序、接口和恢复测试通过。

本地 GPU 0 完成真实 Cube-Triple task2 的 64 步工程运行，使用默认 256×256 网络、
10-Q 随机 2-Q min target、entropy backup 与 UTD=20，训练退出 0。
step 0/64 评估和训练日志完整，参数有限，实际执行 660 次 critic、33 次 actor 和 33 次
温度更新；checkpoint 能通过该 learner API 恢复并产生有限、合法动作。
验收控制器首次误用系统 Anaconda Python，读取 checkpoint 时触发 JAX/NumPy 兼容错误；
原记录保留，随后在项目虚拟环境复核同一 checkpoint 通过，未重跑训练。

数值报告为数据盘 `reports/action_adapter_validation_20261019/rlpd-reference-equivalence.json`，
GPU 复核记录为 `control/action_adapter_validation_20261019/local-rlpd-reference-engineering-revalidation.json`。
这只验证移植与工程链路；尚无该外部配方的长程 OGBench 性能结果。
两个任务 × 四配对 seed × UTD=1/20 的 16 运行配置及独立执行包已冻结，放在
`control/action_adapter_reference_20261019/`。接续监控已启动：原 72 运行完整验收、原环境
仍为 Running 且独立远程工程检查通过后，复用同一 4 卡实例执行；每卡先运行一个任务。
当前基线长程训练尚未开始，未加入原 72 运行，也未申请额外实例。

独立控制器通过 13 项测试，覆盖真实 episode/success 计数、缺失网格、有效配方、全部动作
边界、精确 optimizer 更新数、非有限 optimizer 状态、重复输出、失败退出码、禁止自动重跑及
缺失配对。还读取了已有真实 GPU checkpoint 复核计数，未再次训练。
新的评估观察器只读取原评估返回的 trajectories，为每个评估点写出实际 episode 数和成功次数；
不增加评估调用，不改变策略、训练更新或 RNG。已有短测缺少该新字段，明确标为旧工程格式，
不冒充正式基线已经满足 episode 记录要求；正式运行必须通过严格计数验收。

### 动作与运行时间分析的准备

首轮已登记的动作范数、协方差、边界占比和 wall time 由独立流式分析器补齐，定义与限制见
[实验说明](experiments.md#已登记的动作与时间次指标)。22 项测试通过，覆盖与 batch covariance
的一致性、近常数动作数值稳定性、warm-up 分离、三种边界量、缺步/重复/非法动作、缺失 seed、
错误源码和退出码，以及拒绝从只有 norm 的旧日志伪造坐标协方差。

已有两份 64 步真实工程日志完成双 CPU worker 的 CLI 验收，每份严格识别出 32 条
post-warmup 动作；其协方差与均值另用直接 batch 计算核对一致。修改 seed 合约的负控被
冻结配置哈希拒绝，未生成结果，也没有新增训练。工程报告在数据盘
`reports/action_adapter_validation_20261019/engineering-action-geometry.json`。
正式 72+16 个运行的几何统计仍待训练完成；这些工程检查不提供新的性能或机制结论。

## 当前可以说什么

| 研究项 | 已有判断 | 边界 |
|---|---|---|
| 固定 `K=5` 方向保持 | 当前 Cube Double/Scene 配对结果不支持提高样本效率 | 不能推广为所有方向–速度或自适应时间方法无效 |
| Cube-Triple 扰动匹配 tangent DS | 部分任务存在正向估计 | 尚无跨任务族、跨算法的一致收益结论 |
| Cube Double/Scene 两 seed 噪声筛选 | 未通过原扩展门 | 不扩展原两矩控制器来追求显著性 |
| 三矩控制器与 residual/offset 变体 | 暴露容量坍缩或边界附近矩匹配耦合 | 几何接线通过不代表性能收益 |

具体数值、统计条件、失败和旧实验规模见[历史结果报告](../archive/results.md)。
这些窄结论不支持“DS 普遍优于 Cartesian”或“已经完成 FQL/RoboMimic 通用性验证”。

## 最近归档的三组 RLPD 筛选

| 批次 | 精简归档 | 科学解释状态 |
|---|---|---|
| Semantic-subspace uncertainty | 48 次运行 | 保留原报告中的未通过假设/确认判断 |
| Critic-flat semantic exploration | 48 次运行 | 保留原报告中的未通过假设/确认判断 |
| State-conditional controllability | 48 次运行 | 执行完成，不能仅据归档检查认定方法成立 |

三组共 144 次运行通过了专用导出器的精简预算、网格、有限值和 incident 检查。
该检查不覆盖所有退出码、checkpoint 和 source/data 身份要求。
证据入口：[归档说明](../data/summaries/2026-10-19/README.md)、
[逐运行检查](../data/summaries/2026-10-19/run_checks.json)、
[原分析报告](../data/summaries/2026-10-19/reports/)。

## 下一步

优先完成本轮完整 action-adapter 的 72 运行复现与消融、critic 等效动作诊断及独立外部
基线比较；随后按冻结规则确定确认配方，扩展 fresh seed、开发任务之外的任务和算法/领域。
旧状态条件批次继续保留为边界证据，其分析状态单独记录，不替代完整 adapter 主线。

写论文时从这里追溯来源，工作稿见[论文参考](../reference/paper_draft.md)。
