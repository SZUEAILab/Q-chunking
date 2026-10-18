# 结果：已知结论和待解释实验

[文档入口](../index.md)

更新：2026-10-19。本页汇总已有记录，不作为实时训练监控。性能解释主要来自此前已分析报告，
新归档中的运行完整性检查不自动升级为科学结论。

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

授权申请的 8×4090 开发环境 ID 为 `a21561187422171136807711`，最近查询处于 Queuing、
排队第 1；该页面不是实时平台监控。已启动批次专用等待/部署进程，分配后先做远程验收，
通过才启动筛选，结束后自动输出配对分析。当前尚无这 72 个长程运行的新性能结果。

配置、原始工程尝试、源码/数据 SHA-256、等待进程状态及后续验收记录统一放在数据盘
`control/action_adapter_validation_20261019/`；生成报告放在 `reports/action_adapter_validation_20261019/`。

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

优先对已完成的状态条件批次做完整验收和配对分析，区分方向、幅值与倒置反事实的作用；
然后按预先规定的门决定确认或修改机制。先使用已有数据，不因文档整理增加 seed、重跑或启动新队列。

写论文时从这里追溯来源，工作稿见[论文参考](../reference/paper_draft.md)。
