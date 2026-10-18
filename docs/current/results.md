# 结果：已知结论和待解释实验

[文档入口](../index.md)

更新：2026-10-19。本页汇总已有记录，不作为实时训练监控。性能解释主要来自此前已分析报告，
新归档中的运行完整性检查不自动升级为科学结论。

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
