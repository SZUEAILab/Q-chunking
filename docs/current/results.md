# 结果：阶段总结与证据边界

[文档入口](../index.md) · [下一步实验](experiments.md#阶段计划2026-10-19)

更新：2026-10-19。结论：**动作几何干预可以真实实现，但尚未证明能稳定改善跨任务学习效率。**
当前应收尾 Action Adapter/critic 表示消融，不继续扩大已失败的噪声打分映射。
本页不是实时监控；运行完成、验收完整、科学假设通过分别记录。

## 已完成研究的性能面板

下表是各候选相对其登记基线的归一化回报 AUC / 成功率 AUC 差，数值为 0–1 尺度。
任务、预算、seed 面板不同，不能合并成一个总效应或按表中均值排名。

| 候选与证据 | Δ return AUC | Δ success AUC | 阶段判断 |
|---|---:|---:|---|
| 扰动匹配流形动作，741xx，16 个配对单元；对照 Cartesian-L2 | +0.03056 | +0.00981 | 机制通过；原登记主张未通过，success-AUC 区间跨零 |
| 固定 DS 行为噪声，741xx，16 个配对单元；对照 isotropic | −0.01353 | −0.01238 | 机制通过；原登记主张未通过 |
| 不确定性分配 SS-VoI，753xx，48 runs | −0.01643 | +0.01271 | 两项主指标方向冲突，且未分离倒置反事实；关闭该映射 |
| Critic-flat 分配，754xx，48 runs | +0.00839 | −0.00208 | 两项主指标方向冲突；关闭该映射 |
| 全局一步可控性，756xx，48 runs | −0.00333 | −0.01244 | 仅 3/12 单元同时改善，Cube 最终成功率下降 7.5 pp；关闭该映射 |
| 状态条件可控性，759xx，48 runs | +0.00818 | +0.01468 | 仅 4/12 单元同时改善，未达 9/12；不进入确认，终态凭证仍有缺口 |

前五行来自[原综合报告](../data/summaries/2026-10-19/reports/direction-speed-evidence-synthesis-20261018.json)：
5/5 候选机制门通过，0/5 原登记性能主张通过。这是描述性回顾，不是新的多重检验。
第六行来自本次[冻结分析与凭证审计](../data/summaries/2026-10-19-phase-review/state-local-review.json)。

## 最新状态条件实验：为什么不扩展

RLPD，Cube Double task5 / Scene task3 / Puzzle 3×3 task3，预算分别为 250k / 125k / 175k。
四臂 isotropic、static tangent、state-high、pointwise inverted-low，配对 seed 75901–75904；
共 48 次、8.8M 步。5k warmup，每 25k 步评估 50 episodes，包含 step 0；请求 XYZ 噪声 trace 固定 0.0675。

| state-high − isotropic | Δ return AUC | Δ success AUC | Δ final success |
|---|---:|---:|---:|
| Cube Double | +0.00874 | +0.06725 | +0.085 |
| Scene | +0.02049 | −0.00750 | −0.025 |
| Puzzle 3×3 | −0.00468 | −0.01571 | +0.130 |

等任务族均值的 95% 分层 bootstrap 区间：return AUC `[−0.01688,+0.03047]`，
success AUC `[−0.03883,+0.07617]`，均跨零。high−low 的两项均值为 `+0.01236/+0.02048`，
也不能弥补 high−isotropic 的跨任务与 9/12 配对门失败。48/48 固定 trace、动作界及非退化机制检查通过。
Puzzle 最终成功率提高而 AUC 下降，说明“最后学会”与“学得更快”不能混为一谈。

**验收边界：**冻结分析器在登记工作树和解释器下通过，但 scheduler.exit 的时间为
10-18 21:51:51，早于预启动清单 21:56:09，不能作为本次终态退出凭证；最终 checkpoint 已参与哈希，
尚未反序列化核验完整 learner 身份。因此不宣称终态验收闭合，不补写退出码、不重跑或追加 seed。

## Action Adapter：新线索，不是新性能结论

200 次历史运行的评估证据已重新核对，覆盖 Cube-Triple 五任务 × H=1/5 × 两方法 × 十 seed。
10 个任务/时域单元的 observed-AUC 差中 8 正、2 零，7 个描述性 bootstrap 区间下界为正；
但没有 step 0，窗口仅 100k–1M，源码、进程退出及评估 RNG 隔离未由这些文件认证。
它不是新方法确认，也不证明旧 posthoc 映射等同于当前 adapter。

64 步、33 次优化更新的配对工程诊断只改变 critic 坐标：执行动作完全相同，latent critic 的
等价动作 Q 差均值约 0.00727、径向 Q 导数 RMS 约 0.01836；canonical critic 分别为 0 和约 1e−9。
这支持“冗余坐标可导致 critic 别名”的机制线索，**不证明修复后成功率会提高**。
RLPD learner 更新等价性测试通过，但不包含完整环境驱动/RNG 等价性；历史 GAC 名称亦不代表官方端到端复现。

本地启动回执登记了 72-run 新筛选：两任务 × 九方法 × 四 seed × 1M 步；最新纳入快照记录了
32 个 running receipt，未提供完整终态结果。本轮没有 SSH 实时核验、启动新实验或查看该批次部分性能。
证据及来源哈希统一在[本次阶段证据包](../data/summaries/2026-10-19-phase-review/README.md)。

## 历史边界与后续

固定 K=5 方向保持在 Cube Double/Scene 的六个主要早期配对 AUC 对比均为负；
三矩控制与 residual/offset 诊断暴露容量坍缩或边界耦合，不继续扫 gain。
历史数值见[归档报告](../archive/results.md)；FQL/RoboMimic 尚无一致迁移结论。
下一步按[阶段计划](experiments.md#阶段计划2026-10-19)收尾、验证、确认；不承诺顶会录用。
