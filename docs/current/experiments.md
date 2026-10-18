# 实验：一批实验怎么设计

[文档入口](../index.md)

一批实验（代码中叫 `cohort`）共同回答一个问题。先冻结设计，再启动训练。
本文件是日常实验规则；旧协议中的“推荐”“待做”不是新实验的启动指令。

## Action-adapter 冻结筛选

批次为 `action_adapter_validation_20261019`，配置、源码摘要、工程尝试与平台记录放在
数据盘 `control/action_adapter_validation_20261019/`。用户授权 8×4090 不限时长；使用已有
兼容镜像、持久化挂载与独立源码发布目录，不覆盖其他批次。

| 方法 | 速度/支持 | Critic |
|---|---|---|
| Cartesian native | 原 TanhNormal cube | 真实环境动作 |
| Cartesian projection | TanhNormal 后 L2 投影 | 投影后的环境动作 |
| Legacy latent / env / canonical | Cartesian D+1、tanh/log、参考半径 1、逐轴 clip | 三种接口分别运行 |
| Direct latent / env / canonical | Cartesian D+1、sigmoid/direct、cube 径向边界 | 三种接口分别运行 |
| Legacy common entropy | 同 legacy latent，target entropy 固定 −2.5 | latent |

两个 adapter 版本是完整配方对比，同时改变速度映射和执行支持，不能写成纯坐标消融。
同一版本的 critic 对照保持 actor、熵目标、网络宽度、初始化、数据、预算与评估一致。
Canonical 的 replay 与 sampled Q 查询统一用执行动作的 canonical 编码；环境动作相同的
潜动作必须给出相同 critic 特征。普通 legacy/direct target entropy 为 −3，native 为 −2.5；
共同熵目标对照显式检验这项差异。

主指标为真实 step 0–1M、每 100k 固定评估网格的 success AUC；50 episodes，固定独立
`eval_seed=20261019`，另报最终值、末三次均值和右删失的 time-to-50%。训练 seeds 10–13，
不参与确认阶段；四 seed 区间仅作描述，首批两个开发任务不能当作跨域通用性证明。
所有失败、零成功率和缺失运行保留，预注册主比较为 legacy−native、canonical−latent
（分别在两个速度配方内），其他比较为机制诊断，不在看到结果后挑选显著性 family。

新增 `--eval_at_start=True` 用于该批次，旧配置默认保持关闭；运行额外保存初始化后解析的
`effective_agent_config.json`。调度状态 `done` 不能替代验收：必须核对真实 returncode、
完整评估轴、有限损失、精确最终预算、必需 checkpoint 与源码/数据身份。失联进程的退出码
记为未知，不纳入成功统计。

## 批次最少登记什么

| 项目 | 要写清楚的内容 |
|---|---|
| 问题 | 想验证的假设、反事实与可能的否定结果 |
| 矩阵 | 算法、任务、方法、相同训练 seed、预算、评估网格 |
| 来源 | 实际执行代码及摘要、有效配置、依赖、数据指纹 |
| 分析 | 主指标、配对方式、区间、多比较与扩展条件 |
| 运行 | 输出根、并发、保存策略、恢复规则、盲法范围 |

单次训练由任务、方法、seed 和配置确定；基础设施故障后的恢复另记 attempt，
保留父 attempt、恢复模型和恢复步。改变科学行为意味着新身份，不能覆盖旧输出。

## 公平比较

- 同一任务的方法使用相同 seed、预算、数据和评估协议；评估 RNG 与训练 RNG 分离。
- 表示、噪声和时间相关性分别消融；幅值、协方差、饱和和初始化差异需要披露。
- 动作几何既看状态平均，也看固定状态条件；匹配几个矩不等于匹配完整分布。
- 外部强基线改变 learner、熵目标或正则时，明确标为算法配方差异。
- 固定时间实验包含 reactive、direction hold、translation repeat、full repeat。
  真正多步 option 的折扣与 backup 按实际持续时间处理。

## 评估和统计

- 真实执行 step 0 评估，固定评估网格，不补造数值、不挑最好 checkpoint。
- 主指标预先指定：例如 success AUC 或 normalized-return AUC；同时报告最终表现和删失。
- 统计单位是任务内的训练 seed/block，不能把同一 seed 的 episodes 当成独立重复。
- 先算配对差，再汇总；报告原始 seed 值、均值、中位数、正配对数及配对区间。
- 两对 seed 只做描述性比较；不同任务的方向和异质性需要展示。
- 不利结果、零成功率、失败 attempt、缺失评估和排除原因都要保留。

## 四种判断分开

| 判断 | 回答的问题 |
|---|---|
| 执行 | 有没有结束，是否达到预算？ |
| 验收 | 退出码、身份、评估、数值、必需模型和 incident 是否符合协议？ |
| 科学结论 | 支持假设、不支持，还是证据不足？ |
| 发布 | 是否上传并固定 HF revision？ |

PID 消失或队列写 `done` 不证明验收通过。必需检查包括退出码、精确预算、评估轴和 episode 数、
有限值、动作合法性、source/config/data 身份、可读的必需模型及未解决 incident。
低回报和方法劣势属于科学结果，不触发自动重跑。

reward-blind 校准/工程阶段只读取允许的状态、几何和运行诊断，不查看 reward/success。
性能结果只能在预定的可见阶段分析；存储归档不能擅自解除盲法。

## 当前示例

状态条件可控性噪声批次：3 个任务 × 4 种噪声 × 4 个 seed，共 48 次训练。
比较普通各向同性、固定 DS、正向状态条件和倒置反事实；各任务预算在批次配置中固定。
执行源码位于独立工作树，其身份和主目录 `main` 分别记录。

运行、存储和恢复见[运行手册](runbook.md)。追溯具体旧方法时再看
[动作表示](../archive/action_space.md)、[噪声](../archive/noise_shaping.md)、
[合成机制](../archive/synthetic.md)或[机器人](../archive/robotics.md)协议合集。
