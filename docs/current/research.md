# 研究：我们要证明什么

[文档入口](../index.md)

目标是用动作方向与幅值的结构改善强化学习的探索和样本效率。OGBench 是主要实验环境，
RLPD 是首个研究算法；FQL 用于跨算法检验，RoboMimic 用于跨域检验。已有 pilot 不等于迁移结论成立。
环境中的“speed”通常是归一化动作幅值，不一定是真实物理速度。

## 原版 posthoc 当前验证主线（2026-10-19）

本分支优先检验已有正向任务结果的完整 posthoc（action-adapter），保留六月 tanh/log-speed 原版，
并与后来的 sigmoid/direct/cube 版本分开登记。首先回答完整方法能否复现、能否迁移，
再解释方向–幅值表示、critic 接口与熵目标各自的贡献。
已完成固定 mapper 的前向探针、Jacobian 与先验编码核验，见
[原版 posthoc 机制](../reference/method.md#511-原版-posthoc-优势的可检验解释)。当前优先假设是
幅值行为先验、价值表示与潜坐标正则的组合；不将它简化为更大动作或 critic 冗余单因。
冻结网络的固定状态初始化检查进一步观察到：posthoc 平均幅值较小，但状态内幅值
波动较大；同一组网络 heads 的映射反事实保留该差异。它是初始化证据，不是性能归因。
同一批初始化进一步测到小动作概率明显不同：原始噪声下，native 有约 0.583% draw
的 XYZ 范数小于 0.3；posthoc 的连续 Gaussian/理想映射概率上界约 `2.59e-19`。
这个界保留近零方向分支，不声称小动作全局不可达；固定噪声缩小时的分配差异，仍需
训练后与性能对照验证，不能单独认定为收益来源。
单步物理探针也观察到该响应差异：同 heads 的执行映射差在噪声 0.2 下增加约 9.27 mm
校正位移，在噪声 1 下减少约 2.11 mm，四个初始化 seed 的方向各自一致。
这限定为匹配完整物理起点的未训练策略干预，尚未建立训练收益或状态覆盖归因。
后续固定状态诊断已登记为训练后次分析：只读取原筛选中的 native 和原版 posthoc 共
40 个最终 checkpoint，继续用相同状态面板和共同噪声，检查初始化差异是否在学习后保留。
该分析在训练启动后登记，尚未读取中途成功率；不替代原科学预注册。

当前机制筛选固定 Cube-Triple task2 与 Cube Double task2，四个配对训练 seed（10–13）、
每运行 1M 步。比较 native、Cartesian L2 projection、两个 adapter 版本各自的
latent/env/canonical critic，以及原版的共同熵目标对照，共九方法、72 运行。
该四 seed 批次属于筛选，不足以单独支撑顶会性能结论；不根据中途结果追加 seed。

Canonical critic 仍读取 D+1 维特征，但所有策略样本先经过实际执行 compose/clipping，
再按与 replay 相同的 decompose 编码。它与 latent critic 同宽同初始化，检验冗余方向范数
能否影响 actor/target Q；保留原策略的 log-prob 和温度，不宣称已经消除了冗余熵自由度。
Env critic 与 canonical critic 的对照进一步检验环境动作的非线性表示作用。

若筛选支持完整方法，后续用预先冻结的十个 fresh seed 做确认，扩展开发批次以外的
OGBench 任务，再做 FQL 与 RoboMimic。原生算法配方和几何强基线需分别实现与标记，
不能将本项目的 RLPD fork 直接当作原论文完整复现。
原版 posthoc 已独立接入原论文 RLPD learner 配方，数值与真实短程工程验收通过，见
[同配方适配](../reference/rlpd_baselines.md#原版-posthoc-在同一-rlpd-配方中的适配)。同两个开发任务、
seed 10–13、UTD1/20 的 16 次原版 posthoc 已冻结并接续原实例，配对既有 16 个 native；
仍等待原批次与分析验收，尚无新长程结果。该开发验证不替代 fresh-seed 确认。
它准备排查 learner 配方依赖，不替代后续 FQL 跨算法检验。
FQL 跨算法准备使用独立的[官方参考 learner 与显式适配](../reference/fql_baseline.md)，
与现有 action-chunking fork 分开。新增本地 H1 driver 与 log/sphere 潜动作 posthoc，
保留 FQL 硬 clip、flow/蒸馏/Q 目标；数值和短程接线不替代跨算法性能证据。
两个开发任务上 native/posthoc 的 16 次 FQL 长程比较已另行冻结，接续现有实例；
它检验是否存在不依赖 SAC 熵温度的收益路径，不直接归因历史 SAC 优势，也不替代十个
fresh seed 的确认。离线预训练与在线效率分别统计，见[实验](experiments.md#fql-跨算法准备)。
相关工作已包含 [GAC（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e58fa42d4b7a798eef8d0d75098f87ad-Abstract-Conference.html)
与 [Action Space Design（CoRL）](https://proceedings.mlr.press/v270/esser25a.html)，
因此单纯方向分解或改变动作空间不足以构成新颖性。论文论证取决于受控机制与跨任务性能。
固定官方 GAC 代码与旧项目适配的区别、17 项数值核对和边界差异见
[GAC 基线来源](../reference/gac_baseline.md)。旧 GAC 分组策略的支持匹配不能自动用于
当前允许 XYZ 单位球外动作的 legacy adapter，旧对照也不替代当前完整方法的性能比较。
独立的完整官方配方 GAC learner 已通过 CPU/CUDA 各三类逐步数值对照，最大误差为 0；
两种后端均通过真实 64 步单环境工程及 checkpoint 验收。依赖隔离的原项目回归与 CUDA
专项检查通过；云端 GPU 环境及长程比较仍待完成。完整算法对照与共同学习器几何消融
继续分开标记，不能把这些工程证据当作 posthoc 已超过 GAC 的结果。

动作冗余与熵的区别已有 [Action redundancy in reinforcement learning（UAI 2021）](https://proceedings.mlr.press/v161/baram21a.html)
研究；[PLAS（CoRL）](https://proceedings.mlr.press/v155/zhou21b.html) 已利用潜动作空间限制
离线策略的分布外动作；[Actor-Critic Alignment（ICML 2023）](https://proceedings.mlr.press/v202/yu23k.html)
已研究离线到在线过程中的价值/策略对齐。当前候选不应宣称首次发现这些问题。
本项目当前检验的是固定解析动作映射下的 value-coordinate aliasing，以及它和实际任务收益
是否相关。Canonical critic 保留原始潜坐标熵，不能据此声称已经消除动作冗余或提高状态熵。

机制检查预先固定两个面板：训练后策略自身的采样，以及与学习参数无关的共同 Gaussian
经策略 squash 后的动作探针。相同验证状态下将原始方向正向缩放 0.5，核验执行动作相同后，
比较 Q 差和 `u · ∇u Q`。梯度仅针对 ensemble-mean Q，排除熵和其他 actor loss 项。
近零方向导致不再等效的样本明确排除并计数；不能把所有零差值都解释为学习成功。

## 三个问题分别验证

| 问题 | 干预 | 必须保留的对照 |
|---|---|---|
| 动作表示 | Cartesian 与方向–幅值坐标 | 原生 Cartesian、投影、表示消融 |
| 探索噪声 | 切向改变方向，径向改变幅值 | 普通高斯、固定比例、状态条件、倒置反事实 |
| 时间尺度 | 方向或其他分量保持若干步 | 每步响应、方向保持、平移保持、完整动作保持 |

不能一次改变三个轴，再把结果统称为“方向–速度分解的收益”。
局部协方差相同、执行矩相同、完整条件分布相同是不同控制强度。

## 一个核心表达

平移动作写作 `a_xyz = s × d`，其中 `d` 是单位方向，`s` 是非负幅值。
旋转和夹爪单独处理。方向噪声和幅值噪声可以分别设计，执行后仍要检查范数、方向扩散、
协方差、边界饱和及近零方向是否定义。

研究对象包括 DS 策略分布和策略之外的行为噪声；二者分别登记，不能混为一种方法。
SAC 的 Q 和 log-prob 必须对应协议规定的同一动作变量。

## 怎样形成可信证据

1. 在受控任务上隔离几何机制与反例；
2. 接入真实 learner、replay 和 critic，检查机制是否保留；
3. 用相同任务、seed、预算和评估网格比较样本效率；
4. 方法冻结后，在 held-out 任务、FQL 和 RoboMimic 上验证适用边界。

下一步优先解释已经完成的批次，再根据结果决定新实验。工程通过、漂亮的动作图或高 GPU
利用率不能替代性能证据；不能通过追加 seed 或删掉不利任务来获得显著性。

当前已知结果见[结果](results.md)，统一设计规则见[实验](experiments.md)。
需要推导和实现细节时再看[方法参考](../reference/method.md)；早期探索见[历史记录](../archive/index.md)。
