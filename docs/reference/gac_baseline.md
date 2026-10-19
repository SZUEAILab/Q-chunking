# GAC 基线的来源与适配边界

[文档入口](../index.md) · [当前实验](../current/experiments.md) · [RLPD 基线](rlpd_baselines.md)

GAC 是 action-adapter 必须面对的近邻方法。ICLR 2026 的
[Beyond Distributions: Geometric Action Control](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e58fa42d4b7a798eef8d0d75098f87ad-Abstract-Conference.html)
已经使用方向与浓度生成动作，方向分解本身不足以支持新颖性声明。

## 固定官方实现

本次核验固定 [Lin-Zhihao98/GAC 的 `374ae2c`](https://github.com/Lin-Zhihao98/GAC/tree/374ae2c9ef284a6ef715a67c1fdce864ebad6f44)，
源码、MIT 许可与文件哈希保存在数据盘 `references/gac-official-20261019/`。
以下是该代码的默认值和更新路径，不是对论文所有 benchmark 实际配置的认证。

| 项目 | 官方代码 | 旧项目 GAC 适配材料 |
|---|---|---|
| actor / critic 隐藏层 | 256×2 / 256×2 | 配置记录为 512×4 / 512×4 |
| Q 与 actor 聚合 | 双 Q，target 与 actor 均取 min | 配置与工作源码为 10 Q、mean |
| actor / critic 学习率 | 3e-4 / 1e-3 | 配置记录共用 3e-4 |
| 更新节奏 | 每次向量交互更新 critic；每两次交互连续更新两次 actor；随后更新 target | 项目联合更新路径与 prior/online 数据混合 |
| 默认并行环境数 | 8 | 旧 OGBench 单环境入口 |
| 温度 | 固定 alpha=1，不自动调节 | code/paper 配置为 1，分组 DS 为 0.1 |

官方默认 8 个环境的一次向量 step 对应 8 条环境转移；它的 `total_timesteps` 和
`learning_starts` 都按向量循环计数。移植时必须报告转移数、critic/actor 更新数与实际预算，
不能把循环数直接当成与当前单环境 1M 步相同的预算。
上述更新、网络和默认参数均见[固定代码](https://github.com/Lin-Zhihao98/GAC/blob/374ae2c9ef284a6ef715a67c1fdce864ebad6f44/GAC.py)。

## 旧实验可以复用什么

归档已有 task1 的五组 × 六 seed 配置，其中 18 份 flags 将方法标为 GAC-code、GAC-paper、
GAC-DS，各为 seeds 92–97。旧 GAC 动作头、浓度正则和 loss 的工作源码也已保留；
它们提供共同学习器内的几何策略对照，不能直接作为官方完整算法复现。
旧 flags 中嵌套 `agent.gac_variant` 仍是默认值，实际方法选择记录在顶层 `gac_variant`；
核查时必须同时读取入口的配置解析路径，不能仅凭嵌套默认值把它们判为 native。

这些目录没有单独的 `effective_agent_config.json`。已保存的源码快照明确标注为后来的
工作源码，不能倒推证明旧进程执行了这份代码。此次报告逐一绑定 30 份 flags 和三份
归档源码的哈希，保留该来源限制；没有重新训练，也没有把旧分数并入当前 72 运行。
旧性能比较见[历史结果](../archive/results.md)，不是当前完整 action-adapter 的竞争结果。

## 旧适配的数值一致范围

只从固定上游 AST 加载已检查的 actor 方法，不执行其训练入口。共同测试点的方向范数远离零、采用相同固定
Gaussian 噪声下，核对旧移植的采样值、方向/浓度梯度、浓度变换、正则项及其梯度；
另外用原始 AST loss 表达式核对固定输入的 actor 目标与 soft target。
17 项对照在 float32 的 `atol=rtol=3e-5` 内通过。
这不覆盖网络初始化、完整 learner 轨迹、随机数流等价或任务性能。

三个边界差异单独保留，不能被非边界测试覆盖掉：

- 方向为零时，官方 normalize 输出零向量，旧移植回退到第一基向量；
  两者的 epsilon 也不同，范数为 1e-10 的非零方向测试同样产生不同输出。
- 单样本浓度正则中，官方无偏样本方差为 NaN，旧移植将方差保护为零。
- 浓度恰为 0.1 的 clamp 边界，PyTorch 与 JAX 采用不同次梯度；二样本检查的正则梯度
  最大差为 0.125。首次对照因此失败，原日志已保留；后续非边界检查没有取消这项差异。

报告为数据盘 `reports/action_adapter_validation_20261019/gac-reference-audit.json`；
可复核脚本及初次失败记录位于 `control/action_adapter_validation_20261019/`。
官方 PDF 下载未完成，部分文件不用于证据；本次数值审查以固定官方代码为依据。

## 独立官方配方学习器（2026-10-19）

新增 [`agents/gac_reference.py`](../../agents/gac_reference.py)，保留固定上游的完整
Torch actor/Q 网络、初始化顺序、两个独立 Q 及其 targets、两个 Adam、min target/actor Q、
固定 alpha、浓度 surrogate 与正则、actor 梯度裁剪和更新后 target EMA。
使用[官方 MIT 许可](gac-upstream-LICENSE.txt)。它不是旧 GAC-DS，也不是在 RLPD 上替换 actor。
CPU 初版为 `5051c30`；后续 `1d6bef4` 增加可选依赖隔离、设备 RNG 隔离和 CUDA 数值入口。
当前已验收 CPU learner 与本地 CUDA 单环境工程，尚未验收云端 GAC 长程运行。

[`research/check_gac_reference.py`](../../research/check_gac_reference.py) 直接核验下载文件与
固定 commit 的 Git 对象字节，再从 AST 加载完整官方类与原始 `global_step > learning_starts`
更新分支。只用固定 replay stub 替代采样器，不运行上游环境/日志入口，也不重新编写一份
oracle 更新。三类案例采用完整 256×2 网络、batch256，覆盖不同 policy/target 间隔、
alpha/radius 与 3/5 维动作。初始化、交错的六次采集采样、六次完整更新的所有模型、
Adam 状态、Torch RNG 和 loss 均精确一致，最大状态差为 0。

本地 API 用私有 Torch RNG 保留采集与更新的交错流；显式 eval seed 不推进训练 RNG，
也不改调用者的随机状态。完整 checkpoint 包含五个模型、两个优化器、RNG 与更新计数；
CPU 同设备恢复后下一次采样与更新精确一致。原 14 项测试覆盖这些行为、完整动作球支持、
更新时钟、官方零方向与单样本正则边界，以及非法配置拒绝。
官方 helper 的单样本无偏方差仍为 NaN，learner 明确拒绝 batch<2，不将其悄悄修为零。
这里的 `-kappa` 仍是官方浓度 surrogate，不认证真实动作分布的熵。

更新时钟保持零基循环语义：严格在 `global_step > learning_starts` 后更新 critic，
在 policy event 连续更新 `policy_frequency` 次 actor，再按 target 间隔更新 target。
单环境 64 步、learning_starts=32 的真实 Cube-Triple task2 CPU 工程运行退出 0，实测
critic/actor/target 更新为 31/30/31，而非把 warm-up 边界近似成 32 次。
step 0/64 各实际返回 1 个评估 episode，额外评估共 2000 个环境步，独立于 64 个训练步。
未用离线先验；完整 checkpoint 恢复与 16 个策略探针通过，122 个冻结 Python 源文件核对一致。

上述单环境与独立评估是本地 driver 适配，不能冒充默认八环境的官方 benchmark；未来
正式比较须分别报告向量循环数、实际转移、更新和 prior 使用量。云端现有 Torch 仍为
CPU 版，GPU 运行环境、正式矩阵与 GAC/posthoc 长程性能比较仍待完成。

数值报告为数据盘 `reports/gac_reference_20261019/learner-equivalence-v2.json`，真实工程和
checkpoint 证据在 `control/gac_reference_20261019/engineering-receipt.json`。首次 oracle
误把保存源码目录当作 Git 工作树，在数值案例开始前退出；改为验证其对应 Git 对象后通过。
原脚本、失败日志和修正说明均保留，没有重跑任何长程科学实验。

### 本地 CUDA 后端的验收

本地另有可复用的 CUDA 解释器，已据此创建专用 GAC 环境，保留 OGBench 1.2.1、
MuJoCo 3.9.0、Gymnasium 1.3.0 与 NumPy 2.4.4，使用 Torch 2.14.0+cu126。
仅使用空闲 GPU 0（RTX6000 Ada），没有修改云端环境或其他 GPU 上的工作。
三类完整 learner 案例在 CUDA 上采用真实任务的 46 维观测、batch256、六次交错采样与
六次更新，所有参数、优化器、CPU/CUDA RNG 和 loss 与官方 AST 精确一致，最大误差为 0。
新源码在原项目 CPU 环境再次通过三类数值核对；两种后端各自与参考比较，不声称跨设备
随机流或训练轨迹逐位相等。

CUDA 环境首次在数值计算前被 eager agent 包导入的不相关 JAX/TFP 依赖阻止。
代理注册现按需加载，保留原 keys、可变注册与类导出；原 RLPD/FQL/注册/GAC 的 87 项
回归通过，CUDA 专用环境另有 17 项 GAC/依赖隔离检查通过。新增检查确认 CPU agent
也不会重置其他可见 CUDA generator。没有修补云端 JAX/TFP 或替换其依赖。

`1d6bef4` 的 123 文件快照另完成真实 CUDA 64 步工程：critic/actor/target=31/30/31，
step 0/64 各实际返回 1 episode，额外评估 2000 步；完整 GPU checkpoint 从新模板恢复
精确一致，16 个显式 seed 策略探针一致。该记录验证本地 CUDA 路径，不替代云端 RTX4090
验收或长程性能。报告为数据盘 `reports/gac_reference_20261019/learner-equivalence-cuda-v2.json`
和 `control/gac_reference_20261019/engineering-receipt-cuda.json`；CPU 最新数值报告为
`reports/gac_reference_20261019/learner-equivalence-cpu-final.json`。

## 动作支持与后续对照

官方采样把完整动作向量的单位方向与单位球面噪声作未归一化凸组合，权重为
`sigmoid(0.5*min(kappa,20))`，再乘 radius 和环境逐轴缩放/平移。
旧 GAC-paper 标签对应归档中的重新归一化、`sigmoid(kappa)` 变体；它是单独标记的公式
变体，不应选出成绩更好的一种再统称官方 GAC。

在五维、对称单位动作边界且 radius=1 时，可直接从实现推得：

| 方法 | 实际动作域 |
|---|---|
| GAC-code | 完整五维动作在单位球内；半径随方向和噪声的混合而变 |
| 旧 GAC-paper 变体 | 非退化输入在完整五维单位球面上 |
| 旧 GAC-DS | XYZ 方向和独立速度组成三维单位球内动作，另带 yaw/gripper 标量 |
| 当前 legacy tanh/log adapter | 参考半径 1 经过 log-speed 映射与逐轴 clip，可产生 XYZ 单位球外动作 |

因此，旧 GAC-DS 对 tangent DS 的支持匹配，不能自动延伸为对当前 legacy adapter 的
支持匹配。若以后增加共同学习器的分组 GAC 消融，必须明确登记 XYZ 支持是否覆盖 cube、
幅值映射与标量头是否一致，以及浓度 surrogate 与潜坐标熵目标的区别。
官方算法适配和共同学习器内的几何消融分别标记，保留各自能回答的问题。

当前冻结的 72 组筛选与 16 组 RLPD 基线不增加 GAC 组。后续基线实现先固定本页的上游
身份与适配选择，再进行数值和环境工程验收；正式任务、seed、预算及分析在读取新结果前
另行冻结。已有正向 posthoc 结果继续作为完整方法复现的动机，GAC 核验不替代该主线。
