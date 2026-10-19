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
CPU 版，GPU 运行环境与 GAC/posthoc 长程性能比较仍待完成；另行冻结的比较矩阵见下文。

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

## 来源绑定的 GAC 训练入口

[`main_gac_reference.py`](../../main_gac_reference.py) 使用 JSON plan，分别声明
`online_only` 与 `prior_balanced`。前者使用纯在线 replay，后者把每个 critic batch 的
一半替换为相同任务的冻结 prior，标为 GAC+prior。这样可对齐数据比例；与 RLPD/posthoc
相比仍是完整算法配方差异，不能称为纯几何干预。当前入口支持单环境、归一化五维 Cube
动作和 radius=1，原始 actor/Q/优化器保持不变。

Plan 绑定源码、环境/Torch 版本、数据 SHA、实际预算、replay 容量与独立评估，拒绝已有
输出和静默 CPU fallback。终点核对实际 critic/actor/target 与 Adam 计数、有限参数、
实际 episode 记录、完整学习器 checkpoint 和策略恢复。学习器 checkpoint 不含 replay、
物理环境及 NumPy 采样器状态，不能据此声称支持整条训练轨迹的精确续跑。

每次保存首批数据的原始索引及在线 replay 快照，记录 prior/online 带放回抽取行数；
不修改先验数组，不用 decompose/compose 改写环境动作。Bootstrap 使用真实 termination，
truncation 保留为 episode 边界。Actor 和 critic 的更新时钟不同，日志另记 actor loss
实际所属的 global step。完整学习器的三类 CPU/CUDA 数值证明继续使用原报告，不因 driver
接线而升级为性能结果。

源码 `29536cb` 的 125 文件快照在本地 GPU 0 完成两种协议各 64 步，均退出 0。
相同 seed 的五个模型与训练 RNG 初始哈希一致；每次实测 31/30/31 次 critic/actor/target。
纯在线共抽取 7936 次 online 行，prior 协议为 prior/online 各 3968 次；不等于唯一转移数。
19 项入口测试通过。独立新进程恢复每份 checkpoint 的 179 个状态字段，逐项精确一致；
首批 prior/online 数据分别与实际原始索引及 replay 快照精确一致，两点实际 episode 数、
64 条采集动作及 16 个恢复探针通过。没有重跑两次训练。
证据为数据盘 `control/gac_driver_20261019/engineering-receipt.json` 和
`control/gac_driver_20261019/independent-validation.json`。

当前云端环境仍需 CUDA Torch 验收，GAC 长程矩阵尚未启动；近邻对照的长期稳定性、
调参公平性与性能差异仍待正式比较。

### 较大 radius 与执行边界的独立验收

固定 commit 的 `GAC.py` 默认 radius=1，但 `README.MD` 在多数 MuJoCo 示例使用 2.5，
Ant 使用 1。作者这些示例不是 OGBench 超参数证据。后续完整强基线比较不能只凭此前
radius=1 的工程检查声称已覆盖作者建议设置；radius 选择和执行边界须在正式比较前登记。

原 `29536cb` driver 的 radius=1 发布包保持冻结。新入口 `371fe5b` 增加可选
`action_execution="clip"`，允许较大 radius，并对**完整五维请求逐轴裁剪**后交给 Cube。
省略该字段仍是原 radius=1 identity 协议；较大 radius 缺少显式裁剪声明会拒绝启动。
官方训练循环把 actor 的原始请求存入 replay，Q 查询也使用原始请求。新适配保留这个
learner／replay 坐标约定，不把裁剪后的执行动作悄悄替换为 Q 的输入。
prior-balanced 中的先验动作仍是原数据内的归一化命令；超界在线请求属于不同分布，
不能声称两者完全匹配或它等同于只改变几何的消融。

训练日志分别保存 requested_action 与实际传给环境的 action，独立计数裁剪步骤；
评估也使用声明的执行方式并记录每 episode 的裁剪次数。warm-up 仍使用原均匀 Box，
不将它误限制在策略球内。官方 learner 文件字节未变，125 文件快照仅入口与其测试改变。
较大 radius 的裁剪是 Cube 边界适配，不是对官方 MuJoCo 环境执行行为的认证。

新入口通过 CPU 39 项／跳过 1 项、专用 CUDA 环境 40 项检查。radius=2.5 的 46 维观测、
五维动作、batch256 案例另与官方 AST 完成六次交错采样和六次更新，模型、Adam、RNG
和 loss 精确一致，最大状态误差为 0。该数值对照发生在请求坐标，不认证环境裁剪等价。

本地 GPU 0 的三个真实 64 步工程均退出 0：原 identity/online-only，以及 radius=2.5 的
clip/online-only、clip/prior-balanced。每次 critic/actor/target=31/30/31，step 0/64
各实际一 episode，评估共额外 6000 步。两个 clip 案例各有 25/28 个训练步骤发生裁剪，
最大请求绝对值为 2.205/1.777，最大执行绝对值均为 1；这些是工程现象，不是性能指标。
三组五模型与 Torch RNG 初始化哈希一致，不证明物理 reset 状态或后续轨迹一致。

独立新进程恢复每个 checkpoint 的 179 个字段、核对实际 Adam 计数与 16 个策略探针，
并将首批 replay 的 actions 与记录的原始请求逐项核对。两个 clip 案例的首批在线 replay
确实保留了绝对值 1.777 的请求，环境输入日志仍在 Box 内；实际 prior/online 索引与
原始字段也逐项通过。所有训练与批次冻结记录未被覆盖，失败的测试收集与 oracle 导入
尝试保留在数据盘 `control/gac_radius_20261019/`。两次错误均发生在相应验收案例开始前，
没有重跑长程科学实验。当前仍无云端 GPU 验收或 posthoc−GAC 性能结论；比较设计已另行冻结。

### 可复用的长程验收工具

独立分析版本 `45a0998` 的 [`research/gac_reference_validation.py`](../../research/gac_reference_validation.py)
不修改冻结训练源码 `371fe5b`。它要求 supervisor 观测到退出码 0，并核对来源、计划、
依赖和实际有效配置；分别复算零基更新时钟、Adam 次数和含放回的 prior/online 抽样行数。
评估从真实 step 0 积分到预算终点，逐 episode 核对数量、seed、成功聚合与裁剪次数，
报告 AUC、endpoint、tail3 的实际点数，以及保留右删失的 time-to-50%。

动作日志流式核对完整步轴、均匀 Box warm-up、策略请求球界和逐轴执行映射，分别保留
termination/truncation。首批原始索引须为合法整数；数据与在线快照、原始先验逐字段核对，
并用动作前缀复核在线 ring 中存的是 requested action。训练诊断要求实际 actor 更新事件
的时间戳，不能把上一条 actor loss 当作当前 critic 更新。终态从新模板恢复完整 learner，
检查有限模型/优化器、实际 Adam、RNG 与显式 seed 策略探针，并再次核对产物哈希。
checkpoint 仍不包括 replay、物理环境及 NumPy 抽样器的完整轨迹续跑状态。

39 项专项检查及独立导出包的同 39 项测试通过。新进程重新读取上述三次真实 CUDA 64 步
运行，均通过可复用验收，各恢复 179 个状态字段；没有执行训练更新。另在独立复制的
clip/online-only 目录中构造三个破坏案例：负 replay 索引、将原始请求替换为执行动作并
同步修改裁剪计数、篡改 Adam step 并同步更新 checkpoint 哈希。三者分别在索引、原始
replay 一致性和实际优化器计数处被拒绝，原始运行文件哈希不变。

导出的分析源码、真实复核报告、破坏副本及验证结果位于数据盘
`control/gac_validation_20261019/`。这些检查覆盖验收契约与真实短程接线，不构成百万步
稳定性、GAC/posthoc 性能或正式矩阵已启动的证据；后续批次还须绑定该分析版本。

### 云端独立 CUDA 环境准备

现有四卡实例仍执行冻结 RLPD 队列，GAC 依赖在本地另行准备，没有安装到运行中的
`/workspace/venv`。官方 PyTorch wheel index 已定位 Python 3.12 / x86_64 的
`torch-2.14.0+cu126-cp312-cp312-manylinux_2_28_x86_64.whl`，保留发布方 SHA-256。
根依赖固定 OGBench 1.2.1、MuJoCo 3.9.0、Gymnasium 1.3.0、NumPy 2.4.4 与 pytest 7.4.4；
已完成 62 个 wheel 的下载、版本选择与 SHA-256 锁定，约 3.805 GB。

只读取得现有云端 OGBench 的完整声明依赖闭包，按实际 extras/markers 包含 33 个分发包、
54 条依赖边。最初无约束解析有 19 个版本不同，包括云端 SciPy 1.17.1 与新解析的 1.18.1，
因此没有将最初下载集合直接当成正式环境。选择后的 32 个版本与现有云端一致；唯一例外
是构建工具 setuptools：云端 70.2.0，而 Torch 2.14 的发布 metadata 要求至少 77.0.3，
故显式固定最低兼容版 77.0.3。该例外及发布要求另行绑定，不声称所有环境包完全相同。

版本二工具仅解析每个 wheel 顶层的 dist-info/METADATA，核对官方 Torch 哈希、根版本和
声明的 Linux 平台标签后生成逐包哈希锁。初版曾因 setuptools 内含 vendored dist-info
而拒绝解析，失败目录与日志保留，没有生成错误就绪标记；缺 pip 的首次下载入口也保留。
选择后的 wheel 与来源记录在数据盘 `control/gac_cloud_runtime_20261019/matched-v2/`，
未通过检查的旧 staging 目录未被覆盖。包身份与静态 ABI 标签核查不等于云端验收。

本地新建不继承系统包的 GAC 环境，以 `--no-index --require-hashes` 完成离线安装，
`pip check` 退出 0，62 个分发包的实际版本与安装位置逐项核对；没有 JAX/Flax。
GPU 0 矩阵运算、40 项 learner/driver 与 39 项分析检查通过。在该新环境重新完成三次
真实 64 步入口和独立终态验收，各 31/30/31 更新、各 179 个 checkpoint 字段精确恢复；
四类 CUDA 官方 AST 数值案例最大状态误差为 0。此处是专用本地工程，不能从短测判断
方法优劣，也不认证 cloud RTX4090 或跨 runtime 物理轨迹逐位相等。
完整记录位于数据盘 `control/gac_clean_runtime_20261019/readiness.json`；嵌套研究脚本
首次直接调用缺少 source-root 导入路径，发生在数值案例开始前，改用 `python -m` 后通过，
旧失败记录保留，没有重跑长程科学实验。

只读云端核查确认 CPython 3.12.13 的实际解释器及 `/opt` 位于根 overlay，glibc 为 2.35。
后续环境拟在非挂载的 `/opt` 独立创建，仍须验收实际动态库与所有导入路径。
从当前前端 `TaskToImage` 与 service 追溯到“存为镜像”的实际接口为
`POST /gateway/foundation/api/v1/image`，包含 name/tag/content/uploadType/sceneList/jobId/teamId
及 clusterId；开发任务保存的 uploadType 为 4。当前账号已有 Harbor 账号，源任务仍为
Running。来源文件哈希、scene 字典和只读核查保存于控制目录，未发送镜像创建请求。

CUDA 依赖属于环境实质变化，须在现有队列技术终态后完成云端真实入口短测，再保存新
镜像版本并等待其可用。当前完成了本地依赖及真实工程、云端安装与保存路径准备，不能标记为云端验收完成、
镜像可复用或 GAC 正式实验已启动；也没有为此申请额外付费实例。

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
另行冻结。现已冻结的独立 32 运行采用两任务、四 seed、两半径/边界和两数据协议，训练源
`371fe5b`、分析 `45a0998`，详见[比较协议](../current/experiments.md#gac-完整基线的冻结开发比较)。
该设计尚未启动，仍受云端工程与镜像门槛约束。已有正向 posthoc 结果继续作为完整方法
复现的动机，GAC 核验不替代该主线。
