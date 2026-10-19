# 实验：一批实验怎么设计

[文档入口](../index.md)

一批实验（代码中叫 `cohort`）共同回答一个问题。先冻结设计，再启动训练。
本文件是日常实验规则；旧协议中的“推荐”“待做”不是新实验的启动指令。

## Action-adapter 冻结筛选

批次为 `action_adapter_validation_20261019`，配置、源码摘要、工程尝试与平台记录放在
数据盘 `control/action_adapter_validation_20261019/`。用户将资源改为 4×4090，保持不限时长；使用已有
兼容镜像、持久化挂载与独立源码发布目录，不覆盖其他批次。
2026-10-19 因八卡排队时间过长，停止尚未运行的旧任务并提交四卡替换任务；
`resource-amendment-4gpu.json` 记录此次资源修订。原科学预注册与任务配置保持冻结，
首轮使用 GPU 0–3、每卡最多 8 个任务；基线阶段每卡 1 个任务。旧部署包和监控记录保留。

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
首轮 joint optimizer 的计数从零记录为 Adam count，TrainState 则从 1 开始；验收分别核对
`max(0, budget-max(1,start_training)+1)×UTD` 及其加 1，并检查优化器状态有限。
首轮驱动没有实际返回的 episode 计数，该证据限制保留在报告中，不能由配置值冒充观测。

训练后的 critic 诊断由 [`research.action_adapter_diagnostics`](../../research/action_adapter_diagnostics.py)
独立读取冻结源码与 checkpoint。状态来自输入 SHA-256 固定的 validation-prior 非终止观测，
面板 seed 为 20261020；共同动作探针 seed 为 20261021，128 状态，方向缩放 0.5。
分别报告策略条件和共同探针下的等效动作 Q 差、特征差、冗余径向 Q 导数及其梯度占比。
报告绑定源码文件、数据、checkpoint、解析配置和诊断工具哈希；不修改模型参数或 RNG。
该离线工具与已冻结的训练发布包分开版本化，不修改首轮 72 个运行。
动作/价值诊断不是任务性能证据，结果缺失或失败不能当作零差值。

### 终态配对比较的完整汇总

[`research.posthoc_screen_comparisons`](../../research/posthoc_screen_comparisons.py) 在原批次
验收并下载终态报告后，独立重算真实 0–1M 曲线的 AUC、endpoint、tail3 和 threshold/censoring。
保留原六项主比较；另在训练启动后、尚未查看中途成功率时登记 22 项描述性次比较，
分别是两个任务内各八项机制对照和三项 legacy/direct 配方对照。具体左右方法、问题与
解释范围固定在批次 `screen-comparisons/` 的 SHA 绑定计划中，不根据结果方向更改。

机制对照覆盖 env/latent、canonical/env、共同熵目标、projection/native 和 legacy/projection；
共同熵目标相对 native 也单独报告。配方对照按三种 critic 接口分别比较 direct 与 legacy，
不能称为只改变 log-speed 的消融。输入配置差异逐 seed 列出，不能仅凭组名声称因素隔离。

每项保留四个训练 seed 的配对值，只有四对全部有效才生成组均值、中位数、正负配对数和
描述性 percentile bootstrap 区间。枚举全部 `4^4=256` 个经验重采样以去掉 Monte Carlo
波动；这不提供小样本覆盖率或多比较保证，也不把次分析升级为确认性检验。
缺失与失败显式保留，不用可用子集估计完整四配对效应；阈值时间逐 seed 保留删失标记，
不只对成功运行取平均。该工具依赖上游生命周期验收，不冒充再次读取原始日志或恢复模型。
单独本地 CPU 监控在首轮终态后立即生成 JSON 和 Markdown 汇总，不必等待 reference 完成。

### 原论文 RLPD 的独立参考配方

原论文 RLPD learner 的独立配方移植已完成数值对照，见[基线来源与验证](../reference/rlpd_baselines.md)。
它与项目 fork、FQL 作者的 SAC/RLPD 是不同配方；后续正式比较需单独冻结，报告 UTD、
网络、目标聚合、熵备份和真实计算量，不能替换首轮方法组或称所有差异都来自动作表示。

独立批次 `action_adapter_reference_20261019` 已冻结两个任务、seed 10–13、UTD=1/20，
共 16 运行，复用首轮 4 卡实例并等待原 72 运行全部验收后启动；两个批次都保留自己的终态。
训练源为 `2c2d0c3`，独立运行包有逐文件哈希，50 episodes、真实 step 0–1M 评估网格与
原批次相同。与 native、legacy/direct latent、legacy/direct canonical 的全部 20 个
task/UTD/method 对照都作描述性报告，缺失配对显式显示，不用部分 seed 估计完整四配对效应。
正式基线每个评估点必须记录实际返回的 episode 数与成功次数，并核对聚合 success。
GPU 预验收、配置、退出码、有限值、精确 critic/actor/温度更新次数和 checkpoint 通过后才纳入。

原版 posthoc 现已另行接入同一原论文 RLPD learner 配方，见[适配范围与数值验证](../reference/rlpd_baselines.md#原版-posthoc-在同一-rlpd-配方中的适配)。
它保持原版 tanh/log/sphere latent 方法，支持 H1、UTD=1/20 与显式共同数值熵目标；
六维潜动作正则也进入 entropy backup。新源码 `e20834c` 与原两个批次分开，不改写已冻结
的 72/16 运行或预先指定后续优胜配方。

独立批次 `posthoc_reference_20261019` 已冻结 16 个原版 posthoc 运行：同两个开发任务、
seed 10–13、UTD=1/20、每次 1M 步；与已冻结的 16 个 native 配对，不重跑 native。
四项同 RLPD 配方的 posthoc−native 比较分别报告；另报 UTD1 的两项方法×learner 配方
交互，以及同 RLPD 内两项方法×UTD 交互，共八项描述性比较。每项都要求全部四个 seed
及其所有输入方法可用，缺失时不输出完整组估计；实际 0–1M AUC 为主，endpoint/tail3 为次。
经验配对 bootstrap 枚举 `4^4` 个重采样，不作确认性、多比较覆盖或纯坐标因果主张。

两份源码共有的 90 个 Python 文件中，88 个字节相同；另两项是 native 初始化/校验
函数重构和数值核对工具扩展，均另有上游数值证据。新批次使用 `e20834c` 的 106 文件
快照；14 项控制文件与配置、数据、计划和输入证据分别绑定哈希。native 与 posthoc
分批执行，记录环境与时长；这不证明长程轨迹逐位相等，也不是 fresh-seed 确认。

接续监控等待原 72 运行、16 个 native，以及 critic、动作几何、终点策略和配对汇总四类
分析完整验收后，复用同一 Running 四卡实例，每卡一个任务。门槛只依赖技术完整性与
身份核验，不按成功率或比较方向决定是否启动。远程仍须通过 UTD1/20 真实短测、实际
episode 计数、精确 TrainState/Adam 更新数和可恢复 checkpoint；不创建额外实例、不自动
重试、覆盖、续跑或替换 seed。控制包和计划位于数据盘 `control/posthoc_reference_20261019/`。

### 已登记的动作与时间次指标

[`research.action_adapter_geometry`](../../research/action_adapter_geometry.py) 独立流式读取逐步
`env.csv`，实现首轮计划中的执行动作范数、协方差、边界占比与运行时长分析。它核对每个
环境步、全部动作分量和日志 norm/Linf 的一致性，并将原始 CSV 哈希绑定到结果。
每次分析还核对冻结任务配置、训练源码、有效配置及上游真实退出码；不重新训练。

随机 warm-up（`step <= start_training`）与策略收集动作分别统计。1M 步、warm-up 5000
的 H1 运行有 995000 条后续策略动作；该数量不同于 optimizer 更新数。另按预先固定的
100k 环境步窗口描述变化，不根据成功率选择时间段。使用分块、中心化 float64 矩合并，
给出各 seed 的样本协方差和等 seed 权重的条件汇总，不池化数百万转移冒充独立重复。

三个边界量分别定义，阈值容差均为 `1e-6`：`abs(||a_xyz||₂−1)<=tol` 是单位球面附近，
`||a_xyz||₂>1+tol` 是单位球外，`any(abs(a_xyz_i)>=1−tol)` 是 cube 平移边界附近；
同时报告所有动作分量的边界占比。边界占比不是对 pre-clipping 事件的直接观测。
原日志的 `action_xyz_at_unit_bound` 表示 `norm>=1−tol`，不能当作“仅在球面上”的占比。

这些协方差是访问状态与时间上的动作边际统计，不是固定状态条件下的策略探索协方差，
不能据此单独认定探索改善或解释成功率差。receipt 时长包含启动、JIT、数据载入、训练、
评估与保存；共享 GPU 上的各运行时长不能相加当作独占 GPU compute hours。

独立监控在两个训练批次终态后，用原实例最多四个 CPU worker 分析 72+16 个身份；
若原筛选未通过扩展门，原批次可用结果与未执行的 reference 批次会明确区分。
失败、缺失或非法日志保留，某组未齐全时不生成完整 seed 组均值。

### 训练后策略的固定状态次分析

[`research.posthoc_checkpoint_policy`](../../research/posthoc_checkpoint_policy.py) 延续初始化诊断，
只读取原 72 运行中 native 8 个和原版 tanh/log posthoc 32 个最终 1M checkpoint。
后者包含 latent/env/canonical/common-entropy 四种接口或设置；direct 和 projection 保留为
不适用记录，不与原版混算。40 个身份、预算、工具、helper 和计划分别绑定 SHA-256。

每个任务固定 128 个 validation-prior 非终止状态、每状态 64 次共同 Gaussian draw，
面板 seed=20261020，噪声 seed=20261023；原始噪声倍率固定为 0.2/0.5/1/2。
对已学 actor 同时计算实际执行映射和同 heads 的 Cartesian 映射反事实，报告固定状态下的
范数均值、范数 SD、XYZ 协方差及裁剪诊断。它不是重训对照，不能单独确定成功率收益来源。

该次分析在训练已启动、尚未读取中途成功率时登记；不是原始预注册中的确认性检验。
它在 72+16 训练批次与已有动作几何分析结束后复用原实例，单进程、CPU 执行，不追加训练。
只接受通过上游验收的 checkpoint，核对冻结源码、真实退出码、有效配置、精确优化器计数
及文件哈希；失败和缺失不补 seed，四个 seed 未齐全时不输出该组汇总均值。

### FQL 跨算法准备

FQL 后续批次需先通过[固定官方 learner 核对](../reference/fql_baseline.md)，再验收真实
数据、执行 mapper、replay、actor/target 查询、评估和 checkpoint。现有 QC/FQL fork、
官方 native 参考与 posthoc 适配分别标记；alpha 是蒸馏权重，不沿用 SAC 熵目标解释。
新增独立 H1 driver 和潜动作 posthoc 后，不插入既有 RLPD 矩阵；离线预算、在线采样配比和
正式比较单独冻结。FQL 硬 clip 与 SAC tanh 的输出分布差异必须披露，不能把执行函数
相同解释为完整随机策略相同。

独立开发批次 `fql_posthoc_transfer_20261019` 现已冻结：Cube-Triple task2 / Cube Double task2、
native/posthoc、相同 seeds 10–13，共 16 运行；训练源码固定 `8e2e850` 的 117 个 Python 文件。
两个方法都使用官方 learner 的完整 512×4、两个 Q、mean target、flow10、alpha300、
Q 归一化关闭、batch256；每运行 1M 离线更新和 1M 在线环境步/更新，在线无随机 warm-up。
alpha300 取自固定官方 Cube Double 命令，对两个方法及 Triple 原样使用，不事后分方法调参。

全程真实总更新轴为 0–2M，每 100k 评估 50 episodes，独立 eval seed=20261019。
主指标只积分真实在线阶段 `[1M,2M]`，减去离线预算后映射为在线环境步 `[0,1M]`；
真实离线终点评估就是在线 step 0，不补零、不把离线更新当作在线样本效率。
另报在线 endpoint/tail3、离线 AUC/endpoint、在线 time-to-50% 及删失、更新数和 wall time。
两个 task-wise posthoc−native 比较都要求全部四对；枚举 `4^4` 个描述性重采样，不作确认检验。

两方法统一采用 uniform combined replay，请求容量 2M，实际按官方逻辑取
`max(2M, prior_size+1)`。Triple 先验为 3M 转移，实际容量 3,000,001；Double 为 1M，
实际容量 2M，因此先验保留与覆盖过程跨任务不同，但每任务内两方法一致。原始环境动作
保留在 replay，取 batch 时再编码；没有据性能切换到 balanced 模式。

该批次接续同一四卡实例，每卡一个运行；任务与 native/posthoc 顺序按 seed/GPU 预先
平衡。只在既有 RLPD posthoc 及其上游训练/分析技术完整验收后启动，没有收益方向门槛。
云端仍须检查四张 4090、冻结源码/数据、解包测试，并完成 native/posthoc 的真实短测与
独立 checkpoint 验收；不得自动重试、覆盖、续跑或替换 seed。
本批仍是两个开发任务上的跨算法验证，不能称为 fresh-seed 确认或广任务泛化。
计划、逐运行配置、部署包和接续控制器位于数据盘 `control/fql_posthoc_transfer_20261019/`。

### GAC 近邻对照的训练准备

[`main_gac_reference.py`](../../main_gac_reference.py) 保留官方配方 learner，另行声明本地
单环境 OGBench driver。`online_only` 不声明或加载 prior；`prior_balanced` 每个 critic
minibatch 使用一半冻结训练先验、一半在线 replay，作为 GAC+prior 协议单独标注。
该比例与现有 RLPD 对照的数据比例对齐，网络、目标、温度 surrogate 与更新节奏仍不同。

两协议的预算、数据 SHA、版本、源码、独立评估和 learner 配置由 JSON plan 固定；
原始环境动作保持五维，结束与截断分开记录，bootstrap 使用 termination 而非 truncation。
在线 replay 使用独立 ring，prior 保持只读。记录抽样行数及首批原始索引；抽样含放回，
同一 critic batch 会被 actor 重用，不能把计数当作唯一经验数或独立统计样本数。

更新保持官方零基循环条件；1M 环境步、learning_starts=5000、policy frequency=2、
target frequency=1 时，预期 critic/actor/target 次数为 994999/994998/994999。
评估使用真实 step 0、间隔点及预算终点；actor loss 保留其实际更新 global step，
避免将上一次 actor loss 标记成当前 critic 步。

源码 `29536cb` 的 125 文件快照通过两种协议各一次真实 CUDA 64 步工程，初始化模型/RNG
一致，更新次数与数据来源计数通过独立复核；19 项入口契约测试通过。
这些为开发验收，不提供 GAC/posthoc 性能判断；GAC 云端 CUDA 环境尚未验收，正式比较尚未启动，
不向既有 RLPD/FQL 批次插入运行。证据位于数据盘 `control/gac_driver_20261019/`。

新入口 `371fe5b` 单独增加显式 clip 执行，保留请求坐标 replay/Q，用于覆盖作者 MuJoCo
示例中的较大 radius，而非将原 radius=1 工程当作全部强基线。三项本地 CUDA 64 步、
独立数据/checkpoint 核验和 radius=2.5 的原始 AST 数值对照通过，见
[执行边界](../reference/gac_baseline.md#较大-radius-与执行边界的独立验收)。两种 radius、裁剪
与数据协议的正式比较仍须另行冻结，不能从工程裁剪比例或短测结果选择优胜设置。

可复用终态工具 [`research.gac_reference_validation`](../../research/gac_reference_validation.py)
另以 `45a0998` 固定分析来源，验收完整预算、实际 episode、请求/执行动作、原始批次、
更新与 checkpoint；39 项专项检查、三份真实 CUDA 工程复核及三种破坏副本拒绝检查通过。
工具与训练发布分开绑定，尚未用于百万步 GAC 运行；见[验收范围](../reference/gac_baseline.md#可复用的长程验收工具)。

CUDA 依赖已形成 62-wheel 哈希锁并完成本地独立离线安装与真实 GPU 工程，OGBench 声明
依赖中 32/33 个版本与现有云端一致，setuptools 的 Torch 必需版本例外显式登记。云端
安装、GPU 验收与镜像仍待完成；已核对根文件系统位置和当前前端保存接口，详见
[环境准备边界](../reference/gac_baseline.md#云端独立-cuda-环境准备)。现有训练环境和批次保持冻结。

### GAC 完整基线的冻结开发比较

`gac_reference_comparison_20261019` 已单独冻结 32 个运行：两个开发任务 Cube Triple task2／
Cube Double task2、seeds 10–13、两种 radius/执行配方、两种数据协议，每次 1M 在线环境步，
没有离线更新。训练源固定 `371fe5b` 的 125 个 Python 文件，独立验收分析固定 `45a0998`。

| 配方 | 执行与学习坐标 | 数据协议 |
|---|---|---|
| radius=1 | identity；完整五维球内请求，Q/replay 使用请求 | online-only 与 prior-balanced 分开运行 |
| radius=2.5 | 逐轴 clip 后执行；Q/replay 仍使用原始请求 | 同上 |

其余保留完整官方 learner 配方，不按短测裁剪比例或当前筛选结果调整参数。online-only 是
数据访问参照；与 RLPD 的主要完整算法比较使用 prior-balanced。后者每次 critic update
抽取 128 prior／128 online，含放回且 actor 重用同批次；在线 ring 容量 1M，prior 独立只读。
两种配方均不称为与 posthoc 支持完全匹配的几何消融，也不将作者 MuJoCo 半径写成 OGBench
最优值。预期 critic/actor/target 次数分别为 994999/994998/994999，保留与 RLPD 时钟的差异。

固定真实 step 0–1M 的 100k 网格、每点 50 episodes、独立 eval seed 20261019，主指标为
归一化 success AUC；另报 endpoint、tail3、右删失 time-to-50%、实际更新数和时长。
24 项描述性对照包括任务内八项 GAC radius/边界或数据协议比较，以及原论文配方 native／
原版 posthoc 的 UTD1/20 相对两个 prior-balanced GAC 配方的十六项完整算法比较。
各项必须有全部四个配对 seed 才生成组估计；枚举 `4^4` 个经验 bootstrap，不作确认性或
同时覆盖主张。失败、零结果和缺失均保留，不重跑、续跑、替换 seed 或只汇总可用子集。

四个 GPU lane 每个固定一个 seed、串行八个条件；XOR offsets 0/3/5/6 使每个队列位置的
任务、数据协议、半径各 2/2，任意因素对覆盖全部四种组合。配置与调度表已生成并核验。
它们尚未接入启动控制器：等待既有 FQL 及上游技术终态，再复用当前四卡实例；完整离线
依赖、云端四卡预验收、真实短测与独立核验、新镜像保存可用均为启动前条件，没有成功率
方向门槛。数据与完整配置、24 项对照、来源和冻结哈希位于数据盘
`control/gac_reference_comparison_20261019/`。该批次不替代 fresh-seed、held-out 或跨域确认。

固定依赖归档与 runtime v2 已分开导出并完成真实解包验收，源/分析 40+39 项检查及
解包后一次 CUDA 64 步入口通过。指定上游 Git 的浅克隆边界已正确打包与还原，首个失败
归档保留；科学文件未修改。runtime v2 已在原实例完成静态暂存校验，依赖包仍在传输；
云端安装、GPU 工程、镜像和正式启动控制器仍待完成，见
[部署范围](../reference/gac_baseline.md#部署归档与暂存)。

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
- 仅给定 reset seed 不自动证明完整求解器状态相同；单步物理反事实先检查完整起点与重放，报告 seed 的实际覆盖范围。
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
