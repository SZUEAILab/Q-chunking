# 实验：一批实验怎么设计

[文档入口](../index.md)

一批实验（代码中叫 `cohort`）共同回答一个问题。先冻结设计，再启动训练。
本文件是日常实验规则；旧协议中的“推荐”“待做”不是新实验的启动指令。

## 当前执行状态（2026-10-20）

首轮 72/72 全部通过真实退出、完整曲线、预算、checkpoint 与身份验收；28 项固定比较、
56 个 critic 诊断及两张完整图已生成。具体性能与限制见[完整结果](results.md#冻结筛选的完整结果2026-10-20)。
官方 native RLPD 接续的首个 GPU 工程短测因 `--agent.field` 早于显式 `--agent` 而在
解析阶段退出，正式 16 运行尚未因此启动。失败日志、receipt 和原 launch lock 保留。
独立 `launch-recovery-r2` 只修复操作包装的参数顺序，保持训练源码、16 个命令的参数值、
种子与预算；UTD1/20 工程现已分别通过 33/660 次 critic、各 33 次 actor/温度更新、
实际 episode 计数与 checkpoint 验收。2026-10-20 08:32（北京时间）首次启动正式 16 运行，
已核验四个真实训练进程分别使用 GPU 0–3，每卡一个任务。RLPD、几何诊断、posthoc、
FQL 四个监控进程均已恢复。现场证据在该目录的 `formal-live-verification.json` 和
`observer-recovery/verification.json`。同类 posthoc 包装在启动前作相同操作修订，旧部署
及绑定保存在 `config-order-amendment-r2`，科学协议不变。
这不是科学运行重试；后续 posthoc/FQL/GAC 继续按原技术完整性门槛等待，不根据成绩决定启动。
参数修复新增 7 项测试通过，posthoc 操作层 30 项回归通过；两个 UTD 的实际冻结 posthoc
入口仅解析参数的本地验收也通过，未训练。平台只读 API 的重试事件另外写入观察日志，
不再覆盖已完成的主批次阶段；合成 GET 重试和 POST 不重试均验证，无外部请求。
对应记录为原批次 `platform-observation-isolation-20261020/verification.json`，历史 owner
进程与失败证据保持不变；该操作修订不改变训练发布包。

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

首轮冻结摘要没有保存完整 Q 方差与全动作梯度尺度；其零差值只表示本面板未检出不一致，
不能单独区分有用 critic 与常数 critic。后续工具版本补入 Q 均值、标准差、绝对均值、范围，
以及全潜动作/方向 Q 梯度尺度，并显式排除没有方向范数冗余的 Cartesian 辅助头对照。
这些上下文量仍不是价值误差校准，Q 标准差混合了面板中的状态与动作变化。
原冻结工具 `2eb394d` 及接续绑定保持不变，新字段未自动加入首轮 56 个 checkpoint 的报告。

### Actor 参数链的事后局部导数诊断（2026-10-20）

原冻结 critic 56 项仍保留。新工具 `73e9a84` 另问：被动作执行层消去的径向 Q 梯度，
经过 actor 参数链后是否仍有执行响应。固定首轮七个 adapter 方法、两个任务、四个
训练 seed 的全部 56 个最终 checkpoint；复用旧 128 状态面板，噪声 key 固定 20261023，
每状态一次重参数采样。分别计算样本平均 Q 的动作 cotangent、径向/剩余参数梯度、
熵和归档 actor loss 对全部 actor 参数的梯度，并计算局部 JVP 执行响应。

范数比不是浪费百分比，也不具有跨参数化不变性；两个参数分量可能抵消。单位方向的
固定数值下限为 `max(1e-6,1e-6*total_q_actor_gradient_norm)`，低于下限只将相应单位响应
标为未定义，原始量、样本和 checkpoint 全部保留。方向范数低于归一化下限的状态不作
径向等价主张，其完整/剩余梯度仍保留。只允许当前 H1、普通 SAC、无 BC、无额外阶段
损失的接口；不把这一诊断用到没有该方向冗余的辅助头对照。

15 项解析反例、等距控制、真实小网络、零 critic 与接口拒绝检查在本地通过；同一测试
在云端原 84 文件冻结源码上实际退出 0。56 项只读分析已在原实例启动，一次一个 worker，
限定两个 CPU 核，backend=CPU；不新增环境步、优化器更新或 GPU 占用。仅传约 35 KB
压缩诊断包，checkpoint 不传出挂载。每项保留独立进程真实退出和失败，不自动重跑或
替换 seed；owner 实际等待 supervisor 退出后才写终态。独立本地收集进程已启用。

完整计划、输入 checkpoint/config/flags/面板 SHA 与工具版本在数据盘
`control/posthoc_actor_radial_response_20261020/plan.json`、`deployment-binding.json`；实际
云端入口与收集进程在 `remote-started.json`、`watch-process.json`。终态输出为
`reports/action_adapter_validation_20261019/posthoc-actor-radial-response.json`，与原
`critic-invariance.json` 分开。该计划在首轮成绩解封后固定，只提供有限状态/噪声上的
探索性局部导数，不是实际优化器步、学习改进或任务因果证据；现有训练和接续门槛不变。

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

另有独立 CPU 接续进程在上述终态汇总完成后导出固定图表：两任务的全部九种方法按
原版/critic/完整配方分面展示，保留四个 seed 细线与完整四 seed 均值；全部 28 项 AUC
配对差逐 seed 展示，并区分原主比较与两类次分析。没有新增比较或推断区间。
缺失运行保留身份与缺失标记，不画部分 seed 的组均值。输入、工具和 PNG/PDF 输出均
绑定 SHA；完整及缺失两类合成布局已渲染并目视检查，合成图明确水印，不作科学结果。
真实终态图表现已全部生成并目视验收，见[完整结果](results.md#冻结筛选的完整结果2026-10-20)。
输入与产物绑定保存在数据盘 `reports/action_adapter_validation_20261019/figures/figure-receipt.json`；
最初接续绑定与启动证据仍保留，不改变后续实验门槛。

### 将投影加入共同接口对照（2026-10-20）

完整首轮结果中简单投影的两个任务均值均高于原版，因此共同六维 actor 的后续机制
设计需要投影对照。源码 `311d7af` 增加
`--ds_adapter_execution_map=cartesian_projection_aux`：去掉辅助第 4 坐标后，将前三维
XYZ 投影到单位 L2 球，后两维 scalar 保留。它与 `cartesian_aux` 和原版 env-critic
映射共用六维 Gaussian/tanh heads、全部六维潜坐标 log-prob、目标熵 −3、五维 env
critic/replay、参数宽度和初始化；接口同样限定 ACRLPD、H1、普通 SAC、无 BC。

Prior 和在线 replay 保留真实五维环境动作，不因为策略执行受球约束而投影先验。
辅助零坐标 lift 仅服务维度接口；对于单位球外动作，它不是执行映射的逆，也不是 BC
标签。熵仍是六维潜动作熵，不声称是投影后动作的体积密度。统一 warmup 仍执行原有
Cartesian uniform cube 动作；投影约束只适用于 warmup 后的策略动作。

这构成未来三组映射设计的实现基础：Cartesian auxiliary、projection auxiliary、原版
posthoc env。它检验共同 env 接口下的完整执行映射效应，不能直接替代原版 latent
critic 内的机制估计，也不能把投影与原版的差异缩写成独立幅值头或单个分布矩。
五维 native/projection 和完整 latent 方法仍应作为独立配方参照。正式科学矩阵与统计
规则尚未冻结，本变更没有向当前训练队列追加运行。

60 项专项检查及 30 项相关回归（另 3 个 subtest）通过，覆盖共同完整初始化、采样与
log-prob、辅助坐标的 Q 零导数与熵非零导数、raw replay 路由、真实 learner 更新、
NumPy/JAX 映射及不支持接口拒绝。测试还复现并修复已有 L2 投影在零向量处的 JAX
反向 NaN：将范数的下界放到 sqrt 前，零点 Jacobian 为单位映射，球外 Jacobian 与
解析切向投影一致。只影响未来源码；既有 72 运行及当前 reference 发布包均保持冻结。
本地双任务 CPU 真入口与独立 checkpoint 恢复现已通过：各 64 个环境步、33 次更新、
TrainState=34，step 0/64 各观测到 2 个完整非视频 episode。逐任务完整 agent 初始化 SHA
与早期 Cartesian auxiliary 一致，前 32 步 warmup 动作逐项相同，后 32 步策略动作均在
单位球内；prior 原始五维动作仍包含球外样本。四份原始数据 SHA、141 个冻结 Python
文件和 24 项工程产物已核对并绑定。记录为数据盘
`control/posthoc_projection_control_20261020/readiness.json`；该记录只覆盖 CPU 工程。
首次 EGL 导入失败及 warmup 范围误判的验收失败均保留；修订 validator 后重新读取同一
Triple checkpoint，再首次运行 Double，没有重跑训练。

相同冻结训练源 `311d7af` 又在本机 GPU 0（RTX 6000 Ada）完成单任务 Cube Double 的
三组映射短测：direction-speed、Cartesian auxiliary、projection auxiliary 各 64 步。
GPU 上 90 项相关检查及 3 个 subtest 通过；三个真实驱动、三个独立恢复验证和测试进程
均观察到退出 0。每组 33 次更新、TrainState=34、step 0/64 各实际 2 个非视频 episode。
三组可序列化参数/优化器/RNG 初始化与前 32 步 warmup 命令哈希一致；它们的映射配置、
物理模拟器状态和初始执行分布不能据此称为相同。源文件 141 项未变，证据绑定 37 个
工程产物。本次只使用本机 GPU 0，不占用云端队列；记录在数据盘
`control/posthoc_projection_gpu_local_20261020/readiness.json` 和 `evidence-binding.json`。
新增六维投影对照尚无 4090 验收或长程性能结果，正式科学矩阵仍未冻结。

### 执行映射因果对照的实现准备

新增 `--ds_adapter_execution_map=cartesian_aux`，用于共同六维 actor 下的 Cartesian 执行
对照。它保留原版 cartesian/tanh/log actor 的参数化、所有六个潜坐标的 log-prob 与温度，
只在执行与 actor/target Q 查询中去掉第 4 个辅助幅值坐标；前 3 个 XYZ 和后 2 个 scalar
直接进入环境。两组 critic 均读五维真实环境动作，prior/replay 不补造观测到的辅助动作。
接口限定 ACRLPD、H1、env critic、无 BC 的普通 SAC；辅助零坐标只用于代表性 lift，
不是有观测依据的行为克隆标签。checkpoint 视频恢复也读取显式执行映射标记。

源码 `7058825` 通过 50 项相关检查，包括相同 actor/critic/优化器/RNG 初始化、共同动作
采样与 log-prob、辅助坐标的零 Q 输入导数和保留的熵导数、真实 learner 更新及不支持的
接口拒绝。两个开发任务各两种映射的 CPU 64 步入口全部退出 0，独立恢复逐项核对
64 个环境步、33 个优化器更新、TrainState=34、六维 actor/五维 critic、目标熵 −3 和
step 0/64 评估轴；每任务的两种映射初始完整 agent 状态 SHA 相同，包含网络、优化器和
RNG，不包括模拟器物理状态。评估实际 episode 计数仍受
原 driver 限制，仅有 flags 依据，不能称为已观测计数。工程证据在数据盘
`control/posthoc_map_control_20261020/readiness.json`；该记录只覆盖 CPU 工程。

后续源码 `a087495` 为新在线运行补入实际评估计数：从返回的完整非视频轨迹计算 episode
数和 terminal success 次数，并核对聚合 success；缺失、非二值或不一致标签不能当作失败
补零。36 项本地相关检查通过，包括视频排除、计数不符、部分标签缺失和 float32 均值舍入。
这不回填 `7058825` 工程或原冻结批次的日志，既有 `flags_only` 限制继续保留。

该后续源码在同一四卡实例的空闲 GPU 0 完成 36 项检查，以及 Cube Double 上两种映射各
64 步的真实入口与独立恢复。五个实际子进程均退出 0；每组 33 次更新、TrainState=34，
step 0/64 各实际返回 2 个非视频 episode，二值成功次数与聚合 success 一致。两组完整
agent 初始化 SHA 相同，37 个已有控制/科学配置文件哈希不变。只上传约 264 KB 的独立
源码及工具包，没有新增实例；该验收不覆盖 Cube Triple 的 GPU 工程或新长程科学比较。
证据在数据盘 `control/posthoc_map_control_gpu_20261020/cloud-receipt.json`。

该入口独立于现有 72 运行和后续已冻结基线源码，尚未登记新的长程训练矩阵。
现有源码保持默认 direction-speed 执行路径；新对照与原版 env 版本将需在共同源码、
数据与完整配置下配对，不能直接把两个来源的成绩差全部归于映射。短程工程与初始化
等价性只能验收接口，不证明幅值机制导致任务收益。研究假设及反证见
[研究计划](research.md#幅值机制仍缺的训练干预)。

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
独立云端启动控制器已生成并完成逻辑验收，新增本地接续 owner 已启用：等待既有 FQL 及上游技术终态，再复用当前四卡实例；完整离线
依赖、云端四卡预验收、真实短测与独立核验、新镜像保存可用均为启动前条件，没有成功率
方向门槛。数据与完整配置、24 项对照、来源和冻结哈希位于数据盘
`control/gac_reference_comparison_20261019/`。该批次不替代 fresh-seed、held-out 或跨域确认。

固定依赖归档与 runtime v2 已分开导出并完成真实解包验收，源/分析 40+39 项检查及
解包后一次 CUDA 64 步入口通过。指定上游 Git 的浅克隆边界已正确打包与还原，首个失败
归档保留；科学文件未修改。runtime v2 已在原实例完成静态暂存校验；依赖包于北京时间
2026-10-20 00:20 完成远端哈希与读回校验。云端安装、GPU 工程、镜像和正式运行仍待完成，见
[部署范围](../reference/gac_baseline.md#部署归档与暂存)。

操作工具另以 `fc33356` 版本化：[配对汇总](../../research/gac_comparisons.py) 实现上述原定
24 项对照，不改变对照身份或指标；[执行核心](../../research/gac_execution.py) 逐卡按冻结
位置串行，分别观察驱动与独立验收进程的退出。执行核心没有 launch CLI；独立云端
控制器 `13c2ba8` 提供来源绑定的 inspect/launch/own/supervise，核验上游、云端工程及可用
镜像，owner 实际等待 supervisor 退出后才写终态。环境准备已有独立远端接续进程，
实际 PID 与脚本哈希已核验，当前等待 FQL 完整终态及四卡空闲。它只提交一次环境准备，
观察子进程退出后核对八类工程回执；失败保留原 attempt，不保存镜像或启动科学矩阵。
镜像保存与科学启动的本地接续现已启用，当前停在 waiting-for-FQL 门槛；实际镜像和科学训练仍未开始。
62 项核心测试及本地真实 64 步 CUDA 工程通过，云端只验证了
纯逻辑/模拟进程测试与全缺失汇总，范围见[工具验收](../reference/gac_baseline.md#四卡执行核心与完整配对汇总)。
已有科学配置、32 个身份和当前队列保持冻结；正式 GAC 训练尚未启动。

包括新控制器的 142 项逻辑检查在本地导出包和云端通过；真实云端 inspect 返回
`waiting_for_fql`，launch 入口按预期拒绝提前启动，科学输出及环境路径在检查前后均不存在。
这些是门槛/逻辑证据，不代表 CUDA 安装、镜像已保存或科学实验已启动，详见
[完整控制器范围](../reference/gac_baseline.md#完整云端启动与终态门槛)。

### GAC 镜像与正式启动的持续接续（2026-10-20）

源码 `d4c42c5`（实际列表字段适配 `0bc033b`）增加[接续逻辑](../../research/gac_image_continuation.py)、
[本机 API/SSH 适配](../../research/gac_continuation_transport.py)及
[本地 owner](../../research/gac_continuation_owner.py)。它使用已暂存的 frozen GAC controller，
不修改训练、验证或科学矩阵。源绑定发布包已在本机独立目录启动；真实 PID/命令已核验，
状态为 `waiting_for_fql`。已有远端环境 owner 继续负责一次环境准备，两个进程职责分开。

接续顺序为 FQL 完整技术终态 → 云端八类 CUDA 工程全部验收 → GPU 空闲 → 保存镜像
并同时核对详情及可用列表 → frozen controller 再验收所有门槛 → 首次启动 32 个科学运行。
每次等待均核验依赖 owner 的实际进程；本地规格、代码和前端证据持续核对 SHA。
API 使用前重新取得当前控制台主入口和三份已检查的前端源码，构建变化时留待检查；
只读接口允许有界观察重试，保存镜像只提交一次。提交结果不明时仅查询该唯一版本，
不重复 POST；科学启动结果不明时只观察原 source-bound started/terminal，不重复启动。
镜像 proof 仅保留白名单字段，令牌和 SSH 私钥留在本机。真实列表的 `jobId/uploadType`
为空，来源以详情严格核验，列表按 ID/版本/引用对账并保留原始空值；已用现有可用
镜像真实验证两种列表。旧等待进程及绑定保存在 `before-list-schema-recovery/`，替换前
确认没有保存/启动 intent，未改动训练进程。

61 项新增检查及既有 142 项回归通过；来源绑定的实际云端 inspect、原任务 Running/
团队/集群身份、镜像全状态及可用状态分页均通过真实只读预检。启用前后 35 个科学
文件 SHA 完全一致，环境目录、镜像回执及 GAC 科学输出均不存在，未申请额外实例。
本地记录位于数据盘 `control/gac_image_continuation_20261020/`：`owner-binding.json`、
`owner-process.json`、`preflight.json`、`arming-verification.json`；状态为 `owner-r2/status.json`。
正式终态后才下载 32 项运行记录及 24 项完整比较，保存到
`reports/gac_reference_comparison_20261019/`。这项接续的启用不代表新环境、镜像或长程
结果已经完成，也不表示已在新实例验证拉取；当前只复用原四卡实例。

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
