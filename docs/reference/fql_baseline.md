# FQL 参考学习器的来源与验证范围

[文档入口](../index.md) · [当前实验](../current/experiments.md)

跨算法检验需要先固定 FQL 配方。项目现有 `agents/acfql.py` 是带 action chunking 的
QC/FQL fork；它的 H1 更新仍与官方 FQL 存在差异，不能仅凭 H1 就称为官方参考实现。
新增 [`agents/fql_reference.py`](../../agents/fql_reference.py) 独立保留固定官方版本的
状态观测、单步 native learner；另提供显式本地 driver 与潜动作 posthoc 适配。
尚无 posthoc 的 FQL 长程性能结果。

## 固定来源与配方

来源为 [seohongpark/fql 的 `e8cd16e`](https://github.com/seohongpark/fql/tree/e8cd16eb490332924dfa2492097219f181765933)，
使用[官方 MIT 许可](fql-upstream-LICENSE.txt)。数值 oracle 直接读取干净 checkout 的
`agents/fql.py`、`utils/networks.py`、`utils/encoders.py` 和 `utils/flax_utils.py`，绑定逐文件 SHA。
本地参考类复用项目网络和 TrainState；本地增加的 Fourier 分支保持关闭，是否等价由
独立加载的官方模块核对，不用本地类同时充当参考答案。

| 项目 | 固定官方 FQL / 新参考类 | 现有 QC/FQL fork |
|---|---|---|
| 训练目标 | 单步 TD | 按 H 构造序列目标 |
| Q 归一化开关 | 参与 actor Q loss，保留 stop-gradient 系数 | 同名配置未在 actor loss 中使用 |
| 策略采样 key | 外部 key 先 split，再采样 flow base noise | 直接使用外部 key 采样 |
| 初始化模块顺序 | critic、target、BC flow、一步 actor | BC flow、一步 actor、critic、target |
| teacher/student | Euler BC flow、一阶蒸馏 actor，均逐轴 clip | 另有 best-of-n 与时间特征开关 |
| alpha 默认值 | 10，蒸馏权重 | 100，蒸馏权重 |
| target EMA | 使用梯度更新前 critic | 使用梯度更新前 critic |

新类保留官方两个 Q、mean/min target 聚合、BC flow matching、蒸馏、actor Q loss、
Adam、RNG 拆分与更新前 EMA。alpha 不是 SAC 熵温度；源码默认值也不是任一任务的
已调参最优值。正式批次需另行固定 alpha、是否归一化、离线预算和在线 replay 配比。
不把官方 README 配方、本地默认值和新任务调参混成同一基线。

## API 适配与边界

参考类接受单个或 batched 示例，内部以 batched 示例初始化，保留官方参数与 key 身份。
`sample_actions` 支持本地 `rng=` 别名和官方 `seed=`，同时传入两者会拒绝；采样不推进
learner RNG。官方 temperature 参数不参与计算，新类仅接受默认 1，避免暗示 SAC 温度语义。

`batch_update` 把本地 UTD×batch×H=1 序列桥接为顺序官方更新；每个 minibatch 同时
更新 critic、teacher 和 student。只接受有效、无 padding 的 H1 数据，保留两种 RNG 的区别。
该桥接不规定采集方式、prior/online 比例或离线预训练步数，也不改变官方 `update`。
自定义 TrainState 从 1 开始，三次 update 后为 4，不能据计数器直接推断四次优化。

Native 类继续拒绝 adapter；两类均拒绝 projection、integrated DS、H>1、像素 encoder、
Fourier 特征、weight decay、best-of-n 和非两个 Q 的设置。FQL 的硬 clip 输出与
原 SAC tanh 输出分别标记，不能把 native learner 数值通过直接描述为 posthoc 性能验证。

## 数值与行为验收

[`research.check_fql_reference`](../../research/check_fql_reference.py) 每例在独立 CPU 进程
加载未修改的官方模块。固定 synthetic 数据包含非零 reward、终止与非终止 mask；
核对初始化、外部 key 采样、BC flow 积分、三次更新的所有参数、优化器状态、TrainState、
RNG 和 loss。覆盖 mean/min、Q 归一化开/关、1/3/10 个 flow step、5/6 维 native 动作、
alpha 10/100/300 以及完整 512×4 网络。六维案例只是通用 learner 数值覆盖，不是 posthoc。

每例核对本地与官方输入文件在执行前后的哈希，并拒绝覆盖已有报告；精确通过情况与
最大误差记录在数据盘 `reports/fql_reference_20261019/fql-reference-equivalence.json`。
15 项本地行为测试覆盖采样 API/边界、无 padding 的 sequence bridge、checkpoint 后续更新、
更新前 EMA 及不支持配方拒绝。

上述初版验证仅覆盖 native learner，报告和源码身份保留在原批次，不被新验收覆盖。

## Posthoc 适配与独立训练路径

[`agents/fql_posthoc_reference.py`](../../agents/fql_posthoc_reference.py) 继承相同 update、
actor/critic loss 和 flow 积分，只单独验证与初始化 D+1 潜动作网络。执行端采用原版
Cartesian 方向、log 幅值、sphere 参考半径与最终逐轴 clip；critic、teacher 数据、actor
查询与 target 查询都在潜坐标。FQL 的一步 actor 和 teacher 保持官方硬 clip 输出，
没有替换为 tanh Gaussian，也没有 SAC 熵或温度损失。

共享 codec 的 `adapter_speed_squash=tanh` 只固定原编码约定；解析配置另记
`policy_output_transform=clip`，明确实际 FQL 输出。该适配保留原执行函数，但不能称为
SAC 完整策略分布的逐项移植。数据中 log-radius 小于 −1 的 canonical 编码保留原值；
teacher 输出仍受 clip 限制，这项支持差异须披露，不通过截断 replay 隐藏。

[`main_fql_reference.py`](../../main_fql_reference.py) 是独立的本地 H1 OGBench driver。
JSON plan 显式登记源码、数据 SHA、native/posthoc、离线与在线预算、learner 配置、
uniform combined replay 或 50/50 balanced replay，以及固定评估协议。两种方法均把真实
环境动作保存到 prior/replay，只在取 batch 时编码；不对先验先 decompose 再 compose 后
写回，以免把编码的数值边界改写为原始动作。

在线从第一步策略采集，每个环境步一次联合更新，没有 RLPD 的随机 warm-up 或仅 critic
UTD。官方 uniform combined replay 和显式 balanced 模式分别登记，不把二者默认为相同。
评估保留真实 step 0、离线终点、固定间隔及总预算终点，记录每个实际返回 episode 的长度
与终点 success；采样 key 与 learner RNG 分离。数据 helper 与评估是本地适配，不能称为
上游实验脚本或旧 benchmark 的完整复现。固定 reset seed 也不证明全部物理缓存逐位相同。

运行拒绝已有输出目录，输入在执行前后核对，错误保留 incident，不自动重试；终点核对
实际环境步、联合 Adam count、TrainState、有限参数/优化器与完整 checkpoint 恢复。
正式长程结果、fresh-seed 确认、广任务结论仍待完成；现有云端三份发布包与矩阵保持冻结。

## 新适配的数值与真实环境验收（2026-10-19）

源码 `8e2e850` 的独立 117 个 Python 文件快照完成十类官方数值核对：五类 native 与
五类潜动作 posthoc，后者把真实环境动作编码为六维，并显式包含零动作对应的
log-radius≈−13.8155。每例三次更新、完整参数/优化器/计数器/RNG/loss 均一致，最大状态
差为 0；覆盖完整 512×4、mean/min、Q 归一化开关与不同 flow/alpha 设置。报告为数据盘
`reports/fql_posthoc_reference_20261019/fql-posthoc-reference-equivalence.json`。

本地空闲 GPU 0 另完成四次真实 Cube-Triple task2 工程检查：native/posthoc 各自采用
uniform combined 或 balanced replay，完整 512×4、batch256、flow10、alpha300，均退出 0。
每次固定 4 个离线更新、64 个在线环境步及更新，实际联合 Adam count=68、TrainState=69；
总更新轴的 0/4/64/68 四个评估点各实际返回 1 个 episode，末状态与完整 checkpoint 检查通过。
这四次短测仅检验接线，不支持性能或 replay 模式优劣判断；没有追加云端实例。

独立 CPU 进程从全新模板恢复四个模型，每个 checkpoint 的 171 个状态字段在值、形状和
类型上精确一致。原始 256 条采集动作与对应执行映射逐条一致；首个离线/在线 batch 编码
与原记录一致，保留原始 replay。44 项本地行为和 driver 契约测试通过。

首次独立检查错误地要求重新序列化后的整段字节相同；诊断确认字段精确一致，只有字典
插入顺序变化。保留失败检查后改用逐字段精确核对，四次训练及原 checkpoint 未重跑或改写。
GPU 编译日志包含 xtile diagnostic dumps，原日志全部保留；通过判断依据真实退出码、
执行计数、有限值与可恢复模型，不宣称日志无诊断信息。验收与检查记录位于数据盘
`control/fql_posthoc_reference_20261019/engineering-receipt.json` 和
`control/fql_posthoc_reference_20261019/engineering-independent-validation-v2.json`。

## 独立开发批次与阶段分离

16 次 native/posthoc FQL 比较已单独冻结，具体矩阵、预算、replay 与接续门槛见
[当前实验](../current/experiments.md#fql-跨算法准备)。训练源码继续使用已验收的 `8e2e850`，
不把新分析工具混入该源码身份；[`research/fql_reference_validation.py`](../../research/fql_reference_validation.py)
单独版本化，核对实际进程退出、完整动作流、评估 episode、有限损失及新模板 checkpoint 恢复。

主指标固定在在线环境步轴；离线曲线单独报告。缺失/失败 seed 不生成完整组均值，
time-to-50% 保留右删失，低成功率不触发重跑。该四 seed 批次属于开发检验，FQL 硬裁剪
和 learner/预训练差异仍在，不能将跨算法结果称为历史 SAC 收益的单因素因果解释。
新工具通过 22 项曲线、计数、动作流、序列化、配对及失败处理测试，并成功独立复核
既有四个真实工程 checkpoint/日志，没有追加训练。
