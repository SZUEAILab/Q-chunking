# FQL 参考学习器的来源与验证范围

[文档入口](../index.md) · [当前实验](../current/experiments.md)

跨算法检验需要先固定 FQL 配方。项目现有 `agents/acfql.py` 是带 action chunking 的
QC/FQL fork；它的 H1 更新仍与官方 FQL 存在差异，不能仅凭 H1 就称为官方参考实现。
新增 [`agents/fql_reference.py`](../../agents/fql_reference.py) 独立保留固定官方版本的
状态观测、单步 native learner；尚无正式训练入口或 posthoc 跨算法性能结果。

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

当前拒绝 adapter、projection、integrated DS、H>1、像素 encoder、Fourier 特征、
weight decay、best-of-n 和非两个 Q 的设置。后续 posthoc 迁移必须单独核验真实执行
mapper、replay 编码及 actor/target 查询；不能把当前 native learner 数值通过描述为
已完成 posthoc 迁移。FQL 的硬 clip 输出与原 SAC tanh 输出也必须分别标记。

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

本验证仅覆盖 learner；真实环境工程验收、数据/评估接线、posthoc 适配、正式矩阵与性能
仍待完成。现有 action-adapter 云端三份发布包与长程矩阵不改动，也没有新增算力实例。
