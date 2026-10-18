# RLPD 基线的来源、配方与验证范围

[文档入口](../index.md) · [当前实验](../current/experiments.md)

Action-adapter 的首轮 72 运行使用项目 ACRLPD，保持原方法与 critic 消融可配对。
另提供独立的原论文 RLPD learner 配方入口，用于补充外部算法比较。两者的差异不能归因于
动作表示，外部基线成绩也不能替代完整 adapter 的同配方对照。

## 三种来源分别标注

| 项目 | 本项目 ACRLPD 冻结筛选 | 原论文 RLPD 配方移植 | FQL 作者的 SAC/RLPD 实现 |
|---|---|---|---|
| 来源 | `agents/acrlpd.py`，发布包 `ccab449` | [ikostrikov/rlpd](https://github.com/ikostrikov/rlpd/tree/c90fd4baf28c9c9ef40a81460a2e395092844f88) | [seohongpark/fql](https://github.com/seohongpark/fql/tree/e8cd16eb490332924dfa2492097219f181765933) |
| 默认隐藏层 | 512×4 | 256×2 | 512×4 |
| Q 网络 / target 聚合 | 10 个取均值 | 10 个中无放回随机抽 2 个取最小值 | 2 个取最小值 |
| 熵进入 Bellman target | 否 | 默认是，可显式关闭 | 默认否 |
| actor 使用的 Q | 全部 Q 的均值 | 全部 Q 的均值 | 全部 Q 的均值 |
| EMA 使用哪个 critic | 此次梯度更新前的参数 | 此次梯度更新后的参数 | 此次梯度更新前的参数 |
| UTD 内更新顺序 | 每个 minibatch 同时更新 critic、actor、温度 | 每个 minibatch 更新 critic；最后一个 minibatch 更新一次 actor、温度 | 原入口一次同时更新 critic、actor、温度 |
| 熵目标默认值 | −dim(A)/2 | −dim(A)/2 | −dim(A)/2 |

FQL 作者实现是另一种公开 RLPD 基线，不能因它没有采用原论文默认的随机 Q 子集就称其
实现错误。本项目也不能把自己的 10-Q mean 配方标成任一上游的逐行复现。
原论文 README 的 locomotion、antmaze、Adroit 配方不同；目前移植的默认配置对应通用
`configs/rlpd_config.py`。UTD=20 来自其公开实验命令，不是 learner 类的内置默认值。
原仓库没有 OGBench Cube 的已调参标准结果。

## 独立入口与适配范围

[`agents/rlpd_reference.py`](../../agents/rlpd_reference.py) 保留原版 actor head 名称和初始化、
TanhNormal 密度、RNG 拆分、独立 Adam、目标 Q 子集和更新顺序。
使用原版 MIT 许可，见[许可文本](rlpd-upstream-LICENSE.txt)。

[`main_rlpd_reference.py`](../../main_rlpd_reference.py) 复用项目的 OGBench 数据、50/50
prior/online 混合、日志、保存与固定评估 RNG。默认 H1、UTD=20；仅支持 native 动作，
不支持 adapter、投影、阶段标签干预或其他 actor loss。配置中可显式覆盖网络、Q 子集、
backup entropy、学习率、UTD；科学批次必须把这些值登记，不根据中途成绩改配方。

交互动作的 RNG 由本项目 driver 外置管理；原仓库 driver 通过 agent 自身 RNG 采样。
评估也是本项目的独立固定 seed 随机策略评估，而非原版 driver 的 mode 评估。
这些适配使其能与 adapter 在同一 OGBench 协议下比较，但不构成原始 D4RL 实验的完整复现。
该移植尚不提供 FQL 作者 SAC/RLPD 配方的独立实现。

```bash
# 工程示例；正式比较先冻结批次、源码、数据、矩阵和分析。
python main_rlpd_reference.py \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --agent=agents/rlpd_reference.py --horizon_length=1 --utd_ratio=20 \
  --seed=10 --online_steps=1000000 --start_training=5000 \
  --eval_at_start=True --eval_seed=20261019 --eval_episodes=50 \
  --eval_interval=100000 --save_interval=1000000
```

上述示例没有自动加入已经冻结的 72 运行。UTD=20 与 UTD=1 比较时，同时披露环境步数、
critic/actor/温度实际更新数、运行时长和硬件。原版 Flax TrainState 计数从 0 开始；
项目自定义 TrainState 从 1 开始，不能直接比较未校正的计数器。

## 数值对照

[`research.check_rlpd_reference`](../../research/check_rlpd_reference.py) 从固定 commit 的
干净上游 checkout 加载原始 learner、分布、MLP、ensemble 与 value 源码。
只绕过会预加载像素/D4RL 的 package 初始化，并把现代 Flax 初始化返回的 dict 转回
原版使用的 FrozenDict。上游算法文件不改写；报告记录每个读取文件的 SHA-256。

```bash
JAX_PLATFORMS=cpu python -m research.check_rlpd_reference \
  --upstream-root=/path/to/ikostrikov-rlpd \
  --output=/path/to/rlpd-reference-equivalence.json
```

固定 synthetic batch 覆盖非零 reward、终止和未终止 mask；每类连续更新三次。
覆盖 UTD=1/20、backup entropy 开/关、Q 子集 1/2/全部，以及完整 256×256 网络。
当前六类对照的参数、优化器状态、计数器、RNG 和 loss 全部一致，最大状态绝对差为 0。
另有 10 项测试覆盖更新次数、EMA、sequence bridge、外置 RNG、checkpoint 恢复及非法配置。

报告保存在数据盘 `reports/action_adapter_validation_20261019/rlpd-reference-equivalence.json`。
数值一致只证明已覆盖设置下的 learner 移植；成功率、调参公平性与长程稳定性须由正式实验验证。

真实环境工程验收已完成 64 步、UTD=20，记录了 660 次 critic 与各 33 次 actor/温度更新，
step 0/64 评估、可恢复 checkpoint 与有限动作；验收过程和解释边界见[当前结果](../current/results.md)。
