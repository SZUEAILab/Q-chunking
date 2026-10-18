# 旧研究计划、疑问与检查记录

[历史归档索引](index.md) · [文档入口](../index.md)

> 历史归档：各节保留原文的日期、实验身份和结论范围。旧文中的“当前”“推荐”“待做”不代表今天的执行状态；日常规则以 [实验](../current/experiments.md)、[运行与存储](../current/runbook.md) 为准。

## 内容

- [Direction–Speed 研究计划](#research-program)
- [Direction-Speed 疑难点跟踪](#ds-open-questions)
- [DS (Direction-Speed) 实验验证清单](#ds-exp-checklist)
- [实验与证据协议](#experiment-protocol)

<a id="research-program"></a>
## Direction–Speed 研究计划

原文位置：`docs/research/research_program.md`。正文按原记录保留，原文件名用于历史追溯。

更新日期：2026-10-13。

<a id="research-program--目标"></a>
### 目标

研究机器人强化学习中动作方向与幅值（速度）是否具有不同的探索几何和有效时间尺度，并将
这种结构用于提高样本效率。OGBench 是主证据场，RLPD 是第一算法；方法冻结后再迁移到 FQL，
最后在 Robomimic 上做跨域确认。

当前世界量是 OGBench 归一化动作增量，不应直接称为真实物理速度。论文应使用
“direction–magnitude”或“direction–speed action coordinates”，并在环境定义处说明单位。

<a id="research-program--核心假设"></a>
### 核心假设

最初的简单假设是：方向应比幅值保持更久。现有实验否定了其固定 `K=5` 版本。修订后的假设是：

> 方向与幅值需要不同但状态相关的噪声和时间尺度。方向只应在策略方向不确定性低、且没有
> 接触/目标切换事件时短暂承诺；幅值保持响应。固定重复完整动作、固定保持方向和完全逐步
> 反应分别是这一自适应机制的边界对照。

这个命题包含三个必须分开的干预轴：

| 轴 | 问题 | 主要对照 |
|---|---|---|
| Action representation | Cartesian 与 direction–speed 坐标是否改变学习？ | 原生 Cartesian、DS 重参数化 |
| Noise shaping | 轴对齐噪声与切空间/径向噪声是否不同？ | 对齐执行动作矩、协方差和范数的控制 |
| Temporal shaping | 哪些动作分量应保持多久？ | reactive、方向保持、平移保持、完整动作保持、自适应保持 |

不能把三个轴同时改变后得到的差异全部归因于“方向–速度分解”。

<a id="research-program--候选主方法"></a>
### 候选主方法

候选方法由四个部件组成：

1. 将平移动作写成 `a_xyz = s d`，其中 `d` 是单位方向、`s` 是非负幅值；
2. 在球面切空间中产生方向噪声，在径向坐标中独立产生幅值噪声；
3. 对执行后动作的范数、协方差、饱和率和局部角扩散做显式诊断；
4. 使用不确定性和接触事件控制方向承诺时长，幅值、旋转和夹爪保持响应。

优先测试的组件消融：

- Cartesian native；
- Cartesian + L2 projection / moment-matched noise；
- DS representation only；
- DS + geometry-aware noise；
- DS + adaptive temporal control；
- 完整方法；
- 现有 fixed-D、translation-repeat、full-repeat 时间对照。

<a id="research-program--研究阶段"></a>
### 研究阶段

<a id="research-program--阶段-0闭合既有时间实验"></a>
#### 阶段 0：闭合既有时间实验

补齐 frozen fourth8 的八个原定案例，形成 original40 的 40/40。保持原 seed、预算、评估轴和
运行身份；当前单卡汇总不能替代缺失案例。

<a id="research-program--阶段-1候选方法筛选"></a>
#### 阶段 1：候选方法筛选

在 Cube Double 和 Scene 上用预先固定的两个 seed、200k 预算筛选四个候选。筛选只决定是否
进入正式实验，不用于最终显著性结论。候选至少应在两个任务都不明显退化，并在一个任务上
带来有意义的早期 AUC 改善。

<a id="research-program--阶段-2rlpd-正式消融"></a>
#### 阶段 2：RLPD 正式消融

使用五个 matched seeds、500k 预算运行通过筛选的方法和关键组件消融。主指标是按原始环境步
积分的 success AUC；同时报告最终成功率、达到阈值所需步数、动作几何和资源开销。

<a id="research-program--阶段-3ogbench-任务扩展"></a>
#### 阶段 3：OGBench 任务扩展

在看不到结果前冻结至少四个 held-out manipulation 任务。只比较 native、最强固定时间基线和
完整方法，避免在 held-out 任务上再次调方法。

<a id="research-program--阶段-4跨算法与跨域"></a>
#### 阶段 4：跨算法与跨域

- FQL：先验证实际执行的一步 student，而不是只改 teacher；纯离线 FQL 不支持“在线探索效率”
  的措辞。
- Robomimic：在 Lift、Can、Square 上验证迁移；先只分解平移，旋转和夹爪保持独立，不能把米
  和弧度放进未经限定的统一范数。

<a id="research-program--决策门"></a>
### 决策门

完整方法进入扩展阶段前应满足：

- 相对 native 的 matched-seed 早期 AUC 效果方向稳定；
- 两个开发任务不存在一个提高、另一个严重退化；
- 最终成功率下降不超过预先规定的容忍度；
- 改善不是由动作幅值更小、clipping 更少或不同初始化单独解释；
- 无 nonfinite、越界、缺失评估或生命周期异常；
- 所有失败 seed 和负结果保留。

如果没有候选通过，停止扩展到 FQL/Robomimic，先修改机制。GPU 数量不能替代清晰的反事实。

<a id="research-program--论文贡献边界"></a>
### 论文贡献边界

当前可争取的贡献不是“球坐标天然优于 Cartesian”，而是：

1. 把表示、瞬时噪声几何和时间相关性拆成可辨识干预；
2. 给出方向/幅值噪声和时间承诺的匹配控制；
3. 证明固定方向保持为何失败，并提出状态相关机制；
4. 在 RLPD、FQL 和跨域机器人任务中验证样本效率与成功率，而不是只展示动作几何。

是否达到顶会标准取决于多任务、多 seed 的正向性能证据；工程门通过或漂亮的合成机制图本身
不足以支撑投稿结论。

<a id="ds-open-questions"></a>
## Direction-Speed 疑难点跟踪

原文位置：`docs/research/ds_open_questions.md`。正文按原记录保留，原文件名用于历史追溯。

本文档整理 DS 实现讨论中的关键疑问，目的是后续按条验证、补实验或改代码。主线设计见 [`approach.md`](../reference/method.md)。

<a id="ds-open-questions--0-当前结论"></a>
### 0. 当前结论

当前保留四种 DS 实现：

| 实现 | 入口 | 当前定位 |
|--|--|--|
| action-adapter D+1 | `--ds_mode=adapter` | 非可逆表示消融；FQL 可用；RLPD 只作 approximate ablation |
| Integrated DS / spherical | `--ds_mode=integrated --ds_integrated_type=spherical` | Jacobian-corrected RLPD/SAC 对照 |
| Integrated DS / stereographic | `--ds_mode=integrated --ds_integrated_type=stereographic` | Jacobian-corrected RLPD/SAC 推荐主线 |
| Integrated DS / tangent | `--ds_mode=integrated --ds_integrated_type=tangent` | 均值方向切平面采样；角距离探索旋转不变 |

当前推荐主实验：

```bash
--ds_mode=integrated --ds_integrated_type=stereographic --horizon_length=1 --agent.action_chunking=False
--ds_mode=integrated --ds_integrated_type=stereographic --horizon_length=5 --agent.action_chunking=True
--ds_mode=integrated --ds_integrated_type=spherical --horizon_length=1 --agent.action_chunking=False
--ds_mode=integrated --ds_integrated_type=spherical --horizon_length=5 --agent.action_chunking=True
```

<a id="ds-open-questions--1-horizon_length-和-action_chunking"></a>
### 1. `horizon_length` 和 `action_chunking`

**疑问。** `horizon_length=5` 和 `action_chunking=True/False` 分别控制什么？

**当前结论。**

- `horizon_length=H` 始终控制 sequence 长度和 H-step TD target。
- `action_chunking=True` 时，actor/critic action 维度从 `D` 扩成 `HD`。
- `action_chunking=False` 且 `H>1` 时，只做 n-step TD，不做动作块。

| 设置 | actor 输出 | critic action | 执行 |
|--|--:|--:|--|
| `H=1` | `D` | `D` | 每步采样 |
| `H=5, action_chunking=False` | `D` | `D` | 每步采样，5-step TD |
| `H=5, action_chunking=True` | `HD` | `HD` | 一次采样，队列执行 H 步 |

**后续检查。**

- [ ] 在启动日志里打印 `raw_action_dim`、`full_action_dim`、`horizon_length`、`action_chunking`。
- [ ] 在 mini test 中 assert `D=5,H=5,action_chunking=True` 时 actor sample shape 为 `25`。

<a id="ds-open-questions--2-ds-和-action-chunking-形状兼容性"></a>
### 2. DS 和 action chunking 形状兼容性

**疑问。** DS 是否只输出单步 `D`，导致 critic 期待 `HD` 时形状不匹配？

**当前结论。** 当前代码已经支持 chunk。`action_chunking=True` 时，DS action groups 会按 horizon 复制。对于 `D=5,H=5`：

```text
每步: direction-speed group + yaw scalar + gripper scalar -> 5D
5 步: 重复 5 份 -> 25D
```

因此 spherical/stereographic DS actor 输出 `HD`，critic 输入也为 `HD`。

**后续检查。**

- [ ] 为 `build_action_groups` 加单元测试或 smoke test。
- [ ] 文档中明确 “DS chunk 不是单个 head 复用，而是 H 个独立 group”。

<a id="ds-open-questions--3-reshapehd---h-x-d-和-xla-精度"></a>
### 3. `reshape(HD -> H x D)` 和 XLA 精度

**疑问。** `action_chunking=True` 时把 `25D` reshape 成 `(5,5)` 是否会切断 XLA 数值链路，导致 float32 掉到 float16？

**当前结论。** `reshape` 不改变 dtype，也不是 `stop_gradient`，不会单独导致精度降级。它只是 view/shape transformation。真正需要关注的是：

- spherical 极点附近 `sin(theta)` 小，Jacobian 可能很负。
- sigmoid 饱和会导致梯度变小。
- mixed precision 或外部框架配置才可能改变 dtype。

**后续检查。**

- [ ] 在 debug 日志或测试中检查 actor sample dtype。
- [ ] 记录 spherical/stereographic 的 `log_prob`、entropy、alpha、Q 范围，排查 NaN/Inf。

<a id="ds-open-questions--4-log_prob-的对象到底是谁"></a>
### 4. `log_prob` 的对象到底是谁

**疑问。** 为什么 critic 使用变换后的 action 时，`log_prob` 也必须是变换后 action 的概率？

**当前结论。** SAC/RLPD actor loss 同时包含：

```text
alpha * log pi(a | s) - Q(s, a)
```

这两个项必须对应同一个动作变量 `a`。如果 critic/env 使用：

```text
a = f(raw)
```

则 entropy 项也应使用：

```text
log pi_action(a | s)
```

而不是：

```text
log pi_raw(raw | s)
```

对于可逆 bijector：

```text
log pi_action(a | s)
= log pi_raw(f^{-1}(a) | s) - log |det J_f(f^{-1}(a))|
```

**后续检查。**

- [ ] 在文档中区分 “raw latent density” 和 “env action density”。
- [ ] 在实验图注中标明是否为 Jacobian-corrected `log_prob`。

<a id="ds-open-questions--5-为什么-bijector-放在-actor-内部"></a>
### 5. 为什么 bijector 放在 actor 内部

**疑问。** MLP 输出仍然是 `theta_raw, phi_raw, speed_raw`，为什么不在 actor 外部 compose 成 xyz？

**当前结论。** MLP 只输出 raw Gaussian 的 `mean/log_std`。在 `IntegratedDirectionSpeedNormal` 内部：

```text
base_dist = Normal(mean, std)
dist = TransformedDistribution(base_dist, DirectionSpeedBijector)
```

因此：

- `dist.sample()` 返回 env action。
- `dist.log_prob(action)` 自动 inverse 回 raw 并减去 `log_det`。

如果把 compose 放到外部，只能解决 “动作能执行”，不能让 `log_prob` 自动变成 env action density。

**后续检查。**

- [ ] 在代码注释中明确 `sample_actions()` 对 spherical/stereographic 返回的是 env action。
- [ ] 在文档中标出 MLP 输出 raw 参数，distribution sample 输出 env action。

<a id="ds-open-questions--6-action-adapter-d1-的多对一问题"></a>
### 6. action-adapter D+1 的多对一问题

**疑问。** action-adapter 中 `u` 先 L2 normalize，从 `D+1` 压到 `D`，影响是什么？

**当前结论。** 这是多对一映射。例如：

```text
u, 2u, 100u
```

normalize 后方向相同，compose 出来的环境动作也相同。真实 `p_action(a)` 需要沿被丢弃的径向自由度积分，而不是简单取某个 `log p_raw(raw)`。因此 action-adapter 没有普通 bijector 的 Jacobian 修正。

影响：

- actor 会优化环境不可见的径向自由度。
- critic/BC 若吃 raw 分解动作，需要学习等价 raw 的相同 Q。
- RLPD/SAC entropy 语义不严格。

**后续检查。**

- [ ] action-adapter RLPD 结果统一标为 approximate ablation。
- [ ] 不把 action-adapter 曲线和 spherical/stereographic 作为同等概率模型比较。

<a id="ds-open-questions--7-critic-如果吃-raw-action-是否还需要-bijector"></a>
### 7. critic 如果吃 raw action 是否还需要 bijector

**疑问。** 如果 critic 对 raw action 打分，是否三种 DS 都可以用外部 decompose/compose，且不需要 bijector？

**当前结论。** 可以让 actor loss 在 raw 空间闭合：

```text
alpha * log pi_raw(raw | s) - Q_raw(s, raw)
```

这时不强制需要 bijector。但环境 reward 仍来自：

```text
env.step(compose(raw))
```

所以 critic 学的是：

```text
Q_raw(s, raw) = Q_env(s, compose(raw))
```

对 action-adapter 多对一 raw，Q 在等价径向维度上应相同，函数逼近更难，actor 也可能 exploit critic 在无效维度上的误差。

**后续检查。**

- [ ] 如果做 raw-critic 消融，需要单独命名为 raw-action-space MDP ablation。
- [ ] 比较 raw-critic 与 env-action-critic 时，不把它们称为同一个 SAC/RLPD 概率模型。

<a id="ds-open-questions--8-三种-ds-的实验定位"></a>
### 8. 三种 DS 的实验定位

**疑问。** 三种 DS 如何公平比较？

**当前结论。**

- `adapter`：验证 D+1 表示是否有经验收益，尤其 FQL。
- `spherical`：验证 “exact bijector + 球坐标” 是否能工作。
- `stereographic`：验证 “exact bijector + 更平滑方向坐标” 是否优于 spherical。
- `tangent`：验证固定角方差是否消除方向相关的探索尺度。

最低限度矩阵：

| Agent | H | chunk | ds_mode | ds_integrated_type | 目的 |
|--|--:|--:|--|--|--|
| RLPD | 1 | false | none | — | baseline |
| RLPD | 1 | false | integrated | spherical | exact DS 对照 |
| RLPD | 1 | false | integrated | stereographic | exact DS 主线 |
| RLPD-AC | 5 | true | none | — | chunk baseline |
| RLPD-AC | 5 | true | integrated | spherical | exact DS chunk 对照 |
| RLPD-AC | 5 | true | integrated | stereographic | exact DS chunk 主线 |
| FQL | 5 | true | adapter | — | 表示消融 |

**后续检查。**

- [ ] 每张实验图标明 `ds_mode`、H、chunk、是否 Jacobian-corrected。
- [ ] 不再使用未标注 provenance 的旧图作为主结论。

<a id="ds-open-questions--9-flag-整理"></a>
### 9. flag 整理

**疑问。** `use_integrated_ds`、`ds_max_speed`、`ds_integrated_type` 是否都有用？

**当前结论。**

- 用户主入口是 `--ds_mode`；Integrated DS 的方向分布由 `--ds_integrated_type` 选择。
- `use_integrated_ds` 仍是 agent 内部开关。
- `ds_integrated_type` 决定 `spherical`、`stereographic` 或 `tangent`。
- `tangent` 使用均值方向处的切平面 wrapped Gaussian，并计算 exponential-map Jacobian。
- `ds_max_speed` 只缩放 Integrated DS 的 sphere bound；cube bound 始终覆盖 normalized cube，不使用该缩放。

**后续检查。**

- [ ] 文档命令使用 `--ds_mode`，Integrated DS 命令同时显式标注 `--ds_integrated_type`。

<a id="ds-open-questions--10-外部噪声"></a>
### 10. 外部噪声

**疑问。** action-adapter 外部噪声是否还有用？

**当前结论。** 已移除外部加噪。探索应来自 actor 本身的随机策略。外部噪声会引入额外不可控分布，混淆 DS 表示和探索策略两个因素。

**后续检查。**

- [ ] 确认代码里不再有 `dir_noise`、`speed_noise`、`other_noise`。
- [ ] 如果后续需要探索消融，应作为单独实验变量，不和 DS 主实验混在一起。

<a id="ds-open-questions--11-数值稳定性清单"></a>
### 11. 数值稳定性清单

**疑问。** 如何避免 JAX 梯度消失/爆炸、NaN/Inf、精度损失？

**当前措施。**

- spherical/stereographic 使用 bounded sigmoid，避免精确到边界。
- inverse/log/log-det 路径做 clip。
- `reshape` 不改变 dtype。
- chunk DS 按 group 复制，避免单步输出和 critic 维度不匹配。

**后续检查。**

- [ ] smoke test：`none/spherical/stereographic` x `H=1/H=5 chunk` 一步 update 全 finite。
- [ ] 训练日志：`actor_loss`、`entropy`、`alpha`、`log_prob`、`q_min/q_max`。
- [ ] 如果 spherical 不稳定，优先比较 stereographic，而不是回退到 action-adapter 主线。

<a id="ds-exp-checklist"></a>
## DS (Direction-Speed) 实验验证清单

原文位置：`docs/experiments/ds_exp_checklist.md`。正文按原记录保留，原文件名用于历史追溯。

> ✅ = 已完成 | 🔄 = 运行中 | ⬜ = 待做 | ❌ = 不可行

表中的 `adapter` 指 Action-adapter DS；`spherical`、`stereographic`、`tangent` 是 Integrated DS 的方向分布标签。归档实验生成时使用的原始 flags 保留在 `docs/data/`，不回写历史元数据。

<a id="ds-exp-checklist--1-rlpdsac--历史纯在线结果-main_onlinepy"></a>
### 1. RLPD/SAC — 历史纯在线结果 (main_online.py)

以下保留重构前实验结果，仅作参考，不视为当前 Integrated/Adapter 实现的完成项。新实现统一按第 6 节重跑。env: cube-triple-task2, seed: 0-3。

<a id="ds-exp-checklist--11-h1-无-chunk--推荐配置"></a>
#### 1.1 H=1 (无 chunk) — 推荐配置

| #     | 状态 | method       | 步数 | 说明                      |
| ----- | :--: | ------------- | ---- | ------------------------- |
| 1-4   |  ✅  | none          | 1M   | RLPD baseline             |
| 5-8   |  ✅  | adapter       | 1M   | DS D+1                    |
| 9-12  |  ✅  | stereographic | 1M   | Integrated DS (球极投影)    |
| 13-16 |  ✅  | spherical     | 1M   | Integrated DS (球坐标)      |

> **结果** (4 seed, 50 eval): stereo 91% ≈ adapter 89% > spherical 64% > baseline 59%
> **升级** (10 seed, 100 eval): adapter 91.0% > stereo 61.6% > spherical 45.8% > baseline 37.3%

<a id="ds-exp-checklist--12-h5-chunk--探索性配置"></a>
#### 1.2 H=5 (chunk) — 探索性配置

| #     | 状态 | method       | 步数 | 说明             |
| ----- | :--: | ------------- | ---- | ---------------- |
| 17-20 |  ✅  | none          | 1M   | RLPD-AC baseline |
| 21-24 |  ✅  | adapter       | 1M   | DS D+1           |
| 25-28 |  ✅  | stereographic | 1M   | Integrated DS      |
| 29-32 |  ✅  | spherical     | 1M   | Integrated DS      |

> **结果** (4 seed, 50 eval): adapter 32% > stereo 19% > spherical 7% > baseline 3%
> **升级** (10 seed, 100 eval): stereo 23.5% ≈ adapter 22.0% > spherical 9.4% > baseline 2.0%

---

<a id="ds-exp-checklist--2-fql--离线在线-mainpy"></a>
### 2. FQL — 离线→在线 (main.py)

验证 DS 在 flow-based 算法上的表示消融效果。FQL 仅支持 adapter。env: cube-triple-task2

<a id="ds-exp-checklist--21-h5-chunk--主实验"></a>
#### 2.1 H=5 (chunk) — 主实验

| #     | 状态 | seed | method | 步数  | 结果                                  |
| ----- | :--: | :--: | ------- | ----- | ------------------------------------- |
| 33-36 |  ✅  | 0-3 | none    | 1M+1M | 96%/84%/90% =**90%**            |
| 37-40 |  ✅  | 0-3 | adapter | 1M+1M | 100%/98%/4% =**67%** (中位 98%) |

<a id="ds-exp-checklist--22-h1--已完成--fql-h1-不适合此任务"></a>
#### 2.2 H=1 — 已完成 (⚠ FQL H=1 不适合此任务)

| #     | 状态 | seed | method | 步数  |               结果               |
| ----- | :--: | :--: | ------- | ----- | :------------------------------: |
| 41-44 |  ✅  | 0-3 | none    | 1M+1M | 0% / 0% / 0% / 0% =**0%** |
| 45-48 |  ✅  | 0-3 | adapter | 1M+1M | 6% / 8% / 8% / 6%* =**7%** |

> * H=1 + chunk=True → 实际等价于无 chunk (1×5=5D)。*seed3 未跑，用 seed0 值填充。

<a id="ds-exp-checklist--23-h5--best-of-n-qc-模式--待做"></a>
#### 2.3 H=5 + best-of-n (QC 模式) — 待做

| #     | 状态 | seed | method | 步数  | 关键参数                             |
| ----- | :--: | :--: | ------- | ----- | ------------------------------------ |
| 49-52 |  ⬜  | 0-3 | none    | 1M+1M | actor_type=best-of-n, num_samples=32 |
| 53-56 |  ⬜  | 0-3 | adapter | 1M+1M | actor_type=best-of-n, num_samples=32 |

<a id="ds-exp-checklist--24--旧-fql-h5-错误-5d-归一化--仅供参考"></a>
#### 2.4 ⚠ 旧 FQL H=5 (错误 5D 归一化) — 仅供参考

| #  | 状态 | seed | ds           | 步数  |       结果       |
| -- | :--: | :--: | ------------ | ----- | :---------------: |
| — |  ⚠  |  0  | adapter (5D) | 1M+1M | raw 88% vs DS 98% |

---

<a id="ds-exp-checklist--3-ogbench-多任务验证"></a>
### 3. OGBench 多任务验证

每个任务 4 DS modes × H1+H5 × 10 seeds, 100 eval_episodes。agent: acrlpd, 1M 纯在线。task1/2 H1+H5 已全部升级为 10 seeds 100 eval。

**cube-triple 5 个 fixed task**（3 cubes，200 步/集）：

| Task | 名称 | 操作 | 难度 |
|:---:|------|------|:---:|
| **1** | single_pnp | 仅移动 1 个 cube，其余 2 个原位不动 | ⭐ |
| **2** | triple_pnp | 3 个 cube 并行平移（列→列） | ⭐⭐ |
| **3** | unstack | 从垂直堆叠中逐一抓取分散放置 | ⭐⭐⭐⭐ |
| **4** | cycle | 3 个 cube 顺时针循环换位 | ⭐⭐⭐ |
| **5** | stack | 3 个散 cube 精确垂直堆叠 | ⭐⭐⭐⭐⭐ |

<a id="ds-exp-checklist--31-cube-triple-task1"></a>
#### 3.1 cube-triple task1

> 📌 单 cube 移动，其余原位不动。预期最简单。

| #       | 状态 | env               | H | method       | seeds | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | :---: | ---- |
| 57-66   |  ✅  | cube-triple-task1 | 1 | none          | 10 | 1M   |
| 67-76   |  ✅  | cube-triple-task1 | 1 | adapter       | 10 | 1M   |
| 77-86   |  ✅  | cube-triple-task1 | 1 | stereographic | 10 | 1M   |
| 87-96   |  ✅  | cube-triple-task1 | 1 | spherical     | 10 | 1M   |
| 97-106  |  ✅  | cube-triple-task1 | 5 | none          | 10 | 1M   |
| 107-116 |  ✅  | cube-triple-task1 | 5 | stereographic | 10 | 1M   |
| 117-126 |  ✅  | cube-triple-task1 | 5 | spherical     | 10 | 1M   |
| 127-136 |  ✅  | cube-triple-task1 | 5 | adapter       | 10 | 1M   |

> H1 已升级为 10 seed 100 eval（原 4 seed 50 eval）。结果：H1 adapter 99.7% > baseline 89.4% > stereo 83.4% > spherical 58.3%；H5 adapter 84.9% (旧 10s)。

<a id="ds-exp-checklist--32-cube-triple-task3"></a>
#### 3.2 cube-triple task3

> 📌 从堆叠抓取分散（task5 的逆）。初始 3 cube 垂直叠放，需逐一抓取放至不同位置。MuJoCo 碰撞 bug 高发区（已修复）。

| #       | 状态 | env               | H | method       | seeds | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | :---: | ---- |
| 137-146 |  ✅  | cube-triple-task3 | 1 | none          | 10 | 1M   |
| 147-156 |  ✅  | cube-triple-task3 | 1 | adapter       | 10 | 1M   |
| 157-166 |  ✅  | cube-triple-task3 | 1 | stereographic | 10 | 1M   |
| 167-176 |  ✅  | cube-triple-task3 | 1 | spherical     | 10 | 1M   |
| 177-186 |  ✅  | cube-triple-task3 | 5 | none          | 10 | 1M   |
| 187-196 |  ✅  | cube-triple-task3 | 5 | stereographic | 10 | 1M   |
| 197-206 |  ✅  | cube-triple-task3 | 5 | spherical     | 10 | 1M   |
| 207-216 |  ✅  | cube-triple-task3 | 5 | adapter       | 10 | 1M   |

> ~~task3 有 MuJoCo 物理 bug（`mj_narrowphase` 碰撞超限），4 seed 反复崩溃。~~ **已解决**，10 seeds 全部完成。结果：H1 adapter 22.4%, H5 adapter 27.5%，仅 adapter 有效。

<a id="ds-exp-checklist--33-cube-triple-task4"></a>
#### 3.3 cube-triple task4

> 📌 顺时针循环换位，无堆叠。3 cube 各挪到相邻位置。

| #       | 状态 | env               | H | method       | seeds | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | :---: | ---- |
| 217-226 |  ✅  | cube-triple-task4 | 1 | none          | 10 | 1M   |
| 227-236 |  ✅  | cube-triple-task4 | 1 | adapter       | 10 | 1M   |
| 237-246 |  ✅  | cube-triple-task4 | 1 | stereographic | 10 | 1M   |
| 247-256 |  ✅  | cube-triple-task4 | 1 | spherical     | 10 | 1M   |
| 257-266 |  ✅  | cube-triple-task4 | 5 | none          | 10 | 1M   |
| 267-276 |  ✅  | cube-triple-task4 | 5 | stereographic | 10 | 1M   |
| 277-286 |  ✅  | cube-triple-task4 | 5 | spherical     | 10 | 1M   |
| 287-296 |  ✅  | cube-triple-task4 | 5 | adapter       | 10 | 1M   |

> 结果：H1 stereo 62.4% ≈ adapter 61.4% > spherical 55.6% > baseline 44.4%；H5 adapter 14.6%。

<a id="ds-exp-checklist--34-cube-triple-task5"></a>
#### 3.4 cube-triple task5

> 📌 精确垂直堆叠（当前最难）。3 cube 从分散位置拾取并叠放至同一位置 (z=0.02, 0.06, 0.10)。1M 步全部 0%。

| #       | 状态 | env               | H | method       | seeds | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | :---: | ---- |
| 297-306 |  ✅  | cube-triple-task5 | 1 | none          | 10 | 1M   |
| 307-316 |  ✅  | cube-triple-task5 | 1 | adapter       | 10 | 1M   |
| 317-326 |  ✅  | cube-triple-task5 | 1 | stereographic | 10 | 1M   |
| 327-336 |  ✅  | cube-triple-task5 | 1 | spherical     | 10 | 1M   |
| 337-346 |  ✅  | cube-triple-task5 | 5 | none          | 10 | 1M   |
| 347-356 |  ✅  | cube-triple-task5 | 5 | stereographic | 10 | 1M   |
| 357-366 |  ✅  | cube-triple-task5 | 5 | spherical     | 10 | 1M   |
| 367-376 |  ✅  | cube-triple-task5 | 5 | adapter       | 10 | 1M   |

> 最难任务，全部 80 runs 成功率 0%。1M 步不够，需更长训练或 offline pretrain。

<a id="ds-exp-checklist--35-cube-double-待做"></a>
#### 3.5 cube-double (待做)

| #       | 状态 | env               | H | method       | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | ---- |
| 377-380 |  ⬜  | cube-double-task2 | 1 | none          | 1M   |
| 381-384 |  ⬜  | cube-double-task2 | 1 | adapter       | 1M   |
| 385-388 |  ⬜  | cube-double-task2 | 1 | stereographic | 1M   |
| 389-392 |  ⬜  | cube-double-task2 | 1 | spherical     | 1M   |
| 393-396 |  ⬜  | cube-double-task2 | 5 | none          | 1M   |
| 397-400 |  ⬜  | cube-double-task2 | 5 | adapter       | 1M   |
| 401-404 |  ⬜  | cube-double-task2 | 5 | stereographic | 1M   |
| 405-408 |  ⬜  | cube-double-task2 | 5 | spherical     | 1M   |

<a id="ds-exp-checklist--36-cube-single-可选"></a>
#### 3.6 cube-single (可选)

| #       | 状态 | env               | H | method       | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | ---- |
| 409-412 |  ⬜  | cube-single-task2 | 1 | none          | 1M   |
| 413-416 |  ⬜  | cube-single-task2 | 1 | stereographic | 1M   |
| 417-420 |  ⬜  | cube-single-task2 | 5 | none          | 1M   |
| 421-424 |  ⬜  | cube-single-task2 | 5 | stereographic | 1M   |

> cube-single 最简单，先跑 stereo vs none，如有显著差异再加 adapter/spherical。

<a id="ds-exp-checklist--37-cube-quadruple-可选"></a>
#### 3.7 cube-quadruple (可选)

| #       | 状态 | env                  | H | method       | 步数 |
| ------- | :--: | -------------------- | :-: | ------------- | ---- |
| 425-428 |  ⬜  | cube-quadruple-task2 | 1 | none          | 1M   |
| 429-432 |  ⬜  | cube-quadruple-task2 | 1 | stereographic | 1M   |
| 433-436 |  ⬜  | cube-quadruple-task2 | 5 | none          | 1M   |
| 437-440 |  ⬜  | cube-quadruple-task2 | 5 | stereographic | 1M   |

> cube-quadruple 极难，先跑 stereo vs none，有信号再加满。

---

<a id="ds-exp-checklist--4-跨域验证-robomimic--7d-动作空间"></a>
### 4. 跨域验证 (RoboMimic — 7D 动作空间)

agent: acrlpd, 4 seeds × 4 DS modes, 1M 纯在线. DS 将位移 3D + 旋转 3D 分别分解.

<a id="ds-exp-checklist--41-lift-桌面升降--最简单"></a>
#### 4.1 lift (桌面升降 — 最简单)

| #       | 状态 | H | method       | 步数 |
| ------- | :--: | :-: | ------------- | ---- |
| 441-444 |  ⬜  | 1 | none          | 1M   |
| 445-448 |  ⬜  | 1 | adapter       | 1M   |
| 449-452 |  ⬜  | 1 | stereographic | 1M   |
| 453-456 |  ⬜  | 1 | spherical     | 1M   |
| 457-460 |  ⬜  | 5 | none          | 1M   |
| 461-464 |  ⬜  | 5 | stereographic | 1M   |
| 465-468 |  ⬜  | 5 | spherical     | 1M   |
| 469-472 |  ⬜  | 5 | adapter       | 1M   |

> 最简单 RoboMimic 任务，基线 ~80%+，先跑 lift 验证 DS 不损害简单任务。

<a id="ds-exp-checklist--42-square-方块装配--高精度"></a>
#### 4.2 square (方块装配 — 高精度)

| #       | 状态 | H | method       | 步数 |
| ------- | :--: | :-: | ------------- | ---- |
| 473-476 |  ⬜  | 1 | none          | 1M   |
| 477-480 |  ⬜  | 1 | adapter       | 1M   |
| 481-484 |  ⬜  | 1 | stereographic | 1M   |
| 485-488 |  ⬜  | 1 | spherical     | 1M   |
| 489-492 |  ⬜  | 5 | none          | 1M   |
| 493-496 |  ⬜  | 5 | stereographic | 1M   |
| 497-500 |  ⬜  | 5 | spherical     | 1M   |
| 501-504 |  ⬜  | 5 | adapter       | 1M   |

> 最难任务，论文中 DS 预期收益最大。

<a id="ds-exp-checklist--43-can-罐头搬运--中等难度"></a>
#### 4.3 can (罐头搬运 — 中等难度)

| #       | 状态 | H | method       | 步数 |
| ------- | :--: | :-: | ------------- | ---- |
| 505-508 |  ⬜  | 1 | none          | 1M   |
| 509-512 |  ⬜  | 1 | adapter       | 1M   |
| 513-516 |  ⬜  | 1 | stereographic | 1M   |
| 517-520 |  ⬜  | 1 | spherical     | 1M   |
| 521-524 |  ⬜  | 5 | none          | 1M   |
| 525-528 |  ⬜  | 5 | stereographic | 1M   |
| 529-532 |  ⬜  | 5 | spherical     | 1M   |
| 533-536 |  ⬜  | 5 | adapter       | 1M   |

---

<a id="ds-exp-checklist--5-libero--复杂操作基准"></a>
### 5. LIBERO — 复杂操作基准

验证 DS 在更真实的机械臂操作场景上的效果。LIBERO 基于 robosuite，7D 动作空间，120D 低维观测（proprio-state 50D + object-state 70D），场景含 5+ 个交互物体。

**数据集**：HuggingFace `yifengzhu-hf/LIBERO-datasets`（LIBERO 作者 Yifeng Zhu 的 HF 镜像），3 suites × 10 tasks = 30 tasks，每 task 50 human demos，HDF5 格式。总大小 ~5GB。

**安装**：详见 [datasets.md](../reference/datasets.md) LIBERO 章节。

**任务命名**：`libero_{suite}/{index}`，如 `libero_spatial/0`、`libero_object/2`。

| suite | 10 tasks | 描述 |
|-------|:---:|------|
| `libero_spatial` | ✅ | 空间关系推理（碗+盘子+烤杯等） |
| `libero_object` | ✅ | 不同物体操作（罐头、牛奶、黄油等） |
| `libero_goal` | ✅ | 目标条件任务（抽屉、炉灶、柜子等） |

<a id="ds-exp-checklist--51-libero-spatial-rlpd"></a>
#### 5.1 LIBERO Spatial (RLPD)

> 📌 RLPD 纯在线，H=1 + H=5，4 DS modes × 4 seeds × 10M。

| #       | 状态 | H | method       | seeds | 步数 |
| ------- | :--: | :-: | ------------- | :---: | ---- |
| 558-561 |  ⬜  | 1 | none          | 4 | 10M   |
| 562-565 |  ⬜  | 1 | adapter       | 4 | 10M   |
| 566-569 |  ⬜  | 1 | stereographic | 4 | 10M   |
| 570-573 |  ⬜  | 1 | spherical     | 4 | 10M   |
| 574-577 |  ⬜  | 5 | none          | 4 | 10M   |
| 578-581 |  ⬜  | 5 | adapter       | 4 | 10M   |
| 582-585 |  ⬜  | 5 | stereographic | 4 | 10M   |
| 586-589 |  ⬜  | 5 | spherical     | 4 | 10M   |

---

<a id="ds-exp-checklist--6-当前实现的消融实验-checklist"></a>
### 6. 当前实现的消融实验 Checklist

<a id="ds-exp-checklist--61-固定实验协议"></a>
#### 6.1 固定实验协议

可直接运行的 H=1 screening 配置见仓库根目录 `tasks.ds_ablation.example.json`；它包含 13 个唯一配置 × 4 seeds，共 52 runs。

- [ ] 环境固定为 `cube-triple-play-singletask-task2-v0`。
- [ ] Agent 固定为 ACRLPD，主筛选使用 H=1、1M online steps、100 eval episodes。
- [ ] 第一阶段使用 seeds `0-3`；只把候选最优配置扩展到 seeds `0-9`。
- [ ] 除被消融变量外，网络、UTD、数据比例、温度和评估协议完全一致。
- [ ] 每个 run 保存完整的新 flags：`ds_mode`、方向类型/表示、speed squash、speed interpretation、speed bound、H 和 seed。
- [ ] 报告 mean、std、median 和每 seed 原始成功率，不用单 seed 判断方法优劣。

<a id="ds-exp-checklist--62-实现预检"></a>
#### 6.2 实现预检

| 项目 | 状态 | 通过条件 |
|---|:---:|---|
| Adapter compose/decompose | ✅ | cartesian/spherical/stereographic 在各 bound 的可表示域内往返误差有限 |
| Mixed squash | ✅ | identity/tanh/sigmoid 的 forward、inverse、log-det 有限 |
| Integrated distribution | ✅ | spherical/stereographic/tangent 可采样且 `log_prob` 有限 |
| H=1/H=5 shape | ✅ | policy action、critic input、chunk reshape 维度一致 |
| 真实环境 smoke | ⬜ | 每个架构至少执行一次短 rollout，无 NaN、越界或 compose 错误 |

<a id="ds-exp-checklist--63-核心表示筛选方向与架构"></a>
#### 6.3 核心表示筛选：方向与架构

统一使用 `--ds_speed_bound=cube`；所有非 legacy 配置使用 `sigmoid+direct` 速度流水线。

| ID | 状态 | 架构 | 方向表示 | 关键配置 | seeds |
|---|:---:|---|---|---|:---:|
| A0 | ⬜ | Baseline | Cartesian XYZ | `--ds_mode=none` | 0-3 |
| A1 | ⬜ | Integrated | spherical | `--ds_mode=integrated --ds_integrated_type=spherical` | 0-3 |
| A2 | ⬜ | Integrated | stereographic | `--ds_mode=integrated --ds_integrated_type=stereographic` | 0-3 |
| A3 | ⬜ | Integrated | tangent | `--ds_mode=integrated --ds_integrated_type=tangent` | 0-3 |
| A4 | ⬜ | Adapter | cartesian | `--ds_mode=adapter --ds_adapter_representation=cartesian --ds_adapter_speed_squash=sigmoid --ds_adapter_speed_interpretation=direct` | 0-3 |
| A5 | ⬜ | Adapter | spherical | `--ds_mode=adapter --ds_adapter_representation=spherical --ds_adapter_speed_squash=sigmoid --ds_adapter_speed_interpretation=direct` | 0-3 |
| A6 | ⬜ | Adapter | stereographic | `--ds_mode=adapter --ds_adapter_representation=stereographic --ds_adapter_speed_squash=sigmoid --ds_adapter_speed_interpretation=direct` | 0-3 |

需要单独回答的问题：

- [ ] A1/A2/A3：固定架构后，三种方向分布谁最稳定？
- [ ] A1 vs A5、A2 vs A6：方向公式相同时，distribution-integrated 与 action-adapter 的差异是多少？
- [ ] A4 vs A5/A6：Adapter 的 D+1 Cartesian 冗余是否有收益？
- [ ] 所有 DS 配置相对 A0 是否提升，而不是只做 DS 方法间排名？

<a id="ds-exp-checklist--64-sphere-vs-cube-速度边界"></a>
#### 6.4 Sphere vs Cube 速度边界

只改变 `--ds_speed_bound`，速度比例和方向实现保持不变。Integrated 三种方向全部覆盖，以检验 bound 是否与方向坐标发生交互。

| ID | 状态 | 架构 | 配置 | bound | 复用 |
|---|:---:|---|---|:---:|---|
| B1 | ⬜ | Integrated spherical | `--ds_mode=integrated --ds_integrated_type=spherical` | sphere | 新运行 |
| B2 | ⬜ | Integrated spherical | 同上 | cube | A1 |
| B3 | ⬜ | Integrated stereographic | `--ds_mode=integrated --ds_integrated_type=stereographic` | sphere | 新运行 |
| B4 | ⬜ | Integrated stereographic | 同上 | cube | A2 |
| B5 | ⬜ | Integrated tangent | `--ds_mode=integrated --ds_integrated_type=tangent` | sphere | 新运行 |
| B6 | ⬜ | Integrated tangent | 同上 | cube | A3 |
| B7 | ⬜ | Adapter cartesian | `--ds_mode=adapter --ds_adapter_representation=cartesian --ds_adapter_speed_squash=sigmoid --ds_adapter_speed_interpretation=direct` | sphere | 新运行 |
| B8 | ⬜ | Adapter cartesian | 同上 | cube | A4 |

- [ ] 验证 cube action 每维都在 `[-1,1]`，且斜方向范数可大于 1。
- [ ] 分别在 Integrated 与 Adapter 中报告 cube/sphere 差值，避免把 bound 效果误判为架构效果。

<a id="ds-exp-checklist--65-adapter-速度模块squash--interpretation"></a>
#### 6.5 Adapter 速度模块：Squash × Interpretation

使用 Cartesian Adapter 做完整的 `speed squash × speed interpretation × radial bound` 2×2×2，方向表示固定不变。

| ID | 状态 | Actor squash | Adapter interpretation | bound | 复用 |
|---|:---:|---|---|:---:|---|
| C1 | ⬜ | sigmoid | direct | cube | A4 |
| C2 | ⬜ | sigmoid | direct | sphere | B7 |
| C3 | ⬜ | tanh | log | cube | 新运行 |
| C4 | ⬜ | tanh | log | sphere | 新运行 |
| C5 | ⬜ | tanh | direct | cube | 新运行 |
| C6 | ⬜ | tanh | direct | sphere | 新运行 |
| C7 | ⬜ | sigmoid | log | cube | 新运行 |
| C8 | ⬜ | sigmoid | log | sphere | 新运行 |

- [ ] 记录 log-speed 中 `exp(s)>1` 后被 env clip 的比例。
- [ ] 不在 Integrated DS 上设置 `log`：当前 Integrated speed 定义固定为 bounded sigmoid ratio。
- [ ] 分开报告 squash 与 interpretation 的主效应及交互效应。
- [ ] 将 `tanh+direct` 和 `sigmoid+log` 标记为机制消融，不作为默认速度定义。

<a id="ds-exp-checklist--66-squash-机制消融"></a>
#### 6.6 Squash 机制消融

目标是验证收益来自 DS 坐标本身，还是来自为各坐标匹配正确 support。S0/S2/S4 可复用主矩阵；S1/S3 需要新增 `ds_adapter_squash_profile` 实验开关。

| ID | 状态 | Adapter 表示 | squash profile | 方向/速度 support | 复用 |
|---|:---:|---|---|---|---|
| S0 | ⬜ | stereographic | matched | `identity(px,py) + sigmoid(rho)` | A6 |
| S1 | ⬜ | stereographic | bounded-plane | `tanh(px,py) + sigmoid(rho)` | 需新增开关 |
| S2 | ⬜ | spherical | matched | `sigmoid(theta,phi,rho)` | A5 |
| S3 | ⬜ | spherical | tanh-to-unit | `(tanh+1)/2` 映射 `theta,phi,rho` | 需新增开关 |
| S4 | ⬜ | cartesian | matched-ratio | `tanh(direction) + sigmoid(rho)` | A4 |
| S5 | ⬜ | cartesian | legacy-log | `tanh(direction,log_speed)` | C3 |

- [ ] S0 vs S1：量化把 stereographic 无界平面错误限制到 `[-1,1]^2` 的影响。
- [ ] S2 vs S3：在相同 `[0,1]` 输出 support 下比较 sigmoid 与 rescaled tanh 的数值尺度，而不是声称消除了极点奇异性。
- [ ] S4 vs S5：作为速度坐标与 squash 的联合对照，不能单独归因于 squash。
- [ ] 每组统计 replay/decompose 超出 actor support 的比例；BC 开启时另外检查 target clip。

<a id="ds-exp-checklist--67-distribution-与-jacobian-消融"></a>
#### 6.7 Distribution 与 Jacobian 消融

主矩阵已经覆盖四个 distribution family；先比较正确实现，再用错误-density 对照验证几何 Jacobian 是否必要。

| ID | 状态 | Distribution | 几何设置 | 复用/要求 |
|---|:---:|---|---|---|
| J0 | ⬜ | `TanhNormal` | Cartesian baseline | A0 |
| J1 | ⬜ | `MixedSquashNormal` | Adapter policy density | A4-A6 |
| J2 | ⬜ | `IntegratedDirectionSpeedNormal` | spherical，完整 log-det | A1 |
| J3 | ⬜ | `IntegratedDirectionSpeedNormal` | stereographic，完整 log-det | A2 |
| J4 | ⬜ | `TangentDirectionSpeedNormal` | tangent/exp-map，完整 log-det | A3 |
| J5 | ⬜ | `IntegratedDirectionSpeedNormal` | stereographic，去除方向几何 log-det | 需新增开关 |
| J6 | ⬜ | `TangentDirectionSpeedNormal` | 去除 tangent/exp-map log-det | 需新增开关 |

- [ ] J2/J3/J4 比较方向 distribution，但不把差异错误归因于 Jacobian 单项。
- [ ] J3 vs J5、J4 vs J6 才是 Jacobian 的严格机制对照；forward action 必须完全相同，只改变 `log_prob`。
- [ ] J1 的 density 定义在 `a_pi`，不能与 J2-J4 的 env-action entropy 数值直接横向解释。

<a id="ds-exp-checklist--68-action-chunking"></a>
#### 6.8 Action chunking

完成 H=1 筛选后，只扩展以下配置到 H=5：

| ID | 状态 | 配置 | H | seeds |
|---|:---:|---|:---:|:---:|
| D0 | ⬜ | Baseline A0 | 5 | 0-3 |
| D1 | ⬜ | H=1 最优 Integrated | 5 | 0-3 |
| D2 | ⬜ | Integrated tangent A3 | 5 | 0-3 |
| D3 | ⬜ | H=1 最优 Adapter | 5 | 0-3 |

- [ ] H=5 使用同一个 per-step compose/decompose，再 flatten 给 critic；不能对整个 `H×D` 向量做一次方向 normalize。
- [ ] 如果 H=5 排名与 H=1 不同，再将 H=5 前两名扩展到 10 seeds。

<a id="ds-exp-checklist--69-统计确认与扩展顺序"></a>
#### 6.9 统计确认与扩展顺序

1. [ ] 完成 A0-A6，共 28 个 H=1 screening runs。
2. [ ] 完成 B1/B3/B5/B7/C3/C4；其余复用已有配置，共新增 24 runs。
3. [ ] 从 Integrated、Adapter 和 Baseline 中各选一个候选，补 seeds `4-9`。
4. [ ] 再执行 D0-D3 的 H=5 对比。
5. [ ] 只有 task2 得到稳定结论后，才扩展到 task1/3/4/5、RoboMimic 和 LIBERO。

<a id="ds-exp-checklist--610-需要新增实验开关后才能执行"></a>
#### 6.10 需要新增实验开关后才能执行

| 项目 | 状态 | 原因 |
|---|:---:|---|
| `ds_integrated_log_det=full/none` | ⬜ | 支持 J5/J6；forward 保持不变，只控制 density 修正 |
| `ds_adapter_squash_profile=matched/ablation` | ⬜ | 支持 S1/S3 的错误或替代 squash 对照 |
| Tangent 非各向同性 covariance | ⬜ | 当前角噪声使用共享的二维 angular scale |
| Spherical 去除 epsilon bound | ⬜ | 仅用于数值稳定性压力测试，不属于主性能消融 |

<a id="ds-exp-checklist--7-不适用或禁止混用"></a>
### 7. 不适用或禁止混用

| 项目 | 状态 | 说明 |
|---|:---:|---|
| FQL + Integrated DS | ❌ | FQL 使用 ActorVectorField，当前没有 transformed distribution 接口 |
| Integrated + log-speed | ❌ | Integrated 的速度变量定义为 bounded radial ratio |
| Cube bound 下消融 `ds_max_speed` | ❌ | Cube 固定覆盖 normalized cube；`ds_max_speed` 只缩放 sphere bound |
| 把 Adapter `log_prob(a_pi)` 称为 env-action density | ❌ | compose 不属于 actor distribution，Cartesian codec 还是多对一 |
| 同时改变方向表示、速度压缩、速度解释和 bound | ❌ | 无法归因，必须按上述正交轴逐项改变 |

<a id="experiment-protocol"></a>
## 实验与证据协议

原文位置：`docs/experiments/experiment_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

更新日期：2026-10-13。本文件取代旧 SDD/Superpowers 合同作为项目级通用实验规范；具体批次仍需
有冻结的任务矩阵和配置 manifest。

<a id="experiment-protocol--1-实验身份"></a>
### 1. 实验身份

每个运行由以下字段唯一确定：

```text
cohort + task + method + seed + budget + expanded-config hash + source hash + attempt
```

改变方法行为、数据、预算、seed、评估协议或关键数值处理意味着新的实验身份。不得覆盖旧目录、
清理 incident 后原地重跑或把失败 attempt 改成成功。

<a id="experiment-protocol--2-公平对比"></a>
### 2. 公平对比

- 同一任务内使用相同训练 seed、评估 seed、数据、预算和评估轴；
- 表示、噪声几何和时间相关性分别做反事实；
- action norm、协方差、饱和率或初始化不同必须披露；
- 外部算法基线若改变目标、熵或 critic，不得写成纯参数化消融；
- task 不因结果不利而删除；正式 seed 不用于调参；
- evaluation episode 是 seed 内观测，不是独立训练重复。

<a id="experiment-protocol--3-actionnoise-shaping-控制"></a>
### 3. Action/noise shaping 控制

至少报告执行后平移动作的：

- L2 与 L∞ 范数；
- 方向变化与近零范数未定义比例；
- 径向变化；
- 协方差和主要特征值；
- 边界饱和/clipping 比例；
- 方向覆盖；
- 速度、yaw、gripper 的均值和标准差。

“局部协方差匹配”“状态平均矩匹配”和“完整条件分布匹配”是不同主张，不能混用。球约束会排除
Cartesian cube 中合法的对角动作，因此 sphere/cube 比较同时是支持域干预。

<a id="experiment-protocol--4-temporal-shaping-控制"></a>
### 4. Temporal shaping 控制

固定时间实验至少包含：

- reactive K=1；
- direction-only hold；
- translation-vector repeat；
- full-action repeat。

自适应方法应记录实际 option/hold duration、释放原因、actor 调用频率、接触事件和不确定性。若使用
semi-MDP backup，折扣必须按实际持续时间处理；不能把一步离线数据伪造成 K 步 option transition。

<a id="experiment-protocol--5-评估"></a>
### 5. 评估

- step 0 必须真实评估，不能补造 0；
- 使用固定 checkpoint 网格，不挑最好 checkpoint；
- 主指标为 success curve AUC 或批次预先规定的 normalized-return AUC；
- 同时报告最终 success、最终 return、time-to-threshold 和删失；
- time-to-threshold 未达到时记为右删失，不赋成最大预算；
- 评估 RNG 与训练 RNG 分离；
- stochastic-primary 与 diagnostic/latent-location 评估分开，后者不替代前者。

<a id="experiment-protocol--6-统计"></a>
### 6. 统计

- 单位是 task 内的 matched training seed/block；
- 先计算 seed 内配对差，再汇总；
- 报告原始 seed 值、均值、中位数、正配对数和配对 bootstrap CI；
- task 间异质性必须展示，不用 pooled 均值隐藏；
- 两个配对时只做描述，不报告看似正式的 p 值；
- 多比较需要预先定义 family 和校正；
- 不在看到结果后追加 seed 直到显著。

<a id="experiment-protocol--7-生命周期"></a>
### 7. 生命周期

至少区分：

```text
pending / running / complete / failed / interrupted / invalid / needs_review
```

进程消失、PID 不存在或调度器写入 `done` 都不能单独证明成功。成功需要：

- 可验证的退出码 0；
- 最后训练步等于预算；
- 预定评估轴和 episode 数完整；
- 数值有限且动作合法；
- 配置/source/data 哈希一致；
- 必需 checkpoint 可读；
- 没有未解决 incident。

基础设施故障和科学失败必须分开。科学低回报、零成功率或方法劣势不触发自动重跑。

<a id="experiment-protocol--8-调度和并发"></a>
### 8. 调度和并发

云端 RTX 4090 队列首先使用：

```bash
python -u schedule.py --run=<compiled.json> \
  --gpus=0,1,2,3,4,5,6,7 --gpu_tasks=8 --stagger=30
```

`gpu_tasks=8` 是每卡上限；任务少时调度器应跨卡分散。调度器只负责资源填槽，不代替上述
生命周期验收。OOM、静默退出、nonfinite、评估不完整或聚合吞吐倒退时，只降低尚未启动队列的
并发；保留已发生运行和 incident。

<a id="experiment-protocol--9-结果与证据布局"></a>
### 9. 结果与证据布局

- `docs/`：规范、方法和结果解释；
- `docs/data/summaries/`：Hugging Face submodule 中的精简评估、统计、图表和来源哈希；
- `/data/chainpray/qc/reports/`：生成的分析表、论文图和完整报告；
- `/data/chainpray/qc/results/`：新的本地运行、模型和原始结果；
- `/data/chainpray/qc/control/`：批次配置、运行状态和验收记录；
- `/data/chainpray/qc/logs/`：新的调度/训练文本日志；
- `exp/`、`exp_logs/`：已取消的历史路径，不再作为兼容入口；
- 云端 `/workspace/...`：云端冻结发布和运行结果；
- `/data/chainpray/qc/archive/consolidated-legacy-storage-20261018/project-evidence/evidence/legacy-sdd-20261013/`：只读历史归档，不继续写入。

任何结论都应能追溯到原始文件或哈希；日常文档不复制大模型和训练日志。
使用 `QC_DATA_ROOT` 覆盖本机数据根；发布规则和运行完整性检查范围见
[`result_archiving.md`](operations.md#result-archiving)。

<a id="experiment-protocol--10-结果披露"></a>
### 10. 结果披露

摘要、图表和正文必须同时披露：

- 有利和不利任务；
- success 与 return 的方向不一致；
- 缺失运行、删失、失败和 UNKNOWN；
- 工程门与性能门的区别；
- 当前结果是 pilot、descriptive 还是 formal；
- 没有完成的算法/跨域迁移。

论文不能把机制可行、数值稳定或 GPU 高利用率改写成任务成功率提升。
