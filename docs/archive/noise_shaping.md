# 噪声与速度调节：历史协议

[历史归档索引](index.md) · [文档入口](../index.md)

> 历史归档：各节保留原文的日期、实验身份和结论范围。旧文中的“当前”“推荐”“待做”不代表今天的执行状态；日常规则以 [实验](../current/experiments.md)、[运行与存储](../current/runbook.md) 为准。

## 内容

- [跨任务族噪声整形筛选协议](#cross-family-noise-screen-protocol)
- [Reward-stage allocation of direction--speed regularization](#phase-regularization-protocol)
- [Reward-stage speed-cap: 500k confirmatory protocol](#stage-speed-cap-formal-protocol)
- [Reward-stage speed-cap bridge: pilot protocol](#stage-speed-cap-protocol)

<a id="cross-family-noise-screen-protocol"></a>
## 跨任务族噪声整形筛选协议

原文位置：`docs/experiments/noise_shaping/cross_family_noise_screen_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

冻结日期：2026-10-16。

<a id="cross-family-noise-screen-protocol--动机"></a>
### 动机

既有 Cube-Triple task1–task3 结果表明，在固定 `alpha=0.1` 时，用控制器把 tangent
direction–speed 策略的“状态内平移半径标准差”和“相对策略中心的角偏差”匹配到 Cartesian
L2 projection，可以改善归一化回报 AUC；task2 同时出现成功率改善，task3 的成功率证据则较弱。
这些结果仍属于同一任务族，不能替代 Cube Double 和 Scene 的跨任务族验证。

本协议先做 reward-blind 校准，再做小规模性能筛选。表示、噪声几何和任务差异不得混为一个干预。

<a id="cross-family-noise-screen-protocol--阶段-areward-blind-cartesian-校准"></a>
### 阶段 A：reward-blind Cartesian 校准

- 环境：`cube-double-play-singletask-task2-v0`、`scene-play-singletask-task2-v0`；
- 条件：Cartesian Gaussian + XYZ L2-ball projection；
- 新 seed：`72410`、`72411`，两个任务内配对；
- 每次 200,000 环境步，`horizon_length=1`，RLPD/ACRLPD，`UTD=1`；
- 固定 `alpha=0.1`、`target_entropy=-2.5`；
- 禁止 reward/success evaluation：`eval_interval=0`、`eval_episodes=0`、`eval_at_start=false`；
- 在 step 0 及每 20k 步记录 512 个固定状态、每状态 32 个样本的执行动作几何；
- 仅在 100k 和 200k 保存 checkpoint，避免保留无必要中间权重；
- 每个任务的目标由两个 seed 的 160k 与 200k 四个 reward-blind 点等权平均得到；控制
  within-state radius SD 与 angular deviation，radius mean/second moment 只作诊断。

校准失败条件包括：非有限值、丢失几何轴、panel 身份不一致、动作越界、生命周期异常、任一运行
未达到 200k。失败时保留原运行与 incident，不用性能结果改 seed 或目标。

<a id="cross-family-noise-screen-protocol--校准结果2026-10-16"></a>
#### 校准结果（2026-10-16）

四个校准运行（两个任务各两个 seed）均达到 200k，生命周期正常且 incident 为空。由每个任务
两个 seed 的 160k/200k 几何点冻结得到：

| 任务 | radius SD | angular deviation | 诊断 radius mean | 诊断二阶矩 |
|---|---:|---:|---:|---:|
| Cube Double task2 | 0.0814520 | 0.3726861 rad | 0.9285565 | 0.8866256 |
| Scene task2 | 0.0563291 | 0.2935505 rad | 0.9748991 | 0.9555694 |

目标文件分别为 `docs/results/cube_double_noise_target_20261016.json` 与
`docs/results/scene_noise_target_20261016.json`，其中记录了输入运行、源哈希和目标哈希。

<a id="cross-family-noise-screen-protocol--阶段-b匹配控制工程门"></a>
### 阶段 B：匹配控制工程门

每个任务使用独立工程 seed 和该任务自己的冻结目标，运行 50k。最后五次控制器更新均须满足：

- radius-SD 绝对误差不超过 `0.02`；
- angular-deviation 绝对误差不超过 `0.05 rad`；
- 固定 alpha 恰为 `0.1`，所有训练量有限；
- 最终方向、执行幅值、yaw、gripper 的跨状态中心标准差分别大于
  `0.05/0.01/0.01/0.01`。

<a id="cross-family-noise-screen-protocol--工程门结果2026-10-16"></a>
#### 工程门结果（2026-10-16）

两个任务均通过。Cube Double 最后五次 radius-SD 误差为
`0.019328/0.013350/0.009812/0.007124/0.000504`，角误差最大 `0.007417 rad`；
Scene 对应误差为 `0.013036/0.008572/0.005251/0.003517/0.001173`，角误差最大
`0.014449 rad`。两者 50k 的独立状态面板误差也通过，且方向、执行幅值、yaw、gripper
中心容量均超过门槛。两个进程正常退出，incident 为空。

<a id="cross-family-noise-screen-protocol--阶段-c200k-两-seed-筛选"></a>
### 阶段 C：200k 两-seed 筛选

每个任务、每个 seed 比较三种方法：

1. Cartesian + L2 projection；
2. default tangent DS；
3. task-specific perturbation-matched tangent DS。

主指标为 step 0–200k 的 success AUC；同时报告归一化回报 AUC、最终成功率、最终归一化回报和
执行动作几何。筛选不是正式显著性实验。候选只有在两个任务均无明显退化、至少一个任务有正的
matched-seed success-AUC 改善、且最终成功率平均下降不超过 5 个百分点时，才进入 500k/五-seed
正式实验。所有 seed、零成功率和反向结果都保留。

冻结性能 seed 为 `72413/72414`。每 25k 步评估 20 个 episode，并保留真实 step 0；只在
200k 保存最终 checkpoint。评估 seed 固定为 `20261016`，所有方法和训练 seed 共用相同评估
episode reset 序列。

<a id="cross-family-noise-screen-protocol--执行状态2026-10-16-2114asiashanghai"></a>
#### 执行状态（2026-10-16 21:14，Asia/Shanghai）

12 个运行已在本机两张 RTX 6000 Ada 上启动，调度为每卡 6 并发、同卡相邻启动错开 30 秒。
启动后 12/12 均进入训练，最早运行约 38k 步、最晚运行约 5k 步；GPU 利用率快照为
`93%/98%`，显存占用约 `6.8/13.1 GiB`。当时未检出 traceback、OOM、资源耗尽或进程提前退出。
这只是运行健康检查，不是阶段 C 的性能结论；必须等 12 个运行完成并核验生命周期、评估轴、
checkpoint、geometry 与所有 seed 后再做方法比较。

<a id="cross-family-noise-screen-protocol--阶段-c-结果2026-10-16-2150asiashanghai"></a>
#### 阶段 C 结果（2026-10-16 21:50，Asia/Shanghai）

12/12 运行达到 200k 并正常退出；每个运行均有精确的 9 点评估轴和 9 点几何轴、一个 200k
checkpoint，incident 全为空。两 seed 均值如下：

| 任务 | 方法 | success AUC | normalized-return AUC | 200k success |
|---|---|---:|---:|---:|
| Cube Double | Cartesian projection | 0.39688 | 0.78961 | 1.00 |
| Cube Double | default tangent | 0.34063 | 0.77697 | 0.95 |
| Cube Double | matched tangent | 0.37813 | 0.78835 | 0.85 |
| Scene | Cartesian projection | 0.71250 | 0.82690 | 1.00 |
| Scene | default tangent | 0.65625 | 0.79589 | 1.00 |
| Scene | matched tangent | 0.68750 | 0.80916 | 1.00 |

Matched−Cartesian 的 success-AUC 差在 Cube Double 为 `−0.01875`，Scene 为 `−0.02500`；
200k success 差分别为 `−0.15/0.00`，等任务权重平均为 `−0.075`。因此“至少一个任务
success-AUC 为正”和“最终成功率平均下降不超过 5 pp”两个预设门均失败，**不进入 500k/五-seed
扩展**。只有两个配对 seed，本阶段不做显著性或总体优越性声明。

几何操纵本身成功：四个 matched 运行最后五次控制更新的 radius-SD 最大误差不超过
`0.002793`，angle 最大误差不超过 `0.003334 rad`。但是未受控的 radius mean 在 200k 仍比
对应 Cartesian 低约 `0.046–0.051`，导致二阶矩也明显偏低；所以本结果否定当前“两矩匹配”版本，
尚未构成完整扰动分布匹配的反例。

<a id="cross-family-noise-screen-protocol--阶段-dreward-blind-三矩匹配诊断"></a>
### 阶段 D：reward-blind 三矩匹配诊断

阶段 C 不通过后，只允许做几何诊断，不直接增加性能 seed。冻结一个 50k、无 reward/success
evaluation 的 controller-gain 工程筛选：

- 两任务 × `radius_mean_gain ∈ {1,2,4}` × 新 seed `72415/72416`，共 12 个运行；
- 启用 `ds_moment_control_speed_mean=true`，同时匹配 radius mean、within-state radius SD 和 angle；
- 其余 learner、固定 `alpha=0.1`、面板大小、采样数和目标哈希保持不变；
- 每 5k 更新控制器，每 10k 记录独立几何，只保存 50k 最终 checkpoint；
- 每个运行最后五次控制更新必须同时满足 mean 误差 `≤0.02`、radius-SD 误差 `≤0.02`、
  angle 误差 `≤0.05 rad`，且动作容量与边界门继续通过；
- 在通过全部两任务×两 seed 的 gain 中，选择最小 gain；若多个 gain 同为最小只会有一个，
  不使用 reward/success 决策。若没有 gain 全通过，停止三矩性能验证并修订控制器。

<a id="cross-family-noise-screen-protocol--阶段-d-结果2026-10-16"></a>
#### 阶段 D 结果（2026-10-16）

12/12 运行达到 50k 并正常退出，incident 为空、eval 文件不存在且每个运行只有一个 50k
checkpoint。没有 gain 通过完整工程门：

- gain 1 的最后五次 radius-mean 最大误差约为 Cube `0.104`、Scene `0.123`；
- gain 2 对应约为 `0.048/0.063`；
- gain 4 在 Cube 降至约 `0.018`，但 Scene 仍约 `0.030`，超过 `0.02` 门；
- 所有配置的独立几何面板 `center_speed_std` 都为零，因为全局 speed-mean 控制与
  `ds_moment_speed_residual_scale=0` 一起消除了状态相关速度中心容量。

因此阶段 D 不选择任何 gain，也不允许性能验证。三矩目标是可接近的，但当前执行方式把“匹配
全局 mean”和“保留状态条件速度策略”错误地绑定成了二选一。

<a id="cross-family-noise-screen-protocol--阶段-e状态相关-speed-residual-容量修复"></a>
### 阶段 E：状态相关 speed residual 容量修复

继续只用 reward-blind 几何门。冻结配置为：

- 两任务 × `ds_moment_speed_residual_scale ∈ {0.25,0.5,1.0}` × seed `72417/72418`，共 12 个运行；
- radius-mean gain 固定为 `8.0`，用于在 50k 工程预算内修复阶段 D 的慢收敛；
- 其他三矩目标、固定温度、控制周期、几何面板和训练预算保持阶段 D 不变；
- 最后五次 mean/std/angle 最大误差门仍为 `0.02/0.02/0.05 rad`；
- 最终独立面板 `center_speed_std > 0.01`，direction/yaw/gripper 容量分别大于
  `0.05/0.01/0.01`，动作边界不超过 `1+1e-6`；
- 若多个 residual scale 在两个任务和两个 seed 上全部通过，选择最小值；任何 reward、success、
  critic 或 replay 统计都不参与选择。若无一通过，停止性能验证并修改控制器结构。

<a id="cross-family-noise-screen-protocol--阶段-e-结果2026-10-16"></a>
#### 阶段 E 结果（2026-10-16）

12/12 运行达到 50k 并正常退出，incident 为空、没有 reward/success evaluation。三个 residual
scale 的 mean/std/angle 控制误差均通过：Cube 的最后五次最大误差范围分别约为
`0.0058–0.0086 / 0.0103–0.0159 / 0.0031–0.0054 rad`，Scene 对应约为
`0.0139–0.0170 / 0.0094–0.0154 / 0.0080–0.0088 rad`。但所有运行的最终
`center_speed_std` 都只有约 `1e-7`，远低于 `0.01` 容量门。

轨迹显示这不是初始接线缺失：`residual_scale=1.0` 时，Cube/Scene 的 step-0
`center_speed_std` 约为 `0.014/0.012`，到 10k 已降到约 `2e-5`，20k 后降至约 `1e-7`。
为匹配接近 1 的半径均值，全局 latent speed mean 被推到约 `3.3–5.0`，sigmoid 饱和把同一
latent 空间中的状态残差压扁。因此阶段 E 不选择任何 scale，也不做性能验证。

<a id="cross-family-noise-screen-protocol--阶段-f分数空间-speed-residual-参数化"></a>
### 阶段 F：分数空间 speed residual 参数化

只修复阶段 E 已定位的参数化问题，不改变默认旧行为。新增显式配置
`ds_moment_speed_residual_space=fraction`：先把全局 speed mean 映射为 `[0,1]` 中的速度比例，
在该比例上加状态相关 residual，再映回 Gaussian location。默认 `latent` 路径保持历史公式和
旧 checkpoint 兼容。

冻结 reward-blind 工程筛选为：

- 两任务 × fraction residual scale `{0.02,0.05,0.10}` × 新 seed `72419/72420`，共 12 个运行；
- radius-mean gain、三矩目标、50k 预算、控制与几何频率保持阶段 E 不变；
- 完全关闭 reward/success evaluation，只保留最终 checkpoint；
- 最后五次 mean/std/angle 误差门为 `0.02/0.02/0.05 rad`；最终独立面板仍要求
  `center_speed_std > 0.01`，其余容量和动作边界门不变；
- 只允许在所有任务和 seed 上通过的最小 scale 进入新的性能 cohort；若再次出现速度容量坍缩，
  不追加性能 seed，而转向显式条件速度目标或解耦的 residual regularization。

<a id="cross-family-noise-screen-protocol--阶段-f-结果2026-10-16"></a>
#### 阶段 F 结果（2026-10-16）

12/12 运行达到 50k，生命周期、精确控制/几何轴、配置身份和最终 checkpoint 均通过审计；
incident 为空且没有生成 `eval.csv`。没有任何 scale 通过完整门，因此不进入性能实验：

| fraction residual scale | 四运行最大 mean 误差 | 四运行最小最终 `center_speed_std` | 全部矩门 | 全部容量门 |
|---:|---:|---:|:---:|:---:|
| 0.02 | 0.02366 | 8.94e-8 | 否 | 否 |
| 0.05 | 0.04482 | 5.04e-8 | 否 | 否 |
| 0.10 | 0.05251 | 8.57e-8 | 否 | 否 |

`0.10` 在 Scene 两个 seed 和 Cube 的一个 seed 保留了大于 `0.01` 的速度中心差异，但同时
radius-mean 控制误差约为 `0.048–0.053`；其余大部分运行仍坍缩到约 `1e-7`。这说明单纯把
residual 从 latent 移到 fraction 空间没有解决“全局均值控制与条件速度策略相互竞争”的结构问题。
下一步不应再盲扫 residual scale，也不应绕过预设门读取性能；更合理的工程方向是把全局控制量
改成对原始条件 speed head 的零均值校正，或给条件 speed residual 加显式容量/任务相关目标。

机器可读审计位于
`docs/results/cross_family_noise_screen_20261016/fraction_residual_engineering/`，分析器为
`analyze_fraction_residual_engineering.py`；二者都明确排除了 `env.csv` 与 `online_agent.csv`。

<a id="cross-family-noise-screen-protocol--阶段-g条件-speed-head-加全局-fraction-offset"></a>
### 阶段 G：条件 speed head 加全局 fraction offset

阶段 F 后冻结一个更窄的结构反事实：不再用控制器的绝对均值替换条件 speed head，而是先由
原始 head 产生状态相关速度比例，再叠加一个初值为 0 的全局 fraction-space correction。这样
全局控制器只负责校正总体均值，条件 head 仍直接接受 actor 梯度。

- 两任务 × radius-mean gain `{0.5,1.0,2.0}` × 新 seed `72421/72422`，共 12 个运行；
- `ds_moment_speed_residual_space=fraction_offset`、`ds_moment_speed_mean_init=0.0`，不使用额外
  residual scale；
- 50k、固定 `alpha=0.1`、三矩目标、控制/几何频率和无评估约束保持不变；
- 矩误差、条件容量和动作边界门与阶段 F 完全相同；
- 在所有任务和 seed 上通过的最小 gain 才可进入独立的新性能 cohort。若无 gain 通过，停止
  当前闭环三矩控制路线，转向显式分布距离或扰动教师蒸馏，而不继续调 controller gain。

<a id="cross-family-noise-screen-protocol--阶段-g-结果2026-10-16"></a>
#### 阶段 G 结果（2026-10-16）

12/12 运行达到 50k，生命周期、配置身份、控制/几何轴、空 incident、无评估约束和唯一最终
checkpoint 均通过审计。条件 head 加全局 offset 确实解决了低/中 gain 的容量坍缩，但没有
gain 同时通过矩门：

| radius-mean gain | 四运行最大 mean 误差 | 四运行最大 radius-SD 误差 | 四运行最小最终 `center_speed_std` | 全部容量门 |
|---:|---:|---:|---:|:---:|
| 0.5 | 0.02620 | 0.02659 | 0.02367 | 是 |
| 1.0 | 0.00876 | 0.04123 | 0.02154 | 是 |
| 2.0 | 0.05361 | 0.06407 | 1.06e-7 | 否 |

因此结构反事实的结论不是“offset 无效”，而是：它恢复了条件速度容量，且 gain 1 准确控制了
均值，但 mean correction 改变了接近边界处的径向噪声，原来的独立 radius-SD 闭环无法同时
保持目标；gain 2 又进入裁剪饱和。按冻结规则不继续扫描 controller gain，也不做性能选择，
转向直接最小化执行空间中的条件扰动分布距离。

机器可读审计位于
`docs/results/cross_family_noise_screen_20261016/fraction_offset_engineering/`，分析器
`analyze_fraction_offset_engineering.py` 不读取 reward/success、critic 或 replay 指标。

<a id="cross-family-noise-screen-protocol--与既有时间实验的关系"></a>
### 与既有时间实验的关系

该批次只检验瞬时 noise shaping，动作每步重采样，不引入固定 direction hold、translation repeat
或 full repeat。因此它不能回答时间承诺问题；固定 `K=5` direction hold 的既有负结果保持不变。

<a id="phase-regularization-protocol"></a>
## Reward-stage allocation of direction--speed regularization

原文位置：`docs/experiments/noise_shaping/phase_regularization_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: complete, including all eight 500k-step runs and the common-seed
> 100-episode checkpoint reevaluation.  The screen did not pass its
> preregistered decision rule.

<a id="phase-regularization-protocol--motivation"></a>
### Motivation

Global semantic KL did not improve task2, separate component temperatures
failed, and a promising stereographic discovery did not replicate.  One
remaining mechanism is temporal mismatch: broad translation exploration may
help before any cube is placed, while the later precision stages may need less
direction/speed regularization.  Cube-triple's task reward is an observable
progress count: -3, -2, and -1 mean zero, one, and two cubes currently satisfy
their goals.

<a id="phase-regularization-protocol--intervention"></a>
### Intervention

The control is the corrected S5 spherical semantic policy with one global
temperature and target KL 5.3187063194210715.  The treatment changes only the
actor regularization term.  For a current-state stage reward `r`, define

`progress = clip(r - (-3), 0, 2)` and `w(r) = 0.5 ** progress`.

The direction and speed log-density ratios are multiplied by `w(r)`, giving
weights 1.0, 0.5, and 0.25 across the three stages.  Scalar yaw/gripper
regularization is unchanged.  The alpha update still targets the same global
unweighted semantic KL, so this intervention reallocates the translation
regularization gradient across states rather than silently granting a larger
overall entropy budget.

The actor executed in the environment still observes only the ordinary 46D
state.  Stage reward is used only in the training loss.  Offline labels are
the relabeled reward of the current observation.  Online replay stores the
reward known before the action, avoiding use of the behavior action's future
outcome as a current-state label.

<a id="phase-regularization-protocol--fixed-experiment"></a>
### Fixed experiment

- Environment: `cube-triple-play-singletask-task2-v0`, H=1.
- 500k online steps with the same prior data, network, env-action critic,
  sphere bound, semantic reference, target KL, and evaluation protocol.
- Paired untouched seeds 8, 9, 10, and 11.
- 50 evaluation episodes every 100k steps and a final common 100-episode
  reevaluation.
- Only manipulated factor: global weight 1 versus reward-stage translation
  weights 1/0.5/0.25.

Primary outcome: paired final success at 500k.  Secondary outcomes: fixed-100
success, tail-three success, normalized AUC, episode return, fraction of online
steps that reach reward -2 or -1, and translation norm within each stage.

The screen is considered promising only if its mean endpoint difference is
positive, at least three of four paired seeds favor the treatment, and the
fixed-100 result has the same direction.  A positive screen requires a fresh
four-seed confirmation before any performance claim.  A null/negative screen
rejects this simple reward-stage schedule; it should not trigger a decay-factor
sweep on the same seeds.

<a id="phase-regularization-protocol--verification-before-launch"></a>
### Verification before launch

- Reward frequencies in the 3M-transition prior dataset are 91.95%, 7.86%,
  and 0.18% for -3, -2, and -1.
- Unit and sequence-alignment tests verify weights 1/0.5/0.25 and that stage
  labels match the actor's current observation.
- A 100-step real OGBench smoke completed with finite actor/critic losses,
  alpha, phase metrics, and no post-warm-up unit-sphere violations.

<a id="phase-regularization-protocol--results"></a>
### Results

The ordinary 50-episode evaluations remained at zero through 400k steps for
all eight runs.  At 500k, the global control had success rates 0%, 0%, 0%, and
0% for seeds 8--11, while reward-stage regularization had 0%, 12%, 0%, and 0%.
Thus the endpoint means were 0.0% and 3.0%, but the paired differences were
0, +12, 0, and 0 percentage points: only one of four seeds improved.  The
paired bootstrap interval was [0, 9] points and the exact sign-flip p-value
was 1.0.  Tail-three success was 0.0% versus 1.0%, and normalized success AUC
was 0.0% versus 0.3%; both were likewise entirely due to seed 9.

The common-seed 100-episode reevaluation gave global success rates 1%, 0%, 0%,
and 0%, versus 2%, 12%, 0%, and 0% for reward-stage regularization.  The mean
difference was +3.25 points (bootstrap [0, 9], exact sign-flip p=0.5), with
only two positive pairs and two ties.  Final return was also null: -2054.5 for
global versus -2050.3 for reward-stage, a paired difference of +4.2 with
bootstrap interval [-79.7, 88.1] and two of four positive pairs.

The mechanism did alter behavior.  The treatment's last-ten-update average
stage weight was 0.884, and 20.9% of sampled states were assigned a weight
below one.  Total semantic entropy nevertheless stayed matched (-5.327 versus
-5.332).  Mean translation norms for states with zero, one, and two completed
cubes were 0.747/0.691/0.574 under the global control and
0.757/0.784/0.706 under the treatment.  Despite this large late-stage norm
change, online occupancy of states with at least one completed cube was only
35.0% versus 34.2%, and occupancy with at least two cubes was 10.3% versus
10.5%.  Their paired intervals crossed zero widely.

<a id="phase-regularization-protocol--decision-and-interpretation"></a>
### Decision and interpretation

The screen fails the preregistered rule: the endpoint mean is positive, but
only one of four endpoint pairs is positive rather than the required three,
and the fixed-100 signal is driven mainly by seed 9.  No fresh-seed
confirmation is warranted, and the decay factor must not be tuned on seeds
8--11.

This result rejects the specific heuristic, not every state-dependent
exploration method.  Multiplying the direction/speed regularizer by
1/0.5/0.25 clearly changed late-stage actions, but it increased their mean
magnitude without reliably improving stage occupancy.  A more diagnostic
next experiment should first use a synthetic staged task where the causal need
for broad approach motions and small precision motions is controlled.  On the
robot task, a direct state-dependent speed cap or learned uncertainty budget
is better motivated than another sweep of this reward-stage coefficient.

Artifacts:

- `/data/chainpray/qc/reports/analysis/phase_regularization_screen.png`
- `/data/chainpray/qc/reports/analysis/phase_regularization_per_run.csv`
- `/data/chainpray/qc/reports/analysis/phase_regularization_summary.csv`
- `/data/chainpray/qc/reports/analysis/phase_regularization_paired.csv`

<a id="stage-speed-cap-formal-protocol"></a>
## Reward-stage speed-cap: 500k confirmatory protocol

原文位置：`docs/experiments/noise_shaping/stage_speed_cap_formal_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

This protocol was fixed after the two-seed coefficient pilot selected
`lambda = 10` and before examining seeds 14--17.

<a id="stage-speed-cap-formal-protocol--conditions"></a>
### Conditions

Both conditions use integrated spherical direction-speed actions, sphere radial
bound, semantic-product regularization, target entropy
`-5.3187063194210715`, and horizon 1.

- Zero-penalty control: the full diagnostic/data path is active, but the speed
  cap coefficient is exactly zero.
- Selected cap: base cap `1.0`, per-stage decay `0.5`, coefficient `10`.

The critic and replay buffer always use executed environment actions. The
reward-stage label appears only in the actor's training loss; evaluation uses
ordinary observations only.

<a id="stage-speed-cap-formal-protocol--design"></a>
### Design

- Environment: `cube-triple-play-singletask-task2-v0`.
- Fresh paired seeds: `14, 15, 16, 17`; pilot seeds 12--13 are excluded.
- 500k environment steps.
- Evaluation every 100k with 50 episodes and fixed base seed `20260925`.
- Save the 500k checkpoint; after training, independently reevaluate every
  checkpoint for 100 episodes with the same fixed seed.

<a id="stage-speed-cap-formal-protocol--primary-and-diagnostic-outcomes"></a>
### Primary and diagnostic outcomes

Primary: 500k fixed-seed success rate. Secondary: success AUC over 0--500k,
tail-three evaluation success, and fixed-seed episode return. Mechanism
diagnostics: online executed translation norm at stages 0/1/2, actor-batch
stage-specific translation norm, and cap violation rate.

Report all four paired differences, the paired mean, a paired bootstrap 95%
interval, and the exact two-sided sign-flip p-value. With only four seeds, no
single p-value is treated as decisive; effect consistency and interval width
must be shown.

<a id="stage-speed-cap-formal-protocol--decision-rule"></a>
### Decision rule

Call the bridge promising only if all hold:

1. mean fixed-seed success improves by at least 5 percentage points;
2. at least 3 of 4 paired seeds improve and none drops by more than 5 points;
3. mean success AUC does not decrease;
4. stage-1 and stage-2 executed translation norms decrease without a stage-0
   decrease larger than 0.05;
5. no run fails, diverges, or produces non-finite metrics.

Otherwise retain the result as a negative or mechanism-only ablation and do not
promote it as the paper's robot-performance contribution.

<a id="stage-speed-cap-formal-protocol--outcome"></a>
### Outcome

The automated decision was **not promising**. The 500k training evaluation was
`10/0/0/0%` for the control and `0/0/0/0%` for the cap. Independent 100-episode
checkpoint reevaluation was `8/0/0/0%` versus `0/1/0/0%`, a paired mean change
of `-1.75` percentage points with bootstrap interval `[-6.0,+0.75]`. Success AUC
changed by `-0.25` points.

The intervention was nevertheless mechanically effective. Across all four
pairs, the executed translation norm changed by `-0.285` in stage 1 and
`-0.370` in stage 2, while stage 0 changed by only `-0.0027`. Actor-batch norms
showed the same result. Stage-1 and stage-2 online occupancy changed by `-3.56`
and `-1.23` percentage points, respectively. Thus the cap learned the intended
coarse-to-fine speed schedule but did not improve—and may have slightly
reduced—task progression. No further tuning on seeds 14--17 is permitted.

<a id="stage-speed-cap-protocol"></a>
## Reward-stage speed-cap bridge: pilot protocol

原文位置：`docs/experiments/noise_shaping/stage_speed_cap_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

This experiment is a hyperparameter-selection pilot, not confirmatory evidence.
It asks whether a direct state-conditioned radial constraint can correct the
late-stage large-action behavior observed in the negative phase-regularization
audit.

<a id="stage-speed-cap-protocol--intervention"></a>
### Intervention

For reward-stage progress `p = clip(reward - (-3), 0, 2)`, define the maximum
translation norm

`c(p) = 1.0 * 0.5**p`,

so the caps after zero, one, and two completed cubes are `1.0`, `0.5`, and
`0.25`. The actor receives an additional loss

`lambda * mean_h relu(||a_xyz,h||_2 - c(p))**2`.

The loss is computed on actions after the integrated direction-speed policy has
composed them into environment coordinates. The critic, replay buffer, and
environment all use the same executed action. Reward stage is a training-only
label; inference still conditions only on the ordinary observation.

<a id="stage-speed-cap-protocol--fixed-pilot-design"></a>
### Fixed pilot design

- Environment: `cube-triple-play-singletask-task2-v0`.
- Policy: integrated spherical direction-speed, sphere bound, semantic-product
  regularizer, target entropy `-5.3187063194210715`.
- Horizon: 1.
- Coefficients: `0, 10, 30, 100`.
- Paired tuning seeds: `12, 13`.
- Budget: 200k environment steps; evaluations every 50k using 20 episodes and
  fixed evaluation seed `20260925`.
- The coefficient-zero condition keeps the diagnostic path active but adds
  exactly zero penalty.

<a id="stage-speed-cap-protocol--selection-rule-fixed-before-reading-outcomes"></a>
### Selection rule fixed before reading outcomes

Among nonzero coefficients, retain conditions whose mean 200k success is no
more than 5 percentage points below coefficient zero. Compute the mean reduction
of executed translation norm in the one-complete and two-complete stages. Select
the smallest coefficient achieving at least 80% of the largest positive
reduction. If no coefficient produces a positive reduction without violating
the success constraint, reject this cap schedule.

The selected coefficient must subsequently be compared against an untouched
global-control implementation on fresh confirmatory seeds and 500k steps.

<a id="stage-speed-cap-protocol--pilot-outcome-not-confirmatory-evidence"></a>
### Pilot outcome (not confirmatory evidence)

All four conditions had 0% success and zero success AUC at 200k, so this pilot
does not establish a performance improvement. The predeclared rule selected
`lambda = 10`:

- stage-1 executed translation norm changed from `0.672` to `0.390`
  (`-0.282`; both paired seeds decreased);
- stage-0 norm was essentially unchanged (`+0.002`);
- the actor-batch cap violation fraction changed from `6.60%` to `3.18%`
  (`-3.42` percentage points; both paired seeds decreased);
- final return changed by `+253.6` on average, positive for both paired seeds;
- stage-2 online states were too rare for a reliable estimate (one finite pair).

The coefficient-10 late-stage norm reduction met the 80%-of-best criterion, so
the smallest qualifying coefficient was selected. These two seeds are now
excluded from the 500k confirmation.
