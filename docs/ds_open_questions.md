# Direction-Speed 疑难点跟踪

本文档整理 DS 实现讨论中的关键疑问，目的是后续按条验证、补实验或改代码。主线设计见 [`approach.md`](approach.md)。

2026-06-10 已完成轻量验证和 6 组真实环境 smoke；逐项结论见 [`ds_question_resolution.md`](ds_question_resolution.md)，完整机器输出见 [`ds_validation_results.md`](ds_validation_results.md)。

## 0. 当前结论

当前保留三种 DS 实现：

| 实现 | 入口 | 当前定位 |
|--|--|--|
| post-hoc D+1 | `--ds_mode=posthoc` | 非可逆表示消融；FQL 可用；RLPD 只作 approximate ablation |
| spherical bijector | `--ds_mode=spherical` | Jacobian-corrected RLPD/SAC 对照 |
| stereographic bijector | `--ds_mode=stereographic` | Jacobian-corrected RLPD/SAC 推荐主线 |

当前推荐主实验：

```bash
--ds_mode=stereographic --horizon_length=1 --agent.action_chunking=False
--ds_mode=stereographic --horizon_length=5 --agent.action_chunking=True
--ds_mode=spherical --horizon_length=1 --agent.action_chunking=False
--ds_mode=spherical --horizon_length=5 --agent.action_chunking=True
```

## 1. `horizon_length` 和 `action_chunking`

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

## 2. DS 和 action chunking 形状兼容性

**疑问。** DS 是否只输出单步 `D`，导致 critic 期待 `HD` 时形状不匹配？

**当前结论。** 当前代码已经支持 chunk。`action_chunking=True` 时，DS action groups 会按 horizon 复制。对于 `D=5,H=5`：

```text
每步: direction-speed group + yaw scalar + gripper scalar -> 5D
5 步: 重复 5 份 -> 25D
```

因此 spherical/stereographic DS actor 输出 `HD`，critic 输入也为 `HD`。

**后续检查。**

- [ ] 为 `_chunked_ds_action_groups` 加单元测试或 smoke test。
- [ ] 文档中明确 “DS chunk 不是单个 head 复用，而是 H 个独立 group”。

## 3. `reshape(HD -> H x D)` 和 XLA 精度

**疑问。** `action_chunking=True` 时把 `25D` reshape 成 `(5,5)` 是否会切断 XLA 数值链路，导致 float32 掉到 float16？

**当前结论。** `reshape` 不改变 dtype，也不是 `stop_gradient`，不会单独导致精度降级。它只是 view/shape transformation。真正需要关注的是：

- spherical 极点附近 `sin(theta)` 小，Jacobian 可能很负。
- sigmoid 饱和会导致梯度变小。
- mixed precision 或外部框架配置才可能改变 dtype。

**后续检查。**

- [ ] 在 debug 日志或测试中检查 actor sample dtype。
- [ ] 记录 spherical/stereographic 的 `log_prob`、entropy、alpha、Q 范围，排查 NaN/Inf。

## 4. `log_prob` 的对象到底是谁

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

## 5. 为什么 bijector 放在 actor 内部

**疑问。** MLP 输出仍然是 `theta_raw, phi_raw, speed_raw`，为什么不在 actor 外部 compose 成 xyz？

**当前结论。** MLP 只输出 raw Gaussian 的 `mean/log_std`。在 `DirectionSpeedNormal` 内部：

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

## 6. post-hoc D+1 的多对一问题

**疑问。** post-hoc 中 `u` 先 L2 normalize，从 `D+1` 压到 `D`，影响是什么？

**当前结论。** 这是多对一映射。例如：

```text
u, 2u, 100u
```

normalize 后方向相同，compose 出来的环境动作也相同。真实 `p_action(a)` 需要沿被丢弃的径向自由度积分，而不是简单取某个 `log p_raw(raw)`。因此 post-hoc 没有普通 bijector 的 Jacobian 修正。

影响：

- actor 会优化环境不可见的径向自由度。
- critic/BC 若吃 raw 分解动作，需要学习等价 raw 的相同 Q。
- RLPD/SAC entropy 语义不严格。

**后续检查。**

- [ ] post-hoc RLPD 结果统一标为 approximate ablation。
- [ ] 不把 post-hoc 曲线和 spherical/stereographic 作为同等概率模型比较。

## 7. critic 如果吃 raw action 是否还需要 bijector

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

对 post-hoc 多对一 raw，Q 在等价径向维度上应相同，函数逼近更难，actor 也可能 exploit critic 在无效维度上的误差。

**后续检查。**

- [ ] 如果做 raw-critic 消融，需要单独命名为 raw-action-space MDP ablation。
- [ ] 比较 raw-critic 与 env-action-critic 时，不把它们称为同一个 SAC/RLPD 概率模型。

## 8. 三种 DS 的实验定位

**疑问。** 三种 DS 如何公平比较？

**当前结论。**

- `posthoc`：验证 D+1 表示是否有经验收益，尤其 FQL。
- `spherical`：验证 “exact bijector + 球坐标” 是否能工作。
- `stereographic`：验证 “exact bijector + 更平滑方向坐标” 是否优于 spherical。

最低限度矩阵：

| Agent | H | chunk | ds_mode | 目的 |
|--|--:|--:|--|--|
| RLPD | 1 | false | none | baseline |
| RLPD | 1 | false | spherical | exact DS 对照 |
| RLPD | 1 | false | stereographic | exact DS 主线 |
| RLPD-AC | 5 | true | none | chunk baseline |
| RLPD-AC | 5 | true | spherical | exact DS chunk 对照 |
| RLPD-AC | 5 | true | stereographic | exact DS chunk 主线 |
| FQL | 5 | true | posthoc | 表示消融 |

**后续检查。**

- [ ] 每张实验图标明 `ds_mode`、H、chunk、是否 Jacobian-corrected。
- [ ] 不再使用未标注 provenance 的旧图作为主结论。

## 9. flag 整理

**疑问。** `use_ds_bijector`、`use_direction_speed`、`ds_max_speed`、`ds_bijector_type` 是否都有用？

**当前结论。**

- 用户主入口是 `--ds_mode`。
- `use_ds_bijector` 仍是 agent 内部开关。
- `use_direction_speed` 只是旧兼容别名。
- `ds_bijector_type` 决定 `spherical` 或 `stereographic`。
- `ds_max_speed` 只在 DS bijector 开启时生效。

**后续检查。**

- [ ] 文档命令只使用 `--ds_mode`。
- [ ] 旧 flag 保留兼容，但不要在新实验脚本里继续使用。
- [ ] 考虑启动时 warning：用户直接设置旧 agent direction-speed flag 时提示 deprecated。

## 10. 外部噪声

**疑问。** post-hoc 外部噪声是否还有用？

**当前结论。** 已移除外部加噪。探索应来自 actor 本身的随机策略。外部噪声会引入额外不可控分布，混淆 DS 表示和探索策略两个因素。

**后续检查。**

- [ ] 确认代码里不再有 `dir_noise`、`speed_noise`、`other_noise`。
- [ ] 如果后续需要探索消融，应作为单独实验变量，不和 DS 主实验混在一起。

## 11. 数值稳定性清单

**疑问。** 如何避免 JAX 梯度消失/爆炸、NaN/Inf、精度损失？

**当前措施。**

- spherical/stereographic 使用 bounded sigmoid，避免精确到边界。
- inverse/log/log-det 路径做 clip。
- `reshape` 不改变 dtype。
- chunk DS 按 group 复制，避免单步输出和 critic 维度不匹配。

**后续检查。**

- [ ] smoke test：`none/spherical/stereographic` x `H=1/H=5 chunk` 一步 update 全 finite。
- [ ] 训练日志：`actor_loss`、`entropy`、`alpha`、`log_prob`、`q_min/q_max`。
- [ ] 如果 spherical 不稳定，优先比较 stereographic，而不是回退到 post-hoc 主线。

## 12. 待办优先级

1. 补启动日志和 shape assert，确认 H/chunk/DS 输出维度。
2. 补 smoke test，覆盖 spherical/stereographic 的 H=1 与 H=5 chunk。
3. 统一实验图命名和 caption，标注 `ds_mode` 与 Jacobian 状态。
4. 跑 RLPD H=1 三组：none、spherical、stereographic。
5. 跑 RLPD-AC H=5 chunk 三组：none、spherical、stereographic。
6. 将 post-hoc 保留为 FQL/历史消融，不进入 RLPD 主结论。
