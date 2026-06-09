# 方向+速度动作分解

## 1. 动机

连续控制任务（如 OGBench 5D）的动作向量的前三维是**空间位移**，后两维是**旋转和夹爪**，但标准 SAC 对五维各自独立加噪：

$$a = \underbrace{[a_x, a_y, a_z]}_{\text{机械臂末端位移}} \; \underbrace{[a_{\text{yaw}}]}_{\text{旋转}} \; \underbrace{[a_{\text{grip}}]}_{\text{夹爪}}$$

SAC 的 TanhNormal 在高斯空间独立采样：

$$z_i \sim \mathcal{N}(\mu_i, \sigma^2_i), \quad a_i = \tanh(z_i)$$

各维度的噪声 $\sigma_i$ 虽然可学习，但**采样时互不关联**。问题出在前三维：

```
原始动作:  [0.4, 0.3, 0.1]  →  位移方向 ≈ (0.78, 0.59, 0.20),  位移量 = 0.51
加噪后:    [0.35, 0.42, 0.08] →  位移方向 ≈ (0.64, 0.77, 0.15),  位移量 = 0.55
                                 方向变了 14°              速度也变了 8%
```

**方向和速度总是同时变**——想换个方向，速度也跑偏了；想调个步长，方向也歪了。探索缺乏语义。

## 2. 方法：动作空间的语义重参数化

设连续动作 $a \in [-1,1]^D$ 可以划分为一组空间位移 $v=a_{1:d}$ 和若干标量控制项 $c=a_{d+1:D}$。对于 OGBench 5D 动作，$d=3$，$c$ 对应 yaw 与 gripper。我们的核心假设是：位移的方向和幅值承担不同控制语义，应该在策略分布与探索噪声中被显式分离。

我们定义一个方向-速度坐标系：

$$
m = \|v\|_2,\qquad
u = \frac{v}{\max(m,\epsilon)} \in S^{d-1},\qquad
s = \log \max(m,\epsilon),
$$

并将动作表示为

$$
z = [u,\;s,\;c].
$$

其中 $u$ 位于单位球面，$s$ 是对数速度，$c$ 保留原始标量控制。反变换为

$$
v = \operatorname{norm}(u)\exp(s),\qquad
a = [v,\;c].
$$

注意，逐维 Box 动作的空间位移范数上界是 $\sqrt{d}$ 而不是 1。因此 $s$ 不天然受限于 $[-1,1]$。这一区别决定了后续两种实现：一种把 $z$ 作为实用的策略输出空间，另一种把方向-速度结构写进可逆分布变换。

### 2.1 结构化随机策略

DS 不在环境执行端额外注入噪声；探索仍来自 actor 自身的随机策略。区别在于随机变量的坐标系不同。baseline 在原始动作维度上使用独立 Gaussian，再通过 tanh squash；DS 则让 Gaussian latent 经过方向-速度参数化后生成动作：

$$
\xi \sim \mathcal{N}(\mu_\theta(o), \Sigma_\theta(o)),\qquad
a = f_{\text{DS}}(\xi).
$$

在 post-hoc 模式中，$f_{\text{DS}}$ 是执行前的确定性 compose；在 stereographic/spherical 模式中，$f_{\text{DS}}$ 是带 inverse 和 log-det 的 bijector。这样，actor 的随机性仍然存在，但其作用方向与速度坐标一致，而不是在原始 Cartesian 维度上彼此纠缠。

对数速度坐标的一个好处是，latent 中速度方向的加性变化对应原始速度中的乘性变化：

$$
\exp(s+\epsilon_s)=\exp(s)\exp(\epsilon_s)=m\exp(\epsilon_s).
$$

因此速度噪声具有尺度一致性：同一个 $\epsilon_s$ 对小步长和大步长都表示相同的相对速度变化，而不是固定绝对位移偏移。

### 2.2 与时序扩展动作的结合

对于 action chunking，策略输出 $H$ 个连续动作。方向-速度重参数化逐时间步独立应用：

$$
z_{1:H} = [z_1,\ldots,z_H],\qquad z_t=[u_t,s_t,c_t].
$$

因此该方法不要求关闭时序扩展。实现上只需要保证 critic、actor、replay batch 和环境执行端对同一动作坐标系达成一致：post-hoc 方案在分解空间学习 Q；bijector 方案在原始动作空间学习 Q，但 actor 分布内部使用方向-速度变换。

## 3. 两种 RLPD-valid 参数化实例

上述语义分解可以通过两种可逆策略参数化实现。二者共享方向-速度先验，并且都保持单步动作维度为 $D$，因此 SAC/RLPD 中的 entropy、actor loss 和 BC `log_prob` 都可以包含正确的 change-of-variables Jacobian。

### 3.1 Stereographic Cartesian Bijector

第一种方案避免直接输出 3D 冗余方向，而是在二维平面上采样一个点 $p=(p_x,p_y)$，再通过 inverse stereographic projection 得到单位方向：

$$
\rho^2=p_x^2+p_y^2,\qquad
u(p)=\left[
\frac{2p_x}{1+\rho^2},
\frac{2p_y}{1+\rho^2},
\frac{1-\rho^2}{1+\rho^2}
\right].
$$

速度仍由一个 unconstrained scalar 经过 bounded sigmoid 得到：

$$
r = r_{\max}\sigma(r_{\text{raw}}),\qquad v=r\,u(p).
$$

该变换从 $(p_x,p_y,r_{\text{raw}})$ 到 3D 位移是几乎处处可逆的，并具有闭式 log-det：

$$
\log |J|
= 2\log r
+ \log\frac{4}{(1+\rho^2)^2}
+ \log\left|\frac{\partial r}{\partial r_{\text{raw}}}\right|.
$$

因此它保留 Cartesian 方向向量进入环境动作的形式，同时避免 D+1 post-hoc 归一化导致的非可逆密度问题。

### 3.2 Spherical Bijector 参数化

第二种方案使用两个角度参数化单位球面。对于一个 3D 位移组，actor 在 unconstrained latent space 中输出

$$
\xi = [\theta_{\text{raw}},\phi_{\text{raw}},r_{\text{raw}}],
$$

并通过可逆变换得到原始空间位移：

$$
\theta=\pi\sigma(\theta_{\text{raw}}),\quad
\phi=2\pi\sigma(\phi_{\text{raw}}),\quad
r=r_{\max}\sigma(r_{\text{raw}}),
$$

$$
v=r[\sin\theta\cos\phi,\;\sin\theta\sin\phi,\;\cos\theta].
$$

标量控制项仍使用 tanh bijector。该方案保持单步动作维度为 $D$，action chunking 时维度为 $H D$，并为每个时间步重复相同的 action group。由于 forward、inverse 与 log-det Jacobian 都由 bijector 给出，SAC 的 entropy 和 BC log-prob 在概率模型上更一致。主要风险来自球坐标极点和零半径附近的 Jacobian 退化；当前实现使用 epsilon-bounded sigmoid 和 clipped inverse 降低 NaN/Inf 风险。

### 3.3 非可逆 post-hoc 表示

post-hoc Cartesian 方案保留一个冗余的三维方向向量，让 actor 输出

$$
\hat{a}=[\tilde{u}_x,\tilde{u}_y,\tilde{u}_z,s,c] \in \mathbb{R}^{D+1}.
$$

环境执行前使用确定性 compose：

$$
u=\frac{\tilde{u}}{\max(\|\tilde{u}\|_2,\epsilon)},\qquad
v=u\exp(s),\qquad
a=[v,c]\in\mathbb{R}^D.
$$

这里的关键问题是归一化会丢弃 $\tilde{u}$ 的径向长度。对于任意正数 $\lambda$，

$$
\operatorname{norm}(\lambda\tilde{u})=\operatorname{norm}(\tilde{u}),
$$

因此

$$
[\tilde{u},s,c],\quad [2\tilde{u},s,c],\quad [100\tilde{u},s,c]
$$

都会映射到同一个环境动作。这是一个多对一映射，而不是 bijection。

这带来三个直接影响：

1. **没有普通 change-of-variables log-prob。** SAC/RLPD 的 actor loss 和 entropy 需要 $\log \pi(a|o)$。对于 TanhNormal 或 DS bijector，可以用

   $$
   \log \pi(a)=\log p(\xi)-\log |\det \partial f(\xi)/\partial \xi|.
   $$

   但 post-hoc 映射是 $D+1\rightarrow D$，Jacobian 不是方阵，也没有唯一 inverse，不能得到普通 bijector 的 log-det 修正。实现中只能计算 actor 在 $\hat{a}$ 空间的 `log_prob`，这不是环境动作 $a$ 的严格密度。

2. **方向向量的长度是不可观测自由度。** Critic 和 BC loss 会看到 $\tilde{u}$ 的数值，但环境只看到归一化后的方向。模型可能把容量用在一个执行端会被丢弃的径向自由度上。

3. **训练坐标和执行动作存在近似错位。** post-hoc 路径在分解空间训练 actor/critic，再 compose 成原始动作执行。这个近似可能仍然有效，尤其对不依赖 SAC-style entropy density 的方法更温和，但不能作为概率严格的 RLPD/SAC 对照。

因此 post-hoc 模式保留为两个用途：一是 FQL 或工程消融，二是验证“D+1 表示本身”是否带来经验收益。对于需要正确 `log_prob` 的 RLPD/SAC 主实验，应使用 `stereographic` 或 `spherical` bijector。

### 3.4 对比

| 维度 | Stereographic Cartesian | Spherical Angles |
|--|--|--|
| 方向坐标 | 2D 平面点经 stereographic projection 到 $S^2$ | $(\theta,\phi)$ 两角度 |
| 单步输出维度 | $D$ | $D$ |
| Action chunking | 支持，输出 $HD$ | 支持，输出 $HD$ |
| Critic 输入 | 原始动作 | 原始动作 |
| `log_prob` | 含 stereographic/radial Jacobian | 含 spherical/radial Jacobian |
| 实现入口 | `--ds_mode=stereographic` | `--ds_mode=spherical` |
| 主要风险 | stereographic chart 在一个极点退化 | 球坐标极点/角度周期退化 |

## 4. 实验开关

DS 相关入口统一为 `--ds_mode`：

三种 DS 实现的维度、映射、`log_prob` 和 action chunking 差异，可打开 [`ds_implementations_animation.html`](ds_implementations_animation.html) 查看动画说明。

| `ds_mode` | 实现方式 | 适用场景 | 动作维度 | `log_prob` |
|--|--|--|--|--|
| `none` | 原始动作空间 | baseline | 单步 $D$，chunk $HD$ | TanhNormal |
| `posthoc` | D+1 分解动作，执行前确定性 compose | FQL；RLPD 近似消融 | 单步 $D+1$，chunk $H(D+1)$ | 分解空间近似，非环境动作密度 |
| `stereographic` | stereographic DS bijector | RLPD/SAC 正式 DS | 单步 $D$，chunk $HD$ | Jacobian-corrected |
| `spherical` | spherical-angle DS bijector | RLPD/SAC 正式 DS | 单步 $D$，chunk $HD$ | Jacobian-corrected |

旧 flag 仍可兼容：

- `--direction_speed=True` 等价于 `--ds_mode=posthoc`。
- `--agent.use_ds_bijector=True --agent.ds_bijector_type=...` 仍可用，但推荐改用 `--ds_mode=stereographic` 或 `--ds_mode=spherical`。

RLPD 中用于验证 Jacobian-corrected DS 的两种实现如下：

```bash
# 方案一：Stereographic Cartesian DS，单步动作。
MUJOCO_GL=egl python main_online.py \
  --run_group=ds_stereo_rlpd_h1 \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --sparse=False \
  --horizon_length=1 \
  --agent.action_chunking=False \
  --ds_mode=stereographic

# 方案二：Stereographic Cartesian DS，action chunking。
MUJOCO_GL=egl python main_online.py \
  --run_group=ds_stereo_rlpd_ac_h5 \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --sparse=False \
  --horizon_length=5 \
  --agent.action_chunking=True \
  --ds_mode=stereographic

# 方案三：Spherical-angle DS，单步动作。
MUJOCO_GL=egl python main_online.py \
  --run_group=ds_spherical_rlpd_h1 \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --sparse=False \
  --horizon_length=1 \
  --agent.action_chunking=False \
  --ds_mode=spherical

# 方案四：Spherical-angle DS，action chunking。
MUJOCO_GL=egl python main_online.py \
  --run_group=ds_spherical_rlpd_ac_h5 \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --sparse=False \
  --horizon_length=5 \
  --agent.action_chunking=True \
  --ds_mode=spherical
```

不要在 ACRLPD/RLPD 中使用 `--ds_mode=posthoc` 做严格 log-prob 对照。该模式对应非可逆 D+1 post-hoc 近似；RLPD 的正确性实验应使用 `--ds_mode=stereographic` 或 `--ds_mode=spherical`。

如果需要保留旧的 D+1 post-hoc 路径做近似消融，需要显式 opt-in：

```bash
MUJOCO_GL=egl python main_online.py \
  --run_group=posthoc_ds_rlpd_ac_ablation \
  --env_name=cube-triple-play-singletask-task2-v0 \
  --sparse=False \
  --horizon_length=5 \
  --ds_mode=posthoc \
  --agent.action_chunking=True \
  --allow_posthoc_direction_speed_rlpd=True
```

这条路径的结果应标注为 approximate ablation，不应与 Jacobian-corrected bijector 结果混为同一类。

## 5. 实现流程

下述流程对应 post-hoc Cartesian 方案。`stereographic` 和 `spherical` bijector 方案不预处理数据动作，也不在环境执行前调用 `compose`；它们只在 actor 分布内部替换 TanhNormal。

### 5.1 离线数据预处理

```python
# 数据集动作是 5D raw，一次性 decompose 成 6D
dataset_actions  # (N, 5)
decomposed = decompose(dataset_actions)  # (N, 6)
# Agent 学习 obs → 6D decomposed 的映射
```

### 5.2 在线交互（每步）

```python
decomposed = agent.act(obs)                        # Actor 出 6D
raw = compose(decomposed)                          # 6D → 5D
env.step(raw)                                      # 环境吃 5D
```

### 5.3 Critic 训练

Replay buffer 存 raw action（5D）。训练前 decompose 回 6D 给 Critic：

```python
batch = replay_buffer.sample()
batch['actions'] = decompose_chunked(batch['actions'])  # 5D → 6D
critic.update(batch)  # Critic 在分解空间评估 Q 值
```

### 5.4 探索来源

所有 DS 模式都不再额外注入外部动作噪声。探索来自 actor 自身的随机策略：

```text
baseline/posthoc:       Normal -> TanhNormal action
stereographic/spherical: Normal -> DS bijector -> raw action
```

这样训练时的 `log_prob` 与执行时的 action 分布保持一致；post-hoc 模式也只作为 D+1 表示的近似消融，而不是额外手工噪声消融。

## 6. 数值和 JAX 检查

- `reshape(H * D -> H x D)` 不会切断 XLA 的精度或梯度链路；它不改变 dtype，也不是 `stop_gradient`。
- post-hoc DS 的 compose 是确定性变换；不会额外改变 actor 采样出的动作分布。
- 球坐标 bijector 的 `theta`、`phi`、`speed` 使用 epsilon-bounded sigmoid，避免精确落到 0/1、极点或零半径。
- inverse/log_prob 路径对 `atanh`、`log`、`acos` 输入做 clipping，降低 NaN/Inf 风险。
- 仍需监控的风险：球坐标极点附近的 Jacobian 很小，log-det 会很负；如果 actor 均值长期饱和，可能导致梯度变小。建议实验日志里同时看 `actor/entropy`、`actor/alpha`、`critic/q_max`、`critic/q_min`。
- post-hoc DS 的主要近似不是 XLA，而是概率模型：D+1 分解动作没有完整 Jacobian 修正，且 TanhNormal 对 `log_speed` 的取值范围有约束。

## 7. 实现

```python
# posthoc_direction_speed.py

decompose(a)          # a ∈ R^D     → â ∈ R^{D+1}
compose(â)            # â ∈ R^{D+1} → a ∈ R^D
decompose_chunked()   # H>1 时的批量处理

# rlpd_distributions/direction_speed.py

DirectionSpeedNormal  # ACRLPD 的 Jacobian-corrected DS actor distribution
```

post-hoc helper 见 [`posthoc_direction_speed.py`](../posthoc_direction_speed.py)，RLPD/SAC distribution 见 [`rlpd_distributions/direction_speed.py`](../rlpd_distributions/direction_speed.py)。
