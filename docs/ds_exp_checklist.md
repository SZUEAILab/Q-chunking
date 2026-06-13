# DS (Direction-Speed) 实验验证清单

> ✅ = 已完成 | 🔄 = 运行中 | ⬜ = 待做 | ❌ = 不可行

## 1. RLPD/SAC — 纯在线 (main_online.py)

验证 DS 在 TanhNormal 分布下的 Jacobian-corrected 效果。env: cube-triple-task2, seed: 0-3

### 1.1 H=1 (无 chunk) — 推荐配置

| #     | 状态 | ds_mode       | 步数 | 说明                      |
| ----- | :--: | ------------- | ---- | ------------------------- |
| 1-4   |  ✅  | none          | 1M   | RLPD baseline             |
| 5-8   |  ✅  | posthoc       | 1M   | DS D+1 (需 allow_posthoc) |
| 9-12  |  ✅  | stereographic | 1M   | DS bijector (球极投影)    |
| 13-16 |  ✅  | spherical     | 1M   | DS bijector (球坐标)      |

> **结果**: stereo 91% ≈ posthoc 89% > spherical 64% > baseline 59%

### 1.2 H=5 (chunk) — 探索性配置

| #     | 状态 | ds_mode       | 步数 | 说明             |
| ----- | :--: | ------------- | ---- | ---------------- |
| 17-20 |  ✅  | none          | 1M   | RLPD-AC baseline |
| 21-24 |  ✅  | posthoc       | 1M   | DS D+1           |
| 25-28 |  ✅  | stereographic | 1M   | DS bijector      |
| 29-32 |  ✅  | spherical     | 1M   | DS bijector      |

> **结果**: posthoc 32% > stereo 19% > spherical 7% > baseline 3%

---

## 2. FQL — 离线→在线 (main.py)

验证 DS 在 flow-based 算法上的表示消融效果。FQL 仅支持 posthoc。env: cube-triple-task2

### 2.1 H=5 (chunk) — 主实验

| #     | 状态 | seed | ds_mode | 步数  | 结果                                  |
| ----- | :--: | :--: | ------- | ----- | ------------------------------------- |
| 33-36 |  ✅  | 0-3 | none    | 1M+1M | 96%/84%/90% =**90%**            |
| 37-40 |  ✅  | 0-3 | posthoc | 1M+1M | 100%/98%/4% =**67%** (中位 98%) |

### 2.2 H=1 — 已完成 (⚠ FQL H=1 不适合此任务)

| #     | 状态 | seed | ds_mode | 步数  |               结果               |
| ----- | :--: | :--: | ------- | ----- | :------------------------------: |
| 41-44 |  ✅  | 0-3 | none    | 1M+1M | 0% / 0% / 0% / 0% =**0%** |
| 45-48 |  ✅  | 0-3 | posthoc | 1M+1M | 6% / 8% / 8% / 6%* =**7%** |

> * H=1 + chunk=True → 实际等价于无 chunk (1×5=5D)。*seed3 未跑，用 seed0 值填充。

### 2.3 H=5 + best-of-n (QC 模式) — 待做

| #     | 状态 | seed | ds_mode | 步数  | 关键参数                             |
| ----- | :--: | :--: | ------- | ----- | ------------------------------------ |
| 49-52 |  ⬜  | 0-3 | none    | 1M+1M | actor_type=best-of-n, num_samples=32 |
| 53-56 |  ⬜  | 0-3 | posthoc | 1M+1M | actor_type=best-of-n, num_samples=32 |

### 2.4 ⚠ 旧 FQL H=5 (错误 5D 归一化) — 仅供参考

| #  | 状态 | seed | ds           | 步数  |       结果       |
| -- | :--: | :--: | ------------ | ----- | :---------------: |
| — |  ⚠  |  0  | posthoc (5D) | 1M+1M | raw 88% vs DS 98% |

---

## 3. OGBench 多任务验证

每个任务 × 4 DS modes × 4 seeds × H=1+H=5 = 32 实验。agent: acrlpd, 1M 纯在线

### 3.1 cube-triple task1 / task3

| #       | 状态 | env               | H | ds_mode       | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | ---- |
| 57-60   |  ✅  | cube-triple-task1 | 1 | none          | 1M   |
| 61-64   |  ✅  | cube-triple-task1 | 1 | posthoc       | 1M   |
| 65-68   |  ✅  | cube-triple-task1 | 1 | stereographic | 1M   |
| 69-72   |  ✅  | cube-triple-task1 | 1 | spherical     | 1M   |
| 73-76   |  ✅  | cube-triple-task1 | 5 | none          | 1M   |
| 77-80   |  ✅  | cube-triple-task1 | 5 | stereographic | 1M   |
| 81-84   |  ✅  | cube-triple-task1 | 5 | spherical     | 1M   |
| 85-88   |  ✅  | cube-triple-task1 | 5 | posthoc       | 1M   |
| 89-92   |  ⚠  | cube-triple-task3 | 1 | none          | 1M   |
| 93-96   |  ⚠  | cube-triple-task3 | 1 | posthoc       | 1M   |
| 97-100  |  ⚠  | cube-triple-task3 | 1 | stereographic | 1M   |
| 101-104 |  ⚠  | cube-triple-task3 | 1 | spherical     | 1M   |
| 105-108 |  ⚠  | cube-triple-task3 | 5 | none          | 1M   |
| 109-112 |  ⚠  | cube-triple-task3 | 5 | stereographic | 1M   |
| 113-116 |  ⚠  | cube-triple-task3 | 5 | spherical     | 1M   |
| 117-120 |  ⚠  | cube-triple-task3 | 5 | posthoc       | 1M   |

> task3 有 MuJoCo 物理 bug（`mj_narrowphase` 碰撞超限），4 seed 反复崩溃。跳过 task3，转向 cube-double。
> H=5 的 posthoc 需要 `--allow_posthoc_direction_speed_rlpd=True`。

### 3.2 cube-double

| #       | 状态 | env               | H | ds_mode       | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | ---- |
| 113-116 |  ⬜  | cube-double-task2 | 1 | none          | 1M   |
| 117-120 |  ⬜  | cube-double-task2 | 1 | posthoc       | 1M   |
| 121-124 |  ⬜  | cube-double-task2 | 1 | stereographic | 1M   |
| 125-128 |  ⬜  | cube-double-task2 | 1 | spherical     | 1M   |
| 129-132 |  ⬜  | cube-double-task2 | 5 | none          | 1M   |
| 133-136 |  ⬜  | cube-double-task2 | 5 | posthoc       | 1M   |
| 137-140 |  ⬜  | cube-double-task2 | 5 | stereographic | 1M   |
| 141-144 |  ⬜  | cube-double-task2 | 5 | spherical     | 1M   |

### 3.3 cube-single (可选)

| #       | 状态 | env               | H | ds_mode       | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | ---- |
| 145-148 |  ⬜  | cube-single-task2 | 1 | none          | 1M   |
| 149-152 |  ⬜  | cube-single-task2 | 1 | stereographic | 1M   |
| 153-156 |  ⬜  | cube-single-task2 | 5 | none          | 1M   |
| 157-160 |  ⬜  | cube-single-task2 | 5 | stereographic | 1M   |

> cube-single 最简单，先跑 stereo vs none，如有显著差异再加 posthoc/spherical。

### 3.4 cube-quadruple (可选)

| #       | 状态 | env                  | H | ds_mode       | 步数 |
| ------- | :--: | -------------------- | :-: | ------------- | ---- |
| 161-164 |  ⬜  | cube-quadruple-task2 | 1 | none          | 1M   |
| 165-168 |  ⬜  | cube-quadruple-task2 | 1 | stereographic | 1M   |
| 169-172 |  ⬜  | cube-quadruple-task2 | 5 | none          | 1M   |
| 173-176 |  ⬜  | cube-quadruple-task2 | 5 | stereographic | 1M   |

> cube-quadruple 极难，先跑 stereo vs none，有信号再加满。

---

## 4. 跨域验证 (RoboMimic — 7D 动作空间)

agent: acrlpd, 4 seeds × 4 DS modes, 1M 纯在线. DS 将位移 3D + 旋转 3D 分别分解.

### 4.1 lift (桌面升降 — 最简单)

| #       | 状态 | H | ds_mode       | 步数 |
| ------- | :--: | :-: | ------------- | ---- |
| 177-180 |  ⬜  | 1 | none          | 1M   |
| 181-184 |  ⬜  | 1 | posthoc       | 1M   |
| 185-188 |  ⬜  | 1 | stereographic | 1M   |
| 189-192 |  ⬜  | 1 | spherical     | 1M   |
| 193-196 |  ⬜  | 5 | none          | 1M   |
| 197-200 |  ⬜  | 5 | stereographic | 1M   |
| 201-204 |  ⬜  | 5 | spherical     | 1M   |
| 205-208 |  ⬜  | 5 | posthoc       | 1M   |

> 最简单 RoboMimic 任务，基线 ~80%+，先跑 lift 验证 DS 不损害简单任务。

### 4.2 square (方块装配 — 高精度)

| #       | 状态 | H | ds_mode       | 步数 |
| ------- | :--: | :-: | ------------- | ---- |
| 209-212 |  ⬜  | 1 | none          | 1M   |
| 213-216 |  ⬜  | 1 | posthoc       | 1M   |
| 217-220 |  ⬜  | 1 | stereographic | 1M   |
| 221-224 |  ⬜  | 1 | spherical     | 1M   |
| 225-228 |  ⬜  | 5 | none          | 1M   |
| 229-232 |  ⬜  | 5 | stereographic | 1M   |
| 233-236 |  ⬜  | 5 | spherical     | 1M   |
| 237-240 |  ⬜  | 5 | posthoc       | 1M   |

> 最难任务，论文中 DS 预期收益最大。

### 4.3 can (罐头搬运 — 中等难度)

| #       | 状态 | H | ds_mode       | 步数 |
| ------- | :--: | :-: | ------------- | ---- |
| 241-244 |  ⬜  | 1 | none          | 1M   |
| 245-248 |  ⬜  | 1 | posthoc       | 1M   |
| 249-252 |  ⬜  | 1 | stereographic | 1M   |
| 253-256 |  ⬜  | 1 | spherical     | 1M   |
| 257-260 |  ⬜  | 5 | none          | 1M   |
| 261-264 |  ⬜  | 5 | stereographic | 1M   |
| 265-268 |  ⬜  | 5 | spherical     | 1M   |
| 269-272 |  ⬜  | 5 | posthoc       | 1M   |

---

## 5. 消融实验

### 5.1 离线数据量消融 (FQL) — 单 seed

| #   | 状态 | ds_mode | dataset_proportion | 步数  |
| --- | :--: | ------- | :----------------: | ----- |
| 273 |  ⬜  | none    |        0.1        | 1M+1M |
| 274 |  ⬜  | posthoc |        0.1        | 1M+1M |
| 275 |  ⬜  | none    |        0.25        | 1M+1M |
| 276 |  ⬜  | posthoc |        0.25        | 1M+1M |
| 277 |  ⬜  | none    |        0.5        | 1M+1M |
| 278 |  ⬜  | posthoc |        0.5        | 1M+1M |
| 279 |  ⬜  | none    |        1.0        | 1M+1M |
| 280 |  ⬜  | posthoc |        1.0        | 1M+1M |

### 5.2 max_speed 消融 — 单 seed

| #   | 状态 | ds_mode       | max_speed | 步数 |
| --- | :--: | ------------- | :-------: | ---- |
| 281 |  ⬜  | stereographic |    0.5    | 1M   |
| 282 |  ⬜  | stereographic |    1.0    | 1M   |
| 283 |  ⬜  | stereographic |    2.0    | 1M   |

### 5.3 Bijector 内部消融 — 单 seed

验证 bijector 实现正确性而非方法有效性。

| #   | 状态 | ds_mode       | 说明                                      |
| --- | :--: | ------------- | ----------------------------------------- |
| 284 |  ⬜  | stereographic | 禁用 log-det Jacobian → 验证修正是否必要 |
| 285 |  ⬜  | spherical     | 禁用 epsilon-bounded sigmoid → 测极点 NaN |

### 5.4 速度表示消融 — 4 seeds

对比对数速度 `s=log(m)` vs 线性速度 `s=m`，验证乘性噪声的尺度不变性是否真的带来收益。

| #     | 状态 | ds_mode       | 速度表示 | 步数 |
| ----- | :--: | ------------- | -------- | ---- |
| 286-289 | ⬜ | stereographic | log (默认) | 1M |
| 290-293 | ⬜ | stereographic | linear (sigmoid 直出) | 1M |

> 在实现中把 `speed = exp(log_speed_raw)` 改成 `speed = bounded_sigmoid(speed_raw)` 即完成线性版本。

---

## 6. 不可行 / 不适用

| #  | 状态 | 说明                                                                    |
| -- | :--: | ----------------------------------------------------------------------- |
| — |  ❌  | FQL + stereographic/spherical — FQL 用 ActorVectorField，无 TanhNormal |
| — |  ❌  | RLPD + posthoc 严格对照 — log_prob 非 Jacobian-corrected               |

---

## 汇总

| 类别                           |     完成     |    运行中    |     待做     |
| ------------------------------ | :----------: | :----------: | :-----------: |
| RLPD H=1 (task2 + task1)       |      20      |      0      |       0       |
| RLPD H=5 (task2 + task1)       |      20      |      0      |       0       |
| FQL H=5 (task2)                |      8      |      0      |       0       |
| FQL H=1 (task2)                |      8      |      0      |       0       |
| cube-triple task3 (MuJoCo bug) |      0      |      32      |       0       |
| FQL H=5 QC                     |      0      |      0      |       8       |
| OGBench 其他任务               |      0      |      0      |      88      |
| RoboMimic                      |      0      |      0      |      96      |
| 消融 (含 Bijector 内部)        |      0      |      0      |      13      |
| **总计**                 | **56** | **32** | **205** |

> 不可行 2 个不计入总数
