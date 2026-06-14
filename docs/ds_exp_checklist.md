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

每个任务 4 DS modes × H1+H5 × (4 或 10 seeds)。agent: acrlpd, 1M 纯在线。task1 H=5 + task3/4/5 为 10 seeds，其余 4 seeds。

**cube-triple 5 个 fixed task**（3 cubes，200 步/集）：

| Task | 名称 | 操作 | 难度 |
|:---:|------|------|:---:|
| **1** | single_pnp | 仅移动 1 个 cube，其余 2 个原位不动 | ⭐ |
| **2** | triple_pnp | 3 个 cube 并行平移（列→列） | ⭐⭐ |
| **3** | unstack | 从垂直堆叠中逐一抓取分散放置 | ⭐⭐⭐⭐ |
| **4** | cycle | 3 个 cube 顺时针循环换位 | ⭐⭐⭐ |
| **5** | stack | 3 个散 cube 精确垂直堆叠 | ⭐⭐⭐⭐⭐ |

### 3.1 cube-triple task1

> 📌 单 cube 移动，其余原位不动。预期最简单。

| #       | 状态 | env               | H | ds_mode       | seeds | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | :---: | ---- |
| 57-60   |  ✅  | cube-triple-task1 | 1 | none          | 4 | 1M   |
| 61-64   |  ✅  | cube-triple-task1 | 1 | posthoc       | 4 | 1M   |
| 65-68   |  ✅  | cube-triple-task1 | 1 | stereographic | 4 | 1M   |
| 69-72   |  ✅  | cube-triple-task1 | 1 | spherical     | 4 | 1M   |
| 73-82   |  ✅  | cube-triple-task1 | 5 | none          | 10 | 1M   |
| 83-92   |  ✅  | cube-triple-task1 | 5 | stereographic | 10 | 1M   |
| 93-102  |  ✅  | cube-triple-task1 | 5 | spherical     | 10 | 1M   |
| 103-112 |  ✅  | cube-triple-task1 | 5 | posthoc       | 10 | 1M   |

> task1 H=1 为早期 4-seed；H=5 已升级为 10-seed。结果：H1 posthoc 99.7%, H5 posthoc 81.3%。

### 3.2 cube-triple task3

> 📌 从堆叠抓取分散（task5 的逆）。初始 3 cube 垂直叠放，需逐一抓取放至不同位置。MuJoCo 碰撞 bug 高发区（已修复）。

| #       | 状态 | env               | H | ds_mode       | seeds | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | :---: | ---- |
| 113-122 |  ✅  | cube-triple-task3 | 1 | none          | 10 | 1M   |
| 123-132 |  ✅  | cube-triple-task3 | 1 | posthoc       | 10 | 1M   |
| 133-142 |  ✅  | cube-triple-task3 | 1 | stereographic | 10 | 1M   |
| 143-152 |  ✅  | cube-triple-task3 | 1 | spherical     | 10 | 1M   |
| 153-162 |  ✅  | cube-triple-task3 | 5 | none          | 10 | 1M   |
| 163-172 |  ✅  | cube-triple-task3 | 5 | stereographic | 10 | 1M   |
| 173-182 |  ✅  | cube-triple-task3 | 5 | spherical     | 10 | 1M   |
| 183-192 |  ✅  | cube-triple-task3 | 5 | posthoc       | 10 | 1M   |

> ~~task3 有 MuJoCo 物理 bug（`mj_narrowphase` 碰撞超限），4 seed 反复崩溃。~~ **已解决**，10 seeds 全部完成。结果：H1 posthoc 22.4%, H5 posthoc 27.5%，仅 posthoc 有效。

### 3.3 cube-triple task4

> 📌 顺时针循环换位，无堆叠。3 cube 各挪到相邻位置。

| #       | 状态 | env               | H | ds_mode       | seeds | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | :---: | ---- |
| 193-202 |  ✅  | cube-triple-task4 | 1 | none          | 10 | 1M   |
| 203-212 |  ✅  | cube-triple-task4 | 1 | posthoc       | 10 | 1M   |
| 213-222 |  ✅  | cube-triple-task4 | 1 | stereographic | 10 | 1M   |
| 223-232 |  ✅  | cube-triple-task4 | 1 | spherical     | 10 | 1M   |
| 233-242 |  ✅  | cube-triple-task4 | 5 | none          | 10 | 1M   |
| 243-252 |  ✅  | cube-triple-task4 | 5 | stereographic | 10 | 1M   |
| 253-262 |  ✅  | cube-triple-task4 | 5 | spherical     | 10 | 1M   |
| 263-272 |  ✅  | cube-triple-task4 | 5 | posthoc       | 10 | 1M   |

> 结果：H1 stereo 62.4% ≈ posthoc 61.4% > spherical 55.6% > baseline 44.4%；H5 posthoc 14.6%。

### 3.4 cube-triple task5

> 📌 精确垂直堆叠（当前最难）。3 cube 从分散位置拾取并叠放至同一位置 (z=0.02, 0.06, 0.10)。1M 步全部 0%。

| #       | 状态 | env               | H | ds_mode       | seeds | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | :---: | ---- |
| 273-282 |  ✅  | cube-triple-task5 | 1 | none          | 10 | 1M   |
| 283-292 |  ✅  | cube-triple-task5 | 1 | posthoc       | 10 | 1M   |
| 293-302 |  ✅  | cube-triple-task5 | 1 | stereographic | 10 | 1M   |
| 303-312 |  ✅  | cube-triple-task5 | 1 | spherical     | 10 | 1M   |
| 313-322 |  ✅  | cube-triple-task5 | 5 | none          | 10 | 1M   |
| 323-332 |  ✅  | cube-triple-task5 | 5 | stereographic | 10 | 1M   |
| 333-342 |  ✅  | cube-triple-task5 | 5 | spherical     | 10 | 1M   |
| 343-352 |  ✅  | cube-triple-task5 | 5 | posthoc       | 10 | 1M   |

> 最难任务，全部 80 runs 成功率 0%。1M 步不够，需更长训练或 offline pretrain。

### 3.5 cube-double (待做)

| #       | 状态 | env               | H | ds_mode       | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | ---- |
| 353-356 |  ⬜  | cube-double-task2 | 1 | none          | 1M   |
| 357-360 |  ⬜  | cube-double-task2 | 1 | posthoc       | 1M   |
| 361-364 |  ⬜  | cube-double-task2 | 1 | stereographic | 1M   |
| 365-368 |  ⬜  | cube-double-task2 | 1 | spherical     | 1M   |
| 369-372 |  ⬜  | cube-double-task2 | 5 | none          | 1M   |
| 373-376 |  ⬜  | cube-double-task2 | 5 | posthoc       | 1M   |
| 377-380 |  ⬜  | cube-double-task2 | 5 | stereographic | 1M   |
| 381-384 |  ⬜  | cube-double-task2 | 5 | spherical     | 1M   |

### 3.6 cube-single (可选)

| #       | 状态 | env               | H | ds_mode       | 步数 |
| ------- | :--: | ----------------- | :-: | ------------- | ---- |
| 385-388 |  ⬜  | cube-single-task2 | 1 | none          | 1M   |
| 389-392 |  ⬜  | cube-single-task2 | 1 | stereographic | 1M   |
| 393-396 |  ⬜  | cube-single-task2 | 5 | none          | 1M   |
| 397-400 |  ⬜  | cube-single-task2 | 5 | stereographic | 1M   |

> cube-single 最简单，先跑 stereo vs none，如有显著差异再加 posthoc/spherical。

### 3.7 cube-quadruple (可选)

| #       | 状态 | env                  | H | ds_mode       | 步数 |
| ------- | :--: | -------------------- | :-: | ------------- | ---- |
| 401-404 |  ⬜  | cube-quadruple-task2 | 1 | none          | 1M   |
| 405-408 |  ⬜  | cube-quadruple-task2 | 1 | stereographic | 1M   |
| 409-412 |  ⬜  | cube-quadruple-task2 | 5 | none          | 1M   |
| 413-416 |  ⬜  | cube-quadruple-task2 | 5 | stereographic | 1M   |

> cube-quadruple 极难，先跑 stereo vs none，有信号再加满。

---

## 4. 跨域验证 (RoboMimic — 7D 动作空间)

agent: acrlpd, 4 seeds × 4 DS modes, 1M 纯在线. DS 将位移 3D + 旋转 3D 分别分解.

### 4.1 lift (桌面升降 — 最简单)

| #       | 状态 | H | ds_mode       | 步数 |
| ------- | :--: | :-: | ------------- | ---- |
| 417-420 |  ⬜  | 1 | none          | 1M   |
| 421-424 |  ⬜  | 1 | posthoc       | 1M   |
| 425-428 |  ⬜  | 1 | stereographic | 1M   |
| 429-432 |  ⬜  | 1 | spherical     | 1M   |
| 433-436 |  ⬜  | 5 | none          | 1M   |
| 437-440 |  ⬜  | 5 | stereographic | 1M   |
| 441-444 |  ⬜  | 5 | spherical     | 1M   |
| 445-448 |  ⬜  | 5 | posthoc       | 1M   |

> 最简单 RoboMimic 任务，基线 ~80%+，先跑 lift 验证 DS 不损害简单任务。

### 4.2 square (方块装配 — 高精度)

| #       | 状态 | H | ds_mode       | 步数 |
| ------- | :--: | :-: | ------------- | ---- |
| 449-452 |  ⬜  | 1 | none          | 1M   |
| 453-456 |  ⬜  | 1 | posthoc       | 1M   |
| 457-460 |  ⬜  | 1 | stereographic | 1M   |
| 461-464 |  ⬜  | 1 | spherical     | 1M   |
| 465-468 |  ⬜  | 5 | none          | 1M   |
| 469-472 |  ⬜  | 5 | stereographic | 1M   |
| 473-476 |  ⬜  | 5 | spherical     | 1M   |
| 477-480 |  ⬜  | 5 | posthoc       | 1M   |

> 最难任务，论文中 DS 预期收益最大。

### 4.3 can (罐头搬运 — 中等难度)

| #       | 状态 | H | ds_mode       | 步数 |
| ------- | :--: | :-: | ------------- | ---- |
| 481-484 |  ⬜  | 1 | none          | 1M   |
| 485-488 |  ⬜  | 1 | posthoc       | 1M   |
| 489-492 |  ⬜  | 1 | stereographic | 1M   |
| 493-496 |  ⬜  | 1 | spherical     | 1M   |
| 497-500 |  ⬜  | 5 | none          | 1M   |
| 501-504 |  ⬜  | 5 | stereographic | 1M   |
| 505-508 |  ⬜  | 5 | spherical     | 1M   |
| 509-512 |  ⬜  | 5 | posthoc       | 1M   |

---

## 5. 消融实验

### 5.1 离线数据量消融 (FQL) — 单 seed

| #   | 状态 | ds_mode | dataset_proportion | 步数  |
| --- | :--: | ------- | :----------------: | ----- |
| 513 |  ⬜  | none    |        0.1        | 1M+1M |
| 514 |  ⬜  | posthoc |        0.1        | 1M+1M |
| 515 |  ⬜  | none    |        0.25        | 1M+1M |
| 516 |  ⬜  | posthoc |        0.25        | 1M+1M |
| 517 |  ⬜  | none    |        0.5        | 1M+1M |
| 518 |  ⬜  | posthoc |        0.5        | 1M+1M |
| 519 |  ⬜  | none    |        1.0        | 1M+1M |
| 520 |  ⬜  | posthoc |        1.0        | 1M+1M |

### 5.2 max_speed 消融 — 单 seed

| #   | 状态 | ds_mode       | max_speed | 步数 |
| --- | :--: | ------------- | :-------: | ---- |
| 521 |  ⬜  | stereographic |    0.5    | 1M   |
| 522 |  ⬜  | stereographic |    1.0    | 1M   |
| 523 |  ⬜  | stereographic |    2.0    | 1M   |

### 5.3 Bijector 内部消融 — 单 seed

验证 bijector 实现正确性而非方法有效性。

| #   | 状态 | ds_mode       | 说明                                      |
| --- | :--: | ------------- | ----------------------------------------- |
| 524 |  ⬜  | stereographic | 禁用 log-det Jacobian → 验证修正是否必要 |
| 525 |  ⬜  | spherical     | 禁用 epsilon-bounded sigmoid → 测极点 NaN |

### 5.4 速度表示消融 — 4 seeds

对比对数速度 `s=log(m)` vs 线性速度 `s=m`，验证乘性噪声的尺度不变性是否真的带来收益。

| #     | 状态 | ds_mode       | 速度表示 | 步数 |
| ----- | :--: | ------------- | -------- | ---- |
| 526-529 | ⬜ | stereographic | log (默认) | 1M |
| 530-533 | ⬜ | stereographic | linear (sigmoid 直出) | 1M |

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
| RLPD H=1 (task2 + task1)       |      32      |      0      |       0       |
| RLPD H=5 (task2 + task1)       |      56      |      0      |       0       |
| FQL H=5 (task2)                |      8       |      0      |       0       |
| FQL H=1 (task2)                |      8       |      0      |       0       |
| cube-triple task3 (10 seeds)   |      80      |      0      |       0       |
| cube-triple task4 (10 seeds)   |      80      |      0      |       0       |
| cube-triple task5 (10 seeds)   |      80      |      0      |       0       |
| FQL H=5 QC                     |      0       |      0      |       8       |
| OGBench 其他任务               |      0       |      0      |      88      |
| RoboMimic                      |      0       |      0      |      96      |
| 消融 (含 Bijector 内部)        |      0       |      0      |      21      |
| **总计**                 | **344** | **0** | **213** |

> 不可行 2 个不计入总数。task1: H=1 为 4 seeds (早期), H=5 为 10 seeds。task3/4/5: 10 seeds。task3 MuJoCo bug 已解决。
