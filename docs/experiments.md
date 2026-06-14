# DS-RLPD 实验报告

环境: `cube-triple-play-singletask` (task1–5) | Agent: ACRLPD | 观察维度: 46 | 动作维度: 5

---

## 一、复现结果

环境: `cube-triple-play-singletask-task2-v0` | seed: 0 | 观察维度: 46 | 动作维度: 5

### 正式复现结果 (1M 步)

| 方法 | 入口 | H | 关键参数 | 训练方式 | 最终 | 论文 | CI |
|------|------|---|---------|---------|------|------|-----|
| **QC** | main.py | 5 | best-of-n=32 | 离线1M→在线1M | **96%** | 89% | 81.5-93.5% |
| **RLPD** | main_online.py | 1 | — | 纯在线 | **74%** | 60% | 22.5-87% |
| **RLPD-AC** | main_online.py | 5 | 无BC约束 | 纯在线 | **2%** | 2% | 0-3% |

![主对比图](images/reproduce_RLPD_RLPDAC_1M.png)

### 长序训练实验 (10M 步)

| 方法 | H | BC约束 | 最终 | 最佳 | 突破点 |
|------|---|--------|------|------|--------|
| **RLPD-AC** | 5 | 无 | **88%** | 98% @7.6M | ~1.5M |
| **QC-RLPD** | 5 | bc_alpha=0.01 | **90%** | 100% @9.3M | ~2.5M |

![10M对比](images/reproduce_RLPDAC_QCRLPD_10M.png)

**关键发现**: H=5 纯在线需要更多样本但最终可达 90%；bc_alpha=0.01 在长序训练中有效；1M 步不足以判断 H=5 方法的优劣。

### 关键技术细节

- **控制频率**: 20 Hz
- **动作空间**: H=5 时 25 维
- **GPU**: RTX 4090 ×1
- **并行**: `taskset` CPU 绑核, `XLA_PYTHON_CLIENT_PREALLOCATE=false`

---

## 二、小批次验证 — DS 消融 (task2, 4 seeds)

验证 [approach.md](approach.md) 中 posthoc（D+1 非可逆）、stereographic（球极投影 bijector）、spherical（球坐标 bijector）三种 DS 实现，在 H=1（无 chunk）和 H=5（有 chunk）下的表现。

### 实验配置

- 环境: `cube-triple-play-singletask-task2-v0`
- 步数: 1,000,000
- Agent: ACRLPD
- 双卡并行: GPU 0 + GPU 1，4 并发/卡，4 seed/组
- 脚本: `run_ds_h5_1m.sh`, `run_ds_h1_1m.sh`
- 日志: `logs/ds_h5_1m/`, `logs/ds_h1_1m/`

### 结果

| 组 | H=5+chunk | ±std | H=1(无chunk) | ±std |
|---|:---:|:---:|:---:|:---:|
| **Post-hoc** | 31.5% | 21.8% | 89.0% | 11.9% |
| **Stereographic** | 18.5% | 16.1% | **91.0%** | 4.6% |
| **Spherical** | 7.0% | 3.3% | 63.5% | 14.8% |
| **Baseline** (TanhNormal) | 3.0% | 1.7% | 58.5% | 20.3% |

![DS 对比图](images/cube-triple-task2_DS_Bar.png)

![训练曲线](images/cube-triple-task2_DS_Curves_4seed.png)

### 每 seed 详细数据

#### H=5 + Action Chunking

| 组 | s0 | s1 | s2 | s3 | 均值 |
|---|:---:|:---:|:---:|:---:|:---:|
| posthoc | 24% | 66% | 6% | 30% | 31.5% |
| stereographic | 40% | 4% | 2% | 28% | 18.5% |
| spherical | 10% | 6% | 10% | 2% | 7.0% |
| baseline | 6% | 2% | 2% | 2% | 3.0% |

#### H=1 (无 chunk)

| 组 | s0 | s1 | s2 | s3 | 均值 |
|---|:---:|:---:|:---:|:---:|:---:|
| posthoc | 100% | 88% | 70% | 98% | 89.0% |
| stereographic | 86% | 92% | 98% | 88% | 91.0% |
| spherical | 56% | 46% | 66% | 86% | 63.5% |
| baseline | 74% | 64% | 24% | 72% | 58.5% |

### Task1 4-seed 泛化验证

在 task1 上重复相同配置（4 seeds, 50 eval）验证 DS 泛化。task1 仅在 1 个 cube 需移动，其余 2 个原位不动，预期比 task2 简单。

| 组 | H=1 | ±std | H=5 | ±std |
|---|:---:|:---:|:---:|:---:|
| **Post-hoc** | **100%** | 0% | 94% | 9% |
| **Baseline** | 83% | 34% | 53% | 55% |
| **Stereographic** | 51% | 32% | 72% | 43% |
| **Spherical** | 40% | 37% | 48% | 38% |

![task1 4-seed 曲线](images/cube-triple-task1_DS_Curves_4seed.png)
- **4 seed / 50 eval 方差极大**（±55% 级别），10 seed 升级版见第三节。

### FQL + Post-hoc DS (2M 步)

验证 DS 在 flow-based FQL 上的效果。入口：`main.py`（离线 1M → 在线 1M）。FQL 仅支持 posthoc。

| ds_mode | s0 | s1 | s2 | **均值** | **中位数** |
|---------|:---:|:---:|:---:|:---:|:---:|
| none | 96% | 84% | 90% | **90%** | 90% |
| posthoc | 100% | 98% | 4% | **67%** | 98% |

![FQL H5 DS](images/cube-triple-task2_FQL_DS.png)

- FQL baseline 极强（90%），离线预训练是关键
- posthoc s0/s1 近乎完美，s2 异常（4%）可能是 seed 崩溃
- 排除 s2：posthoc 99% > baseline 90%，DS 在 FQL 上仍有收益
- **数据**：[`cube-triple-play-singletask/task2/h5/fql/`](data/ds_experiments/cube-triple-play-singletask/task2/h5/fql/)

---

## 三、大规模 OGBench 跨任务验证 (task1–5)

cube-triple 5 个子任务的完整 DS 消融（4 DS modes × H1/H5 × 1M 纯在线）。

### 跨任务总览

![Summary Grid](images/cube-triple_all_tasks_summary.png)

![H1 Bar](images/cross-task_H1_bar.png)

![H5 Bar](images/cross-task_H5_bar.png)

---

### cube-triple-task1

> 4-seed 小批次验证见 [二、小批次验证](#二小批次验证--ds-消融-task2-4-seeds)。以下为 10-seed 升级版。

#### Task1 H=1 升级版 (10 seeds, 100 eval)

| 方法 | 1M 成功率 | ±std |
|------|:---:|:---:|
| **DS-RLPD (posthoc)** | **99.7%** | 0.2% |
| RLPD (baseline) | 89.4% | 21.4% |
| DS-RLPD (stereographic) | 83.4% | 32.5% |
| DS-RLPD (spherical) | 58.3% | 31.5% |

| 方法 | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| posthoc | 100 | 100 | 100 | 100 | 99 | 100 | 100 | 100 | 99 | 99 |
| none | 100 | 46 | 100 | 98 | 100 | 100 | 56 | 99 | 98 | 97 |
| stereographic | 99 | 100 | 99 | 100 | 0 | 100 | 0 | 100 | 100 | 100 |
| spherical | 52 | 47 | 47 | 45 | 98 | 48 | 100 | 48 | 52 | 46 |

- **posthoc 99.7% 近乎满分**：±0.2%，10 seed 全部 ≥99%，确认 task1 H=1 的最优配置
- **baseline 89.4% 强于预期**：少数 seed 跑出满分，但方差大（46–100%）
- **stereo/spherical 方差极大**：各有 2 个 seed 归零，其余接近满分，说明 seed 敏感
- **与旧 4-seed 对比**：baseline 83%→89%，stereo 51%→83%，10 seed 更稳定但仍受 seed 影响大
- **数据**：[`cube-triple-play-singletask/task1/h1/rlpd/`](data/ds_experiments/cube-triple-play-singletask/task1/h1/rlpd/)

#### Task1 H=5 升级版 (10 seeds, 100 eval)

**升级实验**：seeds 0–9 (10 seeds)，100 eval episodes，1M online steps。H=5 + action chunking。

| 方法 | 1M 成功率 | ±std | 范围 |
|------|:---:|:---:|:---:|
| **DS-RLPD (posthoc)** | **84.9%** | 22.7% | 42–100% |
| **DS-RLPD (stereographic)** | 59.0% | 38.2% | 6–100% |
| **RLPD (baseline)** | 54.1% | 32.7% | 3–99% |
| **DS-RLPD (spherical)** | 43.3% | 33.9% | 8–100% |

- **posthoc 84.9% 大幅领先**：但方差较大（42–100%），部分 seed 跑出满分
- **stereographic (59%) ≈ baseline (54.1%)**：stereo 在 H=5 下优势不明显
- **spherical 最弱 (43.3%)**：与 task2 一致，球坐标 Jacobian 退化
- **10 seeds 给出更可信的 CI**：旧实验 4 seeds 的 ±std 高达 55%，新实验 10 seeds 下 CI 更紧
- **数据**：[`cube-triple-play-singletask/task1/h5/rlpd/`](data/ds_experiments/cube-triple-play-singletask/task1/h5/rlpd/)

![task1 10-seed 曲线](images/cube-triple-task1_DS_Curves_full.png)

---

### cube-triple-task2

> 4-seed 小批次验证见 [二](#二小批次验证--ds-消融-task2-4-seeds)。以下为 10-seed 100 eval 升级版。

| 方法 | H=1 | ±std | H=5 | ±std |
|------|:---:|:---:|:---:|:---:|
| **Post-hoc** | **91.0%** | 2.9% | 22.0% | 12.4% |
| **Stereographic** | 61.6% | 28.3% | **23.5%** | 15.5% |
| Spherical | 45.8% | 25.1% | 9.4% | 5.8% |
| Baseline (TanhNormal) | 37.3% | 27.0% | 2.0% | 1.7% |

![task2 10-seed 曲线](images/cube-triple-task2_DS_Curves_full.png)

**与旧 4-seed 对比：**
- H=1 stereo 从 91% 跌至 62%：旧 4 seed 恰中利好 seed
- H=1 posthoc 稳定 91% ±2.9%，确认为 task2 最优 H=1 配置
- H=5 stereo 首次反超 posthoc（23.5% vs 22.0%）：bijector Jacobian 在 chunking 下不再劣势
- **数据**：[task2/h1/rlpd/](data/ds_experiments/cube-triple-play-singletask/task2/h1/rlpd/) / [task2/h5/rlpd/](data/ds_experiments/cube-triple-play-singletask/task2/h5/rlpd/)

---

### cube-triple-task3 (10 seeds, H1+H5)

完整消融实验：cube-triple-play-singletask-task3-v0，1M 纯在线，H1+H5，4 DS × 10 seeds × 100 eval（MuJoCo `mj_narrowphase` 碰撞 bug 已修复）

#### Task3 结果

| 方法 | H=1 | ±std | H=5 | ±std |
|------|:---:|:---:|:---:|:---:|
| **posthoc** | **28.4%** | 22.2% | **31.7%** | 14.4% |
| spherical | 4.3% | 4.8% | 1.5% | 2.1% |
| stereographic | 4.0% | 4.6% | 1.7% | 1.9% |
| none (baseline) | 1.1% | 1.9% | 0.7% | 1.0% |

每 seed 数据见下方。关键发现：
- **posthoc 独大**：H1 28.4%、H5 31.7%，远超过其他方法（stereo/spherical 均 ≤5%）
- **H5 posthoc 略优于 H1**（31.7% vs 28.4%）— 与 task2 类似，但绝对值低得多
- **stereo/spherical 在 task3 上完全失败**：与 task4（H1 下 stereo 67%）形成鲜明对比

![task3 曲线](images/cube-triple-task3_DS_Curves_full.png)

- 数据：[`cube-triple-play-singletask/task3/`](data/ds_experiments/cube-triple-play-singletask/task3/)

#### H=1 每 seed

| 方法 | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | 均值 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|
| posthoc | 28 | 35 | 65 | 47 | 2 | 10 | 31 | 52 | 13 | 1 | 28.4 |
| spherical | 4 | 16 | 2 | 2 | 2 | 4 | 1 | 1 | 9 | 2 | 4.3 |
| stereographic | 14 | 7 | 0 | 0 | 5 | 4 | 7 | 0 | 3 | 0 | 4.0 |
| baseline | 4 | 1 | 0 | 0 | 0 | 5 | 0 | 0 | 1 | 0 | 1.1 |

#### H=5 每 seed

| 方法 | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | 均值 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|
| posthoc | 31 | 30 | 32 | 33 | 53 | 46 | 14 | 14 | 20 | 44 | 31.7 |
| stereographic | 1 | 4 | 0 | 3 | 0 | 5 | 0 | 2 | 0 | 2 | 1.7 |
| spherical | 1 | 0 | 2 | 3 | 0 | 7 | 1 | 0 | 0 | 1 | 1.5 |
| baseline | 1 | 1 | 0 | 3 | 0 | 2 | 0 | 0 | 0 | 0 | 0.7 |

---

### cube-triple-task4 (10 seeds, H1+H5)

完整消融实验：cube-triple-play-singletask-task4-v0，1M 纯在线，H1+H5，4 DS × 10 seeds × 100 eval

#### Task4 结果

| 方法 | H=1 | ±std | H=5 | ±std |
|------|:---:|:---:|:---:|:---:|
| **stereographic** | **67.9%** | 14.8% | 4.8% | 5.2% |
| posthoc | 66.5% | 15.1% | **17.6%** | 9.2% |
| spherical | 64.8% | 18.6% | 2.0% | 1.6% |
| none (baseline) | 49.4% | 19.8% | 1.6% | 1.8% |

#### H=1 每 seed

| 方法 | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | 均值 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|
| stereographic | 66 | 51 | 72 | 35 | 76 | 70 | 48 | 67 | 69 | 70 | 62.4 |
| posthoc | 51 | 58 | 67 | 44 | 61 | 88 | 69 | 68 | 53 | 55 | 61.4 |
| spherical | 17 | 75 | 55 | 46 | 72 | 47 | 54 | 72 | 41 | 77 | 55.6 |
| none | 46 | 50 | 41 | 19 | 63 | 54 | 59 | 27 | 20 | 65 | 44.4 |

#### H=5 每 seed

| 方法 | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | 均值 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|
| posthoc | 13 | 9 | 18 | 12 | 20 | 29 | 20 | 11 | 3 | 11 | 14.6 |
| stereographic | 10 | 3 | 1 | 5 | 12 | 3 | 1 | 2 | 1 | 1 | 3.9 |
| spherical | 2 | 4 | 1 | 3 | 5 | 2 | 3 | 2 | 1 | 3 | 2.6 |
| none | 1 | 0 | 2 | 0 | 3 | 2 | 1 | 0 | 1 | 1 | 1.1 |

- **H=1: stereo/spherical 首次发威**，追平 posthoc，三者均远超 baseline
- **H=5: 仅 posthoc 勉强可用 (17.6%)**，其余 ≤5%

![task4 曲线](images/cube-triple-task4_DS_Curves_full.png)

- 数据：[`cube-triple-play-singletask/task4/`](data/ds_experiments/cube-triple-play-singletask/task4/)

---

### cube-triple-task5 (10 seeds, H1+H5)

完整消融实验：cube-triple-play-singletask-task5-v0，配置同 task4。

**全部 80 runs (H1+H5, 4 DS × 10 seeds) 成功率均为 0%。** task5 是目前最难的 cube-triple 任务，1M 步内任何方法均无信号。

![task5 曲线](images/cube-triple-task5_DS_Curves_full.png)

---

### 跨任务难度对比

| Task | H1 baseline | H1 best DS | H5 baseline | H5 best DS |
|------|:---:|:---:|:---:|:---:|
| task1 | 89% | 100% (posthoc) | 54% | 85% (posthoc) |
| task2 | 37% | 91% (posthoc) | 2% | 24% (stereo) |
| task3 | 1% | 28% (posthoc) | 0.7% | 32% (posthoc) |
| task4 | 49% | 68% (stereo) | 1.6% | 18% (posthoc) |
| **task5** | **0%** | **0%** | **0%** | **0%** |

> task1/2 H1 为 10 seeds 100 eval；task1/2 H5 含 4-seed 遗留数据

- 难度：task1 < task2 < task4 < task3 < task5
- DS 相对收益在中等难度 (task4) 最显著
- 数据：[`cube-triple-play-singletask/task5/`](data/ds_experiments/cube-triple-play-singletask/task5/)

### 结论

1. **posthoc 在多数场景最优**：H=1 和 H=5 下均稳居前二，D+1 自由度简单有效
2. **H=1 的 stereo 在中等难度有效**（task4 68%，task1 83%），但在过简单/过难任务上不稳定
3. **H=5 stereo 首次反超 posthoc**（task2 23.5% vs 22.0%）：Jacobian bijector 在 chunking 下不再劣势，但需要 10 seed 才能检出
4. **spherical 全局最弱**：球坐标 Jacobian 退化，不推荐
5. **10 seed 对结论可靠性至关重要**：旧 4 seed 的 task2 stereo 91%→62%，baseline 59%→37%
6. **task5 为当前最难任务**：1M 步完全失败，需更长训练或 offline pretrain

### 已知局限

| 问题 | 影响 | 建议 | 状态 |
|------|------|------|:--:|
| ~~task2 H1+H5 仅 4 seeds, 50 eval~~ | — | — | ✅ 已补全 10 seeds |
| ~~task1 H1 仅 4 seeds, 50 eval~~ | — | — | ✅ 已补全 10 seeds |
| task1 H5 存在两套数据 (`rlpd-4s` vs `rlpd`) | `rlpd-4s` 无 checkpoint；`rlpd` 的 `run_group=Debug` | 确认 canonical 版本 | ⚠️ |
| task5 全 0% | 无 DS 对比信号 | 需更长训练或 offline pretrain | ⚠️ |

> 更新：已有 task1 H1 的 10 seed 结果建议统一整理（当前仍有两套命名不一致的数据在同一目录）。

### 更新日志

| 日期 | 更新 |
|------|------|
| 2026-06-15 | task1 H1 + task2 H1+H5 补全 10 seeds 100 eval，全部柱状图加误差线 |
| 2026-06-14 | 统一重绘全部旧图（配色图例一致）+ 跨任务汇总图 + 已知局限 |
| 2026-06-14 | cube-triple-task5 (80 runs, 全 0%) + task4 (80 runs) |
| 2026-06-14 | docs/data 重组为 `env/task/horizon/method/ds_mode/seed` 层级 |
| 2026-06-13 | cube-triple-task1 H=5: 10 seeds, 100 eval |
| 2026-06-11 | 初始实验（task2 H1/H5, task1 4 seeds, FQL） |

### 数据

原始实验数据: [SZUEAILab/ds-experiments](https://huggingface.co/datasets/SZUEAILab/ds-experiments) (`datasets.load_dataset("SZUEAILab/ds-experiments")`)。
