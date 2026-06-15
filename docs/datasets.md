# 数据集安装与使用

## OGBench

OGBench (Offline Goal-Conditioned RL Benchmark) 是 MuJoCo 机械臂操作基准，基于 UR5e + 2F-85 夹爪。

- 动作: 5D `[dx, dy, dz, dyaw, grip]`
- 观测: 46D (MuJoCo proprioceptive state)
- 格式: `.npz` 单文件，已预处理好

### 安装

```bash
pip install ogbench
```

数据集在首次使用时**自动下载**到 `~/.ogbench/datasets/`：

```python
import ogbench
ogbench.download_datasets(['cube-triple-play-singletask-task2-v0'])
```

### 任务命名

```
{task}-{mode}-singletask{ -taskN}-v0
```

| 任务族 | 方块数 | play 示例 | singletask 示例 |
|--------|:---:|------|------|
| cube-single | 1 | `cube-single-v0` | `cube-single-singletask-task2-v0` |
| cube-double | 2 | `cube-double-v0` | `cube-double-singletask-task2-v0` |
| cube-triple | 3 | `cube-triple-play-singletask-task2-v0` | — |
| cube-quadruple | 4 | `cube-quadruple-v0` | — |
| cube-octuple | 8 | `cube-octuple-v0` | — |
| scene | — | `scene-v0` | — |
| puzzle | 3×3~4×6 | `puzzle-3x3-v0` | — |

`play` = 多任务版（不同 episode 不同目标）。`singletask` = 固定目标，`task1~5` 对应 5 种不同目标位置。还有 `visual-*` 图像版。

### 实验中配置

```jsonc
{ "env_name": "cube-triple-play-singletask-task2-v0" }
```

可选: `"ogbench_dataset_dir": "/custom/path/"` 覆盖数据集路径。

### 数据格式

```python
np.load('task.npz')  # → {observations, actions, rewards, terminals}
# terminals: [0,0,0,1, 0,0,0,1, ...] 标记 episode 边界
# 加载时自动生成 next_observations 和 masks
```

---

## RoboMimic

RoboMimic 是 robosuite 机械臂操作基准，Franka Panda 机器人。

- 动作: 7D `[dx, dy, dz, droll, dpitch, dyaw, grip]`
- 观测: 25D (eef_pos + eef_quat + gripper_qpos + object)
- 格式: `.hdf5`，按 demo 分层

### 安装

```bash
# RoboMimic 已在依赖中
pip install robomimic

# 数据集需手动下载到 ~/.robomimic/
# 路径: ~/.robomimic/{task}/{type}/low_dim_v15.hdf5
```

**数据集路径是硬编码的**——没有 CLI flag。文件必须在 `~/.robomimic/{task}/{type}/low_dim_v15.hdf5`，否则报错。

### 任务命名

```
{task}-{type}-low_dim
```

| task | 描述 | ep 长度 | 难度 |
|------|------|:---:|:---:|
| `lift` | 桌面升降 | 300 | ⭐ |
| `can` | 罐头搬运 | 300 | ⭐⭐ |
| `square` | 方块装配 | 400 | ⭐⭐⭐ |
| `transport` | 双臂运输 | 800 | ⭐⭐⭐⭐ |
| `tool_hang` | 工具挂架 | 1000 | ⭐⭐⭐⭐⭐ |

`{type}`: `ph` = proficient human, `mh` = multi human。

### 实验中配置

```jsonc
{ "env_name": "lift-ph-low_dim" }
```

无需额外 flag。数据集路径不可配。

### 数据格式

```python
h5py.File('low_dim_v15.hdf5')  # → data/demo_0/{actions, obs/{robot0_eef_pos,...}, rewards, dones}
# obs keys: robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, object
# 加载时拼接为 25D，生成 next_observations 和 masks
```

---

## LIBERO

LIBERO (Lifelong Robot Learning Benchmark) 是 robosuite 机器人操作基准，130 个任务（4 suites），7-DoF Franka Panda。

- 动作: 7D `[dx, dy, dz, droll, dpitch, dyaw, grip]`
- 观测: 120D (`robot0_proprio-state` 50D + `object-state` 70D)
- 格式: `.hdf5`，每 task 50 human demos

### 安装

```bash
# 1. clone + pip install
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e .

# 2. 首次 import 时按提示输入数据集路径，或预先创建 ~/.libero/config.yaml
# 3. 从 HuggingFace 下载
python -c "
from libero.libero.utils.download_utils import libero_dataset_download
for ds in ['libero_spatial', 'libero_object', 'libero_goal']:
    libero_dataset_download(datasets=ds, download_dir='<数据集目录>', use_huggingface=True)
"
```

数据来源: `yifengzhu-hf/LIBERO-datasets`（LIBERO 作者 Yifeng Zhu 的 HF 镜像，内容与官方 Box.com 一致）。`<数据集目录>` 改为实际路径。

### 任务命名

```
libero_{suite}/{index}
```

| suite | 任务数 | 描述 |
|-------|:---:|------|
| `libero_spatial` | 10 | 空间关系（碗+盘子+烤杯） |
| `libero_object` | 10 | 不同物体（罐头、牛奶、黄油） |
| `libero_goal` | 10 | 目标条件（抽屉、炉灶、柜子） |

`{index}` 对应 suite 内第 i 个任务（0-based）。完整任务列表见 LIBERO 仓库 `libero/libero/benchmark/libero_suite_task_map.py`。

### 实验中配置

```jsonc
{
  "common": { "online_steps": 1000000, "horizon_length": 1, "entry": "main_online.py" },
  "tasks": [
    {
      "name": "libero_posthoc",
      "env_name": "libero_spatial/0",
      "ds_mode": "posthoc",
      "allow_posthoc_direction_speed_rlpd": true
    }
  ]
}
```

### 数据格式

```python
h5py.File('task_demo.hdf5')  # → data/demo_0/{actions, states, obs/{ee_pos,...}, rewards, dones}
# obs keys (低维): ee_pos, ee_ori, ee_states, gripper_states, joint_states
# states (92D): 完整 MuJoCo qpos
# 首次加载时从 states 重建 120D 观测（proprio-state + object-state），结果缓存
```


---

## 对比总结

| | OGBench | RoboMimic | LIBERO |
|---|---|---|---|
| **动作** | 5D | 7D | 7D |
| **观测** | 46D | 25D | 120D |
| **格式** | `.npz` | `.hdf5` | `.hdf5` |
| **自动下载** | ✅ | ❌ | ❌ (手动 HF) |
| **路径可配** | ✅ flag | ❌ 硬编码 | ⚠️ config.yaml |
| **加载模块** | `envs/ogbench_utils.py` | `envs/robomimic_utils.py` | `envs/libero_utils.py` |
