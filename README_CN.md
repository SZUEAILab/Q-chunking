<div align="center">

# [基于动作分块的强化学习](https://arxiv.org/abs/2507.07969)

## [[项目主页](https://colinqiyangli.github.io/qc/)]      [[论文PDF](https://arxiv.org/pdf/2507.07969)]

</div>

<p align="center">
  <a href="https://colinqiyangli.github.io/qc/">
    <img alt="示意图" src="./assets/teaser.png" width="48%">
  </a>
  <a href="https://colinqiyangli.github.io/qc/">
    <img alt="综合结果" src="./assets/agg.png" width="48%">
  </a>
</p>


## 概述
Q-chunking 在**时间扩展动作（动作分块）空间**上运行强化学习，并通过强表达力的行为约束来利用先验数据，从而改善探索能力和在线样本效率。

## 安装
```bash
pip install -r requirements.txt
```


## 数据集
对于 robomimic，我们假设数据集位于 `~/.robomimic/lift/mh/low_dim_v15.hdf5`、`~/.robomimic/can/mh/low_dim_v15.hdf5` 和 `~/.robomimic/square/mh/low_dim_v15.hdf5`。数据集可以从 https://robomimic.github.io/docs/datasets/robomimic_v0.1.html 下载（参见 Method 2: Using Direct Download Links - Multi-Human (MH)）。

对于 cube-quadruple，我们使用 100M 规模的离线数据集。可以从 https://github.com/seohongpark/horizon-reduction 下载：
```bash
wget -r -np -nH --cut-dirs=2 -A "*.npz" https://rail.eecs.berkeley.edu/datasets/ogbench/cube-quadruple-play-100m-v0/
```
并在命令行中添加 `--ogbench_dataset_dir=[你的/cube-quadruple-play-100m-v0/的真实路径]` 标志，以确保使用 100M 规模的数据集。

## 复现论文结果

我们在下方列出了论文中所有评估方法的示例命令。对于 `scene` 和 `puzzle-3x3` 领域，请使用 `--sparse=True`。我们还发布了绘图数据，详见 [plot_data/README.md](plot_data/README.md)。

```bash
# QC
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5

# BFN-n
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=4 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=False

# BFN
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=4 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=1

# QC-FQL
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5

# FQL-n
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=False

# FQL
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=1

# RLPD
MUJOCO_GL=egl python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=1 

# RLPD-AC
MUJOCO_GL=egl python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5

# QC-RLPD
MUJOCO_GL=egl python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.bc_alpha=0.01
```

## 引用
```
@inproceedings{
  li2025reinforcement,
  title={Reinforcement Learning with Action Chunking},
  author={Qiyang Li and Zhiyuan Zhou and Sergey Levine},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=XUks1Y96NR}
}
```

## 致谢
本代码库基于 [FQL](https://github.com/seohongpark/fql) 构建。两个 rlpd_* 文件夹直接取自 [RLPD](https://github.com/ikostrikov/rlpd)。
