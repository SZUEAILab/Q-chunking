# 实验数据

数据已迁移至 HuggingFace Datasets：

**[SZUEAILab/ds-experiments](https://huggingface.co/datasets/SZUEAILab/ds-experiments)**

```python
from datasets import load_dataset
ds = load_dataset("SZUEAILab/ds-experiments")
```

## 目录结构

```
ds-experiments/
├── ds_experiments/     # DS 实验数据 (task1, task2 RLPD/FQL)
├── reproduce/          # QC 复现实验
├── parallel_benchmark/ # 并行训练压力测试
├── scripts/            # 实验脚本
└── images/             # 图表
```
