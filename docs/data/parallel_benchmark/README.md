# 并行压力测试数据

每 GPU 单独保存 `results.json`。

## 文件

| 文件 | 内容 |
|------|------|
| `RTX4090_24GB/results.json` | 4090 压力测试（1–12 并发，5K 步/实验） |
| `RTX6000Ada_48GB/results.json` | 6000 Ada 压力测试（1–10 并发，20K 步/实验） |

## results.json 格式

```json
{
  "gpu": "NVIDIA GeForce RTX 4090",
  "vram": "24GB",
  "single_baseline_it_s": 228,
  "mem_fraction": 0.05,
  "steps_per_test": 5000,
  "cores_total": 32,
  "results": [
    {
      "concurrency": 1,
      "cores_per": 32,
      "total_it_s": 228,
      "scale": 1.0,
      "per_proc": 228,
      "gpu_util": "33%",
      "vram_used": "2GB",
      "status": "OK"
    }
  ]
}
```

### 字段

| 字段 | 含义 |
|------|------|
| `concurrency` | 并行实验数 |
| `cores_per` | 每进程 CPU 核数 |
| `total_it_s` | 全进程合计吞吐（it/s） |
| `scale` | 相对单进程的倍率 |
| `per_proc` | 每进程平均吞吐 |
| `gpu_util` | GPU 利用率（nvidia-smi） |
| `vram_used` | 显存占用 |
| `status` | OK / CRASHED |

### raw_measurements 字段（results.json 内）

`raw_measurements` 按并发数分组，记录每个进程的实测吞吐：

```json
{
  "8": {
    "total_launched": 8,
    "completed": 8,
    "per_process_it_s": [66.2, 68.1, 67.5, 67.3, 59.4, 58.8, 58.6, 63.1],
    "total_it_s": 504.8,
    "mean_per_proc": 63.1
  }
}
```

| 字段 | 含义 |
|------|------|
| `total_launched` | 启动进程数 |
| `completed` | 成功完成数 |
| `per_process_it_s` | 每个进程的稳态吞吐 |
| `total_it_s` | 合计吞吐 |
| `mean_per_proc` | 进程平均吞吐 |

## 实验信息

`results.json` 顶层包含实验参数：

| 字段 | 4090 | 6000 Ada |
|------|:--:|:--:|
| `experiment` | 并行训练压力测试 | 并行训练压力测试 |
| `method` | RLPD (TanhNormal) | RLPD (TanhNormal) |
| `env` | cube-triple-play-singletask-task2-v0 | cube-triple-play-singletask-task2-v0 |
| `entry` | main_online.py | main_online.py |
| `horizon_length` | 1 | 1 |
| `action_chunking` | False | False |
| `online_steps` | 5000 | 20000 |
| `eval_interval` | 0 | 0 |
| `cores_total` | 32 | 32 |
