# QC / Direction–Speed 工作说明

日常文档入口：[docs/index.md](docs/index.md)。只维护四份：
[研究](docs/current/research.md)、[实验](docs/current/experiments.md)、
[结果](docs/current/results.md)、[运行与存储](docs/current/runbook.md)。
单次实验进度写批次记录；历史协议在 docs/archive，不能直接当成当前执行指令。
Superpowers/SDD 已退役，不新建其目录或恢复旧监督链。

## 代码与实验

- main_online.py：RLPD 在线交互并混合先验数据；main.py：FQL/QC 离线到在线。
- agents/ 实现 learner，direction_speed/ 实现动作表示、分布、几何与执行处理。
- 核对实际执行工作树、有效配置、源码和数据身份；不能用当前 main 替代训练时源码。
- 同任务的方法匹配 seed、预算、数据与评估；表示、噪声和时间相关性分别消融。
- 评估 episodes 不是独立训练 seed；完整性通过不等于科学假设成立。
- reward-blind 阶段不读取 reward/success；零成功率和不利结果不触发重跑。
- 保留失败 attempt、incident 和谱系；恢复前核对来源、checkpoint 和预算。

## 执行与存储

- 本地两卡按每卡 6 并发；云端 RTX 4090 首先每卡 8 并发，错峰 JIT、动态回填。
- 编译配置并审查命令后启动；调度器重排队可能重启任务，异常恢复需要先诊断。
- 默认数据根 /data/chainpray/qc，可用 QC_DATA_ROOT 覆盖。
- results/ 保存原始运行；control/ 保存配置与验收；logs/ 保存日志；reports/ 保存分析。
- docs/images/ 保留并提交正文所需图片；不得因为数据迁移删除文档图片。
- docs/data 是 Hugging Face submodule，发布精简证据并固定 revision；默认不上传权重和逐步日志。
- 清理先列目标与引用关系，保留未解决 incident、论文依赖和必要模型；不自动删除恢复材料。

## 验证

文档：python scripts/check_docs.py。
代码按风险运行相关测试；CPU 测试可设置 JAX_PLATFORMS=cpu、CUDA_VISIBLE_DEVICES=''。
完整规则和命令模板见运行手册，不在本文件重复维护。
