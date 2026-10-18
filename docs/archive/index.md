# 历史归档

[返回文档入口](../index.md)

需要追溯具体实验时再查本目录。原来的 50 份详细文档已合并或移为按需参考；各节保留
原文件名、日期、表格、图片和结论边界。历史记录中的“当前”“推荐”“待做”不覆盖日常规则。

| 合集 | 什么时候查 |
|---|---|
| [动作表示与参考测度](action_space.md) | 投影、critic 空间、坐标图、KL 预算、GAC 的具体协议 |
| [噪声与速度调节](noise_shaping.md) | 跨任务族筛选、阶段正则、speed-cap 协议 |
| [合成机制](synthetic.md) | shell、旋转泛化、critic 桥接和 flow 等效性 |
| [机器人任务](robotics.md) | 支持约束、参数隔离和 moment-control 各次实验 |
| [历史结果](results.md) | 原数值、seed/block、负结果、阶段报告 |
| [历史运维](operations.md) | 调度细节、压力测试、迁移、发布和存储设计稿 |
| [旧研究与检查记录](research_history.md) | 原研究阶段、疑问、清单及旧通用协议 |

精简前的原文另保存在数据盘 `archive/documentation-before-consolidation-20261019/source-documents.tar.gz`。
HF 已发布快照和原始运行均保留原版本。

2026-10-19 主仓库一次性脚本已退役；本目录保留的旧脚本名和命令用于历史追溯，
不代表主仓库仍有该入口。45 份脚本及 65 份旧任务配置/日志的原件和 SHA-256 清单在
`/data/chainpray/qc/archive/script-cleanup-20261019-YLEeDs/`；有效合成诊断的新入口见
[运行手册](../current/runbook.md)。执行工作树及 HF 中的冻结源码未改动。
