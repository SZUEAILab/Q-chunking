# Direction–Speed 项目文档

日常只维护、阅读下面四份文档：

| 文档 | 回答的问题 |
|---|---|
| [研究](current/research.md) | 要解决什么问题，核心假设是什么？ |
| [实验](current/experiments.md) | 一批实验怎么设计、比较和判断？ |
| [结果](current/results.md) | 已知什么、缺什么，下一步依据是什么？ |
| [运行与存储](current/runbook.md) | 怎么运行，数据放哪，怎么归档？ |

<details>
<summary>按需参考与历史记录</summary>

- [方法详解](reference/method.md)：推导、分布与实现细节。
- [RLPD 基线来源](reference/rlpd_baselines.md)：原论文配方移植、项目 fork 与 FQL 作者实现的区别。
- [FQL 参考学习器](reference/fql_baseline.md)：固定官方配方、数值对照与后续跨算法适配边界。
- [GAC 基线来源](reference/gac_baseline.md)：固定官方代码、旧策略适配、数值一致范围和动作支持。
- [数据集安装](reference/datasets.md)：OGBench、RoboMimic、LIBERO。
- [论文工作稿](reference/paper_draft.md)：写作时查，不作为日常运行规则。
- [历史归档](archive/index.md)：已按主题合并的旧协议、结果和运维记录。
- [图片](images/index.md)：主仓库保留的文档结果图。
- [HF 数据](data/README.md)：精简评估、配置、统计和来源哈希。
- [交互演示](ds-visualizer/README.md)：方法可视化前端。

</details>

单次实验配置、状态和进度写入数据盘的批次记录；不再每做一次实验就增加一份日常文档。
正文图片保留在 `docs/images/`。检查链接及索引可达性：`python scripts/check_docs.py`。
