# hydrodiag manuscript 章节归档

本目录统一保存论文章节的运行时分析代码、图表生成器、协议、审计报告与 manuscript 结果索引。

## 目录结构

```text
manuscript/
├── r1/                         # R1 运行时分析包（如有）
├── r2/                         # R2 运行时分析包（如有）
├── r3/                         # R3 synthetic-truth / misspecification 分析包
├── r4/                         # R4 real-basin state consistency 分析包
├── scripts/
│   ├── r1/                     # R1 图表、统计与重现实验脚本
│   ├── r2/                     # R2 图表、稳健性与 ablation 脚本
│   ├── r3/                     # R3 图表、审计与汇总脚本
│   ├── r4/                     # R4 图表、状态导出与三结构分析脚本
│   ├── r5/                     # R5 生产流水线脚本
│   └── shared/                 # 跨章节模型审计、采样和质量门禁
├── figures/                    # 论文图件
├── tables/                     # 论文表格
├── results/                    # 章节分析结果，不等同于训练源结果
└── supplement/                 # 补充材料
```

## 运行时导入路径

R3/R4 运行时包已经归档到 manuscript 下，代码应使用：

```python
from manuscript.r3.common import ...
from manuscript.r4.common import ...
from manuscript.scripts.shared.r1_plot_style import ...
```

项目根目录应位于 `PYTHONPATH` 中，推荐从 `project/hydrodiag` 执行命令：

```bash
cd /home/jingxin/code/dmg-research/project/hydrodiag
```

## 章节入口

### R3

- 运行时包：`manuscript/r3/`
- 图表与汇总：`manuscript/scripts/r3/`
- 相关结果：`results/r3_*`、`manuscript/results/R3/`
- 说明：`manuscript/r3/README.md`、`manuscript/r3/HANDOFF.md`

### R4

- 运行时包：`manuscript/r4/`
- 图表与状态导出：`manuscript/scripts/r4/`
- 正式分析：`results/r4_phase1_soil_official/`
- 说明：`manuscript/r4/HANDOFF.md`、`manuscript/scripts/r4/HANDOFF_R4.md`

### R1/R2/R5 与共享脚本

- R1：`manuscript/scripts/r1/`
- R2：`manuscript/scripts/r2/`
- R5：`manuscript/scripts/r5/`
- 共享质量门禁与审计：`manuscript/scripts/shared/`

## 结果与归档原则

- `.npz`、checkpoint、运行日志和大型 CSV 保留在 `project/hydrodiag/results/`，不直接提交到 Git。
- 章节代码只能写入已声明的结果目录；不要新建 `outputs/` 或临时结果树。
- 已完成结果不可覆盖；新的分析写入新的 `analysis/` 或 `summaries/` 子目录。
- R3/R4 的 handoff 文档记录协议、数据来源、结果边界和可声明结论。
