# 章节化 manuscript scripts

所有 manuscript-facing 脚本按章节归类；运行命令从 `project/hydrodiag` 根目录执行。

```text
manuscript/scripts/
├── r1/       R1 日常推演、统计与 Figure/Table 生成
├── r2/       R2 参数、稳健性、TGD2 specificity 与 ablation
├── r3/       R3 synthetic-truth、misspecification、审计与汇总
├── r4/       R4 状态导出、土壤水一致性、Figure 7/8/S6 与三结构表格
├── r5/       R5 生产流水线
└── shared/   模型质量门禁、Phase-0 采样、原生响应审计与绘图样式
```

## 常用入口

```bash
# R1
python manuscript/scripts/r1/build_r1_statistics.py

# R2
python manuscript/scripts/r2/run_r2_robustness_checks.py

# R3
python manuscript/scripts/r3/generate_table_r3_main.py
python manuscript/scripts/r3/generate_table_r3_si.py

# R4
python manuscript/scripts/r4/build_r4_soil_statistics.py --device cuda
python manuscript/scripts/r4/plot_r4_figure4.py
python manuscript/scripts/r4/generate_table_r4.py
python manuscript/scripts/r4/generate_three_structure_r4_all.py

# 共享质量门禁
python manuscript/scripts/shared/run_model_test_suite.py
```

## 约束

- 训练代码放在 `training/`，章节脚本只负责分析、审计、导出和图表生成。
- 结果写入 `results/<run_id>/` 或既有 `analysis/`、`summaries/` 子目录。
- 不覆盖已完成结果，不创建新的物理 `outputs/` 树。
- R3/R4 运行时包分别位于 `manuscript/r3/` 与 `manuscript/r4/`，导入使用 `manuscript.r3` 和 `manuscript.r4`。
