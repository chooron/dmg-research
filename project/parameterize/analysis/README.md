# Analysis README

这个目录提供一套针对 `project/parameterize/outputs` 下多模型、多 loss、多 seed 结果的稳定性分析流水线，核心目标是回答四类问题：

1. 不同模型在预测指标上谁更好。
2. 同一个模型在不同随机种子下，参数是否稳定。
3. 同一个模型在不同 loss 下，参数是否稳定。
4. 参数和流域属性之间的相关结构，在 seed 和 loss 变化下是否稳定。

默认输出目录是：

- `project/parameterize/outputs/analysis/stability_stats/`

常见输入包括：

- 运行结果目录：`project/parameterize/outputs/{variant-531}/{loss}/seed_{seed}/`
- 参数长表：`params_long.csv` 或等价 long-format csv
- 流域属性表：`basin_attributes.csv`

## 推荐执行顺序

如果你只想一步跑完整流程，优先用：

- `run_all.py`

如果你想分步执行，推荐顺序是：

1. `0_common.py`
2. `1_collect_run_tables.py`
3. `2_metric_accuracy_stats.py`
4. `3_parameter_seed_variance.py`
5. `4_parameter_variance_summary.py`
6. `5_cross_loss_parameter_variance.py`
7. `6_correlation_matrices.py`
8. `7_correlation_seed_stability.py`
9. `8_correlation_loss_stability.py`
10. `9_correlation_aggregate_exports.py`
11. `10_generate_analysis_report.py`

## 各脚本说明

### 入口脚本

#### `0_common.py`

作用：

- 不做正式统计。
- 只检查配置解析后的输入输出路径是否正确。
- 列出当前发现到的模型、loss、seed 和 run 数量。

适用场景：

- 开跑前先确认路径、目录命名和数据覆盖范围是否正常。

#### `1_collect_run_tables.py`

作用：

- 收集基础清单和长表。
- 导出 run inventory、metric manifest、parameter manifest、`metrics_long.csv`、`params_long.csv`、`basin_attributes.csv`。

适用场景：

- 把后续分析依赖的基础表先统一落盘。

#### `2_metric_accuracy_stats.py`

作用：

- 汇总精度指标表现。
- 主要按 `model/loss/seed` 和 `model/loss` 两个层次统计 `NSE`、`KGE`、`bias_abs` 等指标。

回答的问题：

- 哪个模型整体更准。
- 哪个 loss 在同一模型下更优。

#### `3_parameter_seed_variance.py`

作用：

- 计算参数在不同随机种子之间的波动。
- 先按参数物理范围归一化到 `[0, 1]`，再计算 seed 方差和 seed 间平均绝对差。

回答的问题：

- 同一模型同一 loss 下，参数对随机种子是否敏感。

#### `4_parameter_variance_summary.py`

作用：

- 对上一步的 seed 参数方差进一步做摘要统计。
- 输出均值、中位数、P90 和平均 seed 差值。

回答的问题：

- 从整体层面比较不同模型、不同 loss 的 seed 稳定性。

#### `5_cross_loss_parameter_variance.py`

作用：

- 比较参数在不同 loss 之间的波动。
- 支持两种模式：
- `pooled`：直接把 `loss x seed` 当作样本集合比较。
- `seed-first`：先在 seed 内平均，再比较 loss。

回答的问题：

- 模型学到的参数是否会随训练目标函数显著改变。

#### `6_correlation_matrices.py`

作用：

- 为每个 `model/loss/seed` 导出“流域属性 vs 参数”的相关矩阵。
- 默认支持 `spearman`、`pearson`、`kendall`。
- 输出 csv 和 npz 两种格式。

回答的问题：

- 每个 run 下，哪些属性和哪些参数关联最强。

#### `7_correlation_seed_stability.py`

作用：

- 衡量参数-属性相关关系在不同 seed 下是否稳定。
- 默认会先按每个参数挑选 top-k 强相关属性，再计算跨 seed 的方差、范围和绝对差。

回答的问题：

- 同一模型同一 loss 下，相关结构是否会被随机种子打乱。

#### `8_correlation_loss_stability.py`

作用：

- 衡量参数-属性相关关系在不同 loss 下是否稳定。
- 同时给出 pooled 和 seed-first 两种视角的波动统计。

回答的问题：

- 改变 loss 后，模型学到的“参数对应什么流域属性”是否还一致。

#### `9_correlation_aggregate_exports.py`

作用：

- 把相关性结果聚合成更方便看报告的表。
- 导出均值、标准差、方差和 top relationships 等汇总结果。

回答的问题：

- 哪些参数-属性关系最稳定、最强、最值得在报告中展示。

#### `10_generate_analysis_report.py`

作用：

- 生成总报告 `reports/analysis_results.md`。
- 同时生成面向论文 Results 2–4 的关系聚焦报告 `reports/relationship_focus_report.md`。
- 会串起 metric、parameter variance、cross-loss variance、seed correlation stability、loss correlation stability、top relationships 等结果。
- 还会额外导出：
- `core_relationships_summary.csv`
- `pair_seed_stability.csv`
- `pair_loss_stability.csv`
- `relationship_classes.csv`
- `parameter_level_consistency.csv`
- `parameter_feature_importance.csv`
- `stability_significance_summary.csv`

适用场景：

- 需要一个可直接阅读的总览 markdown 报告时使用。

#### `run_all.py`

作用：

- 这是完整流水线的主入口。
- 一次性完成数据加载、统计汇总和 markdown 报告生成。

建议：

- 日常分析优先用这个脚本。
- 只有在排查某一步结果时，才单独运行 `0-10` 编号脚本。

## 各实现模块说明

### `common.py`

作用：

- 放公共工具函数。
- 负责解析参数、定位输出目录、发现 runs、加载长表、参数归一化、保存 csv/npz/md。

可以理解为：

- 这是整个分析目录的“基础设施层”。

### `metrics_analysis.py`

作用：

- 只负责精度指标统计和指标相关输出。
- 不关心参数稳定性和相关矩阵。

### `parameter_analysis.py`

作用：

- 只负责参数层面的统计。
- 包括跨 seed 参数方差、跨 loss 参数方差、以及相应摘要表的生成。

### `correlation_analysis.py`

作用：

- 只负责参数-属性相关结构分析。
- 包括相关矩阵构建、跨 seed 稳定性、跨 loss 稳定性、聚合导出。

### `pipeline.py`

作用：

- 把各个分析模块串起来。
- 为分步脚本和 `run_all.py` 提供统一的 orchestration 函数。

可以理解为：

- 这是“流程调度层”。

### `reporting.py`

作用：

- 把前面各步产生的 summary DataFrame 组装成一个 markdown 报告。

### `__init__.py`

作用：

- 纯包初始化文件，没有实际分析逻辑。

## 什么时候需要 `--parameter-csv` 和 `--attribute-csv`

只依赖运行结果目录就能执行的步骤：

- `0_common.py`
- `1_collect_run_tables.py`
- `2_metric_accuracy_stats.py`

必须额外提供参数表的步骤：

- `3_parameter_seed_variance.py`
- `4_parameter_variance_summary.py`
- `5_cross_loss_parameter_variance.py`

必须同时提供参数表和属性表的步骤：

- `6_correlation_matrices.py`
- `7_correlation_seed_stability.py`
- `8_correlation_loss_stability.py`
- `9_correlation_aggregate_exports.py`
- `10_generate_analysis_report.py`
- `run_all.py`

## 结果目录对应关系

典型输出子目录如下：

- `tables/`：基础长表、manifest、metric summary
- `parameter_variance/`：seed variance 和 cross-loss variance
- `correlation_matrices/`：每个 run 的相关矩阵
- `correlation_summaries/`：相关稳定性摘要和 top relationships
- `reports/`：总报告 `analysis_results.md` 和关系聚焦报告 `relationship_focus_report.md`

## 最常用命令

完整跑一遍：

```bash
python -m project.parameterize.analysis.run_all \
  --config project/parameterize/conf/config_param_paper.yaml \
  --outputs-root project/parameterize/outputs \
  --analysis-root project/parameterize/outputs/analysis/stability_stats \
  --parameter-csv project/parameterize/outputs/analysis/stability_stats/tables/params_long_input.csv \
  --attribute-csv project/parameterize/outputs/analysis/stability_stats/tables/basin_attributes_input.csv
```

只检查当前分析输入是否识别正确：

```bash
python -m project.parameterize.analysis.0_common \
  --config project/parameterize/conf/config_param_paper.yaml \
  --outputs-root project/parameterize/outputs
```
