# S4D/S5D Ablation Comprehensive Comparison

## Scope

本报告汇总两部分结果：

1. `project/bettermodel/ablation/results/` 中的 8 变体单种子 `2 x 2 x 2` factorial ablation  
   轴为：
   - 1D convolution: no / yes
   - normalization: BatchNorm / LayerNorm
   - dynamic activation: Sigmoid / Softsign

2. `S5D-ConvBN-Softsign` 与 `HopeV2` 的 5-seed 综合对比  
   seeds: `111 / 222 / 333 / 444 / 555`

本报告优先关注：
- 参数可靠性
- 物理意义
- seed 稳定性

精度仅作为次级证据。

## Source Files

- 单种子精度汇总：`project/bettermodel/ablation/results/ablation_performance_summary.csv`
- 参数可靠性汇总：`project/bettermodel/ablation/results/ablation_parameter_reliability_summary.csv`
- 5-seed 精度原始表：`project/bettermodel/multiseed/results/test_per_seed_metrics.csv`
- `S5D-ConvBN-Softsign` vs `HopeV2` 5-seed 专项比较：`project/bettermodel/ablation/report/bn_softsign_compare_hopev2.md`

## Part I. 8-Variant Factorial Ablation

### Variants

| Variant | Conv | Norm | Activation |
| --- | --- | --- | --- |
| S4D-baseline | no | BatchNorm | Sigmoid |
| S4D-LN | no | LayerNorm | Sigmoid |
| S4D-Softsign | no | BatchNorm | Softsign |
| S4D-LN-Softsign | no | LayerNorm | Softsign |
| S5D-ConvOnly | yes | BatchNorm | Sigmoid |
| S5D-ConvBN-Softsign | yes | BatchNorm | Softsign |
| S5D-ConvLN-Sigmoid | yes | LayerNorm | Sigmoid |
| S5D-full | yes | LayerNorm | Softsign |

### Accuracy Summary

单种子 `seed=111` 下：

| Variant | median NSE | mean NSE | median KGE | mean KGE |
| --- | ---: | ---: | ---: | ---: |
| S4D-baseline | 0.7495 | 0.6941 | 0.7671 | 0.7102 |
| S4D-LN | 0.7454 | 0.6847 | 0.7635 | 0.7099 |
| S4D-Softsign | 0.7505 | 0.6791 | 0.7642 | 0.7098 |
| S4D-LN-Softsign | 0.7449 | 0.6783 | 0.7674 | 0.7091 |
| S5D-ConvOnly | 0.7577 | 0.6922 | 0.7720 | 0.7104 |
| S5D-ConvBN-Softsign | 0.7538 | 0.6977 | 0.7691 | 0.7125 |
| S5D-ConvLN-Sigmoid | 0.7491 | 0.6892 | 0.7724 | 0.7197 |
| S5D-full | 0.7533 | 0.6945 | 0.7702 | 0.7107 |

精度上的直接结论：

- `S5D-ConvBN-Softsign` 是 `mean NSE` 最优。
- `S5D-ConvOnly` 是 `median NSE` 最优。
- `S5D-ConvLN-Sigmoid` 是 `median KGE / mean KGE` 最优。
- `S5D-full` 处于第一梯队，但不是任何单一精度指标上的绝对最优。

按四个精度指标的 rank sum 排序：

1. `S5D-ConvBN-Softsign`
2. `S5D-full`
3. `S5D-ConvOnly`
4. `S5D-ConvLN-Sigmoid`
5. `S4D-baseline`
6. `S4D-Softsign`
7. `S4D-LN`
8. `S4D-LN-Softsign`

### Parameter Reliability Summary

参数可靠性使用以下指标：

- `mean_variability`：短期日际波动，越低越好
- `mean_roughness`：高频尖刺强度，越低越好
- `mean_long_term_shift`：长期变化幅度
- `mean_trend_to_noise_ratio`：长期趋势 / 短期噪声，越高越好
- `mean_boundary_saturation_ratio`：边界效应，越低越好

### Per-Parameter Findings

#### parBETA

- `S5D-ConvLN-Sigmoid` 最擅长“低短期波动 + 高趋势信噪比”
- `S5D-ConvOnly` 也很强，且 roughness 最低
- `S5D-ConvBN-Softsign` 与 `S5D-full` 在边界效应上明显优于 Sigmoid 系列
- `S4D-Softsign` 的边界效应最低，但波动和粗糙度不占优

解释：

- Sigmoid + conv 分支更偏向“趋势清晰、低噪声”
- Softsign 分支更偏向“避免参数贴边”
- `parBETA` 明显表现出“趋势清晰”和“低边界效应”之间的 trade-off

#### parBETAET

- `S5D-ConvBN-Softsign` 是最平衡的变体
  - `variability` 最低
  - `roughness` 最低
  - `boundary_saturation` 仍保持很低
- `S5D-full` 的长期变化幅度更大，但 roughness 更高
- `S5D-ConvLN-Sigmoid` 的趋势信噪比极高，但边界效应很重

解释：

- 对 `parBETAET` 而言，`BatchNorm + Softsign + Conv` 是最容易 defend 的组合
- 它同时兼顾平滑性和物理安全性

#### parK0

- `S5D-ConvLN-Sigmoid` 在 `variability` 和 `trend_to_noise_ratio` 上最好
- `S5D-ConvBN-Softsign` 在 `boundary_saturation` 上明显更好
- `S4D-Softsign` 的边界效应最低，但整体趋势性一般

解释：

- `parK0` 的主要 trade-off 依旧是：
  - Sigmoid 系列：趋势更清晰，但更容易贴边
  - Softsign 系列：边界更安全，但趋势信噪比偏低

### High-Level Interpretation of the 8-Variant Ablation

可以把 8 变体结果总结成：

- `1D conv` 是最稳定的性能增益来源
- `Softsign` 是最稳定的边界效应抑制来源
- `LayerNorm` 不是普遍优于 `BatchNorm`
  - 在某些参数/组合中可以带来更低噪声或更高趋势信噪比
  - 但往往伴随更强边界效应，且没有形成统一的精度优势

因此，这套 ablation 支持的不是“某个结构在所有维度上绝对最优”，而是：

> 不同动态参数对结构组件的偏好不同，最佳结构取决于你优先强调的是趋势清晰度、短期平滑性，还是边界安全性。

## Part II. `S5D-ConvBN-Softsign` vs `HopeV2` Across 5 Seeds

### Accuracy Comparison

5-seed 平均结果：

| Model | mean(mean NSE) | std(mean NSE) | mean(median NSE) | std(median NSE) | mean(mean KGE) | std(mean KGE) | mean(median KGE) | std(median KGE) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HopeV2 | 0.68593 | 0.00666 | 0.75119 | 0.00418 | 0.71051 | 0.00463 | 0.76961 | 0.00492 |
| S5D-ConvBN-Softsign | 0.69096 | 0.00531 | 0.75503 | 0.00212 | 0.71232 | 0.00228 | 0.77074 | 0.00347 |

逐 seed 胜负：

- `median NSE`: `5 / 5`，`S5D-ConvBN-Softsign` 全胜
- `mean NSE`: `4 / 5`，`S5D-ConvBN-Softsign` 占优
- `mean KGE`: `4 / 5`，`S5D-ConvBN-Softsign` 占优
- `median KGE`: `3 / 5`，`S5D-ConvBN-Softsign` 略占优

结论：

- `S5D-ConvBN-Softsign` 不仅平均精度更高，而且 seed 间波动更小
- 精度上它比 `HopeV2` 更稳

### Reliability Comparison

对比 15 个“seed × parameter”单元（5 seeds × 3 parameters）：

- `bnsoftsign_better_variability`: `1 / 15`
- `bnsoftsign_better_roughness`: `7 / 15`
- `bnsoftsign_better_boundary`: `15 / 15`
- `bnsoftsign_better_trend_noise`: `0 / 15`
- `bnsoftsign_higher_long_term_shift`: `9 / 15`

这说明：

- `S5D-ConvBN-Softsign` 的**最稳定优势**是边界效应抑制
  - 边界饱和 `15/15` 全胜
- `HopeV2` 的**最稳定优势**是趋势信噪比
  - 在所有 seed 和参数上，它的长期变化相对短期噪声更“干净”
- `S5D-ConvBN-Softsign` 常常有更大的长期变化幅度
  - 但这些变化伴随更高的短期波动

### Per-Parameter Comparison

#### parBETA

- `S5D-ConvBN-Softsign`
  - 边界更安全
  - 长期变化幅度更大
- `HopeV2`
  - 趋势信噪比更高
  - 短期波动更小

解释：

- BN-Softsign 的 `parBETA` 更“敢动”
- HopeV2 的 `parBETA` 更“克制”

#### parBETAET

- `S5D-ConvBN-Softsign`
  - 边界更低
  - roughness 更低
- `HopeV2`
  - variability 更低
  - 趋势信噪比更高

解释：

- BN-Softsign 更偏向物理安全和平滑性
- HopeV2 更偏向低噪声保守演化

#### parK0

- `S5D-ConvBN-Softsign`
  - roughness 更低
  - 边界更低
- `HopeV2`
  - variability 更低
  - 趋势信噪比更高

解释：

- `parK0` 上 trade-off 最明显
- 这也是两者哲学差异最大的参数

## Overall Conclusions

### Conclusion 1

在 8 变体 factorial ablation 中，没有单一结构在所有参数可靠性指标上统一最优。

### Conclusion 2

`S5D-ConvBN-Softsign` 是当前最均衡、最容易 defend 的结构之一：

- 在精度上综合最好
- 在 seed 稳定性上更好
- 在 `parBETAET` 上最平衡
- 在边界效应控制上极其稳定

### Conclusion 3

`HopeV2` 的主要强项不在边界安全性，而在：

- 更低的短期波动
- 更高的趋势信噪比

也就是说，它更适合被表述成：

> 参数变化更克制、更低噪声、更“慢变量化”的结构

而不是边界最安全的结构。

### Conclusion 4

如果汇报重点是：

- **物理安全性**
- **避免边界效应**
- **跨 seed 稳定性**
- **整体实用表现**

那么 `S5D-ConvBN-Softsign` 更占优。

如果汇报重点是：

- **长期趋势相对短期噪声更清晰**
- **参数演化更保守**

那么 `HopeV2` 更占优。

## Recommended Short Wording

可以用下面这段作为摘要性结论：

> The complete ablation indicates that no single architecture is uniformly optimal for all dynamic-parameter reliability criteria. Across the factorial variants, convolution is the most consistent source of predictive gain, whereas Softsign is the most reliable mechanism for suppressing boundary saturation. Among the tested variants, S5D-ConvBN-Softsign provides the most balanced trade-off between predictive skill, cross-seed stability, and physically safer parameter behavior. In contrast, HopeV2 exhibits lower short-term parameter noise and higher trend-to-noise ratios, but at the cost of systematically stronger boundary saturation.

