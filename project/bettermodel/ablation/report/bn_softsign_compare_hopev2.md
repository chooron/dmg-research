# `S5D-ConvBN-Softsign` vs `HopeV2` Detailed Comparison

## Scope

本报告专门比较两类模型：

- `S5D-ConvBN-Softsign`
- `HopeV2`

比较范围覆盖 5 个 seeds：

- `111`
- `222`
- `333`
- `444`
- `555`

本报告的分析分为两层：

1. **精度与 seed 稳定性**
2. **动态参数可靠性与物理意义**

其中第二层是主分析对象，精度只作为次级证据。

## Source Files

- `S5D-ConvBN-Softsign` 输出目录  
  `project/bettermodel/outputs/dhbv_ablation_s5d_conv_bn_softsign-531/NseBatchLoss/seed_*/test1995-2010_Ep100/metrics_agg.json`

- `HopeV2` 输出目录  
  `project/bettermodel/outputs/dhbv_hopev2/camels_671/seed_*/test1995-2010_Ep100/metrics_agg.json`

- 多 seed 精度原始表  
  `project/bettermodel/multiseed/results/test_per_seed_metrics.csv`

## Compared Models

### `S5D-ConvBN-Softsign`

- 1D convolution: yes
- normalization: BatchNorm
- dynamic activation: Softsign mapped to `[0, 1]`

### `HopeV2`

- `HopeMlpV2`
- 使用其原始配置和 5-seed 训练结果

## Metrics

### Accuracy

- `median NSE`
- `mean NSE`
- `median KGE`
- `mean KGE`

### Parameter Reliability

对三个动态参数分别计算：

- `parBETA`
- `parK0`
- `parBETAET`

并对每个参数统计：

- `mean_variability`  
  短期日际波动，越低越好

- `mean_roughness`  
  高频尖刺强度，越低越好

- `mean_long_term_shift`  
  长期变化幅度，用于判断参数是否具有明显季节/慢变量趋势

- `mean_trend_to_noise_ratio`  
  长期变化 / 短期噪声，越高越好

- `mean_boundary_saturation_ratio`  
  参数贴近边界的比例，越低越好

## Part I. Accuracy Comparison

### Per-Seed Accuracy

| Model | Seed | median NSE | mean NSE | median KGE | mean KGE |
| --- | ---: | ---: | ---: | ---: | ---: |
| S5D-ConvBN-Softsign | 111 | 0.753804 | 0.697654 | 0.769088 | 0.712476 |
| S5D-ConvBN-Softsign | 222 | 0.756661 | 0.688165 | 0.769492 | 0.711036 |
| S5D-ConvBN-Softsign | 333 | 0.757904 | 0.695700 | 0.766512 | 0.709372 |
| S5D-ConvBN-Softsign | 444 | 0.753053 | 0.687173 | 0.774772 | 0.713368 |
| S5D-ConvBN-Softsign | 555 | 0.753740 | 0.686123 | 0.773849 | 0.715373 |
| HopeV2 | 111 | 0.753195 | 0.692362 | 0.766523 | 0.710547 |
| HopeV2 | 222 | 0.743940 | 0.677063 | 0.762669 | 0.702819 |
| HopeV2 | 333 | 0.753809 | 0.686712 | 0.772981 | 0.715006 |
| HopeV2 | 444 | 0.751287 | 0.681492 | 0.771329 | 0.711257 |
| HopeV2 | 555 | 0.753718 | 0.692021 | 0.774574 | 0.712944 |

### 5-Seed Aggregate Accuracy

| Model | mean(mean NSE) | std(mean NSE) | mean(median NSE) | std(median NSE) | mean(mean KGE) | std(mean KGE) | mean(median KGE) | std(median KGE) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HopeV2 | 0.68593 | 0.00666 | 0.75119 | 0.00418 | 0.71051 | 0.00463 | 0.76961 | 0.00492 |
| S5D-ConvBN-Softsign | 0.69096 | 0.00531 | 0.75503 | 0.00212 | 0.71232 | 0.00228 | 0.77074 | 0.00347 |

### Accuracy Interpretation

直接结论：

- `S5D-ConvBN-Softsign` 的 5-seed 平均精度整体更高
- `S5D-ConvBN-Softsign` 的 seed 间标准差整体更小

逐 seed 胜负统计：

- `median NSE`: `5 / 5`，`S5D-ConvBN-Softsign` 全胜
- `mean NSE`: `4 / 5`，`S5D-ConvBN-Softsign` 占优
- `mean KGE`: `4 / 5`，`S5D-ConvBN-Softsign` 占优
- `median KGE`: `3 / 5`，`S5D-ConvBN-Softsign` 略占优

因此，在精度和 seed 稳定性上：

> `S5D-ConvBN-Softsign` 比 `HopeV2` 更强、更稳。

## Part II. Parameter Reliability Comparison

### Reliability Aggregate Across Seeds

#### `parBETA`

| Model | variability | roughness | long-term shift | trend/noise | boundary saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| HopeV2 | 0.004506 | 0.020693 | 0.180785 | 47.7800 | 0.258499 |
| S5D-ConvBN-Softsign | 0.008463 | 0.021667 | 0.214447 | 30.6150 | 0.068471 |

#### `parBETAET`

| Model | variability | roughness | long-term shift | trend/noise | boundary saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| HopeV2 | 0.005995 | 0.010202 | 0.158172 | 33.5322 | 0.050230 |
| S5D-ConvBN-Softsign | 0.009596 | 0.011899 | 0.166460 | 20.4228 | 0.020297 |

#### `parK0`

| Model | variability | roughness | long-term shift | trend/noise | boundary saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| HopeV2 | 0.004958 | 0.013684 | 0.037780 | 8.0631 | 0.192309 |
| S5D-ConvBN-Softsign | 0.005886 | 0.009604 | 0.027833 | 5.1466 | 0.057852 |

### Reliability Win Counts

在全部 `5 seeds × 3 parameters = 15` 个比较单元上：

- `bnsoftsign_better_variability`: `1 / 15`
- `bnsoftsign_better_roughness`: `7 / 15`
- `bnsoftsign_better_boundary`: `15 / 15`
- `bnsoftsign_better_trend_noise`: `0 / 15`
- `bnsoftsign_higher_long_term_shift`: `9 / 15`

这组统计非常关键：

- `S5D-ConvBN-Softsign` 最稳定的优势是 **边界效应控制**
- `HopeV2` 最稳定的优势是 **趋势信噪比**
- `S5D-ConvBN-Softsign` 往往具有更大的长期变化幅度
- `HopeV2` 往往具有更低的短期日际噪声

## Part III. Per-Parameter Interpretation

### `parBETA`

`S5D-ConvBN-Softsign`：

- 长期变化更大
- 边界效应显著更低

`HopeV2`：

- 短期波动更低
- 趋势信噪比更高

解释：

- BN-Softsign 的 `parBETA` 更“敢动”
- HopeV2 的 `parBETA` 更“克制”

如果优先看物理安全性，BN-Softsign 更好。  
如果优先看慢变量的低噪声趋势性，HopeV2 更好。

### `parBETAET`

`S5D-ConvBN-Softsign`：

- 边界效应更低
- 长期变化稍更大

`HopeV2`：

- 短期波动更低
- 趋势信噪比更高
- roughness 也略低

解释：

- BN-Softsign 更偏向“安全、不贴边”
- HopeV2 更偏向“平稳、保守、趋势更干净”

### `parK0`

`S5D-ConvBN-Softsign`：

- roughness 更低
- 边界效应显著更低

`HopeV2`：

- variability 更低
- 趋势信噪比更高
- 长期变化幅度也更大

解释：

- `parK0` 是二者 trade-off 最明显的参数
- BN-Softsign 更像“稳边界、少尖刺”
- HopeV2 更像“低噪声、慢变量更清晰”

## Part IV. Reliability-Oriented Seed View

如果完全不按精度，而只按参数可靠性在 `HopeV2` 内部选 seed，则最均衡的 seed 是：

- `seed 111`

这说明：

- `HopeV2` 内部 seed 差异确实存在
- 不能只看精度选出代表 seed
- 如果要把 `HopeV2` 作为 `S5D-full` 的来源，更稳妥的代表 seed 是 `111`

但即使如此，把 `HopeV2 seed 111` 放回当前 ablation 家族里比较，它仍然不是所有参数可靠性指标上的统一最优结构。

## Overall Conclusion

这两类模型代表了两种不同的偏好：

### `S5D-ConvBN-Softsign`

优势：

- 精度更高
- seed 更稳定
- 边界效应显著更低
- 参数更不容易贴边

不足：

- 日际波动更大
- 趋势信噪比更低

### `HopeV2`

优势：

- 短期波动更低
- 趋势信噪比更高
- 参数演化更克制

不足：

- 边界效应明显更强
- 精度和跨 seed 稳定性略差

因此，最稳妥的总括是：

> `S5D-ConvBN-Softsign` is the stronger and more stable model if the priority is predictive performance, cross-seed robustness, and suppression of boundary saturation. `HopeV2` is more conservative in its parameter evolution, with lower short-term noise and cleaner long-term trend-to-noise structure, but it pays for that behavior with systematically stronger boundary saturation.

## Recommended Wording

可直接用于正文或回复：

> Across five independent seeds, S5D-ConvBN-Softsign consistently outperforms HopeV2 in mean/median NSE and mean KGE, while also exhibiting smaller cross-seed variability. From the parameter-reliability perspective, the main advantage of S5D-ConvBN-Softsign is its dramatically lower boundary saturation across all dynamic parameters and seeds. In contrast, HopeV2 yields more conservative parameter trajectories, characterized by lower short-term variability and higher trend-to-noise ratios. Thus, the comparison supports a clear trade-off: S5D-ConvBN-Softsign is preferable when predictive stability and physically safer parameter ranges are prioritized, whereas HopeV2 is preferable when smoother, lower-noise temporal parameter evolution is the primary concern.

