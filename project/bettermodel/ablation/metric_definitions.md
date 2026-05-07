# Ablation Metric Definitions

本文档汇总 `project/bettermodel/ablation/s5d_ablation_pipeline.py` 和
`project/bettermodel/parameters/plot_multibasin_parameter_summary_3metrics.py`
中使用的动态参数评价指标。

记：

- `p_t` 表示某一流域、某一动态参数在时间步 `t` 的归一化取值
- `T` 表示时间长度
- `Δp_t = p_{t+1} - p_t`
- `Q_{0.75}(p), Q_{0.25}(p)` 分别表示时间序列的 75% 和 25% 分位数
- `eps` 表示边界阈值，在 `3metrics` 图中通常取 `0.05`，在 ablation 管线里近边界统计常用 `0.02`
- `\bar p_{\text{early}}` 和 `\bar p_{\text{late}}` 分别表示序列前后时间窗口的平均值

## Reliability-Oriented Metrics

| 名称 | 公式表示 | 说明 |
| --- | --- | --- |
| Short-term parameter variability | `median_t(|Δp_t|)` | 短期日际波动强度。越小表示参数在相邻时间步之间更平稳。 |
| Roughness | `mean_t((Δp_t)^2)` | 高频尖刺/粗糙度指标。越小表示参数变化更平滑，异常振荡更少。 |
| Long-term shift | `|\bar p_{late} - \bar p_{early}|` | 长期趋势变化幅度。越大表示参数在慢时间尺度上有更明显的整体漂移或季节性重定位。 |
| Trend-to-noise ratio | `Long-term shift / max(Short-term variability, 1e-6)` | 长期变化相对于短期噪声的强弱比。越大表示趋势更清晰、短期扰动相对更弱。 |
| Boundary saturation ratio | `mean_t(1[p_t < eps \\;\\text{or}\\; p_t > 1-eps])` | 参数贴近归一化边界 `0/1` 的比例。越小越好，表示更少出现边界饱和。 |

## Three-Metric Summary Figure Metrics

| 名称 | 公式表示 | 说明 |
| --- | --- | --- |
| Short-term variability | `median_t(|Δp_t|)` | 与 short-term parameter variability 同口径，强调相邻时间步的典型绝对变化幅度。 |
| Roughness | `(1 / (T - 1)) \sum_t (Δp_t)^2` | 高频尖刺/粗糙度指标。越小表示参数变化更平滑，异常振荡更少。 |
| Boundary saturation ratio | `mean_t(1[p_t < eps \\;\\text{or}\\; p_t > 1-eps])` | 与上表同义，用于衡量参数是否经常贴近归一化边界。 |

## Notes

- `Short-term variability` 与旧版图中的 `Median absolute day-to-day change` 本质上是同一个指标，只是命名与 S5D reliability 诊断保持一致。
- `Roughness` 取代了旧版图中的 `Temporal IQR`，用于强调高频尖刺和局部振荡，而不是整体分布展开程度。
- `Trend-to-noise ratio` 适合和 `Long-term shift`、`Short-term variability` 一起解读，单独看容易误判。
- `Boundary saturation ratio` 越低通常越符合“参数不过度贴边、物理意义更稳健”的目标。
