# `S5D-ConvBN-Softsign` vs `HopeV2` Reliability Comparison Table

说明：

- `Variability`、`Roughness`、`Boundary saturation` 越低越好
- `Long-term shift`、`Trend-to-noise` 越高越好
- 表中数值来自 5-seed 平均汇总
- `5-seed wins` 表示在 `5 seeds` 的逐参数比较中，`S5D-ConvBN-Softsign` 胜出的次数

| Scope | Variability | Roughness | Long-term shift | Trend-to-noise | Boundary saturation | 5-seed wins summary |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | `0.00796 vs 0.00515` | `0.01436 vs 0.01485` | `0.13800 vs 0.12499` | `18.93 vs 29.86` | `0.04896 vs 0.16757` | Variability `1/15`; Roughness `6/15`; Long-term shift `10/15`; Trend-to-noise `0/15`; Boundary `15/15` |
| `parBETA` | `0.00838 vs 0.00449` | `0.02157 vs 0.02063` | `0.21697 vs 0.18060` | `31.07 vs 48.15` | `0.06914 vs 0.25949` | Variability `0/5`; Roughness `0/5`; Long-term shift `5/5`; Trend-to-noise `0/5`; Boundary `5/5` |
| `parBETAET` | `0.00962 vs 0.00599` | `0.01194 vs 0.01019` | `0.16960 vs 0.15671` | `20.63 vs 33.37` | `0.01981 vs 0.05065` | Variability `0/5`; Roughness `1/5`; Long-term shift `4/5`; Trend-to-noise `0/5`; Boundary `5/5` |
| `parK0` | `0.00588 vs 0.00498` | `0.00956 vs 0.01374` | `0.02744 vs 0.03767` | `5.08 vs 8.07` | `0.05795 vs 0.19257` | Variability `1/5`; Roughness `5/5`; Long-term shift `1/5`; Trend-to-noise `0/5`; Boundary `5/5` |

## Short Reading Guide

- `S5D-ConvBN-Softsign` 的最稳定优势是 **boundary saturation**，在整体上是 `15/15` 全胜。
- `HopeV2` 的最稳定优势是 **trend-to-noise ratio**，在整体上是 `15/15` 全胜。
- `S5D-ConvBN-Softsign` 通常具有更大的 **long-term shift**，尤其在 `parBETA` 上是 `5/5` 全胜。
- `HopeV2` 通常具有更低的 **variability**，说明短期日际波动更小。
- `Roughness` 上二者不是完全一边倒，但 `parK0` 上 `S5D-ConvBN-Softsign` 明显更优。
