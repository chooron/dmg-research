# 7-Model Multi-Seed Comparison Report

## Scope

本报告基于 `project/bettermodel/conf` 中以下 7 个配置，对 `camels_671` 测试集进行 5-seed 汇总比较：

- `config_dhbv_lstm.yaml` -> `dhbv_lstm`
- `config_dhbv_tcn.yaml` -> `dhbv_tcn`
- `config_dhbv_transformer.yaml` -> `dhbv_transformer`
- `config_dhbv_tsmixer.yaml` -> `dhbv_tsmixer`
- `config_dhbv_hopev1.yaml` -> `s4d`
- `config_dhbv_hopev2.yaml` -> `s5dv1`
- `config_dhbv_hopev3.yaml` -> `s5dv2`

固定使用 seeds: `111, 222, 333, 444, 555`。

统计重点：

- `NSE`
- `KGE`
- 所有 `seed × basin` 合并后的综合统计
- 5 个 seed 各自的结果
- 基于 basin 级 seed 平均分数的显著性分析

## Output Files

- [combined summary](/workspace/autoresearch/project/bettermodel/ablation/report/multiseed_7model_combined_summary.csv)
- [per-seed summary](/workspace/autoresearch/project/bettermodel/ablation/report/multiseed_7model_per_seed_summary.csv)
- [seed stability summary](/workspace/autoresearch/project/bettermodel/ablation/report/multiseed_7model_seed_stability_summary.csv)
- [comprehensive ranking](/workspace/autoresearch/project/bettermodel/ablation/report/multiseed_7model_comprehensive_ranking.csv)
- [seed-basin long table](/workspace/autoresearch/project/bettermodel/ablation/report/multiseed_7model_seed_basin_metrics.csv)
- [Friedman significance](/workspace/autoresearch/project/bettermodel/ablation/report/multiseed_7model_friedman_significance.csv)
- [pairwise Wilcoxon significance](/workspace/autoresearch/project/bettermodel/ablation/report/multiseed_7model_pairwise_wilcoxon.csv)
- [CAMELS 531 combined summary](/workspace/autoresearch/project/bettermodel/ablation/report/camels531_lstm_s4d_combined_summary.csv)
- [CAMELS 531 per-seed summary](/workspace/autoresearch/project/bettermodel/ablation/report/camels531_lstm_s4d_per_seed_summary.csv)
- [CAMELS 531 seed-basin long table](/workspace/autoresearch/project/bettermodel/ablation/report/camels531_lstm_s4d_seed_basin_metrics.csv)
- [CAMELS 531 pairwise Wilcoxon](/workspace/autoresearch/project/bettermodel/ablation/report/camels531_lstm_s4d_pairwise_wilcoxon.csv)

## Method

1. 从每个模型的 `test1995-2010_Ep100/metrics.json` 读取 basin 级 `NSE` 和 `KGE`。
2. 将 5 个 seed 的所有 basin 指标合并，得到综合统计结果。
3. 分别统计每个 seed 自身的 `mean / median / std / quartiles`。
4. 用 basin 级 seed 平均指标进行显著性分析，避免把同一 basin 的多 seed 重复当作独立样本。
5. 显著性检验包含：
   - 7 模型整体比较：Friedman test
   - 两两比较：Wilcoxon signed-rank test + Holm 校正

## Comprehensive Ranking

综合排名的构成：

- 性能项：`nse_mean`, `nse_median`, `kge_mean`, `kge_median`，越高越好
- 稳定性项：`nse_mean_seed_std`, `nse_median_seed_std`, `kge_mean_seed_std`, `kge_median_seed_std`，越低越好
- `comprehensive_rank_score` 为上述 8 项 rank 的平均值，越低越好

| comprehensive_rank | model | comprehensive_rank_score | nse_mean | nse_median | kge_mean | kge_median | nse_mean_seed_std | kge_mean_seed_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | s5dv2 | 2.5000 | 0.6893 | 0.7514 | 0.7089 | 0.7692 | 0.0037 | 0.0042 |
| 2 | s4d | 3.0000 | 0.6875 | 0.7496 | 0.7118 | 0.7673 | 0.0068 | 0.0025 |
| 3 | s5dv1 | 3.7500 | 0.6859 | 0.7512 | 0.7105 | 0.7700 | 0.0060 | 0.0041 |
| 4 | dhbv_tcn | 4.3750 | 0.6741 | 0.7423 | 0.6972 | 0.7566 | 0.0066 | 0.0026 |
| 5 | dhbv_transformer | 4.5000 | 0.6617 | 0.7418 | 0.6781 | 0.7483 | 0.0039 | 0.0023 |
| 6 | dhbv_lstm | 4.8750 | 0.6721 | 0.7416 | 0.7054 | 0.7637 | 0.0043 | 0.0045 |
| 7 | dhbv_tsmixer | 5.0000 | 0.6715 | 0.7418 | 0.6863 | 0.7547 | 0.0052 | 0.0076 |

主要观察：

- Top 1: `s5dv2` (score=2.5000, NSE mean=0.6893, KGE mean=0.7089)
- Top 2: `s4d` (score=3.0000, NSE mean=0.6875, KGE mean=0.7118)
- Top 3: `s5dv1` (score=3.7500, NSE mean=0.6859, KGE mean=0.7105)

按综合 pooled 指标看，前 3 名如下：

### NSE Mean Top 3
- `s5dv2`: 0.6893
- `s4d`: 0.6875
- `s5dv1`: 0.6859

### KGE Mean Top 3
- `s4d`: 0.7118
- `s5dv1`: 0.7105
- `s5dv2`: 0.7089

## Per-Seed Results

| model | seed | nse_mean | nse_median | kge_mean | kge_median |
| --- | --- | --- | --- | --- | --- |
| dhbv_lstm | 111 | 0.6714 | 0.7384 | 0.7049 | 0.7645 |
| dhbv_lstm | 222 | 0.6697 | 0.7416 | 0.6981 | 0.7570 |
| dhbv_lstm | 333 | 0.6686 | 0.7422 | 0.7120 | 0.7710 |
| dhbv_lstm | 444 | 0.6702 | 0.7447 | 0.7045 | 0.7651 |
| dhbv_lstm | 555 | 0.6805 | 0.7399 | 0.7076 | 0.7639 |
| dhbv_tcn | 111 | 0.6726 | 0.7435 | 0.6954 | 0.7570 |
| dhbv_tcn | 222 | 0.6723 | 0.7447 | 0.6961 | 0.7578 |
| dhbv_tcn | 333 | 0.6840 | 0.7467 | 0.7014 | 0.7534 |
| dhbv_tcn | 444 | 0.6777 | 0.7418 | 0.6988 | 0.7577 |
| dhbv_tcn | 555 | 0.6641 | 0.7362 | 0.6941 | 0.7561 |
| dhbv_transformer | 111 | 0.6573 | 0.7462 | 0.6797 | 0.7493 |
| dhbv_transformer | 222 | 0.6567 | 0.7425 | 0.6762 | 0.7497 |
| dhbv_transformer | 333 | 0.6659 | 0.7406 | 0.6819 | 0.7437 |
| dhbv_transformer | 444 | 0.6637 | 0.7416 | 0.6762 | 0.7502 |
| dhbv_transformer | 555 | 0.6648 | 0.7385 | 0.6766 | 0.7476 |
| dhbv_tsmixer | 111 | 0.6633 | 0.7430 | 0.6792 | 0.7530 |
| dhbv_tsmixer | 222 | 0.6781 | 0.7418 | 0.6856 | 0.7507 |
| dhbv_tsmixer | 333 | 0.6722 | 0.7434 | 0.6967 | 0.7591 |
| dhbv_tsmixer | 444 | 0.6752 | 0.7416 | 0.6928 | 0.7578 |
| dhbv_tsmixer | 555 | 0.6687 | 0.7402 | 0.6770 | 0.7488 |
| s4d | 111 | 0.6941 | 0.7495 | 0.7102 | 0.7671 |
| s4d | 222 | 0.6765 | 0.7429 | 0.7078 | 0.7658 |
| s4d | 333 | 0.6949 | 0.7513 | 0.7141 | 0.7675 |
| s4d | 444 | 0.6838 | 0.7487 | 0.7123 | 0.7665 |
| s4d | 555 | 0.6882 | 0.7524 | 0.7145 | 0.7686 |
| s5dv1 | 111 | 0.6924 | 0.7532 | 0.7105 | 0.7665 |
| s5dv1 | 222 | 0.6771 | 0.7439 | 0.7028 | 0.7627 |
| s5dv1 | 333 | 0.6867 | 0.7538 | 0.7150 | 0.7730 |
| s5dv1 | 444 | 0.6815 | 0.7513 | 0.7113 | 0.7713 |
| s5dv1 | 555 | 0.6920 | 0.7537 | 0.7129 | 0.7746 |
| s5dv2 | 111 | 0.6945 | 0.7533 | 0.7107 | 0.7702 |
| s5dv2 | 222 | 0.6877 | 0.7509 | 0.7022 | 0.7628 |
| s5dv2 | 333 | 0.6906 | 0.7517 | 0.7148 | 0.7747 |
| s5dv2 | 444 | 0.6833 | 0.7506 | 0.7100 | 0.7727 |
| s5dv2 | 555 | 0.6906 | 0.7534 | 0.7065 | 0.7687 |

## Seed Stability Summary

| model | nse_mean_seed_mean | nse_mean_seed_std | nse_median_seed_mean | nse_median_seed_std | kge_mean_seed_mean | kge_mean_seed_std | kge_median_seed_mean | kge_median_seed_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dhbv_lstm | 0.6721 | 0.0043 | 0.7414 | 0.0021 | 0.7054 | 0.0045 | 0.7643 | 0.0044 |
| dhbv_tcn | 0.6741 | 0.0066 | 0.7426 | 0.0036 | 0.6972 | 0.0026 | 0.7564 | 0.0016 |
| dhbv_transformer | 0.6617 | 0.0039 | 0.7419 | 0.0025 | 0.6781 | 0.0023 | 0.7481 | 0.0024 |
| dhbv_tsmixer | 0.6715 | 0.0052 | 0.7420 | 0.0011 | 0.6863 | 0.0076 | 0.7539 | 0.0040 |
| s4d | 0.6875 | 0.0068 | 0.7490 | 0.0033 | 0.7118 | 0.0025 | 0.7671 | 0.0010 |
| s5dv1 | 0.6859 | 0.0060 | 0.7512 | 0.0037 | 0.7105 | 0.0041 | 0.7696 | 0.0044 |
| s5dv2 | 0.6893 | 0.0037 | 0.7520 | 0.0012 | 0.7089 | 0.0042 | 0.7698 | 0.0041 |

## Significance Analysis

- `NSE` Friedman: statistic=327.9121, p=0.000000
- `KGE` Friedman: statistic=280.1899, p=0.000000

若 `p < 0.05`，说明 7 个模型在该指标上整体存在显著差异。

Holm 校正后显著的两两比较：

| metric | model_a | model_b | mean_diff_a_minus_b | p_value_holm | better_model_by_mean_diff |
| --- | --- | --- | --- | --- | --- |
| kge | dhbv_transformer | s5dv1 | -0.032389 | 0.000000 | s5dv1 |
| kge | dhbv_transformer | s5dv2 | -0.030732 | 0.000000 | s5dv2 |
| kge | dhbv_transformer | s4d | -0.033653 | 0.000000 | s4d |
| kge | dhbv_tsmixer | s5dv1 | -0.024262 | 0.000000 | s5dv1 |
| kge | dhbv_tsmixer | s5dv2 | -0.022605 | 0.000000 | s5dv2 |
| kge | dhbv_tsmixer | s4d | -0.025527 | 0.000000 | s4d |
| kge | dhbv_lstm | dhbv_transformer | 0.027308 | 0.000000 | dhbv_lstm |
| kge | dhbv_tcn | s5dv1 | -0.013363 | 0.000000 | s5dv1 |
| kge | dhbv_tcn | s4d | -0.014627 | 0.000000 | s4d |
| kge | dhbv_lstm | dhbv_tsmixer | 0.019182 | 0.000000 | dhbv_lstm |
| kge | dhbv_tcn | s5dv2 | -0.011705 | 0.000000 | s5dv2 |
| kge | dhbv_tcn | dhbv_transformer | 0.019026 | 0.000000 | dhbv_tcn |
| kge | dhbv_lstm | dhbv_tcn | 0.008282 | 0.000000 | dhbv_lstm |
| kge | dhbv_transformer | dhbv_tsmixer | -0.008127 | 0.000103 | dhbv_tsmixer |
| kge | dhbv_tcn | dhbv_tsmixer | 0.010900 | 0.017975 | dhbv_tcn |
| nse | dhbv_transformer | s5dv2 | -0.027683 | 0.000000 | s5dv2 |
| nse | dhbv_transformer | s4d | -0.025832 | 0.000000 | s4d |
| nse | dhbv_tcn | s4d | -0.013361 | 0.000000 | s4d |
| nse | dhbv_tsmixer | s5dv2 | -0.017832 | 0.000000 | s5dv2 |
| nse | dhbv_transformer | s5dv1 | -0.024269 | 0.000000 | s5dv1 |
| nse | dhbv_tcn | s5dv2 | -0.015212 | 0.000000 | s5dv2 |
| nse | dhbv_tsmixer | s4d | -0.015982 | 0.000000 | s4d |
| nse | dhbv_lstm | s4d | -0.015399 | 0.000000 | s4d |
| nse | dhbv_tsmixer | s5dv1 | -0.014418 | 0.000000 | s5dv1 |
| nse | dhbv_tcn | s5dv1 | -0.011797 | 0.000000 | s5dv1 |
| nse | dhbv_lstm | s5dv2 | -0.017250 | 0.000000 | s5dv2 |
| nse | dhbv_lstm | dhbv_transformer | 0.010433 | 0.000003 | dhbv_lstm |
| nse | dhbv_tcn | dhbv_transformer | 0.012472 | 0.000008 | dhbv_tcn |
| nse | dhbv_lstm | s5dv1 | -0.013836 | 0.000081 | s5dv1 |
| nse | dhbv_transformer | dhbv_tsmixer | -0.009851 | 0.000621 | dhbv_tsmixer |
| nse | dhbv_lstm | dhbv_tsmixer | 0.000582 | 0.010246 | dhbv_lstm |
| nse | s4d | s5dv1 | 0.001564 | 0.020082 | s4d |
| nse | dhbv_lstm | dhbv_tcn | -0.002039 | 0.033518 | dhbv_tcn |

## Interpretation Notes

- `combined summary` 是把 5 个 seed 的全部 basin 指标直接拼接后再统计，因此最接近“总体表现”。
- `per-seed summary` 适合看单次训练波动。
- `seed stability summary` 反映模型对随机种子的敏感性。
- 显著性分析基于 basin 级 seed 平均指标，更适合做结构之间的稳健比较。
- 若需要论文写法，优先引用 `combined summary` + `pairwise Wilcoxon significance`。

## CAMELS 531 Supplement: `dhbv_lstm` vs `s4d`

补充说明：这一节只比较 `dhbv_lstm` 和 `s4d` 在 `camels_531` 上的 5-seed 测试结果，用于和上面的 `camels_671` 主结论互相参照。

| model | nse_mean | nse_median | kge_mean | kge_median |
| --- | --- | --- | --- | --- |
| dhbv_lstm | 0.6794 | 0.7470 | 0.7206 | 0.7693 |
| s4d | 0.7010 | 0.7552 | 0.7262 | 0.7706 |

逐 seed 结果：

| model | seed | nse_mean | nse_median | kge_mean | kge_median |
| --- | --- | --- | --- | --- | --- |
| dhbv_lstm | 111 | 0.6884 | 0.7509 | 0.7269 | 0.7770 |
| dhbv_lstm | 222 | 0.6750 | 0.7514 | 0.7215 | 0.7713 |
| dhbv_lstm | 333 | 0.6770 | 0.7497 | 0.7153 | 0.7663 |
| dhbv_lstm | 444 | 0.6831 | 0.7456 | 0.7234 | 0.7714 |
| dhbv_lstm | 555 | 0.6734 | 0.7423 | 0.7160 | 0.7620 |
| s4d | 111 | 0.7026 | 0.7523 | 0.7240 | 0.7725 |
| s4d | 222 | 0.6980 | 0.7556 | 0.7209 | 0.7657 |
| s4d | 333 | 0.7051 | 0.7547 | 0.7308 | 0.7707 |
| s4d | 444 | 0.7002 | 0.7540 | 0.7284 | 0.7736 |
| s4d | 555 | 0.6993 | 0.7574 | 0.7268 | 0.7695 |

补充结论：

- 在 `camels_531` 上，`s4d` 的 pooled `NSE mean` 为 0.7010，高于 `dhbv_lstm` 的 0.6794。
- 在 `camels_531` 上，`s4d` 的 pooled `KGE mean` 为 0.7262，高于 `dhbv_lstm` 的 0.7206。
- 这说明主报告里 `s4d` 相对 `dhbv_lstm` 的优势，不只出现在 `camels_671`，在 `camels_531` 上也保持一致。

基于 basin 级 seed 平均分数的两模型 Wilcoxon 检验：

| metric | model_a | model_b | mean_diff_a_minus_b | p_value_holm | better_model_by_mean_diff |
| --- | --- | --- | --- | --- | --- |
| kge | dhbv_lstm | s4d | -0.005579 | 0.583771 | s4d |
| nse | dhbv_lstm | s4d | -0.021644 | 0.000000 | s4d |

