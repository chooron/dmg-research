# S5D Component-Wise Ablation Results

Seed: `111`
Test epoch: `100`

This is a controlled single-seed diagnostic ablation for the reviewer response.
It is intended to compare component behavior, not to claim a robust accuracy gain.

## Variants

| Variant | Config | Conv | Norm | Activation | Status |
| --- | --- | --- | --- | --- | --- |
| S4D-baseline | `conf/config_dhbv_hopev1.yaml` | False | BatchNorm | Sigmoid | available |
| S4D-LN | `conf/config_dhbv_ablation_s4d_ln.yaml` | False | LayerNorm | Sigmoid | available |
| S4D-Softsign | `conf/config_dhbv_ablation_s4d_softsign.yaml` | False | BatchNorm | Softsign | available |
| S4D-LN-Softsign | `conf/config_dhbv_ablation_s4d_ln_softsign.yaml` | False | LayerNorm | Softsign | available |
| S5D-ConvOnly | `conf/config_dhbv_ablation_s5d_conv_only.yaml` | True | BatchNorm | Sigmoid | available |
| S5D-full | `conf/config_dhbv_hopev3.yaml` | True | LayerNorm | Softsign | available |

## Outputs

- `ablation_performance_summary.csv`
- `ablation_basinwise_metrics.csv`
- `ablation_performance_boxplot.png`
- `ablation_parameter_variability.csv`
- `ablation_parameter_roughness.csv`
- `ablation_boundary_saturation.csv`
- `ablation_parameter_variability_boxplot.png`
- `ablation_parameter_variability_heatmap.png`
- `ablation_boundary_saturation_boxplot.png`
- `parameter_trajectories/`
- `configs/`

## Interpretation Guardrail

Small NSE/KGE differences should be reported as maintaining, reducing, or modestly improving skill.
The main diagnostic value is expected in the learned dynamic-parameter behavior, temporal variability, and boundary saturation ratios.
