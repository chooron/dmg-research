# S4D/S5D Component Ablations

This folder contains the eight component-wise variants in the full 2 x 2 x 2
factorial S4D/S5D reviewer ablation. The variants only switch three components
in the dynamic parameter network:

| Variant | 1D conv | Normalization | Dynamic output activation |
| --- | --- | --- | --- |
| `S4DBaseline` | no | BatchNorm1d | Sigmoid |
| `S4DLN` | no | LayerNorm | Sigmoid |
| `S4DSoftsign` | no | BatchNorm1d | Softsign mapped to `[0, 1]` |
| `S4DLNSoftsign` | no | LayerNorm | Softsign mapped to `[0, 1]` |
| `S5DConvOnly` | yes | BatchNorm1d | Sigmoid |
| `S5DConvBNSoftsign` | yes | BatchNorm1d | Softsign mapped to `[0, 1]` |
| `S5DConvLNSigmoid` | yes | LayerNorm | Sigmoid |
| `S5DFull` | yes | LayerNorm | Softsign mapped to `[0, 1]` |

The shared benchmark settings are in
`project/bettermodel/conf/config_dhbv_s5d_ablation_base.yaml`. The eight runnable
variant configs are:

- `config_dhbv_ablation_s4d_baseline.yaml`
- `config_dhbv_ablation_s4d_ln.yaml`
- `config_dhbv_ablation_s4d_softsign.yaml`
- `config_dhbv_ablation_s4d_ln_softsign.yaml`
- `config_dhbv_ablation_s5d_conv_only.yaml`
- `config_dhbv_ablation_s5d_conv_bn_softsign.yaml`
- `config_dhbv_ablation_s5d_conv_ln_sigmoid.yaml`
- `config_dhbv_ablation_s5d_full.yaml`

Train and evaluate with the existing entrypoint, for example:

```bash
python project/bettermodel/run_experiment.py \
  --config conf/config_dhbv_ablation_s5d_full.yaml \
  --mode train_test
```

The existing evaluation pipeline writes NSE/KGE through `metrics.json`. Use
`export_diagnostics.py` after training/testing to save normalized dynamic
parameter trajectories, median absolute day-to-day parameter change, and
boundary saturation ratios:

```bash
python -m project.bettermodel.implements.neural_networks.ablation.export_diagnostics
```
