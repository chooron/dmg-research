# MC-Parameter Correlation Analysis

This workflow exports Monte Carlo dropout parameter samples from `McMlpModel`
and prepares an analysis-ready table for studying the relationship between:

- basin static attributes,
- learned HBV parameters,
- MC-dropout uncertainty of those parameters.

## Goal

For each basin, run the trained `McMlpModel` multiple times with dropout enabled
(for example 10 times), then compute:

- the parameter sample distribution across MC passes,
- the per-basin parameter mean,
- the per-basin parameter variance.

The exported files are intended for downstream correlation analysis, such as:

- attribute ↔ parameter mean correlation,
- attribute ↔ parameter variance correlation,
- cluster-specific parameter behavior comparisons.

## Export command

Example for a fast analysis run using the 20-epoch checkpoint from held-out
cluster `A` and seed `20260325`:

```bash
uv run python scripts/export_mc_basin_parameters.py \
  --config conf/config_vrex_dhbv.yaml \
  --holdout A \
  --seed 20260325 \
  --epoch 20 \
  --dataset holdout \
  --mc-samples 10
```

If GPU memory is tight, force CPU inference:

```bash
uv run python scripts/export_mc_basin_parameters.py \
  --config conf/config_vrex_dhbv.yaml \
  --holdout A \
  --seed 20260325 \
  --epoch 20 \
  --dataset holdout \
  --mc-samples 10 \
  --device cpu
```

## Output files

By default, files are written to:

- `outputs/.../analysis/`

For `--dataset holdout --mc-samples 10`, the script writes:

- `mc_parameter_samples_holdout_mc10.csv`
  - one row per basin per MC pass
  - includes basin metadata, `pass_index`, `mc_seed`
  - includes normalized NN outputs as `norm_*`
  - includes physical-scale HBV parameters such as `parBETA`, `parFC`, `route_a`

- `mc_parameter_summary_holdout_mc10.csv`
  - one row per basin
  - includes `*_mean` and `*_var` for every exported parameter

- `mc_attribute_parameter_table_holdout_mc10.csv`
  - one row per basin
  - merges raw static attributes with the parameter summary table
  - this is the main file for correlation analysis

- `mc_parameter_export_meta_holdout_mc10.json`
  - records checkpoint, seed, split, and exported variable names

## Which table to analyze

Use:

- `mc_attribute_parameter_table_*.csv` for correlation/regression analysis
- `mc_parameter_samples_*.csv` for uncertainty visualization
- `mc_parameter_summary_*.csv` when only parameter statistics are needed

## Recommended analyses

### 1. Attribute vs parameter mean

Study whether a basin attribute is associated with the central tendency of a
learned HBV parameter.

Examples:

- `aridity` vs `parBETA_mean`
- `frac_snow` vs `parTT_mean`
- `soil_porosity` vs `parFC_mean`
- `slope_mean` vs `route_a_mean`

### 2. Attribute vs parameter variance

Study whether some basin types produce more uncertain parameters under MC
dropout.

Examples:

- `aridity` vs `parK1_var`
- `soil_depth_pelletier` vs `parFC_var`
- `frac_forest` vs `route_b_var`

### 3. Cluster-wise comparison

Compare parameter distributions across effective clusters:

- boxplots of `parBETA_mean` by `effective_cluster`
- boxplots of `parTT_var` by `effective_cluster`

## Suggested workflow in a notebook

```python
import pandas as pd

df = pd.read_csv(
    "outputs/vrex_dhbv_held_out_A_seed20260325/analysis/"
    "mc_attribute_parameter_table_holdout_mc10.csv"
)

target_cols = [
    "parBETA_mean",
    "parFC_mean",
    "parK0_mean",
    "parTT_mean",
    "route_a_mean",
]

corr = df.corr(numeric_only=True).loc[
    ["aridity", "frac_snow", "soil_porosity", "slope_mean"],
    target_cols,
]
print(corr)
```

## Spearman ranking and heatmaps

After exporting `mc_attribute_parameter_table_*.csv`, generate correlation
rankings and heatmaps with:

```bash
uv run python scripts/analyze_mc_parameter_correlation.py \
  --input outputs/vrex_dhbv_held_out_A_seed20260325/analysis/mc_attribute_parameter_table_holdout_mc10.csv \
  --top-k 30
```

This script writes:

- `spearman_parameter_mean_matrix.csv`
- `spearman_parameter_mean_ranking.csv`
- `spearman_parameter_mean_top30.csv`
- `spearman_parameter_mean_top30_per_parameter.csv`
- `spearman_parameter_mean_heatmap.png`
- `spearman_parameter_variance_matrix.csv`
- `spearman_parameter_variance_ranking.csv`
- `spearman_parameter_variance_top30.csv`
- `spearman_parameter_variance_top30_per_parameter.csv`
- `spearman_parameter_variance_heatmap.png`

### What each file means

- `*_matrix.csv`
  - full attribute × parameter Spearman matrix

- `*_ranking.csv`
  - global ranking of all attribute-parameter pairs by `abs(spearman_rho)`

- `*_top30.csv`
  - strongest global pairs only

- `*_top30_per_parameter.csv`
  - strongest attributes for each parameter separately

- `*_heatmap.png`
  - visual summary of the full correlation matrix

### Recommended reading order

1. open `*_top30.csv`
2. inspect `*_top30_per_parameter.csv`
3. use `*_heatmap.png` to spot block structure and shared patterns

### Notes

- the script analyzes physical-scale parameters only:
  - `par*_*`
  - `route_*_*`
- normalized NN outputs `norm_*` are excluded by default
- rankings are computed separately for:
  - parameter means
  - parameter variances

## Interpretation notes

- `norm_*` columns are NN outputs in normalized parameter space `[0, 1]`.
- physical parameter columns are post-processed by `HbvStatic._unpack(...)`
  using the HBV parameter bounds.
- with `nmul=1`, each basin has one scalar value per HBV parameter.
- MC variance reflects dropout-induced epistemic uncertainty, not hydrologic
  process noise.

## Validation

This export path should be checked with:

```bash
uv run python -m py_compile scripts/export_mc_basin_parameters.py
uv run python -m py_compile scripts/analyze_mc_parameter_correlation.py
uv run python scripts/export_mc_basin_parameters.py \
  --config conf/config_vrex_dhbv.yaml \
  --holdout A \
  --seed 20260325 \
  --epoch 20 \
  --dataset holdout \
  --mc-samples 10
uv run python scripts/analyze_mc_parameter_correlation.py \
  --input outputs/vrex_dhbv_held_out_A_seed20260325/analysis/mc_attribute_parameter_table_holdout_mc10.csv \
  --top-k 30
```
