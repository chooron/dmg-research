# Manuscript Plotting Scripts

All scripts in this directory produce publication figures for the δ_dist
parameterisation paper. Figures are saved to:

- **Main figures**: `manuscript/figures/main/`
- **Appendix figures**: `manuscript/figures/appendix/` (where applicable)

Shared style, colours, paths, and helper functions live in `common.py`.

---

## Script inventory

### `common.py`
Shared utilities imported by all other scripts.
- Model colours (`MODEL_COLORS`): deterministic `#4C78A8`, mc_dropout `#F58518`,
  distributional `#2A9D8F`
- Process-group and class colour palettes
- `setup_style()`: applies Times New Roman font, 600 DPI, WRR/HESS-compatible rcParams
- Helper functions: `p_label`, `a_label`, `clean_axes`, `add_panel_label`,
  `draw_heatmap`, `parameter_family`, `math_model_labels`
- Path constants: `TABLE_ROOT`, `CORR_ROOT`, `MAIN_FIG_DIR`, etc.

---

### Figure scripts

| Script | Output figure | Description |
|--------|--------------|-------------|
| `fig01_predictive_performance.py` | `Fig01_predictive_performance` | Boxplots of NSE, KGE, and bias for the three models (δ_base, δ_mcd, δ_dist) across 531 basins. Module-style (imported via `__init__.py`). |
| `fig02_cross_seed_parameter_stability.py` | `Fig02_parameter_stability_boundary_interval` | Cross-seed parameter stability: boundary-interval plots showing parameter variance across seeds for each model. Module-style. |
| `plot_fig02_parameter_stability.py` | `Fig02_*` | Standalone version of Fig02. Kernel-density and interval plots of parameter distributions across seeds and loss functions. |
| `plot_fig03_relationship_seed_stability.py` | `Fig03_cross_seed_relationship_stability` | Cross-seed stability of attribute–parameter Spearman correlations. Heatmap or dot-plot showing ρ mean ± SD across seeds for all three models. |
| `plot_fig04_basin_group_relationship_stability.py` | `Fig04_basin_group_relationship_stability` | Basin-group relationship stability. Clusters basins by dominant parameter controls and shows how relationship classes (shared / partially shared / model-sensitive) vary across groups. |
| `plot_fig05_parameter_spatial_maps.py` | `Fig05_parameter_spatial_maps` | Spatial maps of δ_dist parameter means across the 531 CAMELS basins. One panel per parameter, coloured by normalised mean value. |
| `plot_fig06_mean_attribute_relationships.py` | `Fig06_mean_attribute_relationships` | Heatmap matrix of mean Spearman ρ between all parameters and basin attributes for each model. Annotated with relationship class (robust / supportive / secondary). |
| `plot_fig07_uncertainty_attribute_relationships.py` | `Fig07_uncertainty_attribute_relationships` | Uncertainty structure diagnostics: (a) circle heatmap of uncertainty–attribute correlations; (b) parameter-level uncertainty structure strength; (c) mean–std coupling vs boundary sensitivity scatter. |
| `plot_fig08_key_environmental_gradients.py` | `Fig08_key_environmental_gradients` | 3 x 4 panel figure of key environmental gradients: panels (a-f) show selected parameter-mean gradients, and panels (g-l) show selected parameter-uncertainty gradients. |
| `plot_fig09_attribute_parameter_gradients.py` | `Fig09_attribute_parameter_gradients` | **δ_dist parameter mean gradients.** 5 columns x 4 rows (20 panels). Each column = one process group (Snow, Aridity/ET, Terrain, Soil, Routing). x-axis: basin attribute; y-axis: δ_dist parameter mean. Quantile-binned median + IQR scatter. Spearman rho and seed SD annotated per panel. |
| `plot_fig10_parameter_uncertainty_gradients.py` | `Fig10_parameter_uncertainty_attribute_gradients` | **δ_dist parameter uncertainty gradients.** Same 5 x 4 layout as Fig09. y-axis: δ_dist parameter std (mean over seeds x loss functions). Shows structured uncertainty variation along basin attribute gradients. Includes interpretation caveat in notes. |
| `plot_fig11a_case_basin_map.py` | `Fig11a_case_basin_map` | CONUS map of the representative snow, arid, humid, steep, soil/storage, and routing-sensitive case basins. |
| `plot_fig11bcd_case_basin_parameter_regimes.py` | `Fig11bcd_case_basin_parameter_regimes` | **Representative basin cases.** Panels (b)-(d) show basin attribute percentile ranks, δ_dist parameter mean + std summaries, and per-case horizontal lollipop profiles. Case basins are data-driven selections covering snow, arid, humid, steep, soil/storage, and routing regimes. Analysis outputs in `manuscript/analysis/case_study/`. |

---

## Figure sequence in paper

```
Fig01  Predictive performance (NSE / KGE / bias)
Fig02  Cross-seed parameter stability
Fig03  Cross-seed relationship stability (attribute–parameter ρ)
Fig04  Basin-group relationship stability
Fig05  Parameter spatial maps
Fig06  Mean attribute–parameter relationship matrix
Fig07  Uncertainty structure and diagnostics
Fig08  Key environmental gradients (mean + uncertainty, 12 panels)
Fig09  Reliable attribute–parameter mean gradients (δ_dist, 20 panels)
Fig10  Parameter uncertainty gradients (δ_dist, 20 panels)
Fig11  Representative basin cases (case-study style, panels a-d split across two PNGs)
```

---

## Running a script

All standalone scripts (`plot_fig*.py`) can be run directly from the repo root:

```bash
python project/parameterize/manuscript/plots/plot_fig09_attribute_parameter_gradients.py
python project/parameterize/manuscript/plots/plot_fig10_parameter_uncertainty_gradients.py
```

Module-style scripts (`fig01_*.py`, `fig02_*.py`) are invoked via the package:

```bash
python -m project.parameterize.manuscript.plots
```

or imported individually:

```python
from project.parameterize.manuscript.plots import fig01_predictive_performance
```

---

## Key data sources

| Data | Path |
|------|------|
| Parameter estimates (mean, std) | `outputs/analysis/stability_stats/tables/params_long.csv` |
| Basin attributes (531 CAMELS) | `outputs/analysis/stability_stats/tables/basin_attributes.csv` |
| Seed stability statistics | `outputs/analysis/stability_stats/correlation_summaries/pair_seed_stability.csv` |
| Correlation matrices | `outputs/analysis/stability_stats/correlation_summaries/matrices/` |
| Relationship classes | `outputs/analysis/stability_stats/correlation_summaries/relationship_classes.csv` |

---

## Style reference

| Property | Value |
|----------|-------|
| Font | Times New Roman |
| DPI | 600 |
| Default figure width | 180 mm |
| Axes edge colour | `#333333` |
| Axis label size | 10.5 pt |
| Tick label size | 9.2 pt |
| Model: deterministic | `#4C78A8` |
| Model: mc_dropout | `#F58518` |
| Model: distributional | `#2A9D8F` |
| Process: Snow | `#6FA8C9` |
| Process: Aridity/ET | `#D4956A` |
| Process: Terrain | `#7AAF7A` |
| Process: Soil | `#B89A6E` |
| Process: Routing | `#9B8FBF` |
