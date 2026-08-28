# HydroDiag Manuscript Scripts Reference

All manuscript-facing code is located below this directory and grouped by result family / chapter section.

```text
manuscript/scripts/
├── r0/         Figure 1: CAMELS study area, snow distribution & Q3 high-snow catchment map
├── r1/         Figure 2: Outlet hydrograph timing, signed CT errors, performance along snow gradient
├── r2/         Figure 3 & 4: Internal parameter compensation & state distortions in Base/CN/TGD
├── r3/         Figure 5 & 6: Controlled synthetic truth experiment (structural-gap recovery)
├── r4/         Figure 7 & 8, Figure S1: ERA5-Land SM100 & Snow-17 SWE soil moisture dynamics & timing
├── r5/         Figure 9: Cross-model structural replication across XAJ, GR4J, and SIMHYD
├── shared/     Table 1, Table 2, Table S1, Table S2 generators, asset registry, plot styles & audit tools
└── supplement/ Figure S3 (alt-generating field), Figure S4 (TGD sensitivity), Figure S5 (HUC2 LORO)
```

## Quick Figure Execution

```bash
# Main Figures
python project/hydrodiag/manuscript/scripts/r0/plot_r0_figure1.py
python project/hydrodiag/manuscript/scripts/r1/plot_r1_figure2.py
python project/hydrodiag/manuscript/scripts/r2/plot_r2_figure3_final.py
python project/hydrodiag/manuscript/scripts/r2/plot_r2_figure4_canonical.py
python project/hydrodiag/manuscript/scripts/r3/plot_figure5.py
python project/hydrodiag/manuscript/scripts/r3/plot_figure6.py
python project/hydrodiag/manuscript/scripts/r4/plot_r4_figure7.py
python project/hydrodiag/manuscript/scripts/r4/plot_r4_figure8.py
python project/hydrodiag/manuscript/scripts/r5/plot_r5_figure9.py

# Supplementary Figures
python project/hydrodiag/manuscript/scripts/r4/plot_r4_figure_s1_multibasin.py
python project/hydrodiag/manuscript/scripts/supplement/plot_alt_generating_field_robustness.py
python project/hydrodiag/manuscript/scripts/supplement/plot_tgd_response_sensitivity.py
python project/hydrodiag/manuscript/scripts/supplement/plot_huc2_loro_robustness.py
```

## Quick Table Execution

```bash
python project/hydrodiag/manuscript/scripts/shared/generate_table1_structural_configurations.py
python project/hydrodiag/manuscript/scripts/shared/generate_table2_controlled_recovery.py
python project/hydrodiag/manuscript/scripts/shared/generate_table_s1_parameter_bounds.py
python project/hydrodiag/manuscript/scripts/shared/generate_table_s2_sensitivity.py
```

## Asset Output Locations

- Main & Supplementary Figures: `project/hydrodiag/manuscript/figures/`
- Main & Supplementary Tables (CSV / MD / TeX): `project/hydrodiag/manuscript/stats/tables/`
- Full asset registry: `project/hydrodiag/manuscript/figure_manifests/canonical_assets.json`
