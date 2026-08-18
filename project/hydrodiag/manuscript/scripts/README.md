# Manuscript Statistical and Plotting Scripts Catalog

This directory contains the production-grade, read-only statistical evaluation,
figure generation, and table compilation pipelines for Results R1, R2, and R4.

## Pipeline Map

| Result | Main Build Entry Point | Figure Generators | Table Generators | Primary Outputs |
|---|---|---|---|---|
| **R1** | `build_r1_statistics.py` | `plot_r1_figure1.py` | `generate_table1.py`, `generate_table_s1..s3.py` | `manuscript/tables/Table1*`, `manuscript/figures/figure1*` |
| **R2** | `run_r2_parameter_statistics.py` | `plot_r2_figure3_final.py`, `plot_r2_figure4.py` | `run_r2_robustness_checks.py`, `run_r2_tgd2_specificity.py` | `manuscript/tables/Table2*`, `manuscript/figures/figure3*` |
| **R4** | `build_r4_soil_statistics.py` | `plot_r4_figure4.py` | `generate_table_r4.py` | `manuscript/tables/Table4*`, `manuscript/tables/TableS4*`, `manuscript/figures/figure4_r4*`, `results/r4_phase1_soil_official/` |

## R4 Reproduction Commands

```bash
# 1. End-to-end R4 state consistency & robustness build
python manuscript/scripts/build_r4_soil_statistics.py --device cuda

# 2. Compile Figure 4 (4-panel publication figure, PNG & PDF)
python manuscript/scripts/plot_r4_figure4.py

# 3. Compile Table 4 (Main text) and Table S4 (Supplement) in Markdown & LaTeX
python manuscript/scripts/generate_table_r4.py
```

## Detailed Handoff

See `HANDOFF_R4.md` in this directory for the full statistical methods,
provenance logs, metric formulas, and paper writing guidance.
