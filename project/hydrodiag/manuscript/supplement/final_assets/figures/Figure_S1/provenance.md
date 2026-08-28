# Provenance

Final asset: `manuscript/supplement/final_assets/figures/Figure_S1/Figure_S1.png`
Generated: 2026-08-27
Git hash: `322fb932c922a131a67800bcbc6aa7eb704c7605`

## Inputs

- `results/reviewer2_robustness/regional_loro/r1_huc2_loro.csv`
- `results/reviewer2_robustness/regional_loro/r3_huc2_loro.csv`
- `results/reviewer2_robustness/regional_loro/r5_huc2_loro.csv`
- `manuscript/scripts/supplement/plot_huc2_loro_robustness.py`

## Columns / keys

R1: `paradigm`, `region_removed`, `S5_minus_S1_contrast`. R3: `paradigm`, `period`, `region_removed`, `Delta_F_median`. R5: `paradigm`, `region_removed`, `P_majority_positive`.

## Filters

Only source regions HUC_11–HUC_18 were retained. Full-sample rows were retained only as reference lines. Source HUC_01–HUC_10 were excluded because they are random ten-fold partitions.

## Transformations

Source HUC_11–HUC_18 were displayed as HUC_01–HUC_08. R5 majority agreement was converted from fraction to percent. No source result was recomputed.

## Statistical operations

None beyond the source CSV summaries. No interval or resampling operation was added.

## Plot script

`manuscript/supplement/final_assets/figures/Figure_S1/plot_Figure_S1.py`

## Command

`export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/figures/Figure_S1/plot_Figure_S1.py`

## Output

Three-panel PNG with eight displayed HUC categories per panel, IC/dPL omission points, and full-sample reference lines. Image checksum is recorded in the final execution report.

## Image size / checksum

4073 × 1593 px, RGBA; SHA-256 `83092c031751b89e3f39df2c14f56a5f17804647412c541cb99f5b5eb2c422b5`.

## Known caveats

This is regional omission robustness, not spatial correction. The final labels are submission-facing labels; the source IDs remain HUC_11–HUC_18 in the input CSVs.
