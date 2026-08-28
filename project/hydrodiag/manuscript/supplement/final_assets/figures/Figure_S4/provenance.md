# Provenance

Final asset: `manuscript/supplement/final_assets/figures/Figure_S4/Figure_S4.png`
Generated: 2026-08-27
Git hash: `322fb932c922a131a67800bcbc6aa7eb704c7605`

## Inputs

- `manuscript/results/R3/figure6_summary.json`
- `manuscript/results/R3/fig6_seasonal/fig6_seasonal_meta.json`
- `manuscript/results/R3/fig6_seasonal/fig6_seasonal_input.npz`
- `manuscript/results/R3/fig6_seasonal/fig6_seasonal_state.npz`
- `manuscript/scripts/r3/plot_r3_si_seasonal_trajectories.py`

## Columns / keys

Summary keys: `panel_e_seasonal_input` and `panel_f_seasonal_storage_heatmap`; the latter contains `median_matrix`, `iqr_matrix`, `row_iqr_medians`, `row_labels`, and `months`. Model series are Base, TGD, CN, and Truth for IC/dPL where available.

## Filters

The source-defined high-snow subset was retained: `frac_snow >= Q75`, N = 133. The test period is 1995-10-01 to 2010-09-30, with 12 water-year months from October through September.

## Transformations

The existing summary was plotted under final Figure S4 naming. `P_t^*` is displayed as effective input. The storage panel uses the signed source `Delta W_t = W_t(model) - W_t(truth)` and does not convert it to absolute error or NRMSE.

## Statistical operations

The source metadata defines per-basin monthly means, dPL median across seeds per basin, and across-high-snow-basin medians/IQRs. The renderer uses source `ci_lo`/`ci_hi` bands and source row-wise IQR values; no new resampling is run.

## Plot script

`manuscript/supplement/final_assets/figures/Figure_S4/plot_Figure_S4.py`

## Command

`export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/figures/Figure_S4/plot_Figure_S4.py --out-dir manuscript/supplement/final_assets/figures/Figure_S4`

## Output

Two-panel seasonal trajectory/storage-deviation PNG with a storage heterogeneity inset. Image checksum is recorded in the final execution report.

## Image size / checksum

6540 × 2552 px, RGBA; SHA-256 `4f78dbedeb828978e4051eb8570984ebd4423cc9019efa40ee539b3b69b57908`.

## Known caveats

This asset reuses recorded-forward summary/array outputs. The source metadata has historical dirty-Git/CUDA provenance, but no model computation is performed by the final plotting script. The plotted storage quantity is signed deviation.
