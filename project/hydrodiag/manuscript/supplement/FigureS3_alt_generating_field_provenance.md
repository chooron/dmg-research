# Figure S3 provenance

- **Scientific question:** Does the Base-refit versus TGD recovery ordering persist when the canonical PCA/Ridge generating field is replaced by the direct basin-wise calibrated CN–IC field?
- **Input files:**
  - Canonical: `manuscript/results/R3/figure5_basin_seedmedian.csv`.
  - Alternative: `results/reviewer2_robustness/alt_generating_field/alt_generating_field_basin_seedmedian.csv` and `alt_generating_field_summary.json`.
- **Input columns:** `basin_id`, `paradigm`, `period`, `kge_base_no_refit`, `kge_base`/`kge_tgd`, `kge_cn`, `D`, `G_base`, `G_TGD`/`G_tgd`, `F_close`, `F_TGD_star`/`F_tgd_star`, and `delta_F` where present.
- **Aggregation:** Test-period rows only. Panel a shows median and empirical Q25–Q75 interval of raw `G_Base` and `G_TGD` over all 531 basins. Panel b shows unclipped `F_close` and `F_TGD*` over rows with `D_b > 10^-6`; `ΔF` is the recorded `delta_F` when available and otherwise `F_TGD* − F_close` computed row-wise.
- **N:** Raw panel `N=531` for each field/regime. Fraction panel: canonical IC `N=427`, canonical dPL `N=460`, direct-field IC `N=522`, direct-field dPL **`N=123`**.
- **IC/dPL handling:** Four x-axis groups: canonical IC/dPL and direct CN–IC IC/dPL. IC and dPL are comparison regimes, not a rank ordering.
- **Interval definition:** Empirical basin Q25–Q75; no new bootstrap was run. Fractions are not clipped.
- **Plot script:** `manuscript/scripts/supplement/plot_alt_generating_field_robustness.py`.
- **Output:** `manuscript/supplement/figures/FigureS3_alt_generating_field_robustness.png`.
- **Canonical values checked:** Canonical test values reproduce approximately `ΔF=+0.460` (IC, 91.6% positive) and `+0.443` (dPL, 92.8% positive). Direct-field values are `+0.195` (IC, 72.4%, `N=522`) and `+0.701` (dPL, 91.1%, `N=123`).
- **Claim boundary:** Absolute fractions vary with generating-field construction; the figure supports ordering/sign comparison, not absolute generating-field independence.
