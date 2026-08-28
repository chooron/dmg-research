# Figure S4 — High-snow seasonal liquid-water delivery and tension-water-storage trajectories

## 1. Scientific role

Figure S4 separates seasonal pathway evidence from the R3 main aggregate results. It corresponds to the high-snow controlled subset and belongs in the SI because it shows seasonal delivery/storage trajectories rather than another aggregate recovery ranking.

## 2. What is shown

Panel (a) shows October–September effective liquid-water input for generating CN truth, Base refit, TGD refit, and CN refit, with IC and dPL lines. Panel (b) shows the signed median total tension-water-storage deviation from truth as a monthly heatmap and the row-wise IQR heterogeneity for the six structure/regime combinations.

## 3. Source data

- `manuscript/results/R3/figure6_summary.json`
- `manuscript/results/R3/fig6_seasonal/fig6_seasonal_meta.json`
- `manuscript/results/R3/fig6_seasonal/fig6_seasonal_input.npz`
- `manuscript/results/R3/fig6_seasonal/fig6_seasonal_state.npz`
- Source renderer: `manuscript/scripts/r3/plot_r3_si_seasonal_trajectories.py`

## 4. Sample definition

The high-snow subset is defined by `frac_snow >= Q75`, with N = 133 catchments. The evaluation period is test, 1995-10-01 to 2010-09-30, displayed on a water-year October–September axis. dPL seed values are collapsed per basin before the across-basin summary.

## 5. Metric definitions

Panel (a) displays effective input `P_t^*`. Panel (b) displays signed `Delta W_t = W_t(model) - W_t(truth)`, where `W_t = W_{U,t} + W_{L,t} + W_{D,t}`. The plotted storage summary is not an absolute error or NRMSE.

## 6. Aggregation and uncertainty

The source metadata defines per-basin means for each water-year month across the test period, dPL median across seeds per basin, and median/IQR across high-snow basins. The input summary provides `median`, `q25`, `q75`, `ci_lo`, and `ci_hi`; the renderer shows the source 95% CI bands in panel (a), and median plus row-wise IQR heterogeneity in panel (b). No new resampling is run.

## 7. Generation method

- Script: `manuscript/supplement/final_assets/figures/Figure_S4/plot_Figure_S4.py`
- Command: `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/figures/Figure_S4/plot_Figure_S4.py --out-dir manuscript/supplement/final_assets/figures/Figure_S4`
- Input: frozen Figure 6 summary and recorded-forward seasonal arrays/metadata.
- Output: `Figure_S4.png`.
- **NO MODEL TRAINING. NO RECALIBRATION. NO FULL TEST PIPELINE.**

## 8. Visual encoding

Base is orange, TGD green/teal, CN blue, and generating CN truth dark neutral. Solid/dashed lines distinguish IC/dPL where applicable. The storage heatmap is centered at zero and the right-hand bars show IQR in millimeters.

## 9. Caption-ready factual statements

- The high-snow seasonal subset contains N = 133 catchments.
- The horizontal axis is water-year month, October–September.
- Panel (a) compares effective inputs; panel (b) reports signed total tension-storage deviation from truth.
- Bands are source 95% CI bands; right-hand bars summarize row-wise IQR.

## 10. Interpretation boundary

Seasonal trajectory agreement is descriptive controlled-experiment evidence. It is not evidence of real-catchment truth, causal seasonal attribution, or a new model evaluation.

## 11. Validation

The source summary and metadata both report N = 133 and 12 October–September months. Output rows and labels were checked against the source summary, the renderer completed successfully, and the PNG was visually inspected for panel labels, legend, color encoding, heatmap scale, and clipping. No state export or forward simulation was run.
