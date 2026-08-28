# Figure S5 — Outcome-independent external-state examples and population corroboration

## 1. Scientific role

Figure S5 provides R4 external-state examples and population-level corroboration. It belongs in the SI because it illustrates external consistency across selected snow-burden episodes without replacing the main population estimates.

## 2. What is shown

Panels (a)–(f) show two example catchments from each external Snow-17 SWE burden group: Low, Middle, and High. Each example displays ERA5-Land `SM100` anomaly, Base/TGD/CN XAJ total tension-water storage anomaly, and the snow-active episode. Panel (g) shows population distributions of CN–Base and TGD–Base correlation contrasts against the external `SM100` reference by burden group.

## 3. Source data

- `manuscript/supplement/figures/FigureS1_R4_selection_audit.json`
- `manuscript/supplement/figures/FigureS1_R4_population_audit.csv`
- `results/r4_caravan_soil_reference_v1/caravan_soil_ensemble.npz`
- `results/r4_swe_reference_v1/swe_ensemble.npz`
- Original renderer: `manuscript/scripts/r4/plot_r4_figure_s1_multibasin.py`
- Frozen source image: `manuscript/supplement/figures/FigureS1_R4_multibasin_validation.png`

## 4. Sample definition

The population source contains 531 catchments. Eligibility requires at least 10 snow-active days (`SWE >= 5 mm`) in the snowiest water year. The final population panel contains 442 eligible catchments: Low N = 88, Middle N = 177, High N = 177. Examples are exactly two per group, selected by the external Snow-17 burden protocol; IDs and water years are retained in the audit JSON.

## 5. Metric definitions

`SM100` is the external ERA5-Land 0–100 cm depth-weighted soil-moisture composite. Model storage is `W_t = wu + wl + wd`. Panel (g) reports `Delta r` relative to `SM100`: CN–Base and TGD–Base anomaly-correlation contrasts. Selection does not use KGE, Delta r, CT, or visual appearance.

## 6. Aggregation and uncertainty

Panels (a)–(f) are selected-episode trajectories. Panel (g) uses population catchment values with group medians and source 95% confidence intervals in the audit annotations. No new seed aggregation, resampling, or external-data computation is run.

## 7. Generation method

- Script: `manuscript/supplement/final_assets/figures/Figure_S5/plot_Figure_S5.py`
- Command: `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/figures/Figure_S5/plot_Figure_S5.py`
- Method: validate the frozen audit selection and stage the existing source PNG; the original renderer is recorded for provenance but is not rerun.
- Output: `Figure_S5.png`.
- **NO MODEL TRAINING. NO RECALIBRATION. NO FULL TEST PIPELINE.**

## 8. Visual encoding

External reference `SM100` is dark neutral; Base is orange, TGD green/teal, and CN blue. Snow-active periods are pale blue. Outlined anchors mark the six selected examples. Population panel groups are Low, Middle, and High SWE burden.

## 9. Caption-ready factual statements

- Six example catchments are shown, two per externally ranked SWE burden group.
- The population panel contains 442 eligible catchments, grouped 88/177/177.
- Examples are selected using external Snow-17 SWE burden and snow-active eligibility.
- Population contrasts are CN–Base and TGD–Base against external `SM100`.

## 10. Interpretation boundary

This is external-state consistency corroboration, not proof of true internal storage, causal snow-process recovery, or an outcome-independent guarantee beyond the frozen selection protocol.

## 11. Validation

The audit JSON contains six examples with two Low, two Middle, and two High groups. The population CSV has 531 rows and 442 eligible rows with the expected 88/177/177 group counts. The staging script completed, source/output equality was checked, and the PNG was visually inspected for all seven panels, legends, group labels, and clipping.
