# Figure S2 — TGD temperature-response characteristics and fixed-shape sensitivity

## 1. Scientific role

Figure S2 documents the mathematical response family used by TGD and its completed response-shape sensitivity. It corresponds to R3/reviewer-2 robustness evidence and belongs in the SI because it explains a control formulation without adding another main-text outcome panel.

## 2. What is shown

Panel (a) plots residence time `tau_t` against temperature for six frozen response settings: default, fitted P10, fitted median, fitted P90, upper bound, and lower bound. Panel (b) plots the corresponding daily retention `r_t` and shows the continuous thermal gate. Panel (c) shows one recovery estimand, median `Delta F = F_TGD* - F_close`, for Sharp, Canonical, Warm-shifted, and Broad response shapes with IC and dPL values.

## 3. Source data

- `results/reviewer2_robustness/tgd_response/tgd_response_data.csv`
- `results/reviewer2_robustness/tgd_response/tgd_response_summary.md`
- `results/reviewer2_robustness/tgd_shape_sensitivity/tgd_shape_sensitivity_basin_metrics.csv`
- `results/reviewer2_robustness/tgd_shape_sensitivity/tgd_shape_sensitivity_summary.json`
- Formal equations: `manuscript/methods_supplement_production_audit.md`, Section 3.2

## 4. Sample definition

Panels (a)–(b) use the 351-temperature response grid. Panel (c) uses the frozen basin-level test metrics, with denominator-valid N = 427 for IC and 460 for dPL in the canonical source summary. Shape variants are `s_T = 1`, canonical `s_T = 2`, `T_ref = +2, s_T = 2`, and `s_T = 4` °C.

## 5. Metric definitions

`tau_t` is the temperature-conditioned TGD residence time and `r_t = exp(-1/tau_t)` is the daily retention factor. `Delta F` is the paired recovery contrast between TGD and close/reference recovery. `T_ref` and `s_T` are fixed gate constants in the canonical formulation; panel (c) changes them only in the frozen sensitivity variants.

## 6. Aggregation and uncertainty

Panels (a)–(b) are deterministic response curves. Panel (c) uses basin-level medians and empirical Q25–Q75 intervals from the existing shape-sensitivity CSV. The source metrics already encode the canonical IC and dPL handling; no new seed aggregation or bootstrap is run.

## 7. Generation method

- Script: `manuscript/supplement/final_assets/figures/Figure_S2/plot_Figure_S2.py`
- Command: `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; /home/jingxin/code/dmg-research/.venv/bin/python manuscript/supplement/final_assets/figures/Figure_S2/plot_Figure_S2.py --out manuscript/supplement/final_assets/figures/Figure_S2/Figure_S2.png`
- Input: frozen response and basin-metric CSVs.
- Output: `Figure_S2.png`.
- **NO MODEL TRAINING. NO RECALIBRATION. NO FULL TEST PIPELINE.**

## 8. Visual encoding

The TGD response family uses the canonical green/teal palette; line style identifies response setting. IC is an open circle and dPL a filled triangle in panel (c). Horizontal zero reference and empirical interval bars are shown without additional hue families.

## 9. Caption-ready factual statements

- Panels (a) and (b) contain the 351-temperature mathematical response curves.
- Panel (c) contains four fixed-shape variants and one recovery estimand, `Delta F`.
- Intervals in panel (c) are Q25–Q75; IC and dPL are visually distinct.
- The response is continuous in temperature rather than a discrete snow/rain partition.

## 10. Interpretation boundary

The figure tests response-shape sensitivity within the frozen TGD family. It does not establish a universal bound, physical snow equivalence, or a new hydrological experiment.

## 11. Validation

All six response-column pairs and all four shape variants were found in the source files. The renderer ran successfully, the output was visually inspected for curve completeness and panel-c readability, and the displayed canonical/sensitivity values were checked against the source summary. No panel was blocked.
