# Figure S4 provenance

- **Scientific question:** What continuous temperature response does TGD implement, and how does the tested response-shape choice affect the controlled recovery contrast?
- **Input files:**
  - `results/reviewer2_robustness/tgd_response/tgd_response_data.csv`
  - `results/reviewer2_robustness/tgd_shape_sensitivity/tgd_shape_sensitivity_basin_metrics.csv`
  - Formulation check: `results/reviewer2_robustness/tgd_response/tgd_response_summary.md` and `models/tgd2.py`.
- **Input columns:** Response CSV `temperature_c` plus six recorded `tau_*` and six matching `retention_*` columns. Shape CSV `variant`, `t_ref`, `s_t`, `delta_F_ic`, and `delta_F_dpl` (with the companion KGE/gain/fraction columns retained in the source file).
- **Aggregation:** Panels a–b plot the six recorded response settings at 351 temperatures; no response is re-simulated. Panel c summarizes each variant by median and empirical Q25–Q75 of finite basin metrics.
- **N:** Response panel has 351 temperature rows. Shape panel uses the denominator-valid canonical samples recorded in the source: `N=427` IC and `N=460` dPL.
- **IC/dPL handling:** Panel c uses open circles for IC and filled triangles for dPL; the response curves are a single TGD parameter family and are not split by estimation regime.
- **Interval definition:** Panel c intervals are empirical basin Q25–Q75; no new bootstrap was run. Panel a–b have no statistical interval.
- **Plot script:** `manuscript/scripts/supplement/plot_tgd_response_sensitivity.py`.
- **Output:** `manuscript/supplement/figures/FigureS4_tgd_response_shape_sensitivity.png`.
- **Canonical values checked:** dPL median `ΔF` is +0.4135 (sharp, `s_T=1`), +0.4410 (canonical, `s_T=2`), +0.2553 (warm-shifted, `T_ref=+2`), and −0.0914 (broad, `s_T=4`). IC values are −0.1588, −0.1339, −0.2189, and −0.3657 in the same order.
- **Claim boundary:** TGD is a continuous generic thermal reservoir; the shape results do not establish a universal robustness range or an explicit snow module.
