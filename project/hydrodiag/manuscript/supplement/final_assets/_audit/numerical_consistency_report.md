# Numerical consistency report

Generated: 2026-08-27
Git HEAD at preflight: `322fb932c922a131a67800bcbc6aa7eb704c7605`

## Table S2 / Table S3 locks

- Table S2 Panel A uses KGE thresholds `0.40, 0.50, 0.60, 0.70, 0.80` and CT thresholds `10, 15, 20 d`. It contains 180 rows: two denominator types × two regimes × three structures × five KGE thresholds × three CT thresholds.
- At KGE `0.60`, common-pass N is 321 for IC and 331 for dPL, matching the Figure 2 common-pass definition.
- Table S2 Panel B uses the source denominator grid `1e-6, 1e-4, 1e-3, 0.01, 0.02, 0.05, 0.10`, with N_valid 427 for IC and 460 for dPL at the canonical cutoff.
- Table S3 Panel A uses N_total = 531, N_valid = 427 (IC), and N_valid = 460 (dPL) for the canonical `D_b > 10^-6` audit. It reports `F_close`, `F_TGD`, and paired `Delta F` quantiles and unclipped tails. S1–S5 totals are 165, 156, 121, 34, and 55.
- Table S3 Panel B canonical field rows use `G_Base`, `G_TGD`, `F_close`, `F_TGD*`, `Delta F`, and positive-Delta-F proportions from `summaries/canonical_registry.csv`. Direct-field rows use the alternative-field summary.

## CONFLICT: R3 canonical F_TGD source summaries

- **Source A:** `results/reviewer2_robustness/p0_reporting/recovery_denominator_tail_audit.csv` gives test `F_TGD_median = 0.5456` (IC) and `0.5240` (dPL), with `Delta F = 0.4600` and `0.4432`.
- **Source B:** `results/reviewer2_robustness/summaries/canonical_registry.csv` gives canonical `F_TGD_star = 0.545645` (IC) and `0.522659` (dPL), with `Delta F = 0.4600` and `0.4432`.
- **Likely canonical:** Source B for the canonical field comparison in Table S3 Panel B; Source A is retained unchanged for the full tail-distribution audit in Table S3 Panel A.
- **Reason:** The registry explicitly labels the dPL value as the seed-median canonical value and separately records the pooled-union N = 468; the tail audit is a distinct frozen summary. The values are not silently merged.

## Figure S1: HUC-2 regional omission

The final Figure S1 intentionally retains only source HUC_11–HUC_18 and displays them as HUC_01–HUC_08. Source HUC_01–HUC_10 are random ten-fold partitions.

- Full R1 references: IC 47.400 d; dPL 46.267 d. Retained ranges: IC 45.267–48.533 d; dPL 40.333–46.800 d.
- Full R3 references: IC +0.4600; dPL +0.4432. Retained ranges: IC +0.4547–+0.4960; dPL +0.4330–+0.4634.
- Full R5 references: IC 90.91%; dPL 90.91%. Retained ranges: IC 88.57–95.12%; dPL 90.00–92.68%.
- Retained omission count: 8 source regions per paradigm, 16 rows per CSV.

## Figure S2: TGD response and shape sensitivity

Source response grid: 351 temperature rows. Shape medians and valid N are:

| Variant | IC Delta F | dPL Delta F | IC N | dPL N |
|---|---:|---:|---:|---:|
| Sharp (`T_ref=0`, `s_T=1`) | -0.1588 | +0.4135 | 427 | 460 |
| Canonical (`T_ref=0`, `s_T=2`) | -0.1339 | +0.4410 | 427 | 460 |
| Warm-shifted (`T_ref=+2`, `s_T=2`) | -0.2189 | +0.2553 | 427 | 460 |
| Broad (`T_ref=0`, `s_T=4`) | -0.3657 | -0.0914 | 427 | 460 |

Only `Delta F` is used in panel (c). No uncertain panel is blocked.

## Figure S3: component values versus main aggregate

Final Figure S3 uses `delta_abs_e` for parameter components and `delta_E` for test NRMSE state/flux components. Selected component medians are:

- `xaj_k` parameter: IC Base 0.01542, IC TGD 0.00677; dPL Base 0.00703, dPL TGD 0.00142.
- `W_t` state: IC Base 0.27020, IC TGD 0.20987; dPL Base 0.04171, dPL TGD 0.00363.

Main Figure 6 aggregate anchors are:

- Parameter excess: IC Base 0.09740, IC TGD 0.07648; dPL Base 0.06364, dPL TGD 0.02384.
- `W_t` state excess: IC Base 0.27020, IC TGD 0.20987; dPL Base 0.04040, dPL TGD 0.00366.

The dPL component-source values differ slightly from the Figure 6 summary (`W_t`: 0.04171/0.00363 versus 0.04040/0.00366). This is a source-summary aggregation discrepancy, not silently corrected; the Figure S3 source is documented as `state_excess.csv` and the main aggregate remains documented as `figure6_summary.json`.

## Figure S4: seasonal pathway

Both `figure6_summary.json` and `fig6_seasonal_meta.json` report high-snow N = 133, 12 October–September water-year months, and the recorded-forward seasonal source. The plotted storage quantity is signed `Delta W_t = W_t(model) - W_t(truth)`.

## Figure S5: external-state examples and population

- Population source N = 531; final eligible N = 442.
- Eligible group counts: Low 88, Middle 177, High 177.
- Selected examples and water years: Low 02472000 (1996), Low 07195800 (2009), Middle 05495000 (2008), Middle 03473000 (1999), High 12167000 (1997), High 08377900 (2005).
- Selection is based on external Snow-17 SWE burden and snow-active eligibility; no KGE, Delta r, CT, or visual selection was used.

## Symbol audit summary

- Formal Methods uses `P_t`, `T_t`, `E_{p,t}`, and `P_t^*`; legacy `PET`/`pet` source aliases are not promoted in display labels.
- Formal TGD symbols are `S_t^g`, `tau_t`, and `r_t`; final Figure S2 uses time-indexed labels.
- XAJ storage/flux symbols are `W_{U,t}`, `W_{L,t}`, `W_{D,t}`, `W_t`, `Q_{i,t}`, `Q_{g,t}`, and `Q_t`.
- `s` versus derived `wt` remains a documented protocol/display distinction in `_audit/symbol_registry.md`; Figure S3 uses explicit `wt`.
- Local intermediate mappings `S_mm`, `S_0`, `O_i`, `O_g`, `R_i`, and `R_g` remain `UNRESOLVED_SYMBOL_MAPPING` and are not invented in final labels.

## Validation scope

This report compares existing CSV/JSON summaries and final outputs. No training, recalibration, state export, forward simulation, pytest, unittest, or full evaluation pipeline was run.
