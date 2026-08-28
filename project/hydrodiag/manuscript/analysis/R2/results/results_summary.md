# Canonical R2 Statistical Audit and Rebuild Report

- **Status:** COMPLETED
- **Canonical Promotion Gates:** **PASS** (12/12 Gates PASS)
- **Dataset Dimensions:** 531 basins × 3 structures (Base, CN, TGD) × 2 paradigms (IC, dPL)
- **Raw Ledger:** 310635 rows (IC: 10 restarts, dPL: 3 seeds × 15 parameters)
- **Canonical Vectors:** 3186 rows
- **Bootstrap Settings:** 200 resamples (Seed `20260730`, unit = basin)
- **Execution Time:** 66.52 s (Peak RAM: 2117.8 MB)

## 1. Executive Summary of Canonical Gates & Reconciliation

| Validation Gate | Status | Description |
| :--- | :---: | :--- |
| Gate 01: Provenance | **PASS** | All statistics traceable to explicit restart/seed artifacts. |
| Gate 02: Shared Parameter Definition | **PASS** | 15 shared parameters identities, bounds, and order verified. |
| Gate 03: Normalized Coordinates | **PASS** | z = (phys - lower)/(upper - lower) verified with max diff 0. |
| Gate 04: Canonical Vector Rule | **PASS** | IC best train-KGE and dPL across-seed median verified. |
| Gate 05: Ensemble Formulas | **PASS** | IC 45 within + 100 between, dPL 3 within + 9 between exact. |
| Gate 06: Basin Weighting | **PASS** | Pairwise metrics reduced to basin-level (N=531). |
| Gate 07: Basin Joins | **PASS** | Explicit basin-ID (8-digit string) and parameter name joins. |
| Gate 08: Snow Axis | **PASS** | S1=165, S2=156, S3=121, S4=34, S5=55 matches frozen R1 manifest. |
| Gate 09: Paired Bootstrap | **PASS** | Base-CN, Base-TGD, and Delta_beta paired on same resamples. |
| Gate 10: Historical Conflicts | **PASS** | Prevalence 63.1%/83.8% and slope 0.1542 resolved and proven. |
| Gate 11: All-Parameter Transparency | **PASS** | All 15 parameters computed without significance filtering. |
| Gate 12: Scope | **PASS** | No truth/mechanistic claims, no IC/dPL superiority ranking. |

## 2. Primary Macro: Whole-Parameter-Space Base–CN Response (Figure 3)

### A. Ensemble Within-Adjusted Structural Separation

| Paradigm | Subset | n | within_pooled (median) | between_all (median) | excess (median [95% CI]) | fraction(between > within) [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| IC | Full531 | 531 | 0.4754 | 0.4913 | +0.0075 [+0.0049, +0.0109] | 63.1% [59.5%, 66.9%] |
| IC | ExcludeS5 | 476 | 0.4760 | 0.4850 | +0.0050 [+0.0029, +0.0071] | 59.0% [54.2%, 63.0%] |
| IC | S1 | 165 | 0.4729 | 0.4705 | -0.0018 [-0.0044, +0.0021] | 46.7% [40.0%, 53.9%] |
| IC | S5 | 55 | 0.4520 | 0.5605 | +0.0829 [+0.0701, +0.1104] | 98.2% [94.5%, 100.0%] |
| dPL | Full531 | 531 | 0.1056 | 0.1817 | +0.0611 [+0.0504, +0.0794] | 83.8% [80.6%, 86.8%] |
| dPL | ExcludeS5 | 476 | 0.1046 | 0.1746 | +0.0553 [+0.0457, +0.0730] | 83.4% [80.2%, 86.6%] |
| dPL | S1 | 165 | 0.1052 | 0.1322 | +0.0186 [+0.0071, +0.0260] | 70.3% [63.6%, 77.0%] |
| dPL | S5 | 55 | 0.1194 | 0.3066 | +0.1252 [+0.0787, +0.2053] | 87.3% [78.2%, 94.5%] |

### B. Macro Regression Slopes on Snow Fraction

| Paradigm | Subset | Dependent Variable | Slope beta [95% CI] | Spearman rho [95% CI] |
| :--- | :--- | :--- | :---: | :---: |
| IC | Full531 | excess | +0.1542 [+0.1341, +0.1732] | +0.549 [+0.493, +0.602] |
| IC | ExcludeS5 | excess | +0.1387 [+0.1103, +0.1662] | +0.406 [+0.331, +0.481] |
| dPL | Full531 | excess | +0.1974 [+0.1362, +0.2653] | +0.441 [+0.363, +0.520] |
| dPL | ExcludeS5 | excess | +0.4271 [+0.3302, +0.5261] | +0.465 [+0.379, +0.538] |

## 3. TGD Attribution Control & Paired Slope Differences (Figure 3)

| Paradigm | Subset | beta(Base-CN) [95% CI] | beta(Base-TGD) [95% CI] | Paired Delta_beta [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| IC | Full531 | +0.1542 [+0.1360, +0.1735] | +0.1538 [+0.1260, +0.1812] | **+0.0004 [-0.0292, +0.0303]** |
| IC | ExcludeS5 | +0.1387 [+0.1134, +0.1682] | +0.1155 [+0.0816, +0.1650] | **+0.0232 [-0.0108, +0.0542]** |
| dPL | Full531 | +0.1974 [+0.1388, +0.2629] | +0.1563 [+0.1102, +0.2143] | **+0.0412 [+0.0106, +0.0823]** |
| dPL | ExcludeS5 | +0.4271 [+0.3365, +0.5277] | +0.3410 [+0.2702, +0.4197] | **+0.0861 [+0.0173, +0.1554]** |

## 4. Explanatory Parameter Shifts: All 15 Parameters (Figure 4)

| Parameter | Description | Category | IC Slope beta [95% CI] | IC rho | dPL Slope beta [95% CI] | dPL rho |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `xaj_k` | k | soil | -0.0285 [-0.0909, +0.0333] | -0.13 | -0.0036 [-0.0454, +0.0409] | -0.07 |
| `xaj_b` | b | soil | -0.0059 [-0.1671, +0.2234] | +0.02 | +0.0238 [-0.0568, +0.1118] | +0.09 |
| `xaj_im` | im | soil | -0.3635 [-0.4905, -0.1834] | -0.25 | -0.1418 [-0.2212, -0.0628] | -0.35 |
| `xaj_um` | um | soil | +0.5211 [+0.3529, +0.7064] | +0.22 | +0.5655 [+0.3765, +0.7688] | +0.25 |
| `xaj_lm` | lm | soil | +0.4001 [+0.2208, +0.5621] | +0.19 | +0.0922 [+0.0192, +0.1865] | +0.09 |
| `xaj_dm` | dm | soil | -0.5373 [-0.7832, -0.2724] | -0.12 | -0.1061 [-0.2076, +0.0042] | -0.07 |
| `xaj_c` | c | soil | +0.2783 [+0.0513, +0.5154] | +0.08 | +0.2535 [+0.1081, +0.3805] | +0.24 |
| `xaj_sm` | sm | routing | +0.1615 [-0.0055, +0.3474] | +0.11 | +0.2682 [+0.1936, +0.3692] | +0.15 |
| `xaj_ex` | ex | routing | -0.3702 [-0.5387, -0.2149] | -0.08 | -0.0954 [-0.1580, -0.0392] | -0.10 |
| `xaj_ki` | ki | routing | -0.4746 [-0.6304, -0.3107] | -0.24 | -0.3146 [-0.4532, -0.2103] | -0.33 |
| `xaj_kg` | kg | routing | -0.2029 [-0.4032, +0.0045] | -0.20 | +0.0110 [-0.0648, +0.1062] | -0.09 |
| `xaj_ci` | ci | routing | -0.4138 [-0.6101, -0.1949] | -0.18 | -0.5312 [-0.6903, -0.3681] | -0.29 |
| `xaj_cg` | cg | routing | +0.2925 [+0.1482, +0.4320] | +0.01 | +0.0590 [-0.0311, +0.1607] | -0.12 |
| `xaj_a` | a (UH shape) | routing | +0.0363 [-0.1810, +0.2738] | +0.04 | -0.0122 [-0.1221, +0.1157] | -0.01 |
| `xaj_theta` | theta (UH scale) | routing | -0.0166 [-0.2338, +0.2182] | -0.08 | +0.1304 [+0.0499, +0.2179] | +0.11 |

## 5. Artifact Manifest

All tables and json files saved in `results/` are authoritative, reproducible, and ready for publication figures/text.
