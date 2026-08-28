# Canonical Results 3.2 (R2) Parameter Analysis Package

This package performs the canonical statistical audit and rebuild for Section 3.2 (R2: Real-catchment parameter response layer).

It operates strictly on lowest-level raw parameter artifacts (10 CMA-ES restarts per basin for IC; 3 seeds 42, 123, 2026 for dPL) across the 15 shared host XAJ parameters. It launches **no model training, calibration, inference, or forward simulations**, and enforces strict non-mechanistic, non-superiority observational scientific boundaries.

---

## 1. Modular Architecture

The analysis is decomposed into clean, modular files:

| File | Purpose |
| :--- | :--- |
| `config.py` | Directory paths, schema contracts, frozen S1–S5 strata definitions, seeds, and constants. |
| `shared_parameter_specs.py` | 15 shared parameters identities, physical bounds, Gamma-UH specifications, and normalization transforms ($z = (\theta - \text{lower})/(\text{upper} - \text{lower})$). |
| `parameter_ledger.py` | Raw long-form parameter ledger builder from lowest-level IC raw JSONs and dPL physical parameter NPZs (310,635 rows). |
| `canonical_vectors.py` | Canonical basin-level parameter vector reduction (3,186 rows = 531 basins × 3 structures × 2 paradigms) using verified reduction rules. |
| `macro_whole_space.py` | Macro whole-space Base-CN response: 4A canonical 15-D displacement ($D_{\text{rms}}$ and $D_{\text{euclidean}}$) and 4B ensemble within/between/excess (Figure 3 primary). |
| `parameter_shifts_all15.py` | Primary explanatory all-15 signed parameter shifts across Full (531), Strata (S1–S5), ExcludeS5 (476), and Leave-One-Stratum-Out (Figure 4 primary). |
| `tgd_attribution_control.py` | Macro TGD attribution control (Base-TGD vs Base-CN) and paired basin-bootstrap estimation of $\Delta_\beta = \beta_{\text{CN}} - \beta_{\text{TGD}}$. |
| `diagnostics_and_safeguards.py` | IC restart quality audit (KGE IQR, best-minus-median, Top-3/Top-5), dPL seed stability, and boundary point mass safeguards. |
| `canonical_gates.py` | Formal automated verification of all 12 R2 validation gates. |
| `run_all.py` / `main.py` | End-to-end pipeline runner, historical reconciliation builder, and Markdown audit report generator. |
| `tests/` | Comprehensive test suite (9 test modules) covering bounds, ledger completeness, canonical vectors, macro separation, 15 parameter slopes, TGD paired bootstrap, and gates. |

---

## 2. Command-Line Usage

From the repository root:

```bash
.venv/bin/python project/hydrodiag/manuscript/analysis/R2/run_all.py --draws 10000
```

CLI options:
- `--output-dir`: Path to write downstream artifacts (default: `manuscript/analysis/R2/results/`).
- `--draws`: Number of paired basin-bootstrap resamples (default: `10000`).

---

## 3. Key Scientific Findings & Canonical Statistics

### A. Primary Macro: Whole-Parameter-Space Response (Figure 3)
- **Ensemble Structural Separation Prevalence ($\text{fraction}(\text{between\_all} > \text{within\_pooled})$):**
  - **IC-CMA-ES (10 restarts):** **63.09%** (335/531 basins) [59.13%, 67.04%] (95% CI)
  - **dPL-MLP (3 seeds):** **83.80%** (445/531 basins) [80.60%, 86.82%] (95% CI)
- **Macro Excess OLS Slope on Snow Fraction ($\beta(\text{excess} \sim f_{\text{snow}})$):**
  - **IC-CMA-ES:** $\beta = \mathbf{+0.1542}$ [+0.0898, +0.2185] (Full531); $\beta = \mathbf{+0.4042}$ [+0.2783, +0.5285] (ExcludeS5)
  - **dPL-MLP:** $\beta = \mathbf{+0.1974}$ [+0.1578, +0.2372] (Full531); $\beta = \mathbf{+0.4267}$ [+0.3541, +0.4996] (ExcludeS5)

### B. Primary Explanatory: Key Signed Parameter Shifts (Figure 4)
- **$u_m$ (Upper-layer tension water capacity):**
  - IC: slope $\beta = \mathbf{+0.521}$ [+0.322, +0.717], $\rho = \mathbf{+0.216}$
  - dPL: slope $\beta = \mathbf{+0.566}$ [+0.389, +0.747], $\rho = \mathbf{+0.253}$
- **$k_i$ (Interflow outflow coefficient):**
  - IC: slope $\beta = \mathbf{-0.475}$ [-0.654, -0.309], $\rho = \mathbf{-0.237}$
  - dPL: slope $\beta = \mathbf{-0.315}$ [-0.439, -0.201], $\rho = \mathbf{-0.328}$
- **$c_i$ (Interflow reservoir recession constant):**
  - IC: slope $\beta = \mathbf{-0.414}$ [-0.637, -0.197], $\rho = \mathbf{-0.183}$
  - dPL: slope $\beta = \mathbf{-0.531}$ [-0.694, -0.371], $\rho = \mathbf{-0.290}$
- **$i_m$ (Impervious area fraction):**
  - IC: slope $\beta = \mathbf{-0.363}$ [-0.512, -0.205], $\rho = \mathbf{-0.248}$
  - dPL: slope $\beta = \mathbf{-0.142}$ [-0.235, -0.059], $\rho = \mathbf{-0.345}$

### C. TGD Attribution Control & Paired $\Delta_\beta$ Bootstrap
- **Full531:**
  - IC: $\Delta_\beta = \mathbf{+0.000}$ [-0.032, +0.031]
  - dPL: $\Delta_\beta = \mathbf{+0.041}$ [+0.008, +0.077]
- **ExcludeS5:**
  - IC: $\Delta_\beta = \mathbf{+0.023}$ [-0.013, +0.058]
  - dPL: $\Delta_\beta = \mathbf{+0.086}$ [+0.017, +0.157]

---

## 4. Resolution of Historical Conflicts

1. **Prevalence Conflict (63.1% / 83.8% vs ~97.36% / 100%):**
   - **Root cause:** The draft ~97.36% / 100% resulted from substituting the basin-specific `within_pooled` with a fixed scalar threshold `0.08`.
   - **Canonical verdict:** Exact 10-restart IC cross-structure calculations yield 335/531 = **63.09%**, and 3-seed dPL calculations yield 445/531 = **83.80%**.
2. **Slope Discrepancy:**
   - OLS slope **+0.1542** is proven as the exact Base-CN excess slope on snow fraction in Full531.
   - Parameter slopes (e.g. $u_m = +0.521$ in IC, $+0.566$ in dPL) are strictly reproducible from canonical vectors.

---

## 5. Verification & Promotion Gates

All 12 validation gates pass (`canonical_gates_summary.json`):

1. `gate_01_provenance`: PASS
2. `gate_02_shared_parameter_definition`: PASS
3. `gate_03_normalized_coordinates`: PASS
4. `gate_04_canonical_vector_rule`: PASS
5. `gate_05_ensemble_formulas`: PASS
6. `gate_06_basin_weighting`: PASS
7. `gate_07_basin_joins`: PASS
8. `gate_08_snow_axis`: PASS
9. `gate_09_paired_bootstrap`: PASS
10. `gate_10_historical_conflicts`: PASS
11. `gate_11_all_parameter_transparency`: PASS
12. `gate_12_scope`: PASS

---

## 6. Test Suite

Run pytest to verify the full analysis pipeline:

```bash
.venv/bin/python -m pytest project/hydrodiag/manuscript/analysis/R2/tests/ -v
```
