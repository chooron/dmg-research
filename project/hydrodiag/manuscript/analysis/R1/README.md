# Canonical R1 Downstream Analysis

This package performs canonical cross-basin inferential statistics, same-basin paired contrasts, snow-activity primary summaries, secondary TGD structural controls, threshold prevalence audits, and robustness analyses for R1.

It operates strictly on verified staged compact tables under `manuscript/cache/r1_rebuild_audit_staged/` using vectorized CUDA reductions. It **does not reprocess raw daily Parquet files**, launches **no model training/calibration/forward simulations**, and materializes **no multi-million-row daily tables**.

---

## 1. Modular Architecture

Instead of a monolithic script, the analysis is decomposed into modular files:

| File | Purpose |
| :--- | :--- |
| `config.py` | Central configuration, paths, schema contracts, SHA-256 digests, frozen S1–S5 strata counts, seeds, and constants. |
| `cuda_engine.py` | Vectorized GPU engine for paired basin-level bootstraps, average-rank Spearman correlations, quantile/median reductions, and CPU reference parity checks. |
| `canonical_basin_table.py` | Staged table audit and construction of the frozen canonical evaluation table (`canonical_basin_level.csv`, 3,186 rows = 531 basins × 3 structures × 2 regimes). |
| `paired_contrasts.py` | Same-basin paired estimand calculations (`delta_absCT_Base_CN`, `delta_KGE_Base_CN`, `delta_absCT_TGD_CN`, `delta_KGE_TGD_CN`) with strict alignment verification (1,062 rows; 0 silent drops). |
| `snow_activity_analysis.py` | S1–S5 strata summaries (median, IQR, 95% bootstrap CI), continuous Spearman $\rho$ with `frac_snow`, $S5-S1$ activity endpoint contrast ($D_{\text{activity}}$), and low-snow characterization. |
| `secondary_tgd_control.py` | Secondary TGD structural control summaries across strata and overall (no $F_{\text{TGD}}$ / parameter compensation claims). |
| `threshold_prevalence_audit.py` | Metric cutoff prevalence across structure-specific versus common-pass denominators (KGE 0.40..0.80, CT 10/15/20 d). |
| `robustness_analysis.py` | Leave-One-Region-Out (LORO) spatial sensitivity and dPL multiseed (42, 123, 2026) / IC restart sensitivity. |
| `canonical_gates.py` | Formal evaluation of all 5 canonical gates (Provenance, Alignment, CT definition, Statistical unit, Reproducibility). |
| `run_all.py` / `main.py` | End-to-end orchestration, profiling (RAM/VRAM/time), and generation of `machine_readable_summary.json` and `results_summary.md`. |
| `tests/` | Unit and integration test suite (19 test cases) verifying mathematical correctness, GPU/CPU parity, schemas, and gates. |

---

## 2. Fast GPU Execution

From the repository root:

```bash
.venv/bin/python project/hydrodiag/manuscript/analysis/R1/run_all.py --draws 10000
```

CLI options:
- `--staged-dir`: Path to staged compact inputs (default: `manuscript/cache/r1_rebuild_audit_staged/`).
- `--output-dir`: Path to write downstream artifacts (default: `manuscript/analysis/R1/results/`).
- `--region-dir`: Optional path to authoritative CAMELS region groupings (`group_11.npy` .. `group_17.npy`).
- `--draws`: Number of paired basin-bootstrap resamples (default: `10000`).

---

## 3. Canonical Promotion Gates

All 5 canonical gates must pass before promotion:

1. **Provenance Gate (PASS):** Staged compact inputs verified against pinned SHA-256 digests and exact schemas.
2. **Basin Alignment Gate (PASS):** 531 paired basins across Base, TGD, CN per paradigm; 0 silent drops, 0 duplicates.
3. **CT Definition Gate (PASS):** Basin CT is median of valid water years; $\text{absolute\_CT\_error} = |\text{signed\_CT\_error}|$.
4. **Statistical Unit Gate (PASS):** Inferential unit is basin ($N=531$). Seeds/restarts are aggregated prior to inference.
5. **Reproducibility Gate (PASS):** All outputs reproducible from verified staged tables without daily raw files.

Gate summary artifact: `results/canonical_gates_summary.json`.

---

## 4. Key Scientific Estimands

- **Sign convention:** $\text{positive} = \text{CN relative to baseline improves the metric}$.
  - $\delta_{\text{abs\_ct\_base\_cn}} = |\text{signed\_e}_{\text{Base}}| - |\text{signed\_e}_{\text{CN}}|$
  - $\delta_{\text{kge\_base\_cn}} = \text{KGE}_{\text{CN}} - \text{KGE}_{\text{Base}}$
- **S5 vs S1 Endpoint Contrast ($D_{\text{activity}} = \text{median}(S5) - \text{median}(S1)$):**
  - IC-CMA-ES: **47.0 days** [44.0, 51.0] (95% CI)
  - dPL-MLP: **46.0 days** [40.0, 48.0] (95% CI)
- **Continuous Spearman Association ($\text{frac\_snow}$ vs $\delta_{\text{abs\_ct\_base\_cn}}$):**
  - IC-CMA-ES: $\rho = \mathbf{0.546}$ [0.467, 0.619]
  - dPL-MLP: $\rho = \mathbf{0.459}$ [0.369, 0.543]
- **Low-snow Condition (S1):** Small / centered near zero ($0.0$ d median in both IC and dPL); no equivalence claim made.
- **Secondary TGD Control:** Overall $\delta_{\text{abs\_ct\_tgd\_cn}}$ is $1.0$ d [0.0, 1.0] for IC and $0.0$ d [0.0, 0.0] for dPL.
- **KGE-Qualified Timing Inconsistency Prevalence ($KGE \ge 0.60 \cap |CT| \ge 15\text{ d}$):**
  - **Conditional Prevalence** $P(|CT| \ge 15\text{ d} \mid KGE \ge 0.60)$:
    - Structure-specific: IC Base = **16.92%** (56/331) [12.95%, 21.11%]; dPL Base = **13.37%** (46/344) [9.82%, 17.00%]; IC CN = **5.85%** (25/427); dPL CN = **4.69%** (20/426).
    - Common-pass ($N_{\text{common}}=321$ for IC, $331$ for dPL): IC Base = **17.13%** (55/321); dPL Base = **13.29%** (44/331); IC CN = **5.61%** (18/321); dPL CN = **4.23%** (14/331).
  - **Joint Prevalence** $P(KGE \ge 0.60 \cap |CT| \ge 15\text{ d})$ ($N=531$):
    - IC Base = 10.55% (56/531); dPL Base = 8.66% (46/531); IC CN = 4.71% (25/531); dPL CN = 3.77% (20/531).

---

## 5. Test Suite

Run pytest to verify dataflow integrity, CUDA/CPU parity, and canonical contracts:

```bash
.venv/bin/python -m pytest project/hydrodiag/manuscript/analysis/R1/tests/ -v
```
