# F4 data audit — um / ki / ci paired-shift integrity

Audit of the basin-level data behind Figure 4 (v4). Read-only: no upstream result, no F4 script, no figure was modified. All statistics computed directly from the canonical R2 CSVs in `manuscript/results/R2/`.

## 1. Data chain and Δz definition

- Normalization: `z = (value_physical − lower)/(upper − lower)`, per-parameter linear min–max; bounds from `supplement/results/s2_parameter_bounds_from_code.csv` (code-verified). Key bounds: `um` [5, 50] mm, `ki` [0, 0.7], `ci` [0.1, 1.0].
- No clipping of `z` in the pipeline. IC values come from CMA-ES raw JSON and can sit exactly on bounds (hence exact `z ∈ {0,1}` and `Δz = ±1` occur for IC). dPL values are reconstructed through sigmoid-to-physical mapping and are strictly interior (`z ∈ (0,1)`), so dPL `Δz` never equals exactly ±1 (observed max |Δz| = 0.998).
- Paired shift: `delta_base_minus_cn = z_base − z_cn`; sign convention **Base − CN** everywhere (panel a slope, ridgelines, GIS). Verified `max|(z_base−z_cn)−delta| ≈ 4e−16` (531 × 15 rows per paradigm, no NaN).
- IC canonical = selected restart; dPL canonical = within-basin median of 3 seeds. `ki` primary shift uses the stored pre-runtime value; KI+KG≥1 joint rescaling is audited separately and affects only 4.2% (S1) → 0% (S3–S5) of dPL `ki` rows; none of the IC `ki` near-boundary samples were joint-rescaled.
- Regimes: S1 [0,0.05) n=165, S2 [0.05,0.15) n=156, S3 [0.15,0.30) n=121, S4 [0.30,0.50) n=34, S5 [0.50,1.00] n=55 — matches the F4 y-axis labels; `snow_regime` agrees 100% with `frac_snow` bins.

## 2. Raw-distribution audit (um / ki / ci × IC / dPL × S1–S5)

Full table: `r2_figure4_data_audit_raw_distributions.csv`. Key cells (median; IQR; % positive/negative; % exact zero):
- **um**
  - dPL: median S1/S3/S5 = +0.002 / +0.007 / +0.902
  - IC: median S1/S3/S5 = +0.000 / +0.000 / +0.038
- **ki**
  - dPL: median S1/S3/S5 = +0.006 / -0.340 / -0.413
  - IC: median S1/S3/S5 = +0.000 / -0.061 / +0.000
- **ci**
  - dPL: median S1/S3/S5 = -0.015 / -0.108 / -0.866
  - IC: median S1/S3/S5 = +0.000 / -0.085 / -0.240
- **um (dPL)** — clear directional shift: median +0.002 → +0.902, IQR S5 [+0.54, +0.97] fully positive, %positive 51% → 98%. S4/S5 carry real boundary involvement (see §3).
- **um (IC)** — weak/heterogeneous: median ≈ 0 through S4, +0.038 at S5; dominated by a Δz=0 spike (46% of S4+S5) with a positive tail (Q75 +0.857 at S5).
- **ki (dPL)** — clear directional shift: median +0.006 → −0.413, IQR fully negative from S3 (Q25 −0.573/Q75 −0.124 at S3 → −0.670/−0.239 at S5), %negative 46% → 100%. No boundary mass anywhere (max |Δz| = 0.90).
- **ki (IC)** — not a clean median shift: median 0 (S1,S2,S5), −0.061 (S3), −0.323 (S4); S5 is 49% exact-zero + 49% negative. The negative β (−0.475) is carried by S3/S4, not by a monotonic S1→S5 median movement.
- **ci (dPL)** — clear directional shift: median −0.015 → −0.866, Q75 −0.781 at S5, %negative 59% → 100%; interior concentration near (not at) the bound (§3).
- **ci (IC)** — directional but heterogeneous: median 0 → −0.240 with Q25 −0.800 at S5; zero-spike (18%) + negative boundary cluster.
- Duplication/granularity: **dPL Δz values are all distinct** (n_distinct = n, max frequency = 1 for every cell). IC shows exact-value duplication (e.g. ki S5: 27 basins sharing one value) — these are the Δz=0 / bound-co-location rows, not a rounding artifact of the pipeline.

## 3. Boundary-mass audit

Existing convention reused: continuous boundary distance `min(z,1−z)` with concentration thresholds 0.01 / 0.02 / 0.05 (as in `run_r2_parameter_statistics.py` / `r2_boundary_summary.csv`). Full table: `r2_figure4_data_audit_boundary_mass.csv`.

- **Exact Δz = ±1 exists only for IC** (CMA-ES solutions on bounds); dPL is strictly interior. Counts (exact ±1): um IC S4/S5: 6/12 at +1, 1/2 at −1; ki IC S4/S5: 0/0 at +1, 3/7 at −1; ci IC S4/S5: 0/1 at +1, 2/4 at −1.
- **um dPL S5: 21/55 (38%) of basins within 0.05 of +1 — all with Base at the upper bound and CN at the lower bound (opposite boundaries)**; S4: 6/34 (18%) similarly. This is genuine boundary involvement (Base pushing `um` to 50 mm, CN to 5 mm), confirmed by the raw histogram (mode bin +0.95; 58% of S5 mass |Δz| ≥ 0.85) — **not a KDE or clipping artifact**.
- **ki / ci dPL have essentially zero boundary mass**: near-|1| ≤ 0.05 counts are 0 for ki in all regimes; ci has exactly 1 basin within 0.05 of −1 at S5 (|Δz| = 0.95). Their negative shifts are entirely interior.
- IC boundary mass is small and genuine (parameters at bounds); for the Δz=0 IC spike, **100% of exact-zero rows in S4+S5 are cases where both Base and CN sit at the same parameter bound** (um 73% both@1, ki 95% both@0, ci 94% both@1; interior equality = 0%). I.e., IC Δz=0 means co-location at a shared bound, not “no reorganization” in the interior.

## 4. KDE robustness audit

Current F4 KDE: fixed bandwidth h = 0.06, boundary reflection at ±1, peak-normalized. Sensitivity tested at h = 0.03 and h = 0.10 against the raw histogram (20 bins) and ECDF/quantiles. Full table: `r2_figure4_data_audit_kde_robustness.csv`.

- **dPL modes are stable**: ki S4 mode −0.41/−0.41/−0.42 and ci S4 −0.55/−0.52/−0.51, ci S5 −0.89/−0.88/−0.90 across h = 0.03/0.06/0.10; um S5 mode +1.00 under all three bandwidths (raw histogram mode +0.95). The location shift survives all bandwidths.
- **S4/S5 multi-modality is small-n noise**: h = 0.03 produces 5–13 local maxima in S4/S5; h = 0.06 gives 2–4; h = 0.10 gives 1–3. Only the coarse mode location and the mass fractions are interpretable — do not read fine multi-modal detail in S4/S5.
- **ci IC S5 is KDE-sensitive for the mode**: bimodal (Δz=0 spike + negative boundary cluster); the KDE mode flips 0 (h=0.03) → −1 (h≥0.06). The negative boundary cluster itself is real (raw data: 25% of S5 |Δz| ≥ 0.85, 4 exact −1), but the single “mode” is not a stable summary for this cell.
- **Boundary peaks in um dPL S4/S5 are real, not KDE artifacts**: raw histograms put the mode at +0.95 (S5) / +0.05 with 21% ≥0.85 (S4); the reflection only shapes the falloff, it does not create the mass.

## 5. Cross-panel consistency

Full table: `r2_figure4_data_audit_consistency.csv`.
- **panel (a) β == OLS on the same Δz**: recomputed slopes match `r2_snow_gradients_summary.csv` for all 30 (paradigm, parameter) cells (0 mismatches, max diff ≈ 0).
- **Ridgelines == GIS == panel (a) data**: the F4 ridgelines and GIS both read `delta_base_minus_cn` from `r2_paired_shifts_basin_level.csv` (dPL for GIS); canonical `z` matches paired `z_base`/`z_cn` exactly (max diff 0.0); same 531 basins per (paradigm, parameter).
- **GIS basin alignment**: all 531 study basins present in the CONUS shapefile (0 missing; 140 extra non-study gages unused).
- **Sign/colour convention**: Δz = z_Base − z_CN; GIS diverging scale blue(−1) → white(0) → orange(+1) ⇒ blue = Base < CN, orange = Base > CN, matching the caption and the ridgeline sign.
- Names, units, normalization, snow regime and basin IDs are single-source (one canonical pipeline); no subsetting or sign flip differs between panels.

## 6. Classification and recommendations

| parameter | dPL | IC |
|---|---|---|
| um | **clear directional shift, boundary-involved at S4/S5 (real)** | directional but heterogeneous / boundary-involved |
| ki | **clear directional shift, interior, robust** | heterogeneous; β and S1→S5 median disagree (not a clean median shift) |
| ci | **clear directional shift, robust; S5 concentration near (not at) the bound** | directional but heterogeneous; S5 mode KDE-sensitive |

Answers to the four questions:
1. **um high-snow +1 mass is real boundary involvement**: 38% of dPL S5 within 0.05 of +1, all opposite-boundary (Base@upper / CN@lower), confirmed by raw histogram and stable across bandwidths — not a KDE/clipping artifact.
2. **ki/ci high-snow negative movement is whole-distribution**: IQR fully negative from S3 (ki) and S4 (ci) in dPL, %negative 100% at S5, zero boundary mass — not driven by a few extremes. (IC versions are weaker and partly boundary/tail-driven.)
3. **S4/S5 (n = 34/55) support the direction claim** (sign fractions 97–100%, medians/IQR move, bootstrap CIs exclude 0) **but not fine multi-modal detail** — h = 0.03 spurious modes in S4/S5 are small-n noise.
4. **The ridgeline remains suitable main-text evidence for the dPL regime.** Keep it, with three interpretation cautions: (i) describe IC's Δz=0 spike as both-structures-at-a-shared-bound co-location, not absence of reorganization; (ii) avoid interpreting fine S4/S5 multi-modality; (iii) note that exact Δz=±1 exists only for IC (parameters at bounds), while dPL is strictly interior.

## 7. Diagnostic outputs

- `r2_figure4_data_audit_raw_distributions.csv` — §2 (n, mean, median, Q10/25/75/90, min/max, %>0/<0/=0, distinct/frequency, granularity)
- `r2_figure4_data_audit_boundary_mass.csv` — §3 (exact ±1, near ±1 at 0.01/0.02/0.05, Base/CN touch composition, opposite/same-boundary, ki joint-rescale)
- `r2_figure4_data_audit_kde_robustness.csv` — §4 (mode location & mode count at h = 0.03/0.06/0.10, histogram mode, |Δz| ≥ 0.85/0.95 fractions)
- `r2_figure4_data_audit_consistency.csv` — §5 (β vs recomputed OLS, n, basin counts)
- This report.
