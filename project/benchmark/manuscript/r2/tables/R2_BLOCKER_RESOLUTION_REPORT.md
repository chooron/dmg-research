R2 BLOCKER RESOLUTION = READY FOR WRITING

1. modhydrolog:k3:
   PASS
   authoritative M = 0.3880192475023052
   authoritative R = 0.016289203982133137

   Independent recomputation from the frozen source
   `results/joh_direct_parameter_change_diagnostic_20260905/r2/cache/modhydrolog_normalized_parameter_matrices.npz`
   used 531 basins and coordinate 14 (`k3`):

   ```text
   M = median_b |theta_dPL_norm - theta_IC_norm|
   R = Spearman(theta_IC_norm, theta_dPL_norm)
   M = 0.3880192475023052
   R = 0.016289203982133137
   ```

   `DeltaTheta` matched `dPL - IC` exactly. `F4_ALL_COORDINATE_DIAGNOSTICS.csv`, `F4_CANDIDATE_POOL.csv`, `F4_CANDIDATE_RANKINGS.csv`, the F4 plotting input, and the frozen arrays agree. No repository artifact contains the older exact pair 0.3879 / 0.0164. The older value is superseded, not averaged or merged.

   Detailed provenance: `tables/R2_MODHYDROLOG_K3_PROVENANCE_AUDIT.md`.

2. collie1:
   PASS
   P = 1
   F2 marker treatment = REMOVE
   localization wording = “22/22 eligible multi-parameter models changed in the expected direction; collie1 (P=1) is invariant by construction.”

   `collie1` has one calibrated parameter (`Smax`), is strictly eligible, and remains valid in the F2 estimand. Its localization metrics are structurally fixed at C_eff = 1, top-1 = 1, and top-2 = 1; its exact raw-to-adjusted localization changes are zero. The current `MIXED` provenance classification does not establish corruption, restart duplication, cache failure, or bad data.

   The dagger/open-square treatment was removed from the F2 plotting script and the existing Figure 2 composite/build-note output was regenerated without a special symbol. Numerical F2 values were unchanged.

   Detailed audit: `tables/R2_COLLIE1_STRUCTURAL_AUDIT.md`.

3. F3 audit-artifact synchronization:
   PASS

   The active branch now uses exact common model x basin support:

   ```text
   C_eff: 0.34521687865 -> 0.2933820389
   top-1: 0.5711769539  -> 0.6726335350
   top-2: 0.8799996185  -> 0.9417612386
   ```

   Figure 3a and 3c are now `READY` in `R2_FIGURE_DATA_MANIFEST.csv`. The two former FAIL rows in `R2_HEADLINE_REPRO_CHECK.csv` now validate the exact-support primary values. The earlier 0.346893 / 0.573874 all-basins raw summaries remain explicitly labeled `HISTORICAL/SENSITIVITY ONLY`. The old `R2_FIGURE_DATA_AUDIT.md` and pre-resolution final audit are explicitly marked `SUPERSEDED / HISTORICAL`; no historical provenance was erased.

   Synchronization note: `tables/R2_F3_ARTIFACT_SYNCHRONIZATION.md`.

4. Existing validators:
   PASS

   - `scripts/99_validate_frozen_results.py` -> `R2_VALIDATION_PASS`.
   - `sha256sum -c cache/final/MASTER_CHECKSUMS.sha256` -> all checks `OK`.
   - Existing F2 plotting script completed and regenerated the same Figure 2 output without the collie1 marker.
   - No training, calibration, simulation, new experiment, alternative estimand, or alternative aggregation was run.

Remaining blockers:
None.

No numerical or provenance blocker remains for drafting R2 Results and the F2–F4 captions.
