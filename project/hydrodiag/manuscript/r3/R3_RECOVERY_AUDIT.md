# R3 Recovery Audit (read-first)

Date: 2026-08-15/16. Audit performed against the current filesystem, git state,
and the live AutoDL node. No scientific results were modified; nothing was
regenerated or retrained. One trivial read-only recovery step was performed
(restoring the missing local `data/camels_dates.npy` from the hash-verified
remote copy, and installing `pytest` into the shared `.venv`, which lacked it).

---

## 1. Current repository / worktree / environment state

| Item | Actual state |
|---|---|
| Repo root (main checkout) | `/home/jingxin/code/dmg-research` — branch `master` @ `d59365e`, clean (only pre-existing `M requirements.txt`) |
| R3 worktree | `/home/jingxin/orca/workspaces/dmg-research/hydrodiag-R3-exp` — branch `hydrodiag/R3-exp` @ **`8ef9548`**, clean |
| Other worktrees | `chapter-4` (`task/chapter-4`), `flex-mopex` (`task/flex-mopex`), `unknown` (`task/unknown`) — all @ `d59365e`, only unrelated changes |
| Origin | `github.com/chooron/dmg-research` — `origin/master` still @ `d59365e`; the R3 checkpoint `8ef9548` is **local-only, not pushed** |
| R3 checkpoint | **EXISTS**: `8ef9548 feat(hydrodiag): R3 misspec protocol freeze + Base/TGD2 misspec dPL runs` (27 files, +5007 lines). Committed 2026-08-15 20:58 +0800, i.e., *after* the historical loss window. It is byte-identical (verified file-by-file) to the snapshot tar pulled from the AutoDL node, which in turn is the exact code that produced the truth, gates, and misspec dPL runs. |
| Lost worktree | The historical R3 worktree `/home/jingxin/orca/workspaces/dmg-research/hydrodiag-r3` (branch `task/hydrodiag-r3`, HEAD `d59365e`, uncommitted R3 changes) no longer exists and is not registered in `.git/worktrees`. `git fsck` shows **no dangling commits** — the R3 code was never committed there; the git history holds no R3 content. |
| Shared `.venv` | `/home/jingxin/code/dmg-research/.venv` (symlinked into worktrees), Python 3.10.20, torch 2.9.1+cu128, CUDA available (local GPU: RTX 3060 12GB). `pytest` was absent and has been installed into it during this audit. |
| Local data root | `/home/jingxin/code/dmg-research/data` (symlinked into worktrees). `531sub_id.txt`, `camels_dataset`, `gage_id.npy` hashes **match the truth-manifest fingerprints exactly**. `camels_dates.npy` was **missing locally** and has been restored from the remote copy; its sha256 (`abd5b0cf…`) matches the recorded fingerprint. |
| Local results root | `/home/jingxin/code/dmg-research/project/hydrodiag/results` **does not exist** — the entire local R1/R2/R3 results tree is gone (the `outputs -> results/archive/legacy_outputs_20260730` symlink is dangling). R3 code's `manuscript/r3/common.py::DEFAULT_RESULTS_ROOT` points here. |
| Remote AutoDL node | `connect.westb.seetacloud.com` port **42368** (HANDOFF lists port 20280 — the port changed after a reboot; the current reachable port is 42368), alive. Code at `/root/dmg-research` (git HEAD `7d1132b`, `project/hydrodiag` untracked there — it was synced as files), data at `/autodl-fs/data` (hashes verified == recorded fingerprints), R3 products under `/autodl-fs/r3/`. |

Environment caveat: the historical HANDOFF references a code snapshot tar
`/root/sync_packages/hydrodiag_src_20260814_1806.tar.gz` (sha256
`b6414227…`, verified) — this is the authoritative recoverable code copy and
was used to cross-check the commit.

## 2. Verified scientific / protocol state

All historical headline numbers that could be checked from **primary artifacts**
are confirmed; the ones that depend on now-lost gate outputs are marked.

### 2.1 Synthetic truth v1 — INTACT (remote `/autodl-fs/r3/r3_synthetic_truth_v1/`, 8 files)
Verified by reading the NPZ/JSON content directly:

- `theta_star.npz` `[531,17]` with `parameter_names` (15 shared XAJ + `cn_ctg`,`cn_kf`) and `basin_ids`; `z_normalized`, `clip_mask` (8 clipped entries / 9027 total, 8 basins).
- `q_star.npz` `[531,12418]` `target_mm_day` float64 + `q_star_f32`; finite, non-negative, max 93.87 mm/d.
- `x_star.npz` states `wu,wl,wd,s,fr,qi,qg` `[531,12418]`; `snow_star.npz` `G,eTG,sca,rain,melt,effective_precip`; `final_states.npz` `[531,9]` incl. `prod_final_states`.
- `manifest.json`: protocol `r3_synthetic_truth_v1`, created 2026-08-13, **roundtrip recorded==production = 0.0 max-abs (17 chunks, all 0.0)**, periods warmup 1980-10-01..1981-09-30 / train ..1995-09-30 / test ..2010-09-30.
- `gstar_manifest.json`: `K=15`, `ridge_alpha=316.227766…`, CV seed `20260730`, `frac_snow_index=3`, source field `results/xaj_cn_cmaes_531_batched_paired_v2` (R1/R2-era CN-IC field, best train-KGE restart), code commit `d59365e` (dirty).
- `gstar_diagnostics.json`: `cv_r2_total=0.9391`, `theta_star_equals_g_star=True`, `random_residual=None`, `reconstruction_var_explained=0.9509`, `n_components_k=15`.

→ **Every historical number about the truth (K=15, α≈316.2, CV R²≈0.939, frac_snow index 3, noise-free, roundtrip exact) is verified from the artifact itself.**

### 2.2 Frozen misspec protocol — PRESERVED
`manuscript/r3/protocol_misspec_v1.json` exists (committed + on remote), frozen 2026-08-14
10:02, stating "FROZEN before viewing any 531 Base/TGD2 results". The first
Base/TGD2 531 dPL result completed 16:07 same day → **the predeclared tiers/
estimands were frozen ~6 h before any misspec 531 results existed**. The
estimands (delta_KGE, e, delta_abs_e, delta_e, state delta_E, seed-matched
pairs, primary/secondary parameter tiers, primary states wu/wl/s/qi/qg, wd
secondary, wt derived, cn_only diagnostics) are implemented by
`manuscript/r3/misspec_analysis.py` and match the task's historical description.

### 2.3 Correct-CN gate numbers — RECORDED BUT PRIMARY ARTIFACTS LOST
The HANDOFF + protocol cite: CN-IC median best-restart train KGE 0.9993
(63.4 min, 5310 records), CN-dPL val KGE 0.9953/0.9950/0.9954, oracle identity
3.8e-6 max-abs (IC train path), oracle dPL window KGE median 1.0000, wd NRMSE
1.5–2.5. These numbers appear in the preserved docs, but the result
directories (`r3_gate_ic_xaj_cn_531_v1`, `r3_gate_dpl_xaj_cn_seed_*`) and the
gate-analysis outputs do **not** exist anywhere (local results root gone;
remote `/autodl-fs/r3/results` holds only the misspec dPL dirs). They must be
**regenerated** before any analysis can be re-verified from primary data.

### 2.4 Critical R3 fixes — VERIFIED IN CODE + TESTS (committed `8ef9548`)
1. **Canonical CemaNeige `g_thresh` / `cn_psol_annual`**: present in
   `models/cemaneige.py` (forward + step), `models/composed.py` (XAJ-CN,
   GR4J-CN, SIMHYD-CN), `ablation/ic_core/model_adapter.py` (pass-through),
   `ablation/ic_core/runtime.py` (computes canonical value when
   `canonical_cn_psol_annual` flag set, CN models only). Default behavior
   unchanged when the key/flag is absent (verified by diff vs the unpatched
   local files). Tests `test_cn_model_override_identity_and_effect`,
   `test_cn_override_preserves_recorded_forward_identity` PASS.
2. **dPL synthetic-target basin-ID alignment**: `run_dpl_model.py` selects
   override rows via NPZ `basin_ids` (with a set-equality guard), not by
   position. Regression test `test_reordered_531_list_preserves_membership`
   PASSES; `test_target_override_row_alignment_with_reordered_basin_list`
   (requires truth NPZ locally) is currently skipped only because the truth
   copy is not yet on the local results root.
3. **Recorded-forward state export**: `manuscript/r3/recorded_forward.py` replays the
   production kernels; tests `test_recorded_cn/base/tgd2_matches_production_forward`
   all PASS; the truth manifest records roundtrip 0.0.

### 2.5 Model registry / parameters
- Base XAJ = 15 params, XAJ_CN = 17 (15 + `cn_ctg`,`cn_kf`), XAJ_TGD2 = 17
  (15 + `tgd_tau_warm`,`tgd_delta_tau_cold`); parameter order confirmed from
  the misspec dPL result configs (`parameter_names`) and the truth manifest.
  Bounds recorded in `gstar_manifest.json` (e.g. xaj_k [0.5,2.0],
  xaj_b [0.1,2.0], xaj_theta [0.0,6.5], cn_kf [0.0,10.0], cn_ctg [0.0,1.0]).
- `training/ic/run_tgd2_batched_cmaes_531.py` registers XAJ (15), XAJ_CN (17),
  XAJ_TGD2 (17) and the `--target-npz` synthetic-target path.
- IC protocol (10 starts, 300 generations, population rule
  `max(12, round(25*d/17))`, train-KGE objective, best-train-KGE restart
  selection) is encoded in `run_gate_531.py` / `pilot.py` / HANDOFF.
- dPL protocol (35-attr robust median/IQR clip±5 over full 531 set, 256³ SiLU
  MLP, sigmoid-to-bounds mapping, AdamW/cosine, 100 epochs, seeds 42/123/2026,
  balanced valid-KGE window sampling, 365 d warm-up/365 d windows) is verified
  **from the primary result configs** of the six misspec runs (see §4).

### 2.6 KGE formula
`manuscript/r3/docs/kge_audit.md` documents that all repository KGE implementations
(IC objective, dPL loss/validation, R1 stats, R3 common) use the **standard
KGE** (r/alpha/beta), and the paper's "modified KGE′" wording is a
pre-existing documentation discrepancy that was intentionally not "fixed" in
code. Consistent with the historical claim.

## 3. Code inventory (see R3_RECOVERY_INVENTORY.json for the full table)

All R3 code is **committed in `8ef9548`** and byte-identical to the remote
snapshot tar. Classification:

| Component | Status |
|---|---|
| `manuscript/r3/` package (common, truth_generator, recorded_forward, generate_truth, run_base_no_refit, pilot, analyze_pilot, oracle_identity, oracle_dpl_audit, run_gate_531, launch_d2_parallel, gate_analysis, gate_report_md, misspec_analysis, protocol_misspec_v1.json, README, HANDOFF, docs/kge_audit.md) | PRESENT + current (committed) |
| 6 patched files: `models/composed.py`, `models/cemaneige.py`, `ablation/ic_core/model_adapter.py`, `ablation/ic_core/runtime.py`, `training/dpl/run_dpl_model.py`, `training/ic/run_tgd2_batched_cmaes_531.py` | PRESENT + current (committed); local pre-commit versions were unpatched (diff confirmed) |
| `tests/test_r3_truth_generator.py` (9), `tests/test_r3_forward_identity.py` (7) | PRESENT + current (committed) |
| Repo-root helpers `r3_remote_chain.py`, `r3_watch_remote.sh` (mentioned in HANDOFF) | MISSING locally (never committed, not in tar); functional equivalent `remote_misspec_dpl.py` exists on the remote `/autodl-fs/r3/` |

Test result (run from the committed worktree with pytest installed into the
shared venv): **12 passed, 3 skipped, 1 blocked**. The 3 skipped and the 1
blocked all require the frozen truth NPZ to be present at the local results
root (they are data-dependency skips, not code failures). One of the 16
(two suites) failed only on the missing local `camels_dates.npy`, which was
then restored and is now hash-verified — that specific failure is resolved.

## 4. Result / artifact inventory

| Logical artifact | Actual path | Basins / shape | Status |
|---|---|---|---|
| Synthetic truth v1 | `/autodl-fs/r3/r3_synthetic_truth_v1/` (remote) | 531 × 12418 (θ [531,17]) | **INTACT**, content-verified; **not yet fetched locally** (~250 MB) |
| Base-no-refit | `results/r3_base_no_refit_v1/` | 531 | **LOST** (not on remote, not in git, local root gone); regenerable cheaply (forwards only) |
| CN-IC gate | `results/r3_gate_ic_xaj_cn_531_v1/` | 531 × 10 starts × 300 gens | **LOST**; must re-run (~1 h GPU) |
| CN-dPL gate seeds 42/123/2026 | `results/r3_gate_dpl_xaj_cn_seed_<s>/` | 531 × 17 params | **LOST**; must re-run (~4 h GPU in parallel) |
| Oracle identity / dPL audit | `results/r3_gate_v1/oracle_identity.json`, `oracle_dpl_audit.json` (+window CSV) | 531 | **LOST**; regenerable from truth + scripts (moderate) |
| Gate analysis outputs (13 files: gate_report.json/.md, gate_manifest.json, gate_input_validation.json, parameters_recoverability.csv, parameter_recovery_summary.csv, ic_restart_parameter_dispersion.csv, dpl_seed_parameter_spread.csv, gate_discharge_metrics.csv, kge_vs_parameter_recovery.csv, kge_deficit_vs_frac_snow.csv, gate_state_metrics_basin.csv, gate_state_summary.json) | `results/r3_gate_v1/` | — | **LOST**; regenerated by `manuscript/r3/gate_analysis.py` after the gates are re-run (CPU, cheap) |
| 12-basin engineering pilot | `results/r3_pilot_v1/` + fit dirs | 12 | **LOST**; regenerable (engineering-only) |
| **Base dPL 531 — seeds 42/123/2026** | `/autodl-fs/r3/results/r3_misspec_dpl_xaj_seed_{42,123,2026}/` (remote) | 531 × 15 params, 100 epochs | **COMPLETE, INTACT on remote** — val KGE median 0.9081/0.9092 (seed 2026: 0.9059 final), mean 0.7825/0.7840; `COMPLETE` markers + DONE.json `exit_ok` all true; **not yet fetched locally** |
| **TGD2 dPL 531 — seeds 42/123/2026** | `/autodl-fs/r3/results/r3_misspec_dpl_xaj_tgd2_seed_{42,123,2026}/` (remote) | 531 × 17 params, 100 epochs | **COMPLETE, INTACT on remote** — val KGE median 0.9443 across seeds, mean 0.9017–0.9032; **not yet fetched locally** |
| Base-IC / TGD2-IC 531 | `results/r3_misspec_ic_xaj_531_v1/`, `r3_misspec_ic_xaj_tgd2_531_v1/` | — | **NEVER RUN** (no artifacts anywhere; DONE.json covers dPL only) |
| `r3_gate_chain_report.json` / `r3_gate_configs/` | results root | — | **LOST** (configs regenerable from `run_gate_531.py`/`launch_d2_parallel.py` defaults) |
| Remote chain driver `remote_misspec_dpl.py`, `DONE.json`, `chain.log` | `/autodl-fs/r3/` | — | INTACT (provenance of the six dPL runs) |

The six misspec dPL configs (verified from the primary `config.json` inside
each result dir) match the frozen protocol exactly: model XAJ / XAJ_TGD2,
`target_override_npz = q_star.npz` `[531,12418]`, 531 basins via
`531sub_id.txt`, 100 epochs AdamW lr 1e-3 wd 1e-4 cosine min 1e-4, seeds
42/123/2026, 365 d warm-up + 365 d prediction windows, robust median/IQR
clip±5 over the full 531 set, periods 1980-10-01..2010-09-30, `_protocol =
r3_misspec_dpl_synthetic_target_v1`, `target_override_applied` recorded.

## 5. Comparison vs the historical expected state

### 5.1 Recovered and verified
1. Synthetic truth v1 (remote) — intact; every recorded number re-verified from the artifact.
2. Frozen misspec protocol (`protocol_misspec_v1.json`) — intact; frozen before any 531 Base/TGD2 results.
3. All R3 code incl. the three critical fixes (cn_psol_annual, basin-ID alignment, recorded forward) — committed in `8ef9548`, byte-identical to the authoritative remote snapshot, tests passing.
4. Base/TGD2 dPL 531 full runs — complete and intact on remote (primary artifacts verified).
5. Reproducibility git checkpoint — exists (`8ef9548`), though not yet pushed.
6. Shared-parameter configs, KGE standard, IC/dPL protocol details — verified from configs/manifests.
7. Local data integrity — 531sub_id/camels_dataset/gage_id hash-verified; `camels_dates.npy` restored (hash-verified).

### 5.2 Present but inconsistent / needs attention
- `manuscript/r3/common.py::DEFAULT_RESULTS_ROOT` and `DEFAULT_DATA_ROOT` hard-code the
  main-checkout paths (`/home/jingxin/code/dmg-research/…`). The worktree
  convention (HANDOFF "import trap") makes the code root the worktree, so
  this is consistent *if* the results root is recreated at the main checkout.
- The HANDOFF references remote SSH port 20280 and key `~/.ssh/r3_autodl`;
  the live node is reachable on port 42368 (both recorded in the `unknown`
  worktree's `docs/ssh-connection-guide.md`; the guide contains credentials —
  do not copy secrets into reports/commits).
- Commit `8ef9548` is not pushed; origin/master is unchanged.

### 5.3 Missing and must be rebuilt
1. Correct-CN gate results: `r3_gate_ic_xaj_cn_531_v1` + `r3_gate_dpl_xaj_cn_seed_{42,123,2026}` (the paired-comparison CN baseline — required by `misspec_analysis.py`).
2. `r3_base_no_refit_v1` (raw knockout reference).
3. Oracle identity + dPL audit outputs (`r3_gate_v1/oracle_identity.json`, `oracle_dpl_audit.json`).
4. Gate-analysis outputs (`r3_gate_v1/*` 13 files) — regenerate via `gate_analysis.py` after gates exist.
5. 12-basin engineering pilot (`r3_pilot_v1`) — engineering-only.
6. Repo-root helpers `r3_remote_chain.py` / `r3_watch_remote.sh` (optional; functional equivalent on remote).
7. Local copies of the truth (~250 MB) and the six misspec dPL result dirs (~15 MB) — fetch from remote to the local results root (DONE.json note: "fetch to local results root and verify before shutdown" was never completed).

### 5.4 Uncertain provenance
- The exact CN-gate launch configs (`r3_gate_configs/*.json`, `r3_gate_chain_report.json`) are lost; the gate protocol can only be reconstructed from `run_gate_531.py` defaults (10 starts / 300 gens), `launch_d2_parallel.py`, and the HANDOFF numbers (63.4 min, 5310 records). Re-running with these defaults is the best available reconstruction; the gate runs were on the remote node.
- Whether the pilot ran locally or on the remote is unknown; irrelevant for science (engineering-only).
- The R1/R2-era IC field that anchors g* (`results/xaj_cn_cmaes_531_batched_paired_v2`) is gone from the local results root; the truth itself does not need it (frozen), but any re-derivation of g* would. Remote `/autodl-fs` does not contain it either.
- The pre-commit local R3 files (before `8ef9548`) had unknown content; the commit's content matches the remote tar, which is the authoritative source.

## 6. Safe resume point and next actions (priority order)

The scientific protocol is intact and the six Base/TGD2 dPL jobs are done; the
blocking gap is the lost correct-CN gate baseline + the unfetched remote
results. Recommended earliest safe next phase: **restore data/result copies
and regenerate the correct-CN gate, then re-run gate analysis; only then run
the one-command misspec analysis. Do not regenerate the truth.**

1. **Push `8ef9548` to origin** (with user authorization) so the checkpoint is durable.
2. **Fetch from remote (read-only)** to the local results root:
   `r3_synthetic_truth_v1/` (~250 MB) and `manuscript/r3/results/r3_misspec_dpl_{xaj,xaj_tgd2}_seed_*` (6 dirs). Verify by re-running the 4 currently skipped/blocked R3 tests (they pass once the truth NPZ is local). Channel: AutoDL web console (downlink is unreliable over ssh; base64-over-ssh worked for the 485 KB tar, so small files can be pulled that way).
3. **Rebuild cheap artifacts locally**: `manuscript/r3/run_base_no_refit.py --device cuda`; `manuscript/r3/oracle_identity.py`, `manuscript/r3/oracle_dpl_audit.py` (531-basin forwards).
4. **Regenerate the correct-CN gate** (remote or local GPU): `manuscript/r3/run_gate_531.py` D1 (`--model XAJ_CN --starts 10 --generations 300 --target-npz q_star`) and D2 (`launch_d2_parallel.py`, seeds 42/123/2026). Expected cost ≈ 1 h (IC) + ≈ 4 h (dPL × 3 in parallel) on the remote 3080 Ti.
5. **Re-run gate analysis**: `manuscript/r3/gate_analysis.py --device cuda` → `r3_gate_v1/` (gate_report.json + CSVs), then `manuscript/r3/gate_report_md.py`. Re-verify the frozen-tier rationale (tiers themselves are preserved in `protocol_misspec_v1.json` and must NOT be re-selected).
6. **Optional**: 12-basin pilot re-run (engineering), and Base-IC/TGD2-IC 531 (`r3_misspec_ic_xaj_531_v1`, `r3_misspec_ic_xaj_tgd2_531_v1`) if the IC-regime comparison is wanted (10 starts × 300 gens each; IC was historically optional).
7. **Final analysis**: `manuscript/r3/misspec_analysis.py` (refuses to run on incomplete inputs; implements the frozen protocol; Base-no-refit as raw reference; seed-matched dPL; paired delta_KGE / delta_abs_e / delta_e / delta_E with frac_snow associations; primary/secondary tier labels fixed).

Constraints honored: no experiment launched; truth not regenerated; R1/R2
outputs untouched; shared `.venv` used (only `pytest` added); no credentials
included in this report (see `unknown` worktree ssh guide for connection
details); nothing was committed during this audit (the checkpoint commit
`8ef9548` was authored by the user before the audit was resumed).
