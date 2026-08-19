# R3 Handoff — Controlled Synthetic-Truth Experiment (XAJ-CN)

Last updated: 2026-08-14 (UTC+8). Next session: read this file first, then
`project/hydrodiag/r3/README.md` and `r3/docs/kge_audit.md`.

## 1. Where things live

- **Worktree (code)**: `/home/jingxin/orca/workspaces/dmg-research/hydrodiag-r3`
  (git branch `task/hydrodiag-r3`, HEAD `d59365e`, **uncommitted changes**:
  6 modified files + `r3/` + 2 test files + `r3_remote_chain.py` +
  `r3_watch_remote.sh` at repo root).
- **Data**: `/home/jingxin/code/dmg-research/data` (symlinked; sha256 verified
  identical to the AutoDL node's copy).
- **Results (R1/R2/R3)**: `/home/jingxin/code/dmg-research/project/hydrodiag/results`
  (gitignored; R1/R2 artifacts live here too — never modify them).
- **Python**: `/home/jingxin/code/dmg-research/.venv` (3.10, torch 2.9.1+cu128).
- **Remote AutoDL node**: `connect.westb.seetacloud.com` port **20280**,
  root password `<REDACTED — see local secret store / rotate before reuse>`, SSH key `~/.ssh/r3_autodl` (installed).
  **Node is currently POWERED OFF** (shut down after the gate runs completed
  and results were fetched). Code at `/root/dmg-research/project/hydrodiag`,
  run products at `/autodl-fs/r3/`. If the node reboots, the SSH port changes
  (check the Seetacloud console); data/results persist on `/autodl-fs`.

## 2. R3 design (frozen)

- Generating structure **XAJ-CN**; fitted structures **Base / TGD2 / CN**;
  regimes **IC-CMA-ES / dPL**. 531 basins; truth is noise-free;
  `theta* = g*(A)` exactly (no random residual).
- `g*`: CN-IC parameter field (best train-KGE restart) anchored PCA (K=15,
  95% cumvar) + ridge (alpha=316.2, 5-fold basin CV seed 20260730) on the
  full 35-dim robust-normalized attribute vector (**`frac_snow` index 3
  retained**).
- Truth artifacts (frozen, do not regenerate): `results/r3_synthetic_truth_v1/`
  (`theta_star.npz`, `q_star.npz` key `target_mm_day` [531,12418], `x_star.npz`,
  `snow_star.npz`, manifests). Round-trip recorded==production bitwise (0.0).
- Periods: warmup 1980-10-01..1981-09-30, train ..1995-09-30, test ..2010-09-30.
  Metric: standard KGE everywhere (see `r3/docs/kge_audit.md`).

## 3. Completed

- **Phase A — g_thresh forward identity fix**: CN models accept optional
  `forcings["cn_psol_annual"]` (default unchanged); IC runtime passes
  canonical full-record value when `canonical_cn_psol_annual` config flag set
  (the IC runner sets it exactly when `--target-npz` is given); dPL loader
  computes it from the full record when `target_override_npz` is set.
  Oracle identity over 531 basins: IC train path 3.8e-6 max-abs (float
  level); split/eval-path residuals are the frozen 365-day warm-up
  convention only (see `results/r3_gate_v1/oracle_identity.json`).
- **dPL loader bug fixed**: target-override rows are selected **by basin ID**
  (npz `basin_ids`), not by position — the pilot-first reordered basin list
  had silently mis-paired basins with Q* rows (loss stuck ~0.91 before fix,
  ~0.02 after).
- **dPL-path oracle audit (531 basins)**: `results/r3_gate_v1/oracle_dpl_audit.json`
  — window-path KGE median 1.0000 (84.9% rows >=0.99), eval-path ceiling
  median 1.0000 (96.6% >=0.99), eval-KGE vs frac_snow Spearman -0.001.
  **Verdict: window/warm-up protocol unchanged** (no material snow-dependent
  oracle bias; the removed historical g_thresh bias was strongly
  snow-dependent, Spearman 0.73).
- **531 correct-CN gate (remote AutoDL, results downloaded & verified)**:
  - `r3_gate_ic_xaj_cn_531_v1/`: CN-IC complete, 63.4 min, 5310 records,
    best-restart train KGE median **0.9993** (min 0.9899).
  - `r3_gate_dpl_xaj_cn_seed_{42,123,2026}/`: CN-dPL complete, 100 epochs,
    val_kge_median **0.9953 / 0.9950 / 0.9954**; params [531,17].
- **Tests**: `tests/test_r3_truth_generator.py` (9) +
  `tests/test_r3_forward_identity.py` (7) all pass. Full suite: 303 passed,
  1 known pre-existing failure (`test_active_dpl_registry_covers_all_models`),
  1 skipped.
- **12-basin engineering pilot**: IC stages (CN/Base/TGD2) done; 8/9 dPL
  runs COMPLETE; **last one running**: dPL XAJ_TGD2 seed 2026
  (epoch ~13/100, ETA ~1.5-2h). The pilot manifest (`r3_pilot_v1/pilot_manifest.json`)
  is written only when pilot.py finishes. Pilot is engineering-only.

## 4. Key numbers (531 gate)

| metric | IC (best restart) | dPL (per seed / median) |
|---|---|---|
| train KGE vs Q* | median 0.9993, min 0.9899 | val 0.9950-0.9954 |
| oracle (theta* through IC path) | ~1.0 | eval-path ceiling ~1.0 |

## 5. What remains (next session)

1. Wait for the pilot's last dPL run to finish (XAJ_TGD2 seed 2026); then
   `r3/pilot.py --stage comp-dpl` will complete if needed, else just note
   COMPLETE markers; run `r3/analyze_pilot.py` (optional; pilot is
   engineering-only).
2. **Run the 531 gate analysis**: `python r3/gate_analysis.py --device cuda`
   → produces `results/r3_gate_v1/gate_report.json` + CSVs
   (parameters_recoverability, discharge/state metrics, restart stability,
   attribute identity, per-parameter IC/dPL recoverability, seed spread).
3. Produce the correct-CN gate report (oracle gap, Q/parameter/state
   recovery, 15 shared-XAJ per-parameter table, IC restart & dPL seed
   stability, frac_snow diagnostics). **Do not auto-select the identifiable
   subset** — present evidence, freeze by external review.
4. **Stop**: do NOT launch Base/TGD2 531 runs until the identifiable
   parameter subset is frozen from CN-only results.

## 6. Gotchas (learned the hard way)

- **Import trap**: `manuscript/scripts/r3/common.DEFAULT_PROJECT_ROOT` must be the worktree
  (`Path(__file__).parents[3]`), NOT the main checkout — `load_bundle()`
  inserts it at `sys.path[0]`; using the main checkout silently imports
  unpatched `models/`/`ablation/` (this cost hours of debugging).
- **Channel order**: when building forcing arrays manually, stack in
  `("precip","temp","pet")` order (P,T,PET); a `precip,pet,temp` stack with
  positional assignment swaps temp/pet and silently destroys results.
- **g_thresh semantics**: the CN model historically estimates mean annual
  solid precipitation from the input sequence; all synthetic-protocol paths
  must pass the canonical value (see §3). Truth itself used the full
  12418-day record.
- **Dynamo limits**: tests that exercise many shapes need
  `torch._dynamo.config.recompile_limit/cache_size_limit = 256` (production
  runners already set this).
- **IC runner registration**: `XAJ`/`XAJ_CN`/`XAJ_TGD2` were added to
  `training/ic/run_tgd2_batched_cmaes_531.py` (XAJ_TGD2 lives in
  MODEL_DIMENSIONS → TGD2 seed protocol).
- **Remote watcher**: the previous `r3_watch_remote.sh` rsync had a syntax
  bug (ssh string embedded in source path); it correctly refused to power
  off on verification failure. Download via:
  `rsync -az -e "ssh -p <port> -i ~/.ssh/r3_autodl" root@<host>:/autodl-fs/r3/results/ <local results>/`
- **Warm-up convention**: split/eval paths start 365 d before the scored
  period with default states; residuals vs the continuous truth are small
  but nonzero (bounded ~2.3 mm/day on snowiest basins) — documented, not a
  g_thresh redefinition.
- **KGE**: repo "KGE" is standard KGE everywhere (r/alpha/beta); the paper's
  "modified KGE′" wording is a pre-existing documentation discrepancy,
  recorded in `r3/docs/kge_audit.md`, not silently changed.

## 7. Commands

```bash
cd /home/jingxin/orca/workspaces/dmg-research/hydrodiag-r3/project/hydrodiag
python r3/gate_analysis.py --device cuda          # 531 gate report
python r3/analyze_pilot.py --device cuda          # pilot gate outputs (after pilot finishes)
python -m pytest tests/test_r3_truth_generator.py tests/test_r3_forward_identity.py -q
```

Remote rerun (if needed): power on node → reinstall key (password auth via
paramiko) → rerun `/autodl-fs/r3/remote_chain.py` (D2 3 seeds then D1 IC)
after re-syncing code with rsync (exclude results/outputs/archive/.git).
