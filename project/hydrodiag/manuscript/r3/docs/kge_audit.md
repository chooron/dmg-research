# R3 Phase B1 — KGE / KGE′ audit (actual formulas, not names)

Every KGE implementation in the repository was read directly from code.
All of them implement the **standard (unmodified) KGE**:

```
KGE = 1 - sqrt( (r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2 )
r     = Pearson correlation (population moments)
alpha = std(sim) / std(obs)
beta  = mean(sim) / mean(obs)
```

on paired finite, non-negative values (zero discharge stays valid; no
imputation).  None of them is KGE′ (the Kling et al. 2012 modification that
replaces `alpha` with a coefficient-of-variation ratio).

## Audit table

| Path | Implementation location | Actual formula | Use |
|---|---|---|---|
| R1/R2 IC (CMA-ES fitness + stored metrics) | `training/ic/gpu_kge.py::compute_kge_fp64_matrix_gpu` (+`compute_kge_fp64_batch_gpu`) | standard KGE; population moments; invalid (`-999`) if <30 valid days or obs_std < 1e-10 | maximize on train split (warm-up 365 d excluded); test split evaluated with the same function |
| R1/R2 dPL training loss | `training/dpl/run_dpl_model.py::kge_per_basin` | standard KGE with `eps=1e-6` floored *inside* each squared term (`(r-1)^2+eps^2` etc.); finite/>=0 mask | per 365-day window loss `mean(1-KGE)` |
| R1/R2 dPL validation/final | `training/dpl/run_dpl_model.py::compute_kge_fp64` | standard KGE (numpy); <30 valid or obs_std<1e-10 -> -999 | validation period 1995-2010; `best_checkpoint` selection and final `val_kge` |
| R1 manuscript statistics | `manuscript/scripts/r1/r1_statistics.py::standard_kge` | standard KGE; min 30 valid | R1 tables (explicitly documented in `manuscript/.../README` as "the repository's standard KGE, not modified KGE-prime") |
| R3 IC (pilot + 531 gate) | same as R1/R2 IC, target = Q* (`--target-npz`) | identical formula | identical use |
| R3 dPL (pilot + 531 gate) | same as R1/R2 dPL, target = Q* (`target_override_npz`) | identical formula | identical use |
| R3 gate analysis | `manuscript/r3/common.py::standard_kge` | standard KGE; min 30 valid | oracle and gate metrics vs Q* |

## Verdict

- R1/R2 and R3 use **identical formulas** everywhere (only numerical floors
  differ: `kge_per_basin`'s in-term epsilon vs the min-samples/obs-std
  invalid guards of the other three).  No R1/R2 vs R3 inconsistency.
- The paper's "modified KGE′" wording does not match the code's standard KGE.
  This is a **documentation-level discrepancy that predates R3**; the R1
  README already records the choice of standard KGE explicitly.  R3 keeps
  the repository formula so R3 numbers are comparable with R1/R2 outputs.
  No silent change is made to any historical artifact.
- R3 gate KGEs are computed with `manuscript/r3/common.standard_kge`, which is
  numerically the same formula as the IC objective and the dPL validation.
