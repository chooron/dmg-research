# Rule-based OOB representative-model selection

This directory contains the frozen, non-training selector for the future OOB/PUB experiment. It uses only the completed seen-basin outputs in `project/benchmark/results/seenbasin_remaining_analysis_20260901/`.

## Contract

- Candidate pool: all 36 canonical models; no model is excluded.
- `G = median_b(KGE_IC - KGE_dPL)`; `R` is frozen parameter–attribute reproducibility; `D` is frozen bounds-normalized RMS realization distance; `U` is archived IC restart uncertainty; `K_joint=min(K_IC,K_dPL)`; `P` is parameter count; `A` is the number of BH-FDR-significant G–attribute associations.
- All seven continuous features use deterministic 0–1 percentile ranks.
- Primary selection: median G×R split, two models per quadrant, centroid-nearest representative plus same-quadrant 5D contrast.
- Performance gate: `K_joint >= Q25`; any quadrant relaxation is recorded explicitly.
- A and parameter-count coverage repairs stay within the same quadrant; no manual override is allowed.
- No OOB/PUB information, H1, multi-seed, PUR, training, or checkpoint operation is used.

## Scripts

- `selection_common.py`: frozen inputs, feature construction, percentile ranks.
- `select_models.py`: primary 8, fallback 6, gates, sensitivity, tables, report, provenance.
- `plot_selection_qc.py`: two QC figures and figure-data CSVs.
- `verify_selection.py`: deterministic selection contract QC.
- `revise_mopex_redundancy.py`: Scenario A/B MOPEX gate evaluation and blocked/revised artifacts.
- `structure_qc.py`: auxiliary registry/core structural descriptors.
- `verify_mopex_revision.py`: deterministic MOPEX revision QC.

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python3 project/benchmark/analysis/oob_model_selection_20260901/select_models.py
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg .venv/bin/python3 project/benchmark/analysis/oob_model_selection_20260901/plot_selection_qc.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python3 project/benchmark/analysis/oob_model_selection_20260901/run_mopex_revision.py
```

Outputs are written to `project/benchmark/results/oob_model_selection_20260901/`. Review `OOB_MODEL_SELECTION_REPORT.md` before any future OOB protocol is designed. `HANDOFF.md` records the stop boundary.
