# Parameterize Project README

This directory contains the scientific workflow and paper-facing artifacts for learning hydrological model (HBV) parameters from static catchment attributes across 531 CAMELS basins.

Project archive DOI: `10.5281/zenodo.20389200`.

---

## 1. Scientific Objective

The core objective is investigating whether catchment physical attributes can be used to infer **stable, interpretable, and uncertainty-aware HBV parameters**. Beyond streamflow predictive skill, the emphasis is placed on:
- **Reproducibility of learned parameters** across random seeds and loss functions.
- **Physical plausibility and consistency** of learned attribute-parameter relationships.
- **Uncertainty quantification** through distributional versus deterministic formulations.

The study is conducted over the **CAMELS-US 531-basin dataset** using static catchment attributes coupled to a differentiable lumped hydrological model (HBV). Neural outputs are bounded via sigmoid activations and linearly mapped to physically reasonable parameter intervals.

---

## 2. Model Formulations (Three Paper Variants)

Implemented through `paper_variants.py` and modules in `implements/`:

| Variant | Neural Model | Description & Purpose |
| --- | --- | --- |
| `deterministic` | `DeterministicParamModel` | Point-estimate baseline: predicts one bounded parameter vector per basin. |
| `mc_dropout` | `McMlpModel` | Test-time Monte Carlo dropout sampling as an approximate uncertainty proxy. |
| `distributional` | `DistributionalParamModel` | Predicts full parameter distributions and supports distribution-aware training. |

Canonical configuration: `conf/config_param_paper.yaml`
- **Training Period**: `1989-01-01` to `1998-12-31`
- **Testing Period**: `1999-01-01` to `2009-12-31`
- **Neural Inputs**: 35 static catchment attributes
- **Physical Inputs**: Daily precipitation ($P$), mean temperature ($T$), potential evapotranspiration ($PET$)
- **Output Path Pattern**: `outputs/{variant}-531/{loss}/seed_{seed}/`

---

## 3. Reproducibility & Execution Entry Points

Run all commands from the repository root:

### Single Model Training / Evaluation

```bash
uv run python project/parameterize/train_param_paper.py \
  --config project/parameterize/conf/config_param_paper.yaml \
  --variant distributional \
  --loss HybridNseBatchLoss \
  --seed 111 \
  --mode train_test \
  --device cuda \
  --gpu-id 0
```

### Multi-Seed / Multi-Loss Batch Execution

```bash
bash project/parameterize/scripts/run_param_paper_deterministic.sh
bash project/parameterize/scripts/run_param_paper_mc_dropout.sh
bash project/parameterize/scripts/run_param_paper_distributional.sh
```

Default seeds: `111, 222, 333, 444, 555`  
Default losses: `HybridNseBatchLoss`, `NseBatchLoss`, `LogNseBatchLoss`

---

## 4. Stability & Relationship Analysis Pipeline

The primary multi-run stability and attribute-parameter relationship analysis entry point:

```bash
uv run python project/parameterize/analysis/run_all.py
```

Outputs are written to `outputs/analysis/stability_stats/`. Key analyses evaluated:
1. **Predictive Performance**: Benchmark KGE / NSE metrics across variants and losses.
2. **Parameter Stability**: Cross-seed and cross-loss variability of inferred parameters.
3. **Correlation Structure**: Reproducibility of attribute-parameter Spearman correlation matrices.
4. **Dominant Relationships**: Statistical robustness and identifiability of learned physical rules.

---

## 5. Directory Structure & Organization

```
project/parameterize/
├── README.md                # Project architecture and reproducibility guide
├── train_param_paper.py     # Main training and evaluation CLI
├── paper_variants.py        # Model variant dispatch and validation
├── publication_figures.py   # Publication figure generation entry point
├── conf/                    # YAML configuration files for experiments
├── implements/              # Parameter networks, trainers, losses, and HBV physical models
├── scripts/                 # Automated bash batch runners for the 3 paper variants
├── analysis/                # Multi-run stability and relationship analysis pipeline
├── manuscript/              # Manuscript assets, figure scripts, captions, and tables
│   ├── paper/               # Merged full paper drafts (manu_wrr.md, manu_revised.md)
│   ├── analysis_pipeline/   # Integrated pipeline generating publication artifacts
│   ├── plots/               # Figure generation scripts and shared styling
│   ├── figures/             # Output figure files (main and appendix)
│   ├── captions/            # Main text and appendix figure caption drafts
│   └── tables/              # Main and appendix LaTeX/markdown tables
├── report/                  # Archived procedural reports, QC assessments, and notes (gitignored)
│   ├── qc_and_synthesis/    # Quality-control audits, progress summaries, collinearity checks
│   ├── extends/             # Extended diagnostic and sensitivity summaries
│   └── writing_notes/       # Draft notes for results and discussion sections
├── outputs/                 # Trained model checkpoints and analysis products (gitignored)
├── tests/                   # Integration and regression test suite
└── example/                 # Minimal standalone demonstration scripts
```

> **Note on Version Control**:
> - Core production code, configurations, tests, and plotting scripts are tracked in Git.
> - All `.md` documents under `manuscript/` (paper drafts, caption files, tables) and procedural reports in `report/` are gitignored to maintain a clean codebase and protect drafting privacy.
> - Large experimental outputs, model checkpoints, and caches in `outputs/` are gitignored.

---

## 6. Testing

Verify project integrity and regression tests:

```bash
PYTHONPATH=. .venv/bin/pytest project/parameterize/tests/
```
