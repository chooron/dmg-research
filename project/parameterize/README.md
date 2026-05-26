# Parameterize Project README

This directory contains the paper-facing workflow for learning HBV model
parameters from static basin attributes. It is intended to help reviewers and
editors quickly locate the experiment code, analysis pipeline, and manuscript
artifacts used for the parameter-learning study.

Project archive DOI: `10.5281/zenodo.20389200`.

## Scientific Purpose

The project studies whether catchment attributes can be used to infer stable,
interpretable, and uncertainty-aware HBV parameters. The emphasis is not only on
streamflow prediction accuracy, but also on the reproducibility of learned
parameter values and attribute-parameter relationships across random seeds,
training losses, and model formulations.

The main study uses the CAMELS-US 531-basin subset and a static-attribute neural
parameterizer coupled to an HBV-style differentiable hydrologic model. Neural
outputs are bounded with a sigmoid activation and mapped into HBV parameter
ranges before hydrologic simulation.

## Main Model Variants

The paper workflow compares three parameter-learning formulations, implemented
through `paper_variants.py` and the modules under `implements/`.

| Variant | Neural model | Purpose |
| --- | --- | --- |
| `deterministic` | `DeterministicParamModel` | Predicts one bounded HBV parameter vector per basin. |
| `mc_dropout` | `McMlpModel` | Uses dropout sampling at evaluation time as an approximate uncertainty proxy. |
| `distributional` | `DistributionalParamModel` | Predicts a parameter distribution and supports distribution-aware training. |

The common paper configuration is `conf/config_param_paper.yaml`. By default it
uses:

- training period: 1989-01-01 to 1998-12-31
- testing period: 1999-01-01 to 2009-12-31
- static basin attributes as neural-network inputs
- HBV physical model inputs: precipitation, mean temperature, and PET
- output directory pattern:
  `outputs/{variant}-531/{loss}/seed_{seed}/`

## Reproducibility Entry Points

Run commands from the repository root unless noted otherwise.

Train or evaluate one run:

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

Run the scripted multi-seed, multi-loss batches:

```bash
bash project/parameterize/scripts/run_param_paper_deterministic.sh
bash project/parameterize/scripts/run_param_paper_mc_dropout.sh
bash project/parameterize/scripts/run_param_paper_distributional.sh
```

The batch scripts default to seeds `111 222 333 444 555` and losses
`HybridNseBatchLoss`, `NseBatchLoss`, and `LogNseBatchLoss`. They accept
environment overrides such as `DEVICE`, `GPU_ID`, `SEEDS`, `LOSSES`, `EPOCHS`,
`MC_SAMPLES`, and `MAX_PARALLEL`.

## Analysis Pipeline

The primary stability-analysis entry point is:

```bash
uv run python project/parameterize/analysis/run_all.py
```

This pipeline reads trained runs under `project/parameterize/outputs/` and writes
analysis products to:

```text
project/parameterize/outputs/analysis/stability_stats/
```

It produces inventory tables, predictive metric summaries, parameter stability
statistics, cross-loss stability statistics, attribute-parameter correlation
matrices, relationship-stability summaries, and markdown reports. See
`analysis/README.md` for the step-by-step script inventory.

The core questions addressed by the analysis are:

- Which formulation gives acceptable predictive performance?
- Are inferred parameters stable across random seeds?
- Are inferred parameters stable across training losses?
- Are attribute-parameter correlation structures reproducible?
- Which learned relationships are robust enough to support manuscript claims?

## Manuscript Artifacts

The manuscript-oriented materials are under `manuscript/`.

Important subdirectories include:

- `manuscript/analysis_pipeline/`: integrated analysis modules used to build
  figure-ready tables and reports.
- `manuscript/plots/`: publication figure scripts and shared plotting utilities.
- `manuscript/figures/`: generated main and appendix figures.
- `manuscript/reports/`: quality-control notes, figure role summaries, and
  supporting interpretation reports.
- `manuscript/captions/`: figure caption drafts and attribute abbreviations.

To regenerate the manuscript analysis pipeline:

```bash
uv run python project/parameterize/manuscript/analysis_pipeline/run_all.py
```

To regenerate the publication-style figure suite:

```bash
uv run python project/parameterize/publication_figures.py --all
```

The generated figure manifest is recorded in `manuscript/manifest.json`.

## Directory Guide

| Path | Role |
| --- | --- |
| `conf/` | YAML configurations for paper and test runs. |
| `implements/` | Parameter models, trainers, losses, HBV static model, and DPL assembly. |
| `scripts/` | Batch runners for the three paper variants. |
| `analysis/` | General multi-run stability and relationship-analysis pipeline. |
| `manuscript/` | Manuscript-specific analysis, plots, captions, reports, and figures. |
| `outputs/` | Trained run outputs and derived analysis products. |
| `tests/` | Regression tests for configs, dispatch, trainers, analyses, and plotting pipelines. |
| `example/` | Small integration examples for parameter models and physical-model wiring. |

## Reviewer Notes

- The deterministic model is a point-estimate baseline; it does not provide
  intrinsic parameter intervals.
- MC-dropout uncertainty should be interpreted as an approximate stochastic
  proxy, not as a calibrated posterior.
- Distributional outputs are bounded parameter samples on the normalized search
  scale before conversion to HBV physical parameter ranges.
- Reported stability metrics evaluate reproducibility of learned relationships;
  they should not be read as direct proof of hydrologic causality or unique
  parameter identifiability.
- Some generated outputs and figures can be large. The main code path can be
  inspected through the entry points listed above even when full trained outputs
  are stored separately or archived.

## Quick File Checklist

For a fast editorial or review pass, start with:

1. `train_param_paper.py` - main training and evaluation entry point.
2. `conf/config_param_paper.yaml` - canonical paper configuration.
3. `paper_variants.py` - variant normalization and validation logic.
4. `analysis/run_all.py` and `analysis/README.md` - stability-analysis pipeline.
5. `publication_figures.py` - manuscript figure generation entry point.
6. `manuscript/manifest.json` - generated figure and report manifest.
