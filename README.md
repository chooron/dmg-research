# dmg-research

`dmg-research` is a research workspace built around differentiable hydrologic modeling, differentiable parameter learning (dPL), and basin-scale parameter inference. It extends the upstream `generic_deltamodel` / `dmg` ecosystem, but the current repository has evolved into two clear project lines:

1. `project/bettermodel`
   Focused on predictive performance, PUB generalization, architectural ablations, and interpretability for hybrid hydrologic models.
2. `project/parameterize`
   Focused on paper-oriented HBV parameter learning from basin attributes, with emphasis on parameter stability, uncertainty, and attribute-parameter relationship analysis.

This repository is not just a collection of training scripts. It is a working research environment that includes training, evaluation, post-processing, statistical analysis, and paper-figure generation.

## Project Overview

| Project | Primary Goal | Current Scope | Main Entrypoints |
| --- | --- | --- | --- |
| `project/bettermodel` | Improve or compare predictive performance of hybrid dHBV models | Multiple neural backbones, PUB training flows, S4D/S5D ablations, interpretability, visualization, multiseed summaries | `run_experiment.py`, `run_pub_experiment.py` |
| `project/parameterize` | Compare parameter-learning strategies for HBV parameter inference | Deterministic / MC-dropout / distributional parameter models, paper experiment configs, stability analysis pipeline, Figure 2 generation suite | `train_param_paper.py`, `analysis/run_all.py`, `figure2/src/api.py` |

## Project 1: `bettermodel`

`bettermodel` is the predictive-model experimentation surface. It keeps the dPL idea of combining a physical model with a neural network, while putting most of the engineering focus on model comparison, generalization, and experimental reproducibility.

The directory currently contains several major work areas:

- `conf/`
  Stores experiment configurations such as `config_dhbv_lstm.yaml`, `config_dhbv_gru.yaml`, `config_dhbv_tcn.yaml`, `config_dhbv_transformer.yaml`, `config_dhbv_tsmixer.yaml`, and HOPE / S4D / S5D variants.
- `run_experiment.py`
  A unified training and evaluation entrypoint that supports `train`, `test`, and `train_test`, with CLI overrides for `seed`, `epochs`, `loss`, `test_epoch`, and related runtime settings.
- `run_pub_experiment.py`
  A dedicated PUB experiment entrypoint built around `PubTrainer`, intended for held-out or grouped generalization evaluation.
- `ablation/`
  Includes `s5d_ablation_pipeline.py` for controlled comparisons of normalization choices, activation functions, and convolution-enhanced variants.
- `multiseed/`
  Contains summary scripts that aggregate `NSE` / `KGE` results across multiple random seeds into CSV tables and plots.
- `interpret/` and `visualize/`
  Provide post-processing, parameter visualization, interpretability utilities, and spatial plotting.
- `tests/`
  Covers config loading, gradient behavior, training resume logic, and S5D ablation-related behavior.

Typical questions this project is designed to answer:

- Which hybrid model architecture performs best on CAMELS-style prediction tasks?
- Do specific architectural choices improve PUB or OOD robustness?
- How do different model families differ in learned states, inferred parameters, or interpretation outputs?

### Common `bettermodel` Commands

Single train/test run:

```bash
uv run python project/bettermodel/run_experiment.py \
  --config project/bettermodel/conf/config_dhbv_lstm.yaml \
  --mode train_test \
  --seed 111
```

PUB experiment:

```bash
uv run python project/bettermodel/run_pub_experiment.py \
  --config project/bettermodel/conf/config_dhbv_lstm.yaml \
  --mode train_test \
  --seed 111
```

Scripted examples:

- `project/bettermodel/scripts/run_dhbv_lstm.sh`
- `project/bettermodel/scripts/run_dhbv_gru.sh`
- `project/bettermodel/scripts/run_dhbv_transformer.sh`

## Project 2: `parameterize`

`parameterize` is the more paper-facing research line. Its purpose is not only to improve streamflow prediction, but to study whether basin attributes can be used to learn HBV parameters in a stable, interpretable, and uncertainty-aware way.

The project is currently organized around a full workflow from training to analysis to figure production.

### Current Model Variants

`project/parameterize/paper_variants.py` currently maintains three paper-oriented parameter-learning variants:

- `deterministic`
  Predicts a single deterministic parameter set.
- `mc_dropout`
  Uses dropout-based sampling to approximate parameter uncertainty at evaluation time.
- `distributional`
  Predicts an explicit parameter distribution and supports distribution-aware training mechanisms such as KL regularization.

### Training and Experiment Structure

- `train_param_paper.py`
  Main entrypoint for paper experiments, with CLI overrides for `--variant`, `--loss`, `--seed`, `--epochs`, and `--mc-samples`.
- `conf/config_param_paper.yaml`
  Core paper configuration. Outputs are written by default to `project/parameterize/outputs/{variant}-531/{loss}/seed_{seed}/`.
- `scripts/run_param_paper_*.sh`
  Batch scripts for deterministic, MC-dropout, and distributional runs across multiple seeds and losses.
- `implements/`
  Holds the parameter-learning trainers, losses, HBV static model, MC-MLP implementation, and dPL assembly logic.
- `tests/`
  Covers configs, dispatch logic, analysis modules, figure pipelines, and behavior of the different parameterization variants.

### Analysis and Figure Production

One of the strongest parts of `parameterize` is that the downstream analysis pipeline is already structured and reusable:

- `analysis/`
  Converts multirun outputs into standardized long-form tables and analyzes:
  predictive metrics,
  cross-seed parameter stability,
  cross-loss parameter stability,
  and cross-seed / cross-loss stability of attribute-parameter correlation structure.
- `figure2/`
  Provides the Figure 2 manuscript pipeline. It currently includes an 11-figure suite, companion tables, manifests, and QC reports.
- `figures/`
  Contains direct plotting scripts for additional analysis outputs.
- `example/`
  Includes smaller integration examples that show how parameter models and differentiable physical models are wired together.

Typical questions this project is designed to answer:

- Which parameter-learning strategy is most stable across random seeds?
- Are attribute-parameter relationships reproducible and hydrologically interpretable?
- Does parameter uncertainty provide meaningful scientific signal rather than only numerical variability?

### Common `parameterize` Commands

Single paper experiment:

```bash
uv run python project/parameterize/train_param_paper.py \
  --config project/parameterize/conf/config_param_paper.yaml \
  --variant mc_dropout \
  --mode train_test \
  --seed 111
```

Batch multiseed / multiloss run:

```bash
bash project/parameterize/scripts/run_param_paper_mc_dropout.sh
```

Run the full analysis pipeline:

```bash
uv run python project/parameterize/analysis/run_all.py
```

Generate the full Figure 2 suite:

```bash
uv run python project/parameterize/figure2/src/api.py \
  --config project/parameterize/conf/config_param_paper.yaml
```

## Shared Foundation

Beyond the two main projects, the repository also contains a shared implementation layer used across training and experiment workflows:

- `implements/`
  Shared trainer and runtime infrastructure, including:
  `causal_trainer.py`,
  `baseline_trainer.py`,
  and `gnann_splitter.py`.
- `docs/`
  Research notes, interface contracts, and result drafts, including:
  `docs/mc_mlp_interface.md`,
  `docs/mc_parameter_correlation_analysis.md`,
  and `docs/results.md`.
- `data/`
  Default location for experiment inputs and dataset-related files.

At a high level:

- `bettermodel` covers predictive-model experiments and architecture comparisons.
- `parameterize` covers the parameter-learning paper workflow and its analysis products.
- Root-level shared modules and documents support both lines of work.

## Suggested Reading Order

If you are new to this repository, the fastest way to build context is:

1. Read this README for the high-level split between the two projects.
2. If you care about predictive experiments, start with `project/bettermodel/conf/` and `project/bettermodel/run_experiment.py`.
3. If you care about the paper workflow, start with `project/parameterize/train_param_paper.py`, `project/parameterize/analysis/README.md`, and `project/parameterize/figure2/`.
4. If you want to inspect current output organization, continue with `docs/results.md` and `project/parameterize/figure2/figures/main_revised/`.

## Environment and Dependencies

The repository targets Python 3.10+ and uses `pyproject.toml` as the main dependency definition. Core libraries include:

- `dmg`
- `dmotpy`
- `torch`
- `hydrodl2`
- `omegaconf`
- `scikit-learn`
- `scipy`
- `seaborn`

Recommended setup:

```bash
uv sync
```

If you plan to use GPU training, verify that the local `torch` build matches your CUDA environment.

## Current Repository State

Based on the current code and checked-in artifacts, this repository has already moved well beyond an early prototype:

- `bettermodel` has a fairly mature setup for training, PUB evaluation, ablations, multiseed summaries, and interpretability work.
- `parameterize` has a complete paper-oriented workflow covering training entrypoints, three parameter-learning variants, stability analysis, and a full Figure 2 production stack.
- Shared modules and documents are now carrying cross-project logic, so future work should prefer reusing existing trainers, analysis modules, and figure-generation utilities rather than rebuilding parallel infrastructure.
