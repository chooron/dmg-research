# S4 Metric and Inference Audit

## Metric

The active IC objective and dPL final evaluation implement aligned standard KGE(Q) algebra in separate GPU and NumPy FP64 functions, with Pearson correlation, population standard deviations, a simulation-to-observation variability ratio, and a simulation-to-observation mean ratio. Finite nonnegative observations and simulations are retained; negative and nonfinite values are masked. IC and final evaluation require at least 30 valid samples and return `-999` for invalid cases. The dPL training loss uses the same component structure but adds `1e-6` stabilizers inside standard deviations, the beta denominator, and the distance root. It is therefore aligned but not byte-identical to final KGE(Q).

The code-level numeric check is recorded in `results/s4_metric_equivalence_checks.csv`. It uses a deterministic synthetic vector because no saved production observation/simulation pair was available for a replay. No log transform was found.

## Pairing and inference

The formal snow tables are basin-wise and preserve the same basin ID across structures. IC starts and dPL seeds are route-specific repetitions, not directly seed-paired. The active analysis implements descriptive means/medians/IQRs and Spearman association; no Wilcoxon, paired permutation, bootstrap, cluster bootstrap, or multiplicity correction implementation was found. A confirmatory test family remains unresolved.

## Key limitation

The dPL training code updates `best_checkpoint` from `evaluate(eval_forcing, eval_obs, ...)` at lines 856--868. In the active configuration these are the 1995-10-01--2010-09-30 evaluation arrays, not a calibration-only validation split. This is an evaluation-period checkpoint-selection risk and must be resolved or explicitly disclosed.
