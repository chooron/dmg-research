# S4 Experimental Design, Analysis and Robustness: Source of Truth

## Executive summary

The project provides a usable but qualified foundation for S4.1-S4.5, S4.10-S4.12. The active metric is standard KGE(Q), not the KGE-prime label used in planning text. IC and dPL final evaluation use aligned standard-KGE algebra in separate FP64 implementations, while dPL training uses an epsilon-stabilized differentiable approximation. Basin-wise tables and route-specific repeats support descriptive paired differences. No formal paired test or multiple-testing correction is implemented.

The principal validity conflict is dPL checkpoint selection: the code evaluates `eval_forcing/eval_obs` to update `best_checkpoint`, and those arrays correspond to the configured evaluation period. This is not a harmless naming issue. S4 must either use a non-leaky selection or explicitly limit the claims.

## S4.1-S4.5

The corrected 531-basin IC/dPL master and official dPL three-seed tables support basin-level fixed-bin, quintile, and continuous snow-fraction summaries. Fixed bins are computationally verified, but their pre-result lock is not. The lowest snow bin is near-snow-free descriptively; a formal negative-control rule is absent. The active source contains no Wilcoxon, paired permutation, bootstrap, FDR, or Bonferroni implementation.

## S4.6-S4.9

S2 parameter bounds and dPL normalized/physical outputs exist. IC task records expose normalized best coordinates but not physical parameters or a completed parameter-organization analysis. S1 SWE metadata explicitly prohibit calling the external product truth, and no executed state-comparison analysis was found. No active synthetic experiment or quantitative cross-host reading rule was found. These sections are blocked or partial, not silently completed from plans.

## S4.10-S4.12

Fixed-bin and quintile results, seed tables, IC restart summaries, and optimizer traces are available. Metric, boundary-threshold, state-window, synthetic-noise, and formal cross-host sensitivities are not. TGD and other model coverage must follow the result inventories; incomplete/provisional results must not be represented as complete 531-basin evidence.

## Evidence products

All section statuses, conflicts, unresolved items, claim wording, and follow-up actions are in `results/s4_section_readiness.csv`, `results/s4_conflicts.csv`, `results/s4_unresolved_items.csv`, `results/s4_claim_evidence_map.csv`, and the companion reports in this directory.
