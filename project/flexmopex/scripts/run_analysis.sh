#!/bin/bash
# Run post-processing analyses on completed experiment results
# Usage: bash scripts/run_analysis.sh [BLOCK]
set -e
BLOCK=${1:-all}
RESULTS_DIR="results"
ANALYSIS_DIR="analysis"

echo "Running analysis for block: $BLOCK"

if [[ "$BLOCK" == "block1" || "$BLOCK" == "all" ]]; then
    echo "--- Block 1: Performance + complexity metrics ---"
    python analysis/collect_metrics.py --results_dir ${RESULTS_DIR}/block1_main --out ${RESULTS_DIR}/block1_main/metrics_by_seed.csv
    python analysis/collect_weights.py --results_dir ${RESULTS_DIR}/block1_alpha_path --out ${RESULTS_DIR}/block1_alpha_path/weights_by_alpha_seed.csv
    python analysis/plot_performance_comparison.py
    python analysis/plot_alpha_path.py
fi

if [[ "$BLOCK" == "block2" || "$BLOCK" == "all" ]]; then
    echo "--- Block 2: Stability + interpretability ---"
    python analysis/run_seed_stability.py --alpha 0.005 0.01 0.03
    python analysis/run_attribute_analysis.py --alpha 0.01
    python analysis/run_parameter_manifold.py --alpha 0.01
fi

if [[ "$BLOCK" == "block3" || "$BLOCK" == "all" ]]; then
    echo "--- Block 3: LORO generalization ---"
    python analysis/collect_loro_metrics.py --results_dir ${RESULTS_DIR}/block3_loro
    python analysis/plot_loro_results.py
fi

echo "Analysis complete."
