from pathlib import Path
import re

import pandas as pd


ROOT = Path("/workspace/autoresearch/project/parameterize")
METRICS_PATH = ROOT / "outputs/analysis/stability_stats/tables/metrics_long.csv"
MANUSCRIPT_PATH = ROOT / "paper/manu_wrr.md"

MODEL_ORDER = [
    ("deterministic", r"\delta_{\mathrm{base}}"),
    ("mc_dropout", r"\delta_{\mathrm{mcd}}"),
    ("distributional", r"\delta_{\mathrm{dist}}"),
]


def format_row(label: str, values: list[float]) -> str:
    formatted = [f"{value:.3f}" for value in values]
    return (
        f"| All complete runs | {label:<28} | "
        f"{formatted[0]:<24} | {formatted[1]:<23} | {formatted[2]:<24} |"
    )


def main() -> None:
    metrics = pd.read_csv(METRICS_PATH)
    missing = {"model", "bias_abs", "pbias_abs"} - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing required metric columns: {sorted(missing)}")

    grouped = metrics.groupby("model", sort=False).agg(
        bias_abs_median=("bias_abs", "median"),
        pbias_abs_median=("pbias_abs", "median"),
    )

    model_keys = [model for model, _ in MODEL_ORDER]
    absent = [model for model in model_keys if model not in grouped.index]
    if absent:
        raise ValueError(f"Missing expected models in metrics table: {absent}")

    bias_values = [grouped.loc[model, "bias_abs_median"] for model in model_keys]
    pbias_values = [grouped.loc[model, "pbias_abs_median"] for model in model_keys]

    text = MANUSCRIPT_PATH.read_text()
    new_bias = format_row("Median absolute bias", bias_values)
    new_pbias = format_row("Median absolute percent bias", pbias_values)

    patterns = {
        r"^\| All complete runs \| Median absolute bias\s+\|.*\|$": new_bias,
        r"^\| All complete runs \| Median absolute percent bias \|.*\|$": new_pbias,
    }
    for pattern, replacement in patterns.items():
        text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Expected exactly one matching row for pattern: {pattern}")

    text = text.replace("## Table 1 placeholder", "## Table 1")
    MANUSCRIPT_PATH.write_text(text)

    print("Filled Table 1 all-complete-runs bias metrics:")
    for model, label in MODEL_ORDER:
        print(
            f"{label}: median |bias|={grouped.loc[model, 'bias_abs_median']:.3f}, "
            f"median |pbias|={grouped.loc[model, 'pbias_abs_median']:.3f}"
        )


if __name__ == "__main__":
    main()
