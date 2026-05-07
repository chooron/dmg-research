from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
ABLATION_RESULTS_DIR = PROJECT_DIR / "ablation" / "results"
PARAMETER_RESULTS_DIR = PROJECT_DIR / "parameters" / "results"
DEFAULT_OUTPUT_CSV = ABLATION_RESULTS_DIR / "baseline_reliability_comparison.csv"
DEFAULT_TREND_WINDOW_DAYS = 365

ABLATION_VARIANT_NAME_MAP = {
    "S4D-baseline": "S4D-baseline",
    "S4D-LN": "S4D-LN",
    "S4D-Softsign": "S4D-Softsign",
    "S4D-LN-Softsign": "S4D-LN-Softsign",
    "S5D-ConvOnly": "S4D-ConvOnly",
    "S5D-ConvBN-Softsign": "S4D-ConvBN-Softsign",
    "S5D-ConvLN-Sigmoid": "S4D-ConvLN-Sigmoid",
    "S5D-full": "S4D-ConvLN-Softsign",
}

BASELINE_NPZ_SPECS = {
    "LSTM": "lstm_seed111_normalized_dynamic_parameters.npz",
    "S5Dv2": "s5dv2_seed111_normalized_dynamic_parameters.npz",
    "Transformer": "transformer_seed111_normalized_dynamic_parameters.npz",
    "TimeMixer": "timemixer_seed111_normalized_dynamic_parameters.npz",
    "TCN": "tcn_seed111_normalized_dynamic_parameters.npz",
}

OUTPUT_COLUMNS = [
    "Variant",
    "Variability",
    "Roughness",
    "Long-term shift",
    "Trend-to-noise ratio",
    "Boundary saturation ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a complete reliability comparison table for ablation variants and baseline models."
    )
    parser.add_argument(
        "--ablation-summary-csv",
        type=Path,
        default=ABLATION_RESULTS_DIR / "ablation_parameter_reliability_summary.csv",
        help="Per-parameter ablation summary exported by s5d_ablation_pipeline.py.",
    )
    parser.add_argument(
        "--baseline-npz-dir",
        type=Path,
        default=PARAMETER_RESULTS_DIR,
        help="Directory containing baseline normalized dynamic-parameter NPZ files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Destination CSV path.",
    )
    return parser.parse_args()


def _trend_window_length(n_time: int, preferred: int = DEFAULT_TREND_WINDOW_DAYS) -> int:
    if n_time <= 1:
        return 1
    if n_time >= 4:
        return max(1, min(preferred, n_time // 4))
    return max(1, n_time // 2 or 1)


def load_ablation_rows(summary_csv: Path) -> list[dict[str, float | str]]:
    df = pd.read_csv(summary_csv)
    rows: list[dict[str, float | str]] = []
    for variant, display_name in ABLATION_VARIANT_NAME_MAP.items():
        subset = df.loc[df["variant"].eq(variant)]
        if subset.empty:
            raise FileNotFoundError(f"Missing ablation summary rows for variant: {variant}")
        rows.append(
            {
                "Variant": display_name,
                "Variability": float(subset["mean_variability"].mean()),
                "Roughness": float(subset["mean_roughness"].mean()),
                "Long-term shift": float(subset["mean_long_term_shift"].mean()),
                "Trend-to-noise ratio": float(subset["mean_trend_to_noise_ratio"].mean()),
                "Boundary saturation ratio": float(subset["mean_boundary_saturation_ratio"].mean()),
            }
        )
    return rows


def _metric_summary_from_npz(npz_path: Path) -> dict[str, float]:
    with np.load(npz_path, allow_pickle=True) as data:
        params = data["normalized_parameters"].astype(np.float32, copy=False)

    diffs = np.diff(params, axis=0)
    trend_window = _trend_window_length(params.shape[0])

    variability = np.nanmedian(np.abs(diffs), axis=(0, 3))
    roughness = np.nanmean(np.square(diffs), axis=(0, 3))
    early = np.nanmean(params[:trend_window], axis=0)
    late = np.nanmean(params[-trend_window:], axis=0)
    long_term_shift = np.nanmedian(np.abs(late - early), axis=2)
    trend_to_noise = long_term_shift / np.maximum(variability, 1e-6)
    saturation = np.nanmean((params < 0.02) | (params > 0.98), axis=(0, 3))

    return {
        "Variability": float(np.nanmean(np.nanmean(variability, axis=0))),
        "Roughness": float(np.nanmean(np.nanmean(roughness, axis=0))),
        "Long-term shift": float(np.nanmean(np.nanmean(long_term_shift, axis=0))),
        "Trend-to-noise ratio": float(np.nanmean(np.nanmean(trend_to_noise, axis=0))),
        "Boundary saturation ratio": float(np.nanmean(np.nanmean(saturation, axis=0))),
    }


def load_baseline_rows(npz_dir: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for label, filename in BASELINE_NPZ_SPECS.items():
        npz_path = npz_dir / filename
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing baseline NPZ: {npz_path}")
        stats = _metric_summary_from_npz(npz_path)
        rows.append({"Variant": label, **stats})
    return rows


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(load_ablation_rows(args.ablation_summary_csv))
    rows.extend(load_baseline_rows(args.baseline_npz_dir))

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(output_csv, index=False, float_format="%.6f")
    print(f"saved {output_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
