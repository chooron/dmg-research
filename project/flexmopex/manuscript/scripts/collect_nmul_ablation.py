from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_DIR / "results" / "block1_nmul_ablation"
WEIGHT_NAMES = ("w_phen", "w_int", "w_snow", "w_sub")
METRIC_NAMES = ("nse", "kge", "r2", "rmse", "pbias")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize nmul ablation outputs.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Ablation output root.")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nmul", type=int, nargs="+", default=[1, 8, 16, 32])
    return parser.parse_args()


def alpha_label(alpha: float) -> str:
    return f"{alpha:g}"


def run_dir(root: Path, alpha: float, nmul: int, seed: int) -> Path:
    return root / "flex" / f"alpha{alpha_label(alpha)}" / f"nmul{nmul}" / f"seed_{seed}"


def find_test_dir(path: Path) -> Path | None:
    candidates = sorted(path.rglob("test*_Ep*/metrics_agg.json"))
    if not candidates:
        return None
    return candidates[-1].parent


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(metrics: dict[str, Any], metric: str, stat: str) -> float:
    value = metrics.get(metric, {})
    if isinstance(value, dict):
        return float(value.get(stat, np.nan))
    arr = np.asarray(value, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if stat == "median":
        return float(np.median(arr))
    if stat == "mean":
        return float(np.mean(arr))
    if stat == "std":
        return float(np.std(arr))
    return float("nan")


def finite_array(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(alpha: float, seed: int, nmul: int, test_dir: Path | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "alpha": alpha,
        "seed": seed,
        "nmul": nmul,
        "status": "complete" if test_dir is not None else "missing",
        "test_dir": str(test_dir) if test_dir is not None else "",
    }
    if test_dir is None:
        return row
    metrics = load_json(test_dir / "metrics_agg.json")
    for metric in METRIC_NAMES:
        for stat in ("median", "mean", "std"):
            row[f"{metric}_{stat}"] = metric_value(metrics, metric, stat)
    return row


def summarize_weights(alpha: float, seed: int, nmul: int, test_dir: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if test_dir is None:
        return rows
    process_means: dict[str, float] = {}
    for name in WEIGHT_NAMES:
        path = test_dir / f"{name}.npy"
        if not path.exists():
            rows.append({"alpha": alpha, "seed": seed, "nmul": nmul, "process": name, "status": "missing"})
            continue
        arr = finite_array(path)
        if arr.size == 0:
            rows.append({"alpha": alpha, "seed": seed, "nmul": nmul, "process": name, "status": "empty"})
            continue
        mean = float(np.mean(arr))
        process_means[name] = mean
        rows.append(
            {
                "alpha": alpha,
                "seed": seed,
                "nmul": nmul,
                "process": name,
                "status": "complete",
                "mean": mean,
                "median": float(np.median(arr)),
                "p10": float(np.quantile(arr, 0.10)),
                "p25": float(np.quantile(arr, 0.25)),
                "p75": float(np.quantile(arr, 0.75)),
                "p90": float(np.quantile(arr, 0.90)),
                "active_share_gt_0_01": float(np.mean(arr > 0.01)),
                "active_share_gt_0_50": float(np.mean(arr > 0.50)),
            }
        )
    if process_means:
        ordered = sorted(process_means.items(), key=lambda item: item[1], reverse=True)
        rows.append(
            {
                "alpha": alpha,
                "seed": seed,
                "nmul": nmul,
                "process": "sum",
                "status": "complete",
                "mean": float(sum(process_means.values())),
                "rank_by_mean": ">".join(name for name, _ in ordered),
            }
        )
    return rows


def write_markdown(path: Path, metric_rows: list[dict[str, Any]], weight_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# nmul ablation summary",
        "",
        "Fixed settings: model=flex, alpha=0.01, seed=42.",
        "",
        "## Test performance",
        "",
        "| nmul | status | NSE median | KGE median | RMSE median |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(metric_rows, key=lambda r: r["nmul"]):
        fmt = lambda key: "nan" if key not in row or not np.isfinite(row[key]) else f"{row[key]:.4f}"
        lines.append(
            f"| {row['nmul']} | {row['status']} | {fmt('nse_median')} | "
            f"{fmt('kge_median')} | {fmt('rmse_median')} |"
        )

    lines.extend(
        [
            "",
            "## Weight pattern",
            "",
            "| nmul | process rank by mean | sum mean |",
            "| ---: | --- | ---: |",
        ]
    )
    sums = [row for row in weight_rows if row.get("process") == "sum"]
    for row in sorted(sums, key=lambda r: r["nmul"]):
        mean = row.get("mean", np.nan)
        mean_text = "nan" if not np.isfinite(mean) else f"{mean:.4f}"
        lines.append(f"| {row['nmul']} | {row.get('rank_by_mean', '')} | {mean_text} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    metric_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for nmul in args.nmul:
        test_dir = find_test_dir(run_dir(root, args.alpha, nmul, args.seed))
        metric_rows.append(summarize_metrics(args.alpha, args.seed, nmul, test_dir))
        weight_rows.extend(summarize_weights(args.alpha, args.seed, nmul, test_dir))

    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "nmul_metrics_summary.csv", metric_rows)
    write_csv(root / "nmul_weight_summary.csv", weight_rows)
    write_markdown(root / "nmul_ablation_summary.md", metric_rows, weight_rows)

    print(f"Wrote {root / 'nmul_metrics_summary.csv'}")
    print(f"Wrote {root / 'nmul_weight_summary.csv'}")
    print(f"Wrote {root / 'nmul_ablation_summary.md'}")


if __name__ == "__main__":
    main()
