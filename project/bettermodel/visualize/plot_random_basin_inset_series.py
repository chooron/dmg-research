from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dmg.core.utils import import_data_loader  # noqa: E402
from project.bettermodel import load_config  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "visualize" / "figures"
MSTTCOREFONTS_DIR = Path("/usr/share/fonts/truetype/msttcorefonts")
DEFAULT_CONFIG = PROJECT_DIR / "conf" / "config_dhbv_lstm.yaml"
DEFAULT_FLOW_DIR = PROJECT_DIR / "outputs" / "dhbv_lstm" / "camels_671" / "seed_111" / "test1995-2010_Ep100"
DEFAULT_GAGE_PATH = REPO_ROOT / "data" / "gage_id.npy"
DEFAULT_WINDOW_DAYS = 730
DEFAULT_RANDOM_SEED = 20260430


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one random CAMELS basin's 730-day forcings and hydrograph for inset use.",
    )
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--flow-dir", default=str(DEFAULT_FLOW_DIR))
    parser.add_argument("--gage-id-path", default=str(DEFAULT_GAGE_PATH))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--basin-id", type=int, default=None, help="Optional fixed basin id.")
    parser.add_argument("--start-index", type=int, default=None, help="Optional fixed start index in aligned test series.")
    return parser.parse_args()


def _set_plot_style() -> str:
    for path in sorted(MSTTCOREFONTS_DIR.glob("*.TTF")) + sorted(MSTTCOREFONTS_DIR.glob("*.ttf")):
        try:
            font_manager.fontManager.addfont(str(path))
        except Exception:
            pass
    font_manager._load_fontmanager(try_read_cache=False)

    preferred_fonts = [
        "Times New Roman",
        "TimesNewRomanPSMT",
        "Nimbus Roman No9 L",
        "Nimbus Roman",
        "STIXGeneral",
        "DejaVu Serif",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next((name for name in preferred_fonts if name in available), "serif")
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.serif": [font_family],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return font_family


def _load_eval_data(config_path: str) -> tuple[dict, dict[str, np.ndarray]]:
    config = load_config(config_path)
    loader_cfg = copy.deepcopy(config)
    loader_cfg["device"] = "cpu"
    loader_cls = import_data_loader(config["data_loader"])
    loader = loader_cls(loader_cfg, test_split=True, overwrite=False)
    loader.load_dataset()
    eval_data = {
        key: value.detach().cpu().numpy()
        for key, value in loader.eval_dataset.items()
    }
    return config, eval_data


def _align_for_hydrograph(
    forcing: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_2d = pred[..., 0] if pred.ndim == 3 else pred
    target_2d = target[..., 0] if target.ndim == 3 else target
    pred_len = pred_2d.shape[0]
    target_aligned = target_2d[-pred_len:, :]
    forcing_aligned = forcing[-pred_len:, :, :]
    return forcing_aligned, target_aligned, pred_2d


def _choose_window(
    n_basins: int,
    n_steps: int,
    basin_ids: np.ndarray,
    window_days: int,
    random_seed: int,
    basin_id: int | None,
    start_index: int | None,
) -> tuple[int, int]:
    if window_days > n_steps:
        raise ValueError(f"Requested window_days={window_days} exceeds available steps={n_steps}")

    rng = np.random.default_rng(random_seed)
    if basin_id is None:
        basin_idx = int(rng.integers(0, n_basins))
    else:
        matches = np.where(basin_ids.astype(int) == int(basin_id))[0]
        if matches.size == 0:
            raise ValueError(f"Basin id {basin_id} not found.")
        basin_idx = int(matches[0])

    if start_index is None:
        max_start = n_steps - window_days
        start = int(rng.integers(0, max_start + 1))
    else:
        if start_index < 0 or start_index + window_days > n_steps:
            raise ValueError("start_index is out of range for the aligned series length.")
        start = int(start_index)
    return basin_idx, start


def _make_dates(config: dict, n_steps: int) -> pd.DatetimeIndex:
    full_dates = pd.date_range(config["test"]["start_time"], config["test"]["end_time"], freq="D")
    return full_dates[-n_steps:]


def _plot_forcings(
    dates: pd.DatetimeIndex,
    forcing_window: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    colors = ["#4C78A8", "#E45756", "#54A24B"]

    for idx, ax in enumerate(axes):
        ax.plot(dates, forcing_window[:, idx], color=colors[idx], linewidth=2.1)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.8)
        ax.spines["bottom"].set_linewidth(1.8)
        ax.tick_params(axis="both", which="both", length=3, labelbottom=False, labelleft=False)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_hydrograph(
    dates: pd.DatetimeIndex,
    obs_window: np.ndarray,
    pred_window: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.plot(dates, obs_window, color="#222222", linewidth=1.4, label="Obs.")
    ax.plot(dates, pred_window, color="#D94F4FFF", linewidth=1.3, alpha=0.95, label="Pred.")

    ax.set_ylabel("Streamflow", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper right", fontsize=10, frameon=False, ncol=2, handlelength=2.2)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    font_family = _set_plot_style()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config, eval_data = _load_eval_data(args.config_path)
    pred = np.load(Path(args.flow_dir) / "streamflow.npy", allow_pickle=True)
    obs = np.load(Path(args.flow_dir) / "streamflow_obs.npy", allow_pickle=True)
    basin_ids = np.load(args.gage_id_path, allow_pickle=True)

    forcing_aligned, obs_aligned, pred_aligned = _align_for_hydrograph(
        eval_data["x_phy"], eval_data["target"], pred
    )
    aligned_dates = _make_dates(config, pred_aligned.shape[0])
    basin_idx, start = _choose_window(
        n_basins=forcing_aligned.shape[1],
        n_steps=forcing_aligned.shape[0],
        basin_ids=basin_ids,
        window_days=args.window_days,
        random_seed=args.random_seed,
        basin_id=args.basin_id,
        start_index=args.start_index,
    )
    end = start + args.window_days

    basin_id = int(basin_ids[basin_idx])
    date_window = aligned_dates[start:end]
    forcing_window = forcing_aligned[start:end, basin_idx, :]
    obs_window = obs_aligned[start:end, basin_idx]
    pred_window = pred_aligned[start:end, basin_idx]

    forcing_path = output_dir / f"random_basin_{basin_id}_forcings_{args.window_days}d.png"
    hydrograph_path = output_dir / f"random_basin_{basin_id}_hydrograph_{args.window_days}d.png"
    metadata_path = output_dir / f"random_basin_{basin_id}_window_metadata.json"

    _plot_forcings(date_window, forcing_window, forcing_path)
    _plot_hydrograph(date_window, obs_window, pred_window, hydrograph_path)

    metadata = {
        "font_used": font_family,
        "basin_id": basin_id,
        "basin_index": basin_idx,
        "start_index": start,
        "end_index": end,
        "window_days": args.window_days,
        "start_date": str(date_window[0].date()),
        "end_date": str(date_window[-1].date()),
        "flow_dir": str(Path(args.flow_dir).resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Font used: {font_family}")
    print(f"Selected basin: {basin_id} (index={basin_idx})")
    print(f"Window: {date_window[0].date()} to {date_window[-1].date()} ({args.window_days} days)")
    print(f"Forcing figure: {forcing_path}")
    print(f"Hydrograph figure: {hydrograph_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
