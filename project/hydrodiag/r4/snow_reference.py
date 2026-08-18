"""CAMELS-US Snow-17 / SAC-SMA SWE reference reader (R4 external reference).

Interface design (target-basin only, never whole-671 processing):

- the reader resolves the CAMELS-US layout by probing candidate paths for
  ``usgs_<8-digit>_model_output.txt`` files and records which layout matched;
- ``load_basin(basin_id)`` parses only that basin's file and returns a
  :class:`BasinSnow` with the SWE series (10-member ensemble when present,
  otherwise a single realization broadcast to an ensemble axis), the
  ensemble median, and date alignment onto the project date axis;
- annual burden metrics (annual max SWE, SWE-positive duration, peak timing,
  depletion timing) are computed per water year on demand;
- CN G-vs-SWE consistency helpers (anomaly correlation, seasonal phase
  alignment) are provided for later analysis steps.

The mount ``G:\\Dataset\\CAMELS_US`` (WSL: ``/mnt/g/Dataset/CAMELS_US``) is a
data dependency; when it is unavailable the module still imports and the
reader reports ``available=False`` with the probed locations — the R4 pipeline
does not block on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

_BASIN_RE = re.compile(r"usgs_(\d{8})")

# Candidate layouts, probed in order.  Each entry is (label, candidate file
# pattern) where {id} is the 8-digit gauge id.
LAYOUT_CANDIDATES: tuple[tuple[str, str], ...] = (
    (
        "camels_us_v1p2_model_output",
        "basin_dataset_public_v1p2/model_output/usgs_{id}_model_output.txt",
    ),
    ("model_output_top", "model_output/usgs_{id}_model_output.txt"),
    ("per_basin_folder", "usgs_{id}/usgs_{id}_model_output.txt"),
    ("basins_folder", "basins/usgs_{id}/usgs_{id}_model_output.txt"),
    ("per_basin_flat", "usgs_{id}_model_output.txt"),
    (
        "basin_mean_forcing_sibling",
        "basin_dataset_public_v1p2/usgs_{id}/usgs_{id}_model_output.txt",
    ),
)

_SWE_COLUMN_PRIORITY = (
    "snow17_swe",
    "swe_snow17",
    "swe",
    "sac_sma_swe",
    "swe_sac_sma",
    "snow_water_equivalent",
    "snwe",
    "SWE",
)


class SnowReferenceUnavailable(RuntimeError):
    """Raised when the CAMELS-US snow reference cannot be resolved/read."""


@dataclass
class BasinSnow:
    basin_id: str
    dates: np.ndarray  # np.datetime64[D] on the file axis
    swe_ensemble: np.ndarray  # [n_days, n_members] float64, mm
    swe_median: np.ndarray  # [n_days] float64, mm
    swe_source_column: str
    n_members: int
    layout: str
    file_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def align_to(self, project_dates: np.ndarray) -> np.ndarray:
        """Align SWE onto the project date axis (NaN where missing)."""
        index = {np.datetime64(d, "D"): i for i, d in enumerate(self.dates)}
        out = np.full(project_dates.shape[0], np.nan, dtype=np.float64)
        for j, day in enumerate(np.asarray(project_dates, dtype="datetime64[D]")):
            i = index.get(day)
            if i is not None:
                out[j] = self.swe_median[i]
        return out

    def annual_metrics(
        self, water_year_start_month: int = 10
    ) -> dict[str, dict[str, float]]:
        """Per water year: annual max SWE, SWE-positive duration (days), peak
        and depletion timing (day of water year, fractional when ambiguous)."""
        from pandas import DataFrame

        df = DataFrame({"date": self.dates, "swe": self.swe_median})
        df = df.dropna()
        if df.empty:
            return {}
        year = df["date"].dt.year.values
        month = df["date"].dt.month.values
        wy = np.where(month >= water_year_start_month, year + 1, year).astype(int)
        df["wy"] = wy
        # water-year day-of-year (Oct 1 = 1), per-row start date
        starts = np.array(
            [
                np.datetime64(f"{int(w) - 1}-{water_year_start_month:02d}-01", "D")
                for w in wy
            ]
        )
        wy_doy = ((df["date"].values - starts) / np.timedelta64(1, "D")).astype(
            float
        ) + 1

        metrics: dict[str, dict[str, float]] = {}
        for w in np.unique(wy):
            mask = wy == w
            swe_w = df["swe"].values[mask]
            doy_w = wy_doy[mask]
            positive = swe_w > 0
            peak_i = int(np.nanargmax(swe_w))
            peak_swe = float(swe_w[peak_i])
            peak_doy = float(doy_w[peak_i])
            # depletion: last day where SWE >= 10% of annual peak after the peak
            tail = np.where(mask)[0]
            depletion_doy = np.nan
            if positive.any():
                thr = 0.1 * max(peak_swe, 1e-6)
                post_peak = np.where(swe_w >= thr)[0]
                if len(post_peak):
                    depletion_doy = float(doy_w[post_peak[-1]])
            metrics[str(w)] = {
                "annual_max_swe_mm": peak_swe,
                "swe_positive_duration_days": float(positive.sum()),
                "peak_timing_wy_doy": peak_doy,
                "depletion_timing_wy_doy": depletion_doy,
            }
        return metrics


class SnowReferenceReader:
    """Target-basin CAMELS-US Snow-17/SAC-SMA SWE reader."""

    def __init__(
        self, root: Path, *, members: int = 10, water_year_start_month: int = 10
    ):
        self.root = Path(root)
        self.members = members
        self.water_year_start_month = water_year_start_month
        self._layout: Optional[str] = None
        self._file_index: dict[str, Path] = {}
        self._probe()

    # -- resolution ---------------------------------------------------------
    def _probe(self) -> None:
        if not self.root.is_dir():
            raise SnowReferenceUnavailable(
                f"CAMELS-US root not found: {self.root} (expected mount "
                f"G:\\Dataset\\CAMELS_US; WSL path /mnt/g/Dataset/CAMELS_US)"
            )
        for label, pattern in LAYOUT_CANDIDATES:
            sample = self.root / pattern.format(id="01013500")
            if sample.is_file():
                self._layout = label
                break
        if self._layout is None:
            # scan the root for any usgs_<id>_model_output.txt file
            hits = sorted(self.root.rglob("usgs_*_model_output.txt"))[:5]
            if hits:
                self._layout = "discovered"
                self._file_index = self._index_basins(hits[0].parent)
            else:
                raise SnowReferenceUnavailable(
                    f"no usgs_<id>_model_output.txt found under {self.root} "
                    f"(probed layouts: {[l for l, _ in LAYOUT_CANDIDATES]})"
                )
        if not self._file_index:
            if self._layout == "discovered":
                layout_dir = hits[0].parent
            else:
                pattern = dict(LAYOUT_CANDIDATES)[self._layout]
                layout_dir = self.root / str(pattern).rsplit("/", 1)[0]
            self._file_index = self._index_basins(layout_dir)

    def _index_basins(self, directory: Path) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for path in sorted(directory.glob("usgs_*_model_output.txt")):
            match = _BASIN_RE.search(path.name)
            if match:
                index[match.group(1)] = path
        return index

    def available_basins(self) -> set[str]:
        return set(self._file_index)

    @property
    def layout(self) -> Optional[str]:
        return self._layout

    # -- loading ------------------------------------------------------------
    def load_basin(self, basin_id: Any) -> BasinSnow:
        import pandas as pd

        basin = str(basin_id).zfill(8)
        path = self._file_index.get(basin)
        if path is None:
            raise SnowReferenceUnavailable(
                f"basin {basin}: no usgs_<id>_model_output.txt in layout "
                f"{self._layout!r} under {self.root}"
            )
        frame = pd.read_csv(path, sep=r"\s+", comment="#", header=0)
        columns = [str(c).strip() for c in frame.columns]
        if "date" not in columns and "Date" not in columns:
            raise SnowReferenceUnavailable(
                f"basin {basin}: no date column in {path} (columns: {columns[:8]}...)"
            )
        date_col = "date" if "date" in columns else "Date"
        dates = pd.to_datetime(frame[date_col]).to_numpy(dtype="datetime64[D]")

        swe_cols = [c for c in columns if c.lower() in _SWE_COLUMN_PRIORITY]
        swe_cols = sorted(swe_cols, key=lambda c: _SWE_COLUMN_PRIORITY.index(c.lower()))
        ensemble_cols = [c for c in columns if re.fullmatch(r"swe_\d{1,2}", c.lower())]
        if ensemble_cols:
            source_col = ensemble_cols[0]
            swe = frame[ensemble_cols].to_numpy(dtype=np.float64)
        elif swe_cols:
            source_col = swe_cols[0]
            swe = frame[source_col].to_numpy(dtype=np.float64).reshape(-1, 1)
        else:
            raise SnowReferenceUnavailable(
                f"basin {basin}: no SWE column in {path} (columns: {columns[:12]})"
            )
        if swe.shape[1] == 1:
            swe = np.repeat(swe, self.members, axis=1)  # single realization
        median = np.nanmedian(swe, axis=1)
        return BasinSnow(
            basin_id=basin,
            dates=dates,
            swe_ensemble=swe,
            swe_median=median,
            swe_source_column=source_col,
            n_members=swe.shape[1],
            layout=self._layout or "unknown",
            file_path=path,
            metadata={"ensemble_reduced": "median", "units": "mm"},
        )

    # -- target-basin selection ---------------------------------------------
    def load_target_basins(self, basin_ids: Iterable[Any]) -> dict[str, BasinSnow]:
        """Load only the requested basins (canonical 531 list or a subset)."""
        return {str(b).zfill(8): self.load_basin(b) for b in basin_ids}


def cn_swe_consistency(
    cn_dates: np.ndarray,
    cn_snow_pack: np.ndarray,
    swe: np.ndarray,
    *,
    min_overlap_days: int = 60,
) -> dict[str, float]:
    """G-vs-SWE consistency statistics on a common date axis (same length).

    - ``anomaly_corr``: Pearson correlation of deseasonalised anomalies
      (monthly-mean removed), masking months without snow activity;
    - ``seasonal_phase_shift_days``: lag (days) maximising cross-correlation
      of the two series (positive = G lags SWE);
    - ``annual_peak_timing_corr``: correlation of water-year peak timings.
    """
    from scipy import stats

    common = np.isfinite(cn_snow_pack) & np.isfinite(swe)
    if int(common.sum()) < min_overlap_days:
        return {
            "n_valid_days": int(common.sum()),
            "anomaly_corr": np.nan,
            "seasonal_phase_shift_days": np.nan,
            "annual_peak_timing_corr": np.nan,
        }
    g = cn_snow_pack[common].astype(np.float64)
    s = swe[common].astype(np.float64)
    months = np.asarray(cn_dates, dtype="datetime64[M]").astype(int)[common]

    # monthly-anomaly correlation over snow-active days
    active = (g > 1e-6) | (s > 1e-6)
    if int(active.sum()) < min_overlap_days:
        anomaly_corr = np.nan
    else:
        g_a = g[active] - np.array(
            [g[months[active] == m].mean() for m in months[active]]
        )
        s_a = s[active] - np.array(
            [s[months[active] == m].mean() for m in months[active]]
        )
        if g_a.std() == 0 or s_a.std() == 0:
            anomaly_corr = np.nan
        else:
            anomaly_corr = float(stats.pearsonr(g_a, s_a)[0])

    # lagged cross-correlation over the full common axis
    lag = 0
    best = -1.0
    for k in range(-30, 31):
        if k >= 0:
            x, y = g[k:], s[: len(g) - k]
        else:
            x, y = g[: len(g) + k], s[-k:]
        if len(x) < min_overlap_days or x.std() == 0 or y.std() == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best:
            best, lag = r, k
    phase = float(lag)

    # water-year peak timing correlation
    try:
        from pandas import DataFrame

        df = DataFrame(
            {
                "g": g,
                "s": s,
                "date": np.asarray(cn_dates, dtype="datetime64[D]")[common],
            }
        )
        year = df["date"].dt.year.values
        month = df["date"].dt.month.values
        df["wy"] = np.where(month >= 10, year + 1, year).astype(int)
        peaks = df.groupby("wy").agg(g_peak=("g", "idxmax"), s_peak=("s", "idxmax"))
        peak_corr = float(stats.spearmanr(peaks["g_peak"], peaks["s_peak"]).statistic)
    except Exception:
        peak_corr = np.nan

    return {
        "n_valid_days": int(common.sum()),
        "anomaly_corr": anomaly_corr,
        "seasonal_phase_shift_days": phase,
        "annual_peak_timing_corr": peak_corr,
    }
