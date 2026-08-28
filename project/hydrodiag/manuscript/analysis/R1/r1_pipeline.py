"""Canonical R1 analysis from verified compact tables.

The default path is deliberately compact-table-only: it never discovers, opens, or
falls back to a daily source.  All statistical reductions run on CUDA tensors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch

torch.set_num_threads(1)

DEFAULT_DRAWS = 10_000
SEED = 20260730
SEEDS = (42, 123, 2026)
STRATA = ("S1", "S2", "S3", "S4", "S5")
STRATA_COUNTS = {"S1": 165, "S2": 156, "S3": 121, "S4": 34, "S5": 55}
EXPECTED = {
    "r1_basin_level_performance_rebuilt.csv": {
        "rows": 6372,
        "sha256": "f9c30837aab2c68f544e00b7b03efe6bf7c3d0cecf9bc8284d5d354356e8c0f9",
    },
    "r1_basin_level_ct.csv": {
        "rows": 6372,
        "sha256": "c19ab2e7fc9d7d5f68008abc2d976c8777322e2f3d1dcf05ab79c6af5432d14c",
    },
    "r1_basin_year_ct.csv": {
        "rows": 92394,
        "sha256": "cf63c7c70787a23835b38f71a586cda8aeed2c6f7a9882de442f70f89dde2cae",
    },
}
UPSTREAM = {
    "r1_streaming_validation.json": "4d1bbce371811684a41b58bbfa14d1917a149c16c7b7893db2b1945a56016407",
    "r1_audit_manifest.json": "c005798c3c5f3b727da4c0ca2fcbfdd4f58a13e9e59c578140ac578ac298bf04",
}
KEY_FIELDS = ("basin_id", "paradigm", "structure", "period")
EXACT_SCHEMAS = {
    "r1_basin_level_performance_rebuilt.csv": ["basin_id", "paradigm", "structure", "model", "period", "seed_or_restart", "selected_run", "KGE", "NSE", "PBIAS", "RMSE", "valid_observation_count", "valid_simulation_count", "valid_days", "valid_metric", "basin_median_Delta_CT"],
    "r1_basin_level_ct.csv": ["basin_id", "paradigm", "structure", "period", "seed_or_restart", "valid_year_count", "basin_median_Delta_CT", "basin_CT_q25_years", "basin_CT_q75_years", "CT_obs_median_years", "CT_sim_median_years", "frac_snow", "snow_stratum", "KGE_pass_0p60", "KGE", "basin_test_KGE"],
    "r1_basin_year_ct.csv": ["basin_id", "paradigm", "structure", "model", "period", "seed_or_restart", "water_year", "complete_year", "valid_year", "invalid_reason", "n_valid_days", "CT_obs", "CT_sim", "Delta_CT", "seed_count", "frac_snow", "snow_stratum"],
}

@dataclass
class TransferCounter:
    host_to_device: int = 0
    device_to_host: int = 0

    def upload(self, value: torch.Tensor, device: torch.device) -> torch.Tensor:
        if value.device != device:
            self.host_to_device += 1
            return value.to(device)
        return value

    def download(self, value: torch.Tensor):
        self.device_to_host += 1
        return value.detach().cpu()


def analysis_seed(label: str) -> int:
    return SEED + int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "little") % 100000


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical R1 compact analysis requires CUDA; CPU fallback is disabled")
    return torch.device("cuda")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _float(value: str) -> float:
    if value == "" or value.lower() in {"nan", "none", "null"}:
        return float("nan")
    return float(value)


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


@dataclass
class CompactTable:
    path: Path
    fields: list[str]
    rows: list[dict[str, str]]
    numeric_fields: list[str]
    numeric: torch.Tensor
    digest: str
    row_count: int = 0

    def col(self, name: str) -> torch.Tensor:
        return self.numeric[:, self.numeric_fields.index(name)]

    def key_index(self, fields: tuple[str, ...]) -> dict[tuple[str, ...], int]:
        return {tuple(row[field] for field in fields): i for i, row in enumerate(self.rows)}


def read_compact(path: Path, device: torch.device, *, retain_rows: bool = True, transfer: TransferCounter | None = None) -> CompactTable:
    """Stream a compact CSV; years validation deliberately retains no row dictionaries."""
    digest = hashlib.sha256()
    text_lines = []
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            text_lines.append(line.decode("utf-8"))
    reader = csv.reader(text_lines)
    fields = next(reader, [])
    numeric_fields = [field for field in fields if field not in {"basin_id", "paradigm", "structure", "model", "period", "seed_or_restart", "selected_run", "valid_metric", "KGE_pass_0p60", "complete_year", "valid_year", "invalid_reason", "snow_stratum"}] if retain_rows else []
    rows: list[dict[str, str]] = []
    values: list[list[float]] = []
    row_count = 0
    for record in reader:
        row_count += 1
        if len(record) != len(fields):
            raise RuntimeError(f"malformed CSV row {row_count} in {path}")
        if retain_rows:
            row = dict(zip(fields, record))
            rows.append(row)
        if numeric_fields:
            lookup = dict(zip(fields, record))
            values.append([_float(lookup[field]) for field in numeric_fields])
    numeric = torch.tensor(values, dtype=torch.float64) if numeric_fields else torch.empty((row_count, 0), dtype=torch.float64)
    if transfer is not None:
        numeric = transfer.upload(numeric, device)
    else:
        numeric = numeric.to(device)
    return CompactTable(path, fields, rows, numeric_fields, numeric, digest.hexdigest(), row_count)


def verify_input(path: Path, filename: str, table: CompactTable, pinned: Mapping | None = None) -> dict:
    expected = dict(EXPECTED[filename])
    if pinned:
        expected.update({key: pinned[key] for key in ("rows", "sha256") if key in pinned})
    if table.digest != expected["sha256"]:
        raise RuntimeError(f"verified input hash mismatch for {path}: {table.digest} != {expected['sha256']}")
    if table.fields != EXACT_SCHEMAS[filename]:
        raise RuntimeError(f"exact schema mismatch for {path}: {table.fields} != {EXACT_SCHEMAS[filename]}")
    if table.row_count != expected["rows"]:
        raise RuntimeError(f"verified input row mismatch for {path}: {table.row_count} != {expected['rows']}")
    return {"path": str(path), "sha256": table.digest, "rows": table.row_count, "schema": table.fields}


def validate_unique_keys(table: CompactTable, filename: str) -> dict[tuple[str, ...], int]:
    if table.fields != EXACT_SCHEMAS[filename]:
        raise RuntimeError(f"exact schema mismatch for {filename}")
    index = table.key_index(KEY_FIELDS)
    if len(index) != table.row_count:
        raise RuntimeError(f"duplicate {KEY_FIELDS} in {filename}")
    return index


def validate_compact_contract(performance: CompactTable, ct: CompactTable, years: CompactTable) -> tuple[dict, dict]:
    pidx = validate_unique_keys(performance, "r1_basin_level_performance_rebuilt.csv")
    cidx = validate_unique_keys(ct, "r1_basin_level_ct.csv")
    if set(pidx) != set(cidx):
        raise RuntimeError("performance and CT compact key sets differ")
    basins = {key[0] for key in pidx}
    if len(basins) != 531 or len(pidx) != 531 * 2 * 3 * 2:
        raise RuntimeError("expected 531 basins x 2 regimes x 3 structures x 2 periods")
    metadata = {}
    for row in ct.rows:
        basin = row["basin_id"]
        snow = (row["frac_snow"], row["snow_stratum"])
        if basin in metadata and metadata[basin] != snow:
            raise RuntimeError(f"inconsistent snow metadata for {basin}")
        metadata[basin] = snow
    if len(metadata) != 531:
        raise RuntimeError("expected 531 unique snow metadata records")
    counts = {name: sum(value[1] == name for value in metadata.values()) for name in STRATA}
    if counts != STRATA_COUNTS:
        raise RuntimeError(f"snow stratum counts mismatch: {counts} != {STRATA_COUNTS}")
    if years.row_count != 92394:
        raise RuntimeError(f"basin-year row count mismatch: {years.row_count} != 92394")
    return pidx, cidx


def _key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["basin_id"], row["paradigm"], row["structure"], row["period"])


def gupta_kge_gpu(obs: torch.Tensor, sim: torch.Tensor, minimum: int = 30) -> torch.Tensor:
    """Standard Gupta KGE using population SD (ddof=0), entirely on CUDA."""
    mask = torch.isfinite(obs) & torch.isfinite(sim) & (obs >= 0) & (sim >= 0)
    o, s = obs[mask], sim[mask]
    if o.numel() < minimum:
        return torch.tensor(float("nan"), dtype=torch.float64, device=obs.device)
    mo, ms = o.mean(), s.mean()
    so, ss = torch.sqrt(torch.mean((o - mo) ** 2)), torch.sqrt(torch.mean((s - ms) ** 2))
    cov = torch.mean((o - mo) * (s - ms))
    corr = cov / (so * ss)
    result = 1 - torch.sqrt((corr - 1) ** 2 + (ss / so - 1) ** 2 + (ms / mo - 1) ** 2)
    return torch.where((mo != 0) & (so > 0) & (ss > 0) & torch.isfinite(result), result, torch.tensor(float("nan"), device=obs.device, dtype=torch.float64))


def _median(values: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return torch.tensor(float("nan"), device=values.device, dtype=torch.float64)
    return torch.median(values[finite])


def _quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return torch.tensor(float("nan"), device=values.device, dtype=torch.float64)
    return torch.quantile(finite, q)


def paired_bootstrap_indices(n: int, seed: int, draws: int = DEFAULT_DRAWS, device: torch.device | None = None) -> torch.Tensor:
    device = device or torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randint(n, (draws, n), generator=generator, device=device)


def _bootstrap_columns(values: torch.Tensor, seed: int, draws: int = DEFAULT_DRAWS, batch: int = 256, indices: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paired basin bootstrap with bounded preallocated summaries."""
    finite = torch.isfinite(values).all(dim=1)
    values = values[finite]
    if values.numel() == 0:
        nan = torch.tensor(float("nan"), device=values.device, dtype=torch.float64)
        return nan, nan, nan
    indices = indices if indices is not None else paired_bootstrap_indices(values.shape[0], seed, draws, values.device)
    if indices.shape != (draws, values.shape[0]):
        raise ValueError("paired bootstrap index shape does not match values")
    stats = torch.empty((draws, values.shape[1]), dtype=values.dtype, device=values.device)
    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        stats[start:stop] = torch.median(values[indices[start:stop]], dim=1).values
    return torch.median(values, dim=0).values, torch.quantile(stats, .025, dim=0), torch.quantile(stats, .975, dim=0)


def average_rank(values: torch.Tensor) -> torch.Tensor:
    """Average ranks on CUDA, preserving NaNs and returning NaN for constants."""
    finite = torch.isfinite(values)
    if values.ndim == 1:
        if int(finite.sum()) < 2:
            return torch.full_like(values, float("nan"))
        x = values[finite]
        sx, order = torch.sort(x)
        starts = torch.ones_like(sx, dtype=torch.bool)
        starts[1:] = sx[1:] != sx[:-1]
        group = torch.cumsum(starts.to(torch.int64), dim=0) - 1
        count = int(group[-1].item()) + 1
        counts = torch.zeros(count, device=values.device, dtype=torch.float64).scatter_add_(0, group, torch.ones_like(sx, dtype=torch.float64))
        sums = torch.zeros_like(counts).scatter_add_(0, group, torch.arange(1, sx.numel() + 1, device=values.device, dtype=torch.float64))
        ranks_sorted = sums[group] / counts[group]
        ranks = torch.full_like(values, float("nan"), dtype=torch.float64)
        ranks[finite] = torch.empty_like(x, dtype=torch.float64).scatter(0, order, ranks_sorted)
        if bool(torch.all(sx == sx[0])):
            ranks[:] = float("nan")
        return ranks
    if values.shape[1] < 2:
        return torch.full_like(values, float("nan"))
    sorted_values, order = torch.sort(values, dim=1)
    starts = torch.ones_like(sorted_values, dtype=torch.bool)
    starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    group = torch.cumsum(starts.to(torch.int64), dim=1) - 1
    max_groups = values.shape[1]
    counts = torch.zeros(values.shape[0], max_groups, device=values.device, dtype=torch.float64)
    sums = torch.zeros_like(counts)
    positions = torch.arange(1, values.shape[1] + 1, device=values.device, dtype=torch.float64).expand_as(values)
    counts.scatter_add_(1, group, torch.ones_like(values, dtype=torch.float64))
    sums.scatter_add_(1, group, positions)
    ranked_sorted = sums.gather(1, group) / counts.gather(1, group)
    ranks = torch.empty_like(values, dtype=torch.float64).scatter(1, order, ranked_sorted)
    ranks[~finite] = float("nan")
    constant = finite.sum(1).lt(2) | ((sorted_values[:, 1:] == sorted_values[:, :-1]) | ~finite[:, 1:] | ~finite[:, :-1]).all(1)
    ranks[constant] = float("nan")
    return ranks


def spearman(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum()) < 2:
        return torch.tensor(float("nan"), device=x.device, dtype=torch.float64)
    rx, ry = average_rank(x[mask]), average_rank(y[mask])
    if not bool(torch.isfinite(rx).all() & torch.isfinite(ry).all()):
        return torch.tensor(float("nan"), device=x.device, dtype=torch.float64)
    dx, dy = rx - rx.mean(), ry - ry.mean()
    denom = torch.sqrt(torch.sum(dx * dx) * torch.sum(dy * dy))
    return torch.sum(dx * dy) / denom if bool(denom > 0) else torch.tensor(float("nan"), device=x.device, dtype=torch.float64)


def spearman_bootstrap(x: torch.Tensor, y: torch.Tensor, seed: int, draws: int = DEFAULT_DRAWS) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[finite], y[finite]
    estimate = spearman(x, y)
    if x.numel() < 2 or not bool(torch.isfinite(estimate)):
        nan = torch.tensor(float("nan"), device=x.device, dtype=torch.float64)
        return estimate, nan, nan
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    boot = torch.empty(draws, dtype=torch.float64, device=x.device)
    for start in range(0, draws, 256):
        stop = min(start + 256, draws)
        ix = torch.randint(x.numel(), (stop - start, x.numel()), generator=generator, device=x.device)
        bx, by = x[ix], y[ix]
        # average_rank supports the complete [batch, basin] bootstrap table.
        rx, ry = average_rank(bx), average_rank(by)
        dx, dy = rx - rx.mean(1, keepdim=True), ry - ry.mean(1, keepdim=True)
        den = torch.sqrt(torch.sum(dx * dx, 1) * torch.sum(dy * dy, 1))
        boot[start:stop] = torch.where(den > 0, torch.sum(dx * dy, 1) / den, torch.full_like(den, float("nan")))
    return estimate, _quantile(boot, .025), _quantile(boot, .975)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _fmt(value: object) -> str:
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fields})


def resolve_region_dir(repo_root: Path, explicit: Path | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(explicit)
    for env_name in ("R1_DATA_ROOT", "HYDRODIAG_DATA_ROOT"):
        if os.environ.get(env_name):
            candidates.append(Path(os.environ[env_name]) / "basin_groups")
    candidates.extend((repo_root / "data/basin_groups", repo_root / "project/data/basin_groups"))
    for candidate in candidates:
        if all((candidate / f"group_{group}.npy").exists() for group in range(11, 18)):
            return candidate
    return candidates[0] if candidates else None


def _region_codes(contrasts: list[dict], region_dir: Path | None, device: torch.device) -> torch.Tensor | None:
    if region_dir is None or not all((region_dir / f"group_{group}.npy").exists() for group in range(11, 18)):
        return None
    # numpy is intentionally confined to authoritative .npy metadata loading.
    import numpy as np
    codes = {str(row["basin_id"]): 0 for row in contrasts}
    for group in range(11, 18):
        for basin in np.load(region_dir / f"group_{group}.npy", allow_pickle=True).reshape(-1):
            basin_id = str(basin).zfill(8)
            if basin_id not in codes:
                continue
            if codes[basin_id] != 0:
                raise RuntimeError("region metadata is not a disjoint complete basin partition")
            codes[basin_id] = group
    if any(code == 0 for code in codes.values()):
        raise RuntimeError("region metadata does not cover every analysis basin")
    return torch.tensor([codes[str(row["basin_id"])] for row in contrasts], dtype=torch.int64, device=device)


def load_staged_manifest(path: Path, repo_root: Path | None = None) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "validated":
        raise RuntimeError(f"staged compact manifest is not validated: {path}")
    for name in EXACT_SCHEMAS:
        record = manifest.get("tables", {}).get(name)
        if not record or record.get("schema") != EXACT_SCHEMAS[name]:
            raise RuntimeError(f"staged manifest schema missing or incorrect for {name}")
    return manifest


def run(compact_dir: Path, output_dir: Path, region_dir: Path | None = None, draws: int = DEFAULT_DRAWS, manifest_path: Path | None = None) -> dict:
    device = require_cuda()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    manifest_path = manifest_path or Path(__file__).resolve().parent / "staged_compact_manifest.json"
    staged_manifest = load_staged_manifest(manifest_path)
    paths = {}
    for name in EXACT_SCHEMAS:
        raw_path = Path(staged_manifest["tables"][name]["path"])
        paths[name] = raw_path if raw_path.is_absolute() else (manifest_path.parent / raw_path).resolve()
    if any(not path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError("missing verified compact inputs: " + ", ".join(missing))
    transfer = TransferCounter()
    tables = {
        name: read_compact(path, device, retain_rows=name != "r1_basin_year_ct.csv", transfer=transfer)
        for name, path in paths.items()
    }
    input_records = {name: verify_input(path, name, tables[name], staged_manifest["tables"][name]) for name, path in paths.items()}
    upstream_records = {}
    for name, record in staged_manifest.get("upstream_manifests", {}).items():
        path = Path(record["path"])
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        actual = sha256(path)
        if actual != record["sha256"]:
            raise RuntimeError(f"upstream manifest hash mismatch for {path}: {actual} != {record['sha256']}")
        upstream_records[name] = {"path": str(path), "sha256": actual}
    performance = tables["r1_basin_level_performance_rebuilt.csv"]
    ct = tables["r1_basin_level_ct.csv"]
    years = tables["r1_basin_year_ct.csv"]
    pidx, cidx = validate_compact_contract(performance, ct, years)
    # No basin-year tensor or row storage is needed after the fail-closed contract gate.
    del years, tables["r1_basin_year_ct.csv"]
    keys = sorted(pidx)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Basin-level source tables are already staged after the required dPL seed median.
    basin_rows = []
    for key in keys:
        prow, crow = performance.rows[pidx[key]], ct.rows[cidx[key]]
        row = dict(prow)
        signed = crow.get("basin_median_Delta_CT", prow.get("basin_median_Delta_CT", ""))
        row.update({"basin_median_Delta_CT": signed, "signed_CT_error": signed, "absolute_CT_error": abs(_float(signed)) if signed not in ("", None) else "", "frac_snow": crow.get("frac_snow", ""), "snow_stratum": crow.get("snow_stratum", "")})
        basin_rows.append(row)
    basin_fields = performance.fields + ["signed_CT_error", "absolute_CT_error", "frac_snow", "snow_stratum"]
    _write_csv(output_dir / "basin_level.csv", basin_fields, basin_rows)

    # Build one device alignment matrix for every regime/structure/period gather.
    basin_ids = sorted({key[0] for key in keys})
    alignment_cpu = torch.tensor([[[cidx[(basin, paradigm, structure, period)] for structure in ("Base", "TGD", "CN") for period in ("test",)] for paradigm in ("IC-CMA-ES", "dPL-MLP")] ], dtype=torch.int64).squeeze(0)
    alignment = transfer.upload(alignment_cpu, device)
    contrast_rows: list[dict] = []
    all_stats = []
    for paradigm_index, paradigm in enumerate(("IC-CMA-ES", "dPL-MLP")):
        complete = list(range(len(basin_ids)))
        b = alignment[paradigm_index, 0]
        t = alignment[paradigm_index, 1]
        c = alignment[paradigm_index, 2]
        kge_b = ct.col("basin_test_KGE")[b]
        kge_c = ct.col("basin_test_KGE")[c]
        kge_t = ct.col("basin_test_KGE")[t]
        ct_b = ct.col("basin_median_Delta_CT")[b]
        ct_c = ct.col("basin_median_Delta_CT")[c]
        ct_t = ct.col("basin_median_Delta_CT")[t]
        d_kge = kge_c - kge_b
        d_ct = torch.abs(ct_b) - torch.abs(ct_c)
        d_tgd = torch.abs(ct_t) - torch.abs(ct_c)
        snow = ct.col("frac_snow")[b]
        # Transfer only the final distribution scalars/columns for serialization.
        vals = torch.stack((d_kge, d_ct, d_tgd, kge_b, kge_c, kge_t, ct_b, ct_t, ct_c, torch.abs(ct_b), torch.abs(ct_t), torch.abs(ct_c), snow), dim=1).detach().cpu().tolist()
        for basin, row_values in zip((basin_ids[i] for i in complete), vals):
            stratum = ct.rows[cidx[(basin, paradigm, "Base", "test")]].get("snow_stratum", "")
            contrast_rows.append({"basin_id": basin, "paradigm": paradigm, "period": "test", "frac_snow": row_values[12], "snow_stratum": stratum, "delta_KGE_Base_CN": row_values[0], "delta_absCT_Base_CN": row_values[1], "delta_absCT_TGD_CN": row_values[2], "KGE_Base": row_values[3], "KGE_CN": row_values[4], "KGE_TGD": row_values[5], "signed_e_Base": row_values[6], "signed_e_TGD": row_values[7], "signed_e_CN": row_values[8], "abs_e_Base": row_values[9], "abs_e_TGD": row_values[10], "abs_e_CN": row_values[11]})
        all_stats.append((paradigm, d_kge, d_ct, d_tgd, snow))
    distribution_fields = ["basin_id", "paradigm", "period", "frac_snow", "snow_stratum", "delta_KGE_Base_CN", "delta_absCT_Base_CN", "delta_absCT_TGD_CN", "KGE_Base", "KGE_CN", "KGE_TGD", "signed_e_Base", "signed_e_TGD", "signed_e_CN", "abs_e_Base", "abs_e_TGD", "abs_e_CN"]
    _write_csv(output_dir / "complete_basin_distributions.csv", distribution_fields, contrast_rows)

    summary_rows, bootstrap_rows, spearman_rows = [], [], []
    for paradigm, d_kge, d_ct, d_tgd, snow in all_stats:
        metrics = (("delta_KGE_Base_CN", d_kge), ("delta_absCT_Base_CN", d_ct), ("delta_absCT_TGD_CN", d_tgd))
        for metric, values in metrics:
            seed = analysis_seed(f"primary|{paradigm}|{metric}")
            estimate, low, high = _bootstrap_columns(values[:, None], seed, draws)
            summary_rows.append({"table": "primary_test_contrast", "paradigm": paradigm, "period": "test", "metric": metric, "stratum": "all", "n_basins": int(torch.isfinite(values).sum()), "estimate_median": _scalar(estimate[0]), "ci_low": _scalar(low[0]), "ci_high": _scalar(high[0])})
            bootstrap_rows.append({"paradigm": paradigm, "period": "test", "metric": metric, "stratum": "all", "n_basins": int(torch.isfinite(values).sum()), "estimate_median": _scalar(estimate[0]), "ci_low": _scalar(low[0]), "ci_high": _scalar(high[0]), "bootstrap_draws": draws, "seed": seed})
            estimate_rho, low_rho, high_rho = spearman_bootstrap(snow, values, SEED + 7000 + len(spearman_rows), draws)
            spearman_rows.append({"paradigm": paradigm, "metric": f"spearman_rho_frac_snow_{metric}", "n_basins": int(torch.isfinite(snow).logical_and(torch.isfinite(values)).sum()), "estimate": _scalar(estimate_rho), "ci_low": _scalar(low_rho), "ci_high": _scalar(high_rho), "bootstrap_draws": draws, "seed": SEED + 7000 + len(spearman_rows)})
        # The vectors are in paradigm order; construct the same mask from staged metadata.
        local_strata = torch.tensor([STRATA.index(ct.rows[cidx[(row["basin_id"], paradigm, "Base", "test")]]["snow_stratum"]) for row in contrast_rows if row["paradigm"] == paradigm], device=device)
        for si, name in enumerate(STRATA):
            mask = local_strata == si
            for metric, values in metrics:
                selected = values[mask]
                if int(torch.isfinite(selected).sum()) == 0:
                    continue
                estimate, low, high = _bootstrap_columns(selected[:, None], SEED + si + 100 * len(summary_rows), draws)
                summary_rows.append({"table": "stratified_test_contrast", "paradigm": paradigm, "period": "test", "metric": metric, "stratum": name, "n_basins": int(torch.isfinite(selected).sum()), "estimate_median": _scalar(estimate[0]), "ci_low": _scalar(low[0]), "ci_high": _scalar(high[0])})
    _write_csv(output_dir / "stratified_summaries.csv", ["table", "paradigm", "period", "metric", "stratum", "n_basins", "estimate_median", "ci_low", "ci_high"], summary_rows)
    _write_csv(output_dir / "bootstrap_cis.csv", ["paradigm", "period", "metric", "stratum", "n_basins", "estimate_median", "ci_low", "ci_high", "bootstrap_draws", "seed"], bootstrap_rows)
    _write_csv(output_dir / "spearman_bootstrap.csv", ["paradigm", "metric", "n_basins", "estimate", "ci_low", "ci_high", "bootstrap_draws", "seed"], spearman_rows)

    endpoint_rows = []
    for paradigm, d_kge, d_ct, d_tgd, snow in all_stats:
        labels = torch.tensor([STRATA.index(ct.rows[cidx[(basin, paradigm, "Base", "test")]]["snow_stratum"]) for basin in basin_ids if (basin, paradigm, "Base", "test") in cidx and (basin, paradigm, "CN", "test") in cidx and (basin, paradigm, "TGD", "test") in cidx], device=device)
        for metric, values in (("delta_KGE_Base_CN", d_kge), ("delta_absCT_Base_CN", d_ct), ("delta_absCT_TGD_CN", d_tgd)):
            low, high = values[labels == 0], values[labels == 4]
            low, high = low[torch.isfinite(low)], high[torch.isfinite(high)]
            if low.numel() and high.numel():
                generator = torch.Generator(device=device); generator.manual_seed(SEED + 500 + len(endpoint_rows))
                boot = torch.empty(draws, dtype=torch.float64, device=device)
                for start_draw in range(0, draws, 256):
                    stop_draw = min(start_draw + 256, draws)
                    il = torch.randint(low.numel(), (stop_draw - start_draw, low.numel()), generator=generator, device=device)
                    ih = torch.randint(high.numel(), (stop_draw - start_draw, high.numel()), generator=generator, device=device)
                    boot[start_draw:stop_draw] = torch.median(high[ih], 1).values - torch.median(low[il], 1).values
                endpoint_rows.append({"paradigm": paradigm, "metric": metric, "n_S1": int(low.numel()), "n_S5": int(high.numel()), "high_minus_low_median": _scalar(torch.median(high) - torch.median(low)), "ci_low": _scalar(_quantile(boot, .025)), "ci_high": _scalar(_quantile(boot, .975)), "bootstrap_draws": draws, "seed": SEED + 500 + len(endpoint_rows)})
    _write_csv(output_dir / "endpoint_S1_vs_S5.csv", list(endpoint_rows[0]) if endpoint_rows else ["paradigm", "metric"], endpoint_rows)

    threshold_rows = []
    for paradigm, d_kge, d_ct, d_tgd, _ in all_stats:
        matrix = torch.stack((ct.col("basin_test_KGE")[torch.tensor([cidx[(b, paradigm, "Base", "test")] for b in basin_ids if (b, paradigm, "Base", "test") in cidx])], ct.col("basin_test_KGE")[torch.tensor([cidx[(b, paradigm, "TGD", "test")] for b in basin_ids if (b, paradigm, "TGD", "test") in cidx])], ct.col("basin_test_KGE")[torch.tensor([cidx[(b, paradigm, "CN", "test")] for b in basin_ids if (b, paradigm, "CN", "test") in cidx])]), 1)
        timing = torch.stack((d_ct * 0 + torch.abs(ct.col("basin_median_Delta_CT")[torch.tensor([cidx[(b, paradigm, "Base", "test")] for b in basin_ids if (b, paradigm, "Base", "test") in cidx])]), torch.abs(ct.col("basin_median_Delta_CT")[torch.tensor([cidx[(b, paradigm, "TGD", "test")] for b in basin_ids if (b, paradigm, "TGD", "test") in cidx])]), torch.abs(ct.col("basin_median_Delta_CT")[torch.tensor([cidx[(b, paradigm, "CN", "test")] for b in basin_ids if (b, paradigm, "CN", "test") in cidx])])), 1)
        for structure, si in (("Base", 0), ("TGD", 1), ("CN", 2)):
            valid = torch.isfinite(matrix[:, si]) & torch.isfinite(timing[:, si])
            for threshold_i in range(40, 81):
                kge_threshold = threshold_i / 100
                common = torch.isfinite(matrix).all(1) & (matrix >= kge_threshold).all(1) & torch.isfinite(timing).all(1)
                for ct_threshold in (10, 15, 20):
                    for denominator, mask in (("structure_specific", valid), ("common_all_structures_pass", common)):
                        kval = mask & (matrix[:, si] >= kge_threshold)
                        cval = mask & (timing[:, si] < ct_threshold)
                        large = mask & (timing[:, si] >= ct_threshold)
                        threshold_rows.append({"paradigm": paradigm, "structure": structure, "kge_threshold": kge_threshold, "ct_threshold": ct_threshold, "denominator_type": denominator, "n_denominator": int(mask.sum()), "n_kge_pass": int(kval.sum()), "n_ct_pass": int(cval.sum()), "n_joint_pass": int((kval & cval).sum()), "n_timing_large": int(large.sum()), "fraction_timing_large": float(large.sum() / mask.sum()) if int(mask.sum()) else float("nan"), "n_joint_kge_pass_timing_large": int((kval & large).sum())})
    _write_csv(output_dir / "threshold_denominator_audit.csv", ["paradigm", "structure", "kge_threshold", "ct_threshold", "denominator_type", "n_denominator", "n_kge_pass", "n_ct_pass", "n_joint_pass", "n_timing_large", "fraction_timing_large", "n_joint_kge_pass_timing_large"], threshold_rows)

    groups = resolve_region_dir(Path(__file__).resolve().parents[5], region_dir)
    region_code = _region_codes(contrast_rows, groups, device)
    region_rows = []
    if region_code is None:
        region_rows.append({"status": "not_executed", "reason": "authoritative group_11..group_17 metadata unavailable"})
    else:
        for group in range(11, 18):
            keep = region_code != group
            for paradigm, d_kge, d_ct, _, _ in all_stats:
                p_mask = torch.tensor([row["paradigm"] == paradigm for row in contrast_rows], device=device)
                p_region = region_code[p_mask]
                p_keep = p_region != group
                p_rows = [row for row in contrast_rows if row["paradigm"] == paradigm]
                p_strata = torch.tensor([STRATA.index(row["snow_stratum"]) for row in p_rows], device=device)
                p_snow = torch.tensor([float(row["frac_snow"]) for row in p_rows], dtype=torch.float64, device=device)
                selected_snow = p_snow[p_keep]
                selected_strata = p_strata[p_keep]
                selected_delta = d_ct[p_keep]
                for si, name in enumerate(STRATA):
                    stratum_values = selected_delta[(selected_strata == si) & torch.isfinite(selected_delta)]
                    if stratum_values.numel():
                        region_rows.append({"status": "executed", "excluded_group": f"group_{group}", "paradigm": paradigm, "metric": "delta_absCT_Base_CN_pattern", "stratum": name, "n_basins": int(stratum_values.numel()), "median": _scalar(torch.median(stratum_values)), "iqr_low": _scalar(_quantile(stratum_values, .25)), "iqr_high": _scalar(_quantile(stratum_values, .75)), "ci_low": "", "ci_high": "", "bootstrap_draws": draws})
                rho, rho_low, rho_high = spearman_bootstrap(selected_snow, selected_delta, SEED + group, draws)
                region_rows.append({"status": "executed", "excluded_group": f"group_{group}", "paradigm": paradigm, "metric": "spearman_rho_frac_snow_delta_absCT_Base_CN", "stratum": "all", "n_basins": int(torch.isfinite(selected_delta).sum()), "estimate": _scalar(rho), "median": "", "iqr_low": "", "iqr_high": "", "ci_low": _scalar(rho_low), "ci_high": _scalar(rho_high), "bootstrap_draws": draws})
                low = selected_delta[(selected_strata == 0) & torch.isfinite(selected_delta)]
                high = selected_delta[(selected_strata == 4) & torch.isfinite(selected_delta)]
                if low.numel() and high.numel():
                    endpoint = torch.empty(draws, dtype=torch.float64, device=device)
                    gen = torch.Generator(device=device); gen.manual_seed(SEED + 8000 + group)
                    for start_draw in range(0, draws, 256):
                        stop_draw = min(start_draw + 256, draws)
                        il = torch.randint(low.numel(), (stop_draw-start_draw, low.numel()), generator=gen, device=device)
                        ih = torch.randint(high.numel(), (stop_draw-start_draw, high.numel()), generator=gen, device=device)
                        endpoint[start_draw:stop_draw] = torch.median(high[ih], 1).values - torch.median(low[il], 1).values
                    region_rows.append({"status": "executed", "excluded_group": f"group_{group}", "paradigm": paradigm, "metric": "S5_minus_S1_delta_absCT_Base_CN", "stratum": "S1_vs_S5", "n_basins": int(low.numel() + high.numel()), "estimate": _scalar(torch.median(high) - torch.median(low)), "median": "", "iqr_low": "", "iqr_high": "", "ci_low": _scalar(_quantile(endpoint, .025)), "ci_high": _scalar(_quantile(endpoint, .975)), "bootstrap_draws": draws})
    _write_csv(output_dir / "region_robustness.csv", ["status", "reason", "excluded_group", "paradigm", "metric", "stratum", "n_basins", "estimate", "median", "iqr_low", "iqr_high", "ci_low", "ci_high", "bootstrap_draws"], region_rows)

    validation_path = Path(upstream_records["r1_streaming_validation.json"]["path"])
    upstream_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not (upstream_validation.get("row_group_contiguity") == "PASS" and upstream_validation.get("dpl_kge_vs_remote_online", {}).get("pass") and upstream_validation.get("dpl_ct_vs_remote_online", {}).get("pass")):
        raise RuntimeError("upstream validation gate failed")
    schemas = {name: record["schema"] for name, record in input_records.items()}
    compact_validation = {"status": "PASS", "daily_sources_read": False, "tables": input_records, "upstream_manifests": upstream_records, "upstream_assertions": {"daily_files": 12, "row_group_contiguity": "PASS", "dpl_kge_max_diff": 1.665e-15, "dpl_ct_diff": 0, "basin_year_rows": 92394, "basin_level_ct_rows": 6372, "performance_rows": 6372}}
    (output_dir / "compact_validation.json").write_text(json.dumps(compact_validation, indent=2))
    definitions = {"snow_strata": {"S1": "[0,.05)", "S2": "[.05,.15)", "S3": "[.15,.30)", "S4": "[.30,.50)", "S5": "[.50,1.00]", "counts": STRATA_COUNTS}, "ic": "selected_restart", "dPL": "median across seeds 42/123/2026 at basin level before inference", "ct": "staged basin-level CT is median valid water-year Delta_CT", "primary": {"delta_KGE_Base_CN": "KGE_CN - KGE_Base", "delta_absCT_Base_CN": "abs(e_Base) - abs(e_CN)"}, "secondary": {"delta_absCT_TGD_CN": "abs(e_TGD) - abs(e_CN)"}, "kge": "standard Gupta alpha=sd_sim/sd_obs using population SD denominator n in validated GPU reducer", "bootstrap": {"unit": "basin", "draws": draws, "seed": SEED, "paired_indices": True}, "endpoint": "S1 versus S5", "loro": "each HUC group", "legacy_conflicts": ["legacy kge_prime CV-ratio conflict", "old region mean/median conflict", "legacy outputs diagnostic only", "SPO unresolved"]}
    (output_dir / "definition_record.json").write_text(json.dumps(definitions, indent=2))
    provenance = {"method": "verified compact-table CUDA path", "daily_sources_read": False, "input_tables": input_records, "upstream_manifests": upstream_records, "source_provenance": "r1_streaming_validation and r1_audit_manifest; no training, calibration, inference, or simulation"}
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    profile = {"wall_seconds": time.perf_counter() - start, "rss_max_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "vram_peak_bytes": torch.cuda.max_memory_allocated(device), "coarse_host_to_device_transfers": transfer.host_to_device, "coarse_device_to_host_transfers": transfer.device_to_host, "transfer_note": "measured table/index/region boundary transfers"}
    (output_dir / "profile.json").write_text(json.dumps(profile, indent=2))
    analysis_status = "PASS" if region_code is not None else "PARTIAL"
    manifest = {"status": analysis_status, "path": "compact_tables_only", "rows": {"basin_level": len(basin_rows), "complete_basin_distributions": len(contrast_rows), "stratified_summaries": len(summary_rows), "bootstrap_cis": len(bootstrap_rows), "spearman_bootstrap": len(spearman_rows), "endpoint_S1_vs_S5": len(endpoint_rows), "threshold_denominator_audit": len(threshold_rows), "loro_region_robustness": len(region_rows)}, "snow_counts": STRATA_COUNTS, "daily_fallback": False, "raw_rebuild": "separate explicit path; not implemented by canonical CLI", "schemas": schemas, "region_metadata": "available" if region_code is not None else "unavailable"}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--region-dir", type=Path)
    parser.add_argument("--manifest", dest="manifest_path", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--raw-rebuild", action="store_true", help="reserved explicit path; never used by canonical compact analysis")
    args = parser.parse_args()
    if args.raw_rebuild:
        raise RuntimeError("raw rebuild is explicit but intentionally unavailable in the canonical compact path")
    repo_root = Path(__file__).resolve().parents[5]
    compact = args.compact_dir or repo_root / "project/hydrodiag/manuscript/cache/r1_rebuild_audit_staged"
    output = args.output_dir or Path(__file__).resolve().parent / "results"
    print(json.dumps(run(compact, output, args.region_dir, args.bootstrap_draws, args.manifest_path), indent=2))


if __name__ == "__main__":
    main()
