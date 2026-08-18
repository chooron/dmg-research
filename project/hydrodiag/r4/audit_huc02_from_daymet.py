#!/usr/bin/env python3
"""Audit HUC02 by joining CAMELS-US Daymet folder/file names to the 531 basins.

The CAMELS-US Daymet layout is expected to be::

    CAMELS_US/basin_mean_forcing/daymet/<huc02>/<basin_id>...

This uses the directory name as HUC02 and the file stem as basin ID. It never
infers HUC02 from a basin/gauge ID prefix.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import subprocess

import pandas as pd

DEFAULT_DAYMET = r"G:\\Dataset\\CAMELS_US\\basin_mean_forcing\\daymet"
DEFAULT_PAIRED = Path(
    "/home/jingxin/code/dmg-research/project/hydrodiag/results/"
    "r4_phase1_soil_official/paired_structural_effects.csv"
)


def basin_from_name(path: Path) -> str | None:
    """Extract an 8-digit basin ID from a Daymet file name."""
    stem = Path(path).stem
    matches = re.findall(r"(?<!\d)(\d{8})(?!\d)", stem)
    return matches[0] if matches else None


def is_windows_path(value: str | Path) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\\\/]", str(value)))


def windows_dir(path: str, dirs: bool) -> list[str]:
    """List a Windows directory through cmd.exe when the drive is not mounted."""
    attr = "/ad" if dirs else "/a-d"
    cmd = f"dir /b {attr} {path}"
    proc = subprocess.run(["cmd.exe", "/d", "/c", cmd], capture_output=True)
    if proc.returncode not in (0, 1):
        raise RuntimeError(proc.stderr.decode(errors="replace").strip())
    return [line.strip() for line in proc.stdout.decode("utf-8", errors="replace").splitlines() if line.strip()]


def child_dirs(root: str | Path) -> list[tuple[str, str]]:
    if is_windows_path(root):
        root_str = str(root)
        return [(name, root_str.rstrip("\\/") + "\\" + name) for name in windows_dir(root_str, True)]
    path = Path(root)
    return [(p.name, str(p)) for p in sorted(p for p in path.iterdir() if p.is_dir())]


def child_files(root: str) -> list[str]:
    if is_windows_path(root):
        return windows_dir(root, False)
    return [p.name for p in Path(root).iterdir() if p.is_file()]


def audit(daymet_root: str | Path, paired_csv: Path) -> tuple[dict, pd.DataFrame]:
    if is_windows_path(daymet_root):
        if not windows_dir(str(daymet_root), True):
            raise FileNotFoundError(f"Daymet root not found or empty: {daymet_root}")
    elif not Path(daymet_root).is_dir():
        raise FileNotFoundError(f"Daymet root not found: {daymet_root}")

    paired = pd.read_csv(paired_csv, usecols=["regime", "basin_id"])
    basins = (
        paired.loc[paired["regime"].eq("dPL_seed42"), "basin_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
        .drop_duplicates()
        .tolist()
    )

    basin_hucs: dict[str, set[str]] = defaultdict(set)
    files_seen = 0
    for huc02, huc_dir in child_dirs(daymet_root):
        for item in child_files(huc_dir):
            basin = basin_from_name(item)
            if basin is None:
                continue
            files_seen += 1
            basin_hucs[basin].add(huc02)

    rows = []
    for basin in basins:
        hucs = sorted(basin_hucs.get(basin, set()))
        rows.append(
            {
                "basin_id": basin,
                "huc02": hucs[0] if len(hucs) == 1 else None,
                "join_status": "matched" if len(hucs) == 1 else (
                    "missing" if not hucs else "conflict"
                ),
                "candidate_huc02": ";".join(hucs),
            }
        )
    joined = pd.DataFrame(rows)
    matched = joined[joined["join_status"].eq("matched")]
    summary = {
        "daymet_root": str(daymet_root),
        "paired_csv": str(paired_csv),
        "basin_count": len(basins),
        "daymet_basin_files_seen": files_seen,
        "matched_basin_count": int(len(matched)),
        "missing_basin_count": int((joined["join_status"] == "missing").sum()),
        "conflict_basin_count": int((joined["join_status"] == "conflict").sum()),
        "distinct_huc02_count": int(matched["huc02"].nunique()),
        "huc02_sample_counts": {
            str(k): int(v) for k, v in Counter(matched["huc02"]).items()
        },
    }
    return summary, joined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daymet-root", type=Path, default=DEFAULT_DAYMET)
    parser.add_argument("--paired-csv", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()

    summary, joined = audit(args.daymet_root, args.paired_csv)
    prefix = args.output_prefix or args.paired_csv.with_name("huc02_daymet_join")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    joined.to_csv(f"{prefix}.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
