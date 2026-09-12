#!/usr/bin/env python3
"""Step 2: reconstruct the formal full-test IC/dPL paired table by ID join."""
from __future__ import annotations

import pandas as pd

from r1_config import DATA_ROOT, FORMAL_PAIRED_SOURCE, MODEL_REGISTRY, TABLES_DIR, TEST_END, TEST_START
from r1_utils import canonical_basin_id, read_canonical_ids, registry_sort


def main() -> None:
    raw = pd.read_csv(FORMAL_PAIRED_SOURCE, dtype={"model": str, "basin_id": str})
    raw["basin_id"] = raw["basin_id"].map(canonical_basin_id)
    required = {"model", "basin_id", "KGE_IC", "KGE_dPL", "test_period"}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"formal paired source missing columns: {sorted(missing)}")
    if set(raw["model"]) != set(MODEL_REGISTRY):
        raise RuntimeError("formal paired source model set does not match registry_36")
    if raw.duplicated(["model", "basin_id"]).any():
        raise RuntimeError("formal paired source contains duplicate model-basin keys")
    if raw["test_period"].astype(str).nunique() != 1 or raw["test_period"].iloc[0] != f"{TEST_START}..{TEST_END}":
        raise RuntimeError("formal paired source has an unexpected test period")
    canonical_ids = set(read_canonical_ids(DATA_ROOT / "531sub_id.txt"))
    for model_name, group in raw.groupby("model"):
        if set(group.basin_id) != canonical_ids:
            raise RuntimeError(f"{model_name}: formal basin IDs do not exactly match canonical 531 IDs")

    ic = raw[["model", "basin_id", "KGE_IC"]].copy()
    dpl = raw[["model", "basin_id", "KGE_dPL"]].copy()
    paired = ic.merge(dpl, on=["model", "basin_id"], how="outer", validate="one_to_one", indicator=True)
    if not (paired["_merge"] == "both").all():
        raise RuntimeError("IC/dPL basin pairing failed: outer merge is not complete")
    paired = paired.drop(columns="_merge")
    paired["KGE_IC"] = pd.to_numeric(paired["KGE_IC"], errors="coerce")
    paired["KGE_dPL"] = pd.to_numeric(paired["KGE_dPL"], errors="coerce")
    paired["delta_KGE"] = paired["KGE_dPL"] - paired["KGE_IC"]
    if not paired[["KGE_IC", "KGE_dPL", "delta_KGE"]].notna().all().all():
        raise RuntimeError("paired table contains non-finite KGE values")
    counts = paired.groupby("model")["basin_id"].nunique()
    if len(paired) != 36 * 531 or not (counts == 531).all():
        raise RuntimeError(f"expected 36x531 paired rows, got {len(paired)}; counts={counts.to_dict()}")

    paired = registry_sort(paired)
    paired.to_csv(TABLES_DIR / "R1_model_basin_delta_kge.csv", index=False, float_format="%.10f")
    print(f"PASS: outer ID join produced {len(paired)} rows")
    print(f"PASS: wrote {TABLES_DIR / 'R1_model_basin_delta_kge.csv'}")


if __name__ == "__main__":
    main()
