from __future__ import annotations
import csv, json
from pathlib import Path
from .model_registry import NPARAM_INFO_36


def write_blocked_manifest(root: Path, reason: str) -> None:
    results=root/"results"; benchmarks=root/"benchmarks"; results.mkdir(exist_ok=True); benchmarks.mkdir(exist_ok=True)
    rows=[{"model":name,"dimension":dim,"status":"blocked_before_pilot","reason":reason} for name,dim in NPARAM_INFO_36.items()]
    for path in (results/"model_convergence_summary.csv", results/"uncertified_units.csv"):
        with path.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    with (benchmarks/"component_profile.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["component","status","reason"]); writer.writeheader(); writer.writerow({"component":"hydrology_forward","status":"not_profiled_production","reason":reason}); writer.writerow({"component":"streaming_kge","status":"not_profiled_production","reason":reason}); writer.writerow({"component":"full_covariance_update","status":"not_profiled_production","reason":reason}); writer.writerow({"component":"data_transfer","status":"not_profiled_production","reason":reason})
    (results/"run_manifest.json").write_text(json.dumps({"status":"blocked_before_pilot","reason":reason,"models":len(rows),"test_metric_used_for_selection":False},indent=2))
