from __future__ import annotations
from _common import EXPERIMENT
import csv, torch
from src.model_registry import NPARAM_INFO_36, audit_registry, build_model

rows = audit_registry()
for row in rows:
    model = build_model(row["model"], "cpu", warm_up=1, backend="eager")
    dim = row["dimension"]; x = torch.stack([torch.rand(6, 2) * 10, torch.randn(6, 2), torch.rand(6, 2)], -1)
    if row["model"] in {"mopex4", "mopex5"}: x = torch.cat((x, torch.arange(1, 7).view(6, 1, 1).expand(-1, 2, -1).float()), -1)
    y = model({"x_phy": x}, (None, torch.full((2, dim, 2), .5)))["streamflow"]
    row["output_finite"] = bool(torch.isfinite(y).all()); row["output_shape"] = str(tuple(y.shape))
with (EXPERIMENT / "results/model_registry_validation.csv").open("w", newline="") as h:
    out = csv.DictWriter(h, fieldnames=rows[0].keys()); out.writeheader(); out.writerows(rows)
assert len(rows) == len(NPARAM_INFO_36) == 36 and all(r["output_finite"] for r in rows)
print("validated 36 models")
