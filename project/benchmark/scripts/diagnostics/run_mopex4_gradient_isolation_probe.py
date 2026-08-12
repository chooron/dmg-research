#!/usr/bin/env python3
"""Representative-basin benchmark-only gradient-isolation probe.

Compares a small joint M4 head against an equivalent head where alpha/is_time
are computed from detached shared features. This is not production code and
is not a full-basin training experiment.
"""
from __future__ import annotations

import copy
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

BENCHMARK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BENCHMARK), str(BENCHMARK.parents[1]), str(BENCHMARK / "src"),
                str(BENCHMARK / "scripts" / "diagnostics")]
import audit_mopex34_root_cause as A  # noqa: E402
from project.benchmark.scripts.run_dpl_benchmark_dmg_native import compute_differentiable_kge  # noqa: E402

OUT = BENCHMARK / "results/mopex45_phase_fix/root_cause_audit"
OUT.mkdir(parents=True, exist_ok=True)


class SplitM4(nn.Module):
    def __init__(self, source: nn.Module, isolated: bool):
        super().__init__()
        self.trunk = copy.deepcopy(source.net[:-1])
        self.common = nn.Linear(256, 8)
        self.interception = nn.Linear(256, 2)
        self.isolated = isolated
        with torch.no_grad():
            self.common.weight.copy_(source.net[-1].weight[A.M4_COMMON])
            self.common.bias.copy_(source.net[-1].bias[A.M4_COMMON])
            self.interception.weight.copy_(source.net[-1].weight[A.M4_INTERCEPTION])
            self.interception.bias.copy_(source.net[-1].bias[A.M4_INTERCEPTION])

    def forward(self, attrs):
        z = self.trunk(attrs)
        common = torch.sigmoid(self.common(z))
        iz = z.detach() if self.isolated else z
        inter = torch.sigmoid(self.interception(iz))
        return torch.cat((common[:, :4], inter, common[:, 4:]), dim=1)


def train_variant(attrs, x, y, source, isolated: bool, lambda_i: float, steps: int = 20):
    torch.manual_seed(777)
    net = SplitM4(source, isolated=isolated)
    model = A.make_model("mopex4", lambda_i=lambda_i)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xb = x[:A.WARMUP + A.SCORED]; yb = y[A.WARMUP:A.WARMUP + A.SCORED]
    net.train()
    initial = None
    last = None
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        raw = net(attrs)
        q = model({"x_phy": xb}, (None, raw.unsqueeze(-1)))["streamflow"]
        q = q.squeeze(-1) if q.dim() == 3 else q
        loss, kge = compute_differentiable_kge(q, yb, warmup_days=0)
        if initial is None:
            initial = (float(loss.detach()), float(kge.median().detach()))
        last = (float(loss.detach()), float(kge.median().detach()))
        if step == steps:
            break
        loss.backward(); opt.step()
    return {"variant": "isolated_interception_gradient" if isolated else "joint_shared_gradient",
            "lambda_i": lambda_i, "steps": steps, "initial_loss": initial[0],
            "initial_median_kge": initial[1], "final_loss": last[0],
            "final_median_kge": last[1]}


def permanent_off_control(attrs, x, y, source, steps=20):
    # Use the same split-capable network but lambda_i=0; alpha/is_time are
    # structurally inactive, making this a plumbing/architecture control.
    return train_variant(attrs, x, y, source, isolated=False, lambda_i=0.0, steps=steps)


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.manual_seed(321)
    ids, x, y, b = A.load_context()
    attrs_all = __import__("dpl.attributes", fromlist=["CatchmentAttributeBuilder"]).CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cpu", method="zscore")
    attrs = attrs_all[b]
    source = A.mapped_m3_network()[1]
    rows = [permanent_off_control(attrs, x, y, source),
            train_variant(attrs, x, y, source, isolated=False, lambda_i=1.0),
            train_variant(attrs, x, y, source, isolated=True, lambda_i=1.0)]
    with (OUT / "mopex4_gradient_isolation_probe.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(*rows, sep="\n")


if __name__ == "__main__":
    main()
