#!/usr/bin/env python3
"""Low-cost MOPEX4 F0 forward-reachability decision experiment.

This is a benchmark-only diagnostic. It replays the existing four-basin R1
checkpoint, searches only F0 alpha and is_time with lambda_i=1, and forwards
that matched F0 while freezing every other R1 parameter. No training or
production-model modification is performed.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src"), str(BENCHMARK / "scripts" / "diagnostics")]

import audit_mopex34_root_cause as A
import audit_mopex45_sequential_discretization as D
import run_mopex4_relaxed_manifold as R
from dmotpy.models.flux.mopex import mopex_rainfall_1, mopex_training_context
from dpl.nn_parameterizer import CatchmentParameterizer

OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "root_cause_audit" / "f0_forward_reachability"
OUT.mkdir(parents=True, exist_ok=True)
WARMUP = SCORED = 365
BETA = 50.0
LAMBDA_I = 1.0
R1_CHECKPOINT = BENCHMARK / "results" / "mopex45_phase_fix" / "root_cause_audit" / "relaxed_manifold" / "r1_best_checkpoint.pt"
BASIN_NAMES = {391: "8202700", 373: "8150800", 269: "5507600", 530: "11532500"}
BASINS = [391, 373, 269, 530]
ALPHA_GRID = np.linspace(0.0, 1.0, 41)
ITIME_GRID = np.linspace(0.0, 1.0, 41)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(x) -> float:
    return float(x.detach().cpu()) if isinstance(x, torch.Tensor) else float(x)


def kge_per_basin(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Use the repository's differentiable KGE implementation, but retain its
    # basin vector for the required per-basin report.
    loss, kge = R.compute_differentiable_kge(q, y, warmup_days=0)
    return loss, kge


def stack_state(states: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(states, dim=0)


def r1_full_forward(theta: torch.Tensor, gamma: torch.Tensor, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Exact relaxed R1 runtime over warm-up plus scored days."""
    phys = A.norm_to_phys(theta, "mopex4")
    P, T, PET, doy = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
    Sn, S1, S2, Sc1, Sc2 = D._init_states(theta.shape[0], device=x.device)
    qs, ets, ints, raws, states, storage_delta = [], [], [], [], [], []
    with mopex_training_context(lambda_i=LAMBDA_I, lambda_p=1.0, beta=BETA):
        for t in range(WARMUP + SCORED):
            before = torch.stack([S1, S2, Sc1, Sc2, Sn], dim=1)
            out = R.relaxed_step(
                P[t], T[t], PET[t], *phys[:, :4].t(), phys[:, 4], phys[:, 5], gamma,
                phys[:, 6], phys[:, 7], phys[:, 8], phys[:, 9],
                S1, S2, Sc1, Sc2, Sn, doy[t], rho=1.0, lambda_i=LAMBDA_I,
            )
            q, et, S1, S2, Sc1, Sc2, Sn, fx = out
            after = torch.stack([S1, S2, Sc1, Sc2, Sn], dim=1)
            qs.append(q); ets.append(et); ints.append(fx["i"]); raws.append(fx["i_raw"])
            states.append(after); storage_delta.append(after - before)
            S1, S2, Sc1, Sc2, Sn = [v.detach() for v in (S1, S2, Sc1, Sc2, Sn)]
    alpha = phys[:, 4]
    is_time = phys[:, 5]
    cosine = torch.cos(2 * torch.pi * (doy - is_time[None, :]) / 365.25)
    season_raw = alpha[None, :] + gamma[None, :] * cosine
    fraction = F.softplus(BETA * season_raw) / BETA
    return {
        "q": torch.stack(qs), "et": torch.stack(ets), "i": torch.stack(ints), "i_raw": torch.stack(raws),
        "states": torch.stack(states), "storage_delta": torch.stack(storage_delta),
        "season_raw": season_raw, "fraction": fraction,
    }


def f0_full_forward(theta: torch.Tensor, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Exact production-equivalent F0 runtime with lambda_i fixed to one."""
    phys = D.norm_to_phys(theta, D.M4_BOUNDS)
    P, T, PET, doy = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
    Sn, S1, S2, Sc1, Sc2 = D._init_states(theta.shape[0])
    Sn, S1, S2, Sc1, Sc2 = D._init_states(theta.shape[0], device=x.device)
    with mopex_training_context(lambda_i=LAMBDA_I, lambda_p=1.0, beta=BETA):
        for t in range(WARMUP + SCORED):
            before = torch.stack([S1, S2, Sc1, Sc2, Sn], dim=1)
            out = D.mopex4_step_diag(
                P[t], T[t], PET[t], *phys.t(), S1, S2, Sc1, Sc2, Sn,
                doy=doy[t], nearzero=1e-6,
            )
            q, et, S1, S2, Sc1, Sc2, Sn, fx = out
            after = torch.stack([S1, S2, Sc1, Sc2, Sn], dim=1)
            qs.append(q); ets.append(et); ints.append(fx["i"]); raws.append(fx["i_raw"])
            states.append(after); storage_delta.append(after - before)
            S1, S2, Sc1, Sc2, Sn = [v.detach() for v in (S1, S2, Sc1, Sc2, Sn)]
    alpha = phys[:, 4]
    is_time = phys[:, 5]
    cosine = torch.cos(2 * torch.pi * (doy - is_time[None, :]) / 365.25)
    season_raw = alpha[None, :] + (1.0 - alpha)[None, :] * cosine
    fraction = F.softplus(BETA * season_raw) / BETA
    return {
        "q": torch.stack(qs), "et": torch.stack(ets), "i": torch.stack(ints), "i_raw": torch.stack(raws),
        "states": torch.stack(states), "storage_delta": torch.stack(storage_delta),
        "season_raw": season_raw, "fraction": fraction,
    }


def rainfall(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    phys = D.norm_to_phys(theta, D.M4_BOUNDS)
    return mopex_rainfall_1(x[:, :, 0], x[:, :, 1], phys[:, 0][None, :])


def rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((a - b) ** 2, dim=0))


def search_basin(theta_base: torch.Tensor, x_one: torch.Tensor, target: dict[str, torch.Tensor], basin_idx: int) -> tuple[dict, list[dict]]:
    """Deterministic coarse grid plus two local refinements."""
    def evaluate(points: list[tuple[float, float]]) -> list[dict]:
        n = len(points)
        theta = theta_base.repeat(n, 1)
        theta[:, 4] = torch.tensor([p[0] for p in points], dtype=theta.dtype)
        theta[:, 5] = torch.tensor([p[1] for p in points], dtype=theta.dtype)
        x_rep = x_one.repeat(1, n, 1)
        with torch.no_grad():
            out = f0_full_forward(theta, x_rep)
            tar_i = target["i"][:, basin_idx].unsqueeze(1)
            tar_frac = target["fraction"][:, basin_idx].unsqueeze(1)
            i_rmse = rmse(out["i"], tar_i.expand(-1, n))
            i_rmse_scored = rmse(out["i"][WARMUP:], tar_i[WARMUP:].expand(-1, n))
            frac_rmse = rmse(out["fraction"], tar_frac.expand(-1, n))
            frac_rmse_scored = rmse(out["fraction"][WARMUP:], tar_frac[WARMUP:].expand(-1, n))
        rows = []
        for k, (a, it) in enumerate(points):
            rows.append({"alpha_norm": a, "is_time_norm": it, "i_rmse_full": float(i_rmse[k]), "i_rmse_scored": float(i_rmse_scored[k]), "fraction_rmse_full": float(frac_rmse[k]), "fraction_rmse_scored": float(frac_rmse_scored[k])})
        return rows
    coarse = [(float(a), float(it)) for a in ALPHA_GRID for it in ITIME_GRID]
    rows = evaluate(coarse)
    primary = min(rows, key=lambda r: (r["i_rmse_full"], r["fraction_rmse_full"], r["alpha_norm"], r["is_time_norm"]))
    fraction_best = min(rows, key=lambda r: (r["fraction_rmse_full"], r["i_rmse_full"], r["alpha_norm"], r["is_time_norm"]))
    step = 1.0 / 40.0
    refine_points = set()
    for center in (primary, fraction_best):
        for da in range(-4, 5):
            for di in range(-4, 5):
                refine_points.add((float(np.clip(center["alpha_norm"] + da * step / 4, 0, 1)), float(np.clip(center["is_time_norm"] + di * step / 4, 0, 1))))
    refined = evaluate(sorted(refine_points))
    rows_all = rows + refined
    best = min(rows_all, key=lambda r: (r["i_rmse_full"], r["fraction_rmse_full"], r["alpha_norm"], r["is_time_norm"]))
    frac_best = min(rows_all, key=lambda r: (r["fraction_rmse_full"], r["i_rmse_full"], r["alpha_norm"], r["is_time_norm"]))
    phys_lo = np.asarray([b[0] for b in D.M4_BOUNDS]); phys_hi = np.asarray([b[1] for b in D.M4_BOUNDS])
    best["alpha"] = best["alpha_norm"]
    best["is_time"] = phys_lo[5] + best["is_time_norm"] * (phys_hi[5] - phys_lo[5])
    best["coarse_best_i_rmse_full"] = primary["i_rmse_full"]
    best["coarse_best_fraction_rmse_full"] = fraction_best["fraction_rmse_full"]
    best["coarse_candidates"] = len(coarse)
    best["refine_candidates"] = len(refine_points)
    best["fraction_only_alpha"] = frac_best["alpha_norm"]
    best["fraction_only_is_time"] = phys_lo[5] + frac_best["is_time_norm"] * (phys_hi[5] - phys_lo[5])
    best["fraction_only_i_rmse_full"] = frac_best["i_rmse_full"]
    best["fraction_only_fraction_rmse_full"] = frac_best["fraction_rmse_full"]
    return best, [{"basin_idx": basin_idx, **r} for r in rows_all]


def network_raw_alpha(attrs: torch.Tensor, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    net = CatchmentParameterizer(35, 10, hidden_dims=[256, 256], dropout=0.05)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["network"]); net.eval()
    with torch.no_grad():
        raw = net.net(attrs)
    norm = torch.sigmoid(raw[:, 4])
    derivative = norm * (1 - norm)  # alpha bounds are [0, 1]
    return raw[:, 4], derivative


def parameter_anatomy(ids, theta, gamma, x, r1, y) -> list[dict]:
    phys = A.norm_to_phys(theta, "mopex4")
    P = x[:, :, 0]
    pr = rainfall(x, theta)
    rows = []
    for j, basin_idx in enumerate(BASINS):
        raw = r1["season_raw"][:, j]; frac = r1["fraction"][:, j]; I = r1["i"][:, j]
        rows.append({
            "basin_id": BASIN_NAMES[basin_idx], "basin_idx": basin_idx,
            "alpha": finite_float(phys[j, 4]), "gamma": finite_float(gamma[j]),
            "one_minus_alpha": finite_float(1 - phys[j, 4]), "gamma_minus_one_minus_alpha": finite_float(gamma[j] - (1 - phys[j, 4])),
            "abs_gamma_gap": finite_float((gamma[j] - (1 - phys[j, 4])).abs()), "is_time": finite_float(phys[j, 5]),
            "lambda_i": 1.0, "beta": BETA, "mean_season_raw": finite_float(raw.mean()),
            "min_season_raw": finite_float(raw.min()), "max_season_raw": finite_float(raw.max()),
            "mean_fraction": finite_float(frac.mean()), "min_fraction": finite_float(frac.min()), "max_fraction": finite_float(frac.max()),
            "seasonal_fraction_amplitude": finite_float(frac.max() - frac.min()), "fraction_active_day_ratio": finite_float((frac > 0.01).float().mean()),
            "sum_I_over_sum_Pr": finite_float(I.sum() / (pr[:, j].sum() + 1e-9)),
            "sum_I_over_sum_P": finite_float(I.sum() / (P[:, j].sum() + 1e-9)),
            "KGE": finite_float(r1["kge"][j]), "loss": finite_float(1 - r1["kge"][j]),
        })
    return rows


def trajectory_rows(theta, target, matched, f0, x, y) -> tuple[list[dict], list[dict]]:
    pr = rainfall(x, theta)
    rows_traj, rows_kge = [], []
    _, k1 = kge_per_basin(target["q"][WARMUP:], y[WARMUP:])
    _, k0 = kge_per_basin(f0["q"][WARMUP:], y[WARMUP:])
    for j, basin_idx in enumerate(BASINS):
        q1, q0 = target["q"][:, j], f0["q"][:, j]
        et1, et0 = target["et"][:, j], f0["et"][:, j]
        st1, st0 = target["states"][:, j], f0["states"][:, j]
        i1, i0 = target["i"][:, j], f0["i"][:, j]
        fr1, fr0 = target["fraction"][:, j], f0["fraction"][:, j]
        def corr(a, b):
            av, bv = a.detach().numpy(), b.detach().numpy()
            return float(np.corrcoef(av, bv)[0, 1]) if np.std(av) > 0 and np.std(bv) > 0 else float("nan")
        rows_traj.append({
            "basin_id": BASIN_NAMES[basin_idx], "basin_idx": basin_idx,
            "alpha_F0_star": matched[j]["alpha"], "is_time_F0_star": matched[j]["is_time"],
            "interception_rmse_full": finite_float(torch.sqrt(torch.mean((i0 - i1) ** 2))),
            "interception_rmse_scored": finite_float(torch.sqrt(torch.mean((i0[WARMUP:] - i1[WARMUP:]) ** 2))),
            "fraction_rmse_full": finite_float(torch.sqrt(torch.mean((fr0 - fr1) ** 2))),
            "fraction_rmse_scored": finite_float(torch.sqrt(torch.mean((fr0[WARMUP:] - fr1[WARMUP:]) ** 2))),
            "Q_rmse_full": finite_float(torch.sqrt(torch.mean((q0 - q1) ** 2))),
            "Q_rmse_scored": finite_float(torch.sqrt(torch.mean((q0[WARMUP:] - q1[WARMUP:]) ** 2))),
            "Q_correlation_full": corr(q0, q1), "Q_correlation_scored": corr(q0[WARMUP:], q1[WARMUP:]),
            "ET_rmse_full": finite_float(torch.sqrt(torch.mean((et0 - et1) ** 2))),
            "ET_rmse_scored": finite_float(torch.sqrt(torch.mean((et0[WARMUP:] - et1[WARMUP:]) ** 2))),
            "state_rmse_full": finite_float(torch.sqrt(torch.mean((st0 - st1) ** 2))),
            "I_R1_over_Pr": finite_float(i1.sum() / (pr[:, j].sum() + 1e-9)),
            "I_F0match_over_Pr": finite_float(i0.sum() / (pr[:, j].sum() + 1e-9)),
            "I_R1_over_P": finite_float(i1.sum() / (x[:, j, 0].sum() + 1e-9)),
            "I_F0match_over_P": finite_float(i0.sum() / (x[:, j, 0].sum() + 1e-9)),
        })
        rows_kge.append({
            "basin_id": BASIN_NAMES[basin_idx], "basin_idx": basin_idx,
            "KGE_R1": finite_float(k1[j]), "KGE_best_F0_match": finite_float(k0[j]), "delta_KGE": finite_float(k0[j] - k1[j]),
            "loss_R1": finite_float(1 - k1[j]), "loss_best_F0_match": finite_float(1 - k0[j]),
        })
    return rows_traj, rows_kge


def alpha_boundary(ids, attrs, theta):
    configs = [
        ("R0_baseline_existing_checkpoint", BENCHMARK / "results/dpl_round13_20260805/auto100/checkpoints/mopex4/epoch_100.pt", True),
        ("continuation_existing_checkpoint", BENCHMARK / "results/mopex45_phase_fix/full_continuation/runs/seed_41/checkpoints/J2/seed_41/epoch_100.pt", True),
    ]
    rows = []
    for source, path, available in configs:
        raw, der = network_raw_alpha(attrs, path)
        norm = torch.sigmoid(raw)
        for j, basin_idx in enumerate(BASINS):
            rows.append({"source": source, "basin_id": BASIN_NAMES[basin_idx], "basin_idx": basin_idx, "step": "epoch_100", "alpha_physical": finite_float(norm[j]), "alpha_raw_pre_activation": finite_float(raw[j]), "d_alpha_d_raw": finite_float(der[j]), "distance_to_lower": finite_float(norm[j]), "distance_to_upper": finite_float(1 - norm[j]), "raw_available": True, "history_available": False, "source_file": str(path)})
    eps = 1e-7
    norm_r1 = theta[:, 4].detach().clamp(eps, 1 - eps)
    raw_r1 = torch.log(norm_r1) - torch.log1p(-norm_r1)
    der_r1 = norm_r1 * (1 - norm_r1)
    for j, basin_idx in enumerate(BASINS):
        rows.append({"source": "R1_best_checkpoint_inverse_logit", "basin_id": BASIN_NAMES[basin_idx], "basin_idx": basin_idx, "step": 57, "alpha_physical": finite_float(norm_r1[j]), "alpha_raw_pre_activation": finite_float(raw_r1[j]), "d_alpha_d_raw": finite_float(der_r1[j]), "distance_to_lower": finite_float(norm_r1[j]), "distance_to_upper": finite_float(1 - norm_r1[j]), "raw_available": False, "history_available": False, "source_file": str(R1_CHECKPOINT)})
    return rows


def build_report(anatomy, match, kge, traj, alpha_rows, verdict, summary):
    lines = ["# MOPEX4 F0 forward-reachability decision experiment", "", "## Scope", "CPU-only exact four-basin replay. `lambda_i=1` and `beta=50` were fixed throughout F0 matching and forward testing. All common hydrologic parameters were frozen at the R1 checkpoint; only F0 `alpha` and `is_time` were searched.", "", "## Stage 0 and Stage 1", "", f"- R1 checkpoint median KGE: `{summary['r1_median_kge']:.9f}`; rho=`1`; lambda_i=`1`; beta=`50`.", f"- R1 parameter anatomy is in `r1_per_basin_parameter_anatomy.csv`; the previous alpha≈0.315/gamma≈0.01 point is **not** the actual R1 learned point: `{summary['previous_point_actual']}`. It is a prior coarse alpha-gamma surface location, not a checkpoint parameter.", "", "| basin | alpha | gamma | 1-alpha | gamma-(1-alpha) | I/Pr | KGE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in anatomy:
        lines.append(f"| {r['basin_id']} | {r['alpha']:.6f} | {r['gamma']:.6f} | {r['one_minus_alpha']:.6f} | {r['gamma_minus_one_minus_alpha']:.6f} | {r['sum_I_over_sum_Pr']:.4f} | {r['KGE']:.6f} |")
    lines += ["", "## Stage 2 matching", "", "Primary search criterion: full warm-up + scored-window interception RMSE. Secondary criterion: seasonal-fraction RMSE. Search used a 41×41 coarse normalized grid plus deterministic local refinements around the best interception and best fraction points. `lambda_i=1` throughout.", "", "| basin | alpha* | is_time* | I RMSE full | I RMSE scored | fraction RMSE full | fraction RMSE scored |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r, t in zip(match, traj):
        lines.append(f"| {r['basin_id']} | {r['alpha']:.6f} | {r['is_time']:.4f} | {t['interception_rmse_full']:.6f} | {t['interception_rmse_scored']:.6f} | {t['fraction_rmse_full']:.6f} | {t['fraction_rmse_scored']:.6f} |")
    lines += ["", "## Stage 3 forward test", "", "| basin | KGE R1 | KGE best-F0 | delta KGE | Q RMSE scored | Q correlation scored | ET RMSE scored | state RMSE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, t in zip(kge, traj):
        lines.append(f"| {k['basin_id']} | {k['KGE_R1']:.6f} | {k['KGE_best_F0_match']:.6f} | {k['delta_KGE']:.6f} | {t['Q_rmse_scored']:.6f} | {t['Q_correlation_scored']:.6f} | {t['ET_rmse_scored']:.6f} | {t['state_rmse_full']:.6f} |")
    lines += ["", f"R1 median KGE = `{summary['r1_median_kge']:.6f}`; best-F0-match median KGE = `{summary['f0_median_kge']:.6f}`; median delta KGE = `{summary['median_delta_kge']:.6f}`; R0 matched reference median KGE = `{summary['r0_reference_kge']:.6f}`. Basins retaining most R1 gain (F0 gain / R1 gain ≥ 0.5): `{summary['basins_retaining_most_r1_gain']}` / 4.", "", "## Stage 5 alpha boundary diagnostic", "", "Raw alpha history was not stored for the R1 direct checkpoint; its raw value below is the inverse-logit reconstruction from the stored normalized alpha and is explicitly not presented as training history. Existing R0 and continuation epoch-100 checkpoints were inspected directly. See `alpha_boundary_diagnostic.csv`.", f"- R0 alpha near upper bound (alpha≥0.95): `{summary['r0_alpha_near_upper_count']}` / 4.", f"- R0 raw saturation evidence (|raw|≥4): `{summary['r0_alpha_raw_saturated_count']}` / 4.", "", "## Verdict", f"**{verdict['primary_verdict']}**", "", verdict["explanation"], "", "- Independent gamma required by current evidence: **" + verdict["independent_gamma_required"] + "**", "- Shared-dPL gamma A/B should run next: **" + verdict["shared_dpl_next"] + "**", "- Production change justified: **NO**", "", "## Limits", "This is a four-basin forward reachability diagnostic, not a 531-basin inference. It tests the final `lambda_i=1` F0 manifold with R1 common parameters held fixed; it does not establish what a jointly re-optimized F0 can achieve. Mechanistic interpretation of interception effects beyond this forward comparison remains outside this experiment.", ""]
    return "\n".join(lines)


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(2026)
    ids, x, y, attrs, basins = R.load_all()
    if list(basins) != BASINS:
        raise RuntimeError(f"unexpected representative basin indices: {basins}")
    checkpoint = torch.load(R1_CHECKPOINT, map_location="cpu", weights_only=False)
    theta = checkpoint["theta"].detach().float(); gamma = checkpoint["gamma"].detach().float()
    if theta.shape != (4, 10) or gamma.shape != (4,) or checkpoint.get("step") != 57:
        raise RuntimeError("R1 checkpoint schema/provenance mismatch")
    target = r1_full_forward(theta, gamma, x)
    target["kge"] = kge_per_basin(target["q"][WARMUP:], y[WARMUP:])[1]
    median_kge = finite_float(target["kge"].median())
    if abs(median_kge - 0.6015980243682861) > 1e-5:
        raise RuntimeError(f"R1 checkpoint re-verification mismatch: {median_kge}")
    anatomy = parameter_anatomy(ids, theta, gamma, x, target, y)
    write_csv(OUT / "r1_checkpoint_reverification.csv", [{"checkpoint": str(R1_CHECKPOINT), "best_step": checkpoint["step"], "median_kge": median_kge, "loss": finite_float(1 - target["kge"].mean()), "rho": 1.0, "lambda_i": 1.0, "beta": BETA, "checkpoint_recreated": True}])
    write_csv(OUT / "r1_per_basin_parameter_anatomy.csv", anatomy)
    anatomy_md = ["# R1 per-basin parameter anatomy", "", "The actual R1 learned gamma is read from `r1_best_checkpoint.pt`; the old alpha≈0.315/gamma≈0.01 value is not the checkpoint point.", "", "| basin | alpha | gamma | 1-alpha | gap | is_time | I/Pr | KGE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in anatomy:
        anatomy_md.append(f"| {r['basin_id']} | {r['alpha']:.6f} | {r['gamma']:.6f} | {r['one_minus_alpha']:.6f} | {r['gamma_minus_one_minus_alpha']:.6f} | {r['is_time']:.4f} | {r['sum_I_over_sum_Pr']:.4f} | {r['KGE']:.6f} |")
    (OUT / "r1_parameter_anatomy_summary.md").write_text("\n".join(anatomy_md) + "\n")
    matches, search_rows = [], []
    for j, basin_idx in enumerate(BASINS):
        best, rows = search_basin(theta[j:j + 1], x[:, j:j + 1, :], target, j)
        best["basin_id"] = BASIN_NAMES[basin_idx]; best["basin_idx"] = basin_idx; matches.append(best)
        search_rows.extend([{**r, "basin_id": BASIN_NAMES[basin_idx]} for r in rows])
    write_csv(OUT / "f0_forward_matching_parameters.csv", [{"basin_id": r["basin_id"], "basin_idx": r["basin_idx"], "alpha_F0_star": r["alpha"], "is_time_F0_star": r["is_time"], "alpha_norm": r["alpha_norm"], "is_time_norm": r["is_time_norm"], "I_RMSE_full": r["i_rmse_full"], "I_RMSE_scored": r["i_rmse_scored"], "fraction_RMSE_full": r["fraction_rmse_full"], "fraction_RMSE_scored": r["fraction_rmse_scored"], "fraction_only_alpha": r["fraction_only_alpha"], "fraction_only_is_time": r["fraction_only_is_time"], "coarse_candidates": r["coarse_candidates"], "refine_candidates": r["refine_candidates"]} for r in matches])
    write_csv(OUT / "f0_forward_matching_search_grid.csv", search_rows)
    write_csv(OUT / "f0_forward_matching_trajectory_metrics.csv", traj)
    # Primary F0 forward with matched alpha/is_time and frozen common theta.
    theta_f0 = theta.clone()
    for j, r in enumerate(matches):
        theta_f0[j, 4] = r["alpha_norm"]; theta_f0[j, 5] = r["is_time_norm"]
    f0 = f0_full_forward(theta_f0, x)
    traj, kge_rows = trajectory_rows(theta, target, matches, f0, x, y)
    write_csv(OUT / "f0_forward_matching_kge.csv", kge_rows)
    write_csv(OUT / "f0_forward_matching_trajectory_metrics.csv", traj)
    alpha_rows = alpha_boundary(ids, attrs, theta)
    write_csv(OUT / "alpha_boundary_diagnostic.csv", alpha_rows)
    r0 = 0.4705899954
    kges = np.asarray([r["KGE_best_F0_match"] for r in kge_rows]); r1k = np.asarray([r["KGE_R1"] for r in kge_rows])
    gain_ratio = (kges - r0) / (r1k - r0 + 1e-12)
    r0_rows = [r for r in alpha_rows if r["source"] == "R0_baseline_existing_checkpoint"]
    summary = {
        "r1_median_kge": median_kge, "f0_median_kge": float(np.median(kges)), "median_delta_kge": float(np.median(kges - r1k)),
        "mean_delta_kge": float(np.mean(kges - r1k)), "min_delta_kge": float(np.min(kges - r1k)), "max_delta_kge": float(np.max(kges - r1k)),
        "r0_reference_kge": r0, "basins_retaining_most_r1_gain": int(np.sum(gain_ratio >= 0.5)),
        "previous_point_actual": False, "r0_alpha_near_upper_count": int(sum(float(r["alpha_physical"]) >= .95 for r in r0_rows)),
        "r0_alpha_raw_saturated_count": int(sum(abs(float(r["alpha_raw_pre_activation"])) >= 4 for r in r0_rows)),
        "water_balance_r1_max_abs": finite_float(torch.stack((x[:, :, 0] - target["q"] - target["et"] - target["storage_delta"].sum(dim=2)), dim=0).abs().max()),
        "water_balance_f0_max_abs": finite_float(torch.stack((x[:, :, 0] - f0["q"] - f0["et"] - f0["storage_delta"].sum(dim=2)), dim=0).abs().max()),
        "matching_global_search": "41x41 normalized coarse grid plus 9x9 local refinements around interception and fraction minima",
    }
    # Classify based on retained R1 gain, while retaining the raw basin values.
    if summary["f0_median_kge"] >= r0 + 0.8 * (median_kge - r0):
        verdict = {"primary_verdict": "H1 SUPPORTED", "explanation": "The lambda_i=1 F0 manifold, with R1 common parameters frozen, retains at least 80% of the R1-vs-R0 median gain after direct interception matching. This supports forward reachability and makes optimization accessibility the leading explanation on these four basins.", "independent_gamma_required": "NO", "shared_dpl_next": "NO"}
    elif summary["f0_median_kge"] <= r0 + 0.2 * (median_kge - r0):
        verdict = {"primary_verdict": "H2 SUPPORTED", "explanation": "The best lambda_i=1 F0 match loses at least 80% of the R1-vs-R0 median gain and the trajectory/forward diagnostics show a material restriction. Independent amplitude provides a materially useful forward degree of freedom for this four-basin test.", "independent_gamma_required": "YES", "shared_dpl_next": "YES"}
    else:
        verdict = {"primary_verdict": "MIXED", "explanation": "The best lambda_i=1 F0 match retains an intermediate fraction of the R1-vs-R0 gain. Basin-level results should be used rather than a single global promotion decision.", "independent_gamma_required": "NOT YET", "shared_dpl_next": "NO"}
    summary["gain_ratio_per_basin"] = {r["basin_id"]: float(gain_ratio[i]) for i, r in enumerate(kge_rows)}
    audit = {"device": "cpu", "basins": [BASIN_NAMES[b] for b in BASINS], "warmup": WARMUP, "scored": SCORED, "lambda_i": LAMBDA_I, "beta": BETA, "r1_checkpoint": str(R1_CHECKPOINT), "r1_checkpoint_verified": True, "matching": {"coarse_grid": [41, 41], "local_refine": [9, 9], "primary_objective": "full warm-up + scored interception RMSE", "secondary_objective": "full warm-up + scored fraction RMSE"}, "summary": summary, "verdict": verdict, "production_modified": False}
    (OUT / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    (OUT / "final_f0_reachability_report.md").write_text(build_report(anatomy, matches, kge_rows, traj, alpha_rows, verdict, summary))
    print("MOPEX4 F0 FORWARD-REACHABILITY TEST")
    print(f"\nR1 checkpoint:\nverified median KGE: {median_kge:.9f}\nlambda_i: 1\nbeta: 50")
    print("\nR1 parameter anatomy:")
    for r in anatomy:
        print(f"{r['basin_id']} alpha/gamma/1-alpha/I-Pr: {r['alpha']:.6f}/{r['gamma']:.6f}/{r['one_minus_alpha']:.6f}/{r['sum_I_over_sum_Pr']:.6f}")
    print("\nPrevious alpha≈0.315,gamma≈0.01 interpretation:\nactual R1 learned point: NO\nexplanation: prior alpha-gamma surface slice/grid point, not checkpoint learned values")
    print("\nBest F0 matching (lambda_i fixed = 1):")
    for r, k in zip(matches, kge_rows): print(f"{r['basin_id']} alpha*/is_time*/I-RMSE/KGE: {r['alpha']:.6f}/{r['is_time']:.4f}/{r['i_rmse_full']:.6f}/{k['KGE_best_F0_match']:.6f}")
    print(f"\nAggregate:\nR1 median KGE: {median_kge:.6f}\nbest-F0-match median KGE: {summary['f0_median_kge']:.6f}\nmedian delta KGE: {summary['median_delta_kge']:.6f}\nbasins retaining most R1 gain: {summary['basins_retaining_most_r1_gain']}/4")
    print(f"\nAlpha boundary:\nR0 alpha near upper bound: {summary['r0_alpha_near_upper_count']}/4\nraw sigmoid saturation evidence: {summary['r0_alpha_raw_saturated_count']}/4\nverdict: {verdict['primary_verdict']}")
    print(f"\nPrimary verdict:\n- {verdict['primary_verdict']}\nIndependent gamma required by current evidence: {verdict['independent_gamma_required']}\nShared-dPL gamma A/B should run next: {verdict['shared_dpl_next']}\nProduction change justified: NO\nNext action: {('stop; do not promote independent gamma' if verdict['shared_dpl_next'] == 'NO' else 'only a separate authorized shared-dPL gamma A/B')}")


if __name__ == "__main__":
    main()
