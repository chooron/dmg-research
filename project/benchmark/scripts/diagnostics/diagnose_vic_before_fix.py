"""Fast VIC failure diagnosis script for Round 13 failure at epoch 57 batch 145."""
from __future__ import annotations
import importlib.util
import json
import sys
import math
from pathlib import Path
import torch
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

spec = importlib.util.spec_from_file_location("k_full_retrain", ROOT / "scripts/diagnostics/k_full_retrain.py")
k = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(k)

OUT_DIR = ROOT / "results/dpl_round13_20260805/vic_saturation_fix"
DEVICE = torch.device("cuda")

PARAM_NAMES = ['ibar', 'idelta', 'ishift', 'stot', 'fsm', 'b', 'k1', 'c1', 'k2', 'c2']

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in k.load_ids("data/531sub_id.txt")]
    attrs = k.CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    tx, ty, _, _ = k.NATIVE.load_camels_time_series(ids)
    train_x = torch.as_tensor(tx, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(ty, dtype=torch.float32, device=DEVICE)
    catalog, lengths = k.H1.make_catalog(train_y[k.WARMUP:])

    hydro = k.build_model("vic", DEVICE, warm_up=k.WARMUP, backend="compile", parameter_mapping="auto", warmup_grad_mode="detach")
    net = k.CatchmentParameterizer(attrs.shape[1], 10, hidden_dims=[256, 256], dropout=.05).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    checkpoint_path = ROOT / "results/dpl_round13_20260805/auto100/checkpoints/vic/epoch_050.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["network"])
    opt.load_state_dict(payload["optimizer"])
    torch.random.set_rng_state(payload["cpu_rng"])
    torch.cuda.set_rng_state(payload["cuda_rng"], device=DEVICE)

    print("Loaded epoch_050.pt checkpoint successfully.", flush=True)

    first_failure_info = None

    for epoch in range(51, 58):
        print(f"--- Starting Epoch {epoch} ---", flush=True)
        for step in range(k.STEPS):
            basins = torch.randperm(len(ids), device=DEVICE)[:k.BATCH]
            choices = (torch.rand(k.BATCH, device=DEVICE) * lengths[basins]).long()
            starts = catalog[basins, choices]
            x = k.H1.gather_window(train_x, starts, basins)
            y = k.H1.gather_window(train_y, starts, basins)

            opt.zero_grad(set_to_none=True)
            theta = net(attrs[basins])
            theta.retain_grad()

            is_target_step = (epoch == 57 and step == 145)

            out_dict = hydro({"x_phy": x}, (None, theta.unsqueeze(-1)))
            q = out_dict["streamflow"].squeeze(-1).squeeze(-1)

            loss, _ = k.NATIVE.compute_differentiable_kge(q, y[k.WARMUP:], warmup_days=0)

            loss.backward()

            theta_grad = theta.grad  # shape [100, 10]
            g_list = [p.grad.detach().reshape(-1) for p in net.parameters() if p.grad is not None]
            g = torch.cat(g_list) if len(g_list) > 0 else torch.tensor([], device=DEVICE)

            finite_theta = torch.isfinite(theta_grad)
            finite_g = torch.isfinite(g) if g.numel() > 0 else torch.tensor(False, device=DEVICE)

            if not bool(finite_g.all()) or not bool(finite_theta.all()) or is_target_step:
                print(f"CHECK AT Epoch {epoch}, Step {step}!", flush=True)
                num_finite_theta = int(finite_theta.sum())
                total_theta = int(theta_grad.numel())
                num_finite_g = int(finite_g.sum()) if g.numel() > 0 else 0
                total_g = int(g.numel()) if g.numel() > 0 else 0

                print(f"Theta grad finite: {num_finite_theta}/{total_theta}", flush=True)
                print(f"Network grad finite: {num_finite_g}/{total_g}", flush=True)
                print(f"Loss finite: {bool(torch.isfinite(loss))}, Loss value: {float(loss)}", flush=True)
                print(f"Q finite: {bool(torch.isfinite(q).all())}", flush=True)

                if not bool(finite_theta.all()):
                    bad_indices = (~finite_theta).nonzero(as_tuple=False)
                    bad_theta_info = []
                    for idx in bad_indices:
                        b_idx = int(idx[0])
                        p_idx = int(idx[1])
                        basin_id = int(ids[basins[b_idx]])
                        param_name = PARAM_NAMES[p_idx]
                        param_val = float(theta[b_idx, p_idx].detach())
                        bad_theta_info.append({
                            "batch_basin_index": b_idx,
                            "basin_id": basin_id,
                            "param_index": p_idx,
                            "param_name": param_name,
                            "param_value": param_val,
                            "grad_val": str(theta_grad[b_idx, p_idx].detach().cpu().item())
                        })
                    print("Bad theta entries:", bad_theta_info, flush=True)

                    # Extract saturation_2 detailed state step by step for the bad basins
                    import dmotpy.models.core.vic as vic_core
                    params_tensor = theta
                    ibar, idelta, ishift, stot, fsm, b, k1, c1, k2, c2 = [params_tensor[:, i:i+1] for i in range(10)]
                    smmax = fsm * stot

                    S1, S2, S3 = vic_core.create_initial_state(k.BATCH, 1, DEVICE)
                    sat2_records = []
                    nearzero = 1e-6
                    window_len = x.shape[1]

                    for t in range(window_len):
                        P_t = x[:, t:t+1] # [100, 1]
                        PET_t = torch.ones_like(P_t) * 2.0
                        t_idx = torch.tensor(float(t + 1), device=DEVICE).expand_as(P_t)
                        aux_imax = vic_core.phenology_2(ibar, idelta, ishift, t_idx, torch.tensor(365.25, device=DEVICE), nearzero=nearzero)
                        flux_ei = torch.relu(torch.minimum(vic_core.evap_7(S1, aux_imax, PET_t, nearzero=nearzero), S1 - nearzero))
                        flux_peff = torch.clamp(vic_core.interception_1(P_t, S1, aux_imax, nearzero=nearzero), min=torch.zeros_like(P_t), max=P_t)
                        flux_iex = torch.relu(vic_core.excess_1(S1 + P_t - flux_peff, aux_imax, nearzero=nearzero))
                        S1 = torch.clamp(S1 + P_t - flux_ei - flux_peff - flux_iex, min=nearzero)

                        potential_inf = flux_peff + flux_iex
                        s_rel = S2 / (smmax + nearzero)
                        one_minus_s_rel = 1.0 - s_rel
                        term = torch.clamp(one_minus_s_rel, min=0.0, max=1.0)
                        term_plus_nz = term + nearzero
                        pow_val = term_plus_nz.pow(b)
                        out_frac = 1.0 - pow_val
                        flux_qie = torch.clamp(out_frac * potential_inf, min=torch.zeros_like(potential_inf), max=potential_inf)

                        deriv = (b * (term_plus_nz.pow(b - 1.0))).abs()

                        sat2_records.append({
                            "t": t,
                            "S": S2.detach().clone(),
                            "Smax": smmax.detach().clone(),
                            "b": b.detach().clone(),
                            "potential_inf": potential_inf.detach().clone(),
                            "s_rel": s_rel.detach().clone(),
                            "term": term.detach().clone(),
                            "term_plus_nz": term_plus_nz.detach().clone(),
                            "out_frac": out_frac.detach().clone(),
                            "deriv": deriv.detach().clone()
                        })

                        flux_inf = vic_core.effective_1(potential_inf, flux_qie, nearzero=nearzero)
                        pet_rem_s2 = torch.relu(PET_t - flux_ei)
                        flux_et1 = torch.relu(torch.minimum(torch.minimum(vic_core.evap_7(S2, smmax, pet_rem_s2, nearzero=nearzero), S2 + flux_inf - nearzero), pet_rem_s2))
                        flux_qex1 = torch.clamp(vic_core.saturation_1(flux_inf, S2, smmax, nearzero=nearzero), min=torch.zeros_like(flux_inf), max=flux_inf)
                        flux_pc = vic_core.percolation_5(k1, c1, S2, smmax, nearzero=nearzero)
                        S2_tmp = S2 + flux_inf - flux_et1 - flux_qex1
                        flux_pc = torch.relu(torch.minimum(flux_pc, S2_tmp - nearzero))
                        S2 = torch.clamp(S2_tmp - flux_pc, min=nearzero)

                    bad_basin_indices = list(set([info["batch_basin_index"] for info in bad_theta_info]))
                    rows = []
                    for b_idx in bad_basin_indices:
                        basin_id = int(ids[basins[b_idx]])
                        S_arr = torch.stack([rec["S"][b_idx, 0] for rec in sat2_records]).cpu().numpy()
                        Smax_arr = torch.stack([rec["Smax"][b_idx, 0] for rec in sat2_records]).cpu().numpy()
                        b_arr = torch.stack([rec["b"][b_idx, 0] for rec in sat2_records]).cpu().numpy()
                        pinf_arr = torch.stack([rec["potential_inf"][b_idx, 0] for rec in sat2_records]).cpu().numpy()
                        srel_arr = torch.stack([rec["s_rel"][b_idx, 0] for rec in sat2_records]).cpu().numpy()
                        term_arr = torch.stack([rec["term"][b_idx, 0] for rec in sat2_records]).cpu().numpy()
                        termnz_arr = torch.stack([rec["term_plus_nz"][b_idx, 0] for rec in sat2_records]).cpu().numpy()
                        outfrac_arr = torch.stack([rec["out_frac"][b_idx, 0] for rec in sat2_records]).cpu().numpy()
                        deriv_arr = torch.stack([rec["deriv"][b_idx, 0] for rec in sat2_records]).cpu().numpy()

                        for t_step in range(len(sat2_records)):
                            rows.append({
                                "epoch": epoch,
                                "batch_step": step,
                                "timestep": t_step,
                                "batch_basin_index": b_idx,
                                "basin_id": basin_id,
                                "S": float(S_arr[t_step]),
                                "Smax": float(Smax_arr[t_step]),
                                "s_rel": float(srel_arr[t_step]),
                                "one_minus_s_rel": float(1.0 - srel_arr[t_step]),
                                "p1": float(b_arr[t_step]),
                                "incoming_flux": float(pinf_arr[t_step]),
                                "term": float(term_arr[t_step]),
                                "term_plus_nz": float(termnz_arr[t_step]),
                                "out_frac": float(outfrac_arr[t_step]),
                                "local_deriv_magnitude": float(deriv_arr[t_step]),
                                "term_is_zero": bool(term_arr[t_step] == 0.0),
                                "s_rel_approx_1": bool(abs(srel_arr[t_step] - 1.0) < 1e-4)
                            })

                    df_trace = pd.DataFrame(rows)
                    df_trace.to_csv(OUT_DIR / "before_failure_trace.csv", index=False)
                    print(f"Saved before_failure_trace.csv with {len(rows)} records.", flush=True)

                    max_deriv_row = df_trace.loc[df_trace["local_deriv_magnitude"].idxmax()]
                    print(f"Peak local derivative magnitude in saturation_2: {max_deriv_row['local_deriv_magnitude']:.4e} at timestep {max_deriv_row['timestep']}, basin {max_deriv_row['basin_id']}, S/Smax={max_deriv_row['s_rel']}, term={max_deriv_row['term']}, p1={max_deriv_row['p1']}", flush=True)

                    first_failure_info = {
                        "epoch": epoch,
                        "batch_step": step,
                        "loss": float(loss),
                        "q_finite": bool(torch.isfinite(q).all()),
                        "theta_finite_count": num_finite_theta,
                        "theta_total_count": total_theta,
                        "network_finite_count": num_finite_g,
                        "network_total_count": total_g,
                        "bad_theta_info": bad_theta_info,
                        "peak_deriv": dict(max_deriv_row)
                    }
                    break

            if not bool(finite_g.all()):
                break

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()

        if first_failure_info is not None:
            break

    with open(OUT_DIR / "failure_diagnosis_raw.json", "w") as f:
        json.dump(first_failure_info, f, indent=2, default=str)

    print("Diagnosis run finished successfully.", flush=True)

if __name__ == "__main__":
    main()
