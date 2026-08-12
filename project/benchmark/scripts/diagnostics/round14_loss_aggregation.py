"""Round-14 evidence: dPL KGE aggregation, trajectory, and gradient diagnostics.

This is read-only with respect to training artifacts.  It evaluates existing
auto-100 checkpoints and replays deterministic next-step minibatches from
their saved RNG state.  No model, optimizer, or checkpoint is modified.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
SPEC = importlib.util.spec_from_file_location("round14_runner", ROOT / "scripts/diagnostics/k_full_retrain.py")
K = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(K)

OUT = ROOT / "results/dpl_round14_20260808"
MODELS = ("hbv96", "newzealand1", "mopex4")
EPOCHS = tuple(range(10, 101, 10))
GRADIENT_EPOCHS = (10, 30, 60, 90)
SAMPLES_PER_EPOCH = 5
DEVICE = torch.device("cuda")


def native_kge(q_sim: torch.Tensor, q_obs: torch.Tensor, warmup_days: int = 365) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact runner KGE algebra, returning per-basin KGE and validity."""
    if q_obs.shape[0] == q_sim.shape[0] + warmup_days:
        q_obs = q_obs[warmup_days:]
    elif q_sim.shape[0] == q_obs.shape[0] + warmup_days:
        q_sim = q_sim[warmup_days:]
    eps = 1e-6
    mask = torch.isfinite(q_obs) & torch.isfinite(q_sim) & (q_obs >= 0.0) & (q_sim >= 0.0)
    mask_f = mask.to(q_sim.dtype)
    n_valid = mask_f.sum(dim=0).clamp_min(1.0)
    obs = torch.where(mask, q_obs, torch.zeros_like(q_obs))
    sim = torch.where(mask, q_sim, torch.zeros_like(q_sim))
    mean_obs, mean_sim = obs.sum(0) / n_valid, sim.sum(0) / n_valid
    dobs, dsim = (obs - mean_obs) * mask_f, (sim - mean_sim) * mask_f
    eps_sq = eps * eps
    std_obs = torch.sqrt((dobs.square().sum(0) / n_valid) + eps_sq)
    std_sim = torch.sqrt((dsim.square().sum(0) / n_valid) + eps_sq)
    r = ((dobs * dsim).sum(0) / n_valid) / (std_obs * std_sim)
    alpha = std_sim / std_obs
    beta = mean_sim / (mean_obs + eps)
    kge = 1.0 - torch.sqrt((r - 1.0).square() + (alpha - 1.0).square() + (beta - 1.0).square() + eps_sq)
    return kge, (n_valid > 30) & torch.isfinite(kge)


def fixed_bottom_mask(values: torch.Tensor, fraction: float = 0.10) -> torch.Tensor:
    count = max(1, int(np.ceil(values.numel() * fraction)))
    selected = torch.topk(values.detach(), count, largest=False).indices
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[selected] = True
    return mask


def soft_median_score(kge: torch.Tensor) -> torch.Tensor:
    """A local differentiable median surrogate with fixed detached centre."""
    centre = kge.detach().median()
    scale = (kge.detach().quantile(.75) - kge.detach().quantile(.25)).clamp_min(.05)
    weights = torch.softmax(-torch.abs(kge - centre) / scale, dim=0)
    return (weights * kge).sum()


def aggregation_losses(kge: torch.Tensor, valid: torch.Tensor, theta_grad_norm: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    k = kge[valid]
    per_loss = 1.0 - k
    bottom = fixed_bottom_mask(k)
    keep = ~bottom
    result = {
        "A_mean_kge": per_loss.mean(),
        "B_nkge": (1.0 - 1.0 / (2.0 - k)).mean(),
        "C_trim_bottom10": per_loss[keep].mean(),
        "D_soft_median": -soft_median_score(k),
    }
    if theta_grad_norm is not None:
        weights = theta_grad_norm[valid].detach().clamp_min(1e-8)
        result["E_gradnorm_proxy"] = (per_loss / weights).mean()
    return result


def checkpoint_path(model: str, epoch: int) -> Path:
    paths = [
        ROOT / "results/dpl_round13_20260805/auto100/checkpoints" / model / f"epoch_{epoch:03d}.pt",
        ROOT / "results/dpl_full_retrain_20260804/auto100/checkpoints" / model / f"epoch_{epoch:03d}.pt",
    ]
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {model} checkpoint for epoch {epoch}: {paths}")


def load_context(model: str, backend: str) -> dict:
    ids = [int(x) for x in K.load_ids("data/531sub_id.txt")]
    attrs = K.CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    tx, ty, vx, vy = K.NATIVE.load_camels_time_series(ids)
    train_x = torch.as_tensor(tx, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(ty, dtype=torch.float32, device=DEVICE)
    val_x = torch.as_tensor(vx, dtype=torch.float32, device=DEVICE)
    val_y = torch.as_tensor(vy, dtype=torch.float32, device=DEVICE)
    if model in K.CALENDAR_MODELS:
        train_x, _ = K.add_calendar_forcing(train_x, pd.date_range("1980-10-01", "1995-09-30", freq="D"), model_name=model)
        val_x, _ = K.add_calendar_forcing(val_x, pd.date_range("1994-10-01", "2010-09-30", freq="D"), model_name=model)
    catalog, lengths = K.H1.make_catalog(train_y[K.WARMUP:])
    hydro = K.build_model(model, DEVICE, warm_up=K.WARMUP, backend=backend, parameter_mapping="auto", warmup_grad_mode="detach")
    return {"ids": ids, "attrs": attrs, "train_x": train_x, "train_y": train_y, "val_x": val_x, "val_y": val_y,
            "catalog": catalog, "lengths": lengths, "hydro": hydro}


def load_network(context: dict, model: str, epoch: int, training: bool) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(checkpoint_path(model, epoch), map_location="cpu", weights_only=False)
    net = K.CatchmentParameterizer(context["attrs"].shape[1], K.NPARAM_INFO_36[model], hidden_dims=[256, 256], dropout=.05).to(DEVICE)
    net.load_state_dict(payload["network"])
    net.train(training)
    return net, payload


def validation_trajectory(context: dict, model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_basin: list[dict] = []
    score_rows: list[dict] = []
    initial_kge = None
    quartiles = None
    for epoch in EPOCHS:
        net, _ = load_network(context, model, epoch, training=False)
        with torch.no_grad():
            theta = net(context["attrs"])
            q = context["hydro"]({"x_phy": context["val_x"]}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            kge, valid = native_kge(q, context["val_y"], K.WARMUP)
        if initial_kge is None:
            initial_kge = kge.detach().cpu()
            ranking = torch.argsort(initial_kge)
            quartiles = torch.empty_like(ranking, dtype=torch.long)
            for group, indices in enumerate(torch.tensor_split(ranking, 4), start=1):
                quartiles[indices] = group
        cpu_kge = kge.detach().cpu()
        for i, value in enumerate(cpu_kge.tolist()):
            per_basin.append({"model": model, "epoch": epoch, "basin_id": context["ids"][i], "kge": value,
                              "valid": bool(valid[i].item()), "initial_quartile": int(quartiles[i].item())})
        kval = kge[valid]
        scores = aggregation_losses(kge, valid)
        score_rows.append({"model": model, "epoch": epoch, "validation_median_kge": float(kval.median()),
                           "validation_mean_kge": float(kval.mean()),
                           **{name: float(-loss) for name, loss in scores.items()}})
        del net
    basin = pd.DataFrame(per_basin)
    group = basin.groupby(["model", "epoch", "initial_quartile"], as_index=False).agg(
        median_kge=("kge", "median"), mean_kge=("kge", "mean"), basin_count=("kge", "size"))
    return basin, group, pd.DataFrame(score_rows)


def draw_quartiles(group: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4), sharey=False)
    for axis, model in zip(axes, MODELS):
        sub = group[group.model == model]
        for quartile in range(1, 5):
            line = sub[sub.initial_quartile == quartile]
            axis.plot(line.epoch, line.median_kge, marker="o", label=f"Q{quartile}")
        axis.set_title(model)
        axis.set_xlabel("checkpoint epoch")
        axis.set_ylabel("validation median KGE")
        axis.grid(alpha=.25)
    axes[-1].legend(title="initial KGE")
    fig.tight_layout()
    fig.savefig(OUT / "n2_quartile_trajectories.png", dpi=180)
    plt.close(fig)


def sample_gradient_rows(context: dict, model: str, epoch: int) -> tuple[list[dict], list[dict]]:
    net, payload = load_network(context, model, epoch, training=True)
    torch.random.set_rng_state(payload["cpu_rng"])
    torch.cuda.set_rng_state(payload["cuda_rng"], device=DEVICE)
    rows: list[dict] = []
    n4_rows: list[dict] = []
    for sample in range(SAMPLES_PER_EPOCH):
        basins = torch.randperm(len(context["ids"]), device=DEVICE)[:K.BATCH]
        choices = (torch.rand(K.BATCH, device=DEVICE) * context["lengths"][basins]).long()
        starts = context["catalog"][basins, choices]
        x = K.H1.gather_window(context["train_x"], starts, basins)
        y = K.H1.gather_window(context["train_y"], starts, basins)
        # Replaying each VJP from a fresh graph prevents graph accumulation on
        # the long (730-day) recurrent forward.  Restoring this state makes
        # every replay use the identical dropout mask.
        rng_before_forward = torch.cuda.get_rng_state(DEVICE)

        def forward_terms():
            theta = net(context["attrs"][basins])
            q = context["hydro"]({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            kge, valid = native_kge(q, y[K.WARMUP:], warmup_days=0)
            return theta, kge, valid, 1.0 - kge

        theta, kge, valid, raw_loss = forward_terms()
        rng_after_forward = torch.cuda.get_rng_state(DEVICE)
        base_loss = raw_loss[valid].mean()
        theta_grad = torch.autograd.grad(base_loss, theta)[0]
        theta_norm = theta_grad.norm(dim=1).detach()
        valid_kge = kge[valid].detach()
        valid_loss = raw_loss[valid].detach()
        worst10 = fixed_bottom_mask(valid_kge)
        worst5 = fixed_bottom_mask(valid_kge, .05)
        total_theta = theta_norm[valid].sum().clamp_min(1e-12)

        torch.cuda.set_rng_state(rng_before_forward, device=DEVICE)
        theta, _kge, _valid, raw_loss = forward_terms()
        base_loss = raw_loss[_valid].mean()
        params = [p for p in net.parameters() if p.requires_grad]
        total_net = torch.autograd.grad(base_loss, params)
        total_net_norm = torch.sqrt(sum(g.square().sum() for g in total_net)).detach()

        torch.cuda.set_rng_state(rng_before_forward, device=DEVICE)
        theta, _kge, _valid, raw_loss = forward_terms()
        worst_idx = torch.nonzero(_valid, as_tuple=False).squeeze(1)[worst10]
        worst_loss_scaled = raw_loss[worst_idx].sum() / _valid.sum()
        worst_net = torch.autograd.grad(worst_loss_scaled, params)
        worst_net_norm = torch.sqrt(sum(g.square().sum() for g in worst_net)).detach()
        base_record = {
            "model": model, "checkpoint_epoch": epoch, "sample": sample,
            "kge_p10": float(valid_kge.quantile(.10)), "kge_p50": float(valid_kge.median()),
            "kge_p90": float(valid_kge.quantile(.90)), "kge_min": float(valid_kge.min()),
            "kge_lt_0": int((valid_kge < 0).sum()), "kge_lt_neg1": int((valid_kge < -1).sum()),
            "kge_lt_neg5": int((valid_kge < -5).sum()),
            "loss_share_kge_lt_0": float(valid_loss[valid_kge < 0].sum() / valid_loss.sum().clamp_min(1e-12)),
            "loss_share_kge_lt_neg1": float(valid_loss[valid_kge < -1].sum() / valid_loss.sum().clamp_min(1e-12)),
            "loss_share_kge_lt_neg5": float(valid_loss[valid_kge < -5].sum() / valid_loss.sum().clamp_min(1e-12)),
            "theta_gradient_share_worst5": float(theta_norm[valid][worst5].sum() / total_theta),
            "theta_gradient_share_worst10": float(theta_norm[valid][worst10].sum() / total_theta),
            "network_gradient_norm_total": float(total_net_norm),
            "network_gradient_norm_worst10_scaled": float(worst_net_norm),
            "network_gradient_ratio_worst10": float(worst_net_norm / total_net_norm.clamp_min(1e-12)),
        }
        rows.append(base_record)
        for name in ("A_mean_kge", "B_nkge", "C_trim_bottom10", "D_soft_median", "E_gradnorm_proxy"):
            torch.cuda.set_rng_state(rng_before_forward, device=DEVICE)
            theta, _kge, _valid, raw_loss = forward_terms()
            losses = aggregation_losses(_kge, _valid, theta_norm)
            loss = losses[name]
            grad = torch.autograd.grad(loss, theta)[0].norm(dim=1)[_valid]
            n4_rows.append({"model": model, "checkpoint_epoch": epoch, "sample": sample, "scheme": name,
                            "loss": float(loss.detach()), "score": float(-loss.detach()),
                            "theta_gradient_share_worst10": float(grad[worst10].sum() / grad.sum().clamp_min(1e-12))})
        torch.cuda.set_rng_state(rng_after_forward, device=DEVICE)
        del theta, kge
    return rows, n4_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("n2", "n3n4", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for faithful checkpoint replay")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase != "n3n4":
        all_basin, all_group, all_scores = [], [], []
        for model in MODELS:
            print(f"[round14] validation trajectory {model}", flush=True)
            context = load_context(model, backend="compile")
            basin, group, scores = validation_trajectory(context, model)
            all_basin.append(basin); all_group.append(group); all_scores.append(scores)
            del context["hydro"]
            torch.cuda.empty_cache()
        basin = pd.concat(all_basin, ignore_index=True)
        group = pd.concat(all_group, ignore_index=True)
        scores = pd.concat(all_scores, ignore_index=True)
        basin.to_csv(OUT / "n2_per_basin_checkpoint_kge.csv", index=False)
        group.to_csv(OUT / "n2_quartile_trajectories.csv", index=False)
        scores.to_csv(OUT / "n2_validation_aggregation_scores.csv", index=False)
        draw_quartiles(group)
        historical = basin.groupby(["model", "basin_id"], as_index=False).agg(best_kge=("kge", "max"), final_kge=("kge", "last"))
        historical["final_below_historical_best_gt_005"] = historical.final_kge < historical.best_kge - .05
        historical.to_csv(OUT / "n2_final_vs_checkpoint_best.csv", index=False)
    else:
        scores = pd.read_csv(OUT / "n2_validation_aggregation_scores.csv")
        historical = pd.read_csv(OUT / "n2_final_vs_checkpoint_best.csv")
    if args.phase == "n2":
        print("[round14] N2 complete", flush=True)
        return

    n3, n4 = [], []
    for model in MODELS:
        # Eager mode avoids retaining compiled graphs during repeated VJPs.
        context = load_context(model, backend="eager")
        for epoch in GRADIENT_EPOCHS:
            print(f"[round14] gradient replay {model} epoch {epoch}", flush=True)
            rows, alternatives = sample_gradient_rows(context, model, epoch)
            n3.extend(rows); n4.extend(alternatives)
        del context["hydro"]
        torch.cuda.empty_cache()
    gradient = pd.DataFrame(n3)
    alternatives = pd.DataFrame(n4)
    gradient.to_csv(OUT / "n3_gradient_concentration.csv", index=False)
    alternatives.to_csv(OUT / "n4_sampled_alternative_gradients.csv", index=False)

    # A-D use full validation predictions at all 30 checkpoints.  E uses the
    # explicitly-labelled current-step gradient-norm proxy at 12 checkpoints.
    correlation_rows = []
    for scheme in ("A_mean_kge", "B_nkge", "C_trim_bottom10", "D_soft_median"):
        rho = scores[[scheme, "validation_median_kge"]].corr(method="spearman").iloc[0, 1]
        correlation_rows.append({"scheme": scheme, "scope": "30 checkpoint validation predictions", "spearman_with_validation_median_kge": rho})
    n4_with_median = alternatives.merge(scores[["model", "epoch", "validation_median_kge"]], left_on=["model", "checkpoint_epoch"], right_on=["model", "epoch"])
    for scheme, sub in n4_with_median.groupby("scheme"):
        rho = sub[["score", "validation_median_kge"]].corr(method="spearman").iloc[0, 1]
        correlation_rows.append({"scheme": scheme, "scope": "12 checkpoint sampled training batches", "spearman_with_validation_median_kge": rho})
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(OUT / "n4_median_alignment_correlations.csv", index=False)

    summary = {
        "models": list(MODELS), "checkpoint_epochs": list(EPOCHS), "gradient_epochs": list(GRADIENT_EPOCHS),
        "gradient_samples_per_epoch": SAMPLES_PER_EPOCH,
        "n2_final_below_best_gt_005": historical.groupby("model")["final_below_historical_best_gt_005"].sum().to_dict(),
        "n3_mean_theta_gradient_share_worst10": gradient.groupby("model")["theta_gradient_share_worst10"].mean().to_dict(),
        "n3_mean_network_gradient_ratio_worst10": gradient.groupby("model")["network_gradient_ratio_worst10"].mean().to_dict(),
    }
    (OUT / "round14_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
