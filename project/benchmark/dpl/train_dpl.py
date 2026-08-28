"""
Skeleton Training Pipeline for Differentiable Parameter Learning (dPL).
Connects Neural Network Parameterizer -> Hydrological Model -> Autograd Loss.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dpl.optimizer_transaction import FiniteOptimizerTransaction
from src.data_selection import load_ids, load_repeated_warmup_and_train
from src.model_registry import NPARAM_INFO_36, build_model, get_spec
from src.production_config import load_resolved_config


def compute_differentiable_kge(
    q_sim: torch.Tensor,
    q_obs: torch.Tensor,
    warmup_days: int = 0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute differentiable 1 - KGE loss for backpropagation across basins."""
    # Slice off warmup period: (Time, Basins)
    q_sim = q_sim[warmup_days:]
    q_obs = q_obs[warmup_days:]

    # Mask valid finite observation days
    mask = torch.isfinite(q_obs) & torch.isfinite(q_sim)
    
    # Safe masked means
    n_valid = mask.sum(dim=0, keepdim=True).clamp_min(1.0)
    q_sim_clean = torch.where(mask, q_sim, torch.zeros_like(q_sim))
    q_obs_clean = torch.where(mask, q_obs, torch.zeros_like(q_obs))

    mean_obs = q_obs_clean.sum(dim=0, keepdim=True) / n_valid
    mean_sim = q_sim_clean.sum(dim=0, keepdim=True) / n_valid

    # Centered variances
    var_obs = (torch.where(mask, (q_obs - mean_obs) ** 2, torch.zeros_like(q_obs)).sum(dim=0, keepdim=True) / n_valid).clamp_min(eps)
    var_sim = (torch.where(mask, (q_sim - mean_sim) ** 2, torch.zeros_like(q_sim)).sum(dim=0, keepdim=True) / n_valid).clamp_min(eps)

    std_obs = torch.sqrt(var_obs)
    std_sim = torch.sqrt(var_sim)

    # Pearson correlation r
    cov = torch.where(mask, (q_obs - mean_obs) * (q_sim - mean_sim), torch.zeros_like(q_obs)).sum(dim=0, keepdim=True) / n_valid
    r = cov / (std_obs * std_sim + eps)

    alpha = std_sim / (std_obs + eps)
    beta = mean_sim / (mean_obs + eps)

    kge = 1.0 - torch.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    # Filter out basins with insufficient data or non-finite KGE
    valid_basins = torch.isfinite(kge)
    if valid_basins.any():
        loss = torch.mean(1.0 - kge[valid_basins])
    else:
        loss = torch.tensor(0.0, device=q_sim.device, requires_grad=True)
    return loss


def train_dpl_model(
    model_name: str,
    config_path: Path,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    grad_clip_norm: float | None = None,
) -> None:
    """Train dPL with the finite transaction guard (CPU smoke-friendly)."""
    resolved = load_resolved_config(config_path)
    ids = load_ids(resolved["data"]["basin_ids"])

    # 1. Load Forcings & Target Streamflow
    x, y, data_metadata = load_repeated_warmup_and_train(ids, resolved, device)
    warmup_days = data_metadata["warmup_total_days"]

    # 2. Build Catchment Attribute Matrix
    attr_builder = CatchmentAttributeBuilder()
    norm_attr = attr_builder.build_normalized_attributes(ids, device=device)
    n_basins, n_attr = norm_attr.shape

    # 3. Instantiate model and parameterizer.  HydrologyModel owns the
    # normalized-to-physical mapping, so the dPL output stays normalized.
    spec = get_spec(model_name, device=device)
    n_params = NPARAM_INFO_36[model_name]
    hydro_model = build_model(
        model_name, device, warm_up=warmup_days, backend="compile", dtype=torch.float64
    )
    dpl_config = resolved.get("dpl", {})
    parameterizer = CatchmentParameterizer(
        in_features=n_attr,
        out_features=n_params,
        param_bounds=None,
        architecture=dpl_config.get("parameterizer_architecture", "legacy"),
        parameter_names=list(spec.parameter_names),
        parameter_groups=spec.parameter_groups,
        saturation_floor=float(dpl_config.get("saturation_floor", 0.01)),
        saturation_regularizer_weight=float(dpl_config.get("saturation_regularizer_weight", 0.0)),
    ).to(device, dtype=torch.float64)

    optimizer = optim.Adam(parameterizer.parameters(), lr=lr)
    transaction = FiniteOptimizerTransaction(
        optimizer,
        parameterizer.parameters(),
        clip_norm=grad_clip_norm,
        named_parameters=parameterizer.named_parameters(),
    )
    print(f"=== Starting dPL Training for [{model_name}] across {n_basins} basins ===")
    print(f"Attributes Shape: {norm_attr.shape} | Parameters Dim: {n_params}")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Predict parameters theta from catchment attributes: (n_basins, n_params)
        predicted_params, mapping_diagnostics = parameterizer(norm_attr, return_diagnostics=True)
        total_mapping_jacobian = mapping_diagnostics["transform_jacobian"] * hydro_model.normalized_parameter_mapping_jacobian(predicted_params)
        mapping_diagnostics["total_mapping_jacobian"] = total_mapping_jacobian
        mapping_diagnostics["normalized_jacobian"] = total_mapping_jacobian
        raw_params = predicted_params.unsqueeze(-1)

        # Run differentiable forward pass through hydrological model
        q_sim = hydro_model({"x_phy": x}, (None, raw_params))["streamflow"].squeeze(-1).squeeze(-1)
        loss = compute_differentiable_kge(q_sim, y, warmup_days=warmup_days)
        if parameterizer.saturation_regularizer_weight:
            loss = loss + parameterizer.saturation_regularizer_weight * parameterizer.saturation_regularizer_from_diagnostics(
                mapping_diagnostics
            )
        transaction.step(loss, epoch=epoch, batch_index=0, basin_ids=ids)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] Loss (1 - Mean KGE): {loss.item():.4f}")

    print(f"=== dPL Training Complete for [{model_name}] ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Differentiable Parameter Learning (dPL) Skeleton Runner")
    parser.add_argument("--model", default="simhyd", help="Target hydrological model name")
    parser.add_argument("--config", default="configs/full_run_10starts_300gen_warm1980_1981x5.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config_path = Path(args.config) if Path(args.config).is_absolute() else BENCHMARK_ROOT / args.config
    train_dpl_model(
        args.model,
        config_path,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        grad_clip_norm=args.grad_clip_norm,
    )


if __name__ == "__main__":
    main()
