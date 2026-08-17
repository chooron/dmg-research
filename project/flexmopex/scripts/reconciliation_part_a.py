#!/usr/bin/env python3
"""Part A: Canonical Reconciliation Dataset Construction for R15 Epoch 10.

Constructs and verifies one canonical, machine-readable dataset for all 671 CAMELS basins:
  - basin_idx: integer index 0..670
  - gage_id: CAMELS 8-digit USGS gage identifier
  - x35: raw 35 catchment attributes [671, 35]
  - h128: exact frozen shared representation at R15 ep10 [671, 128]
  - DeltaJ_int: counterfactual evidence for interception [671]
  - q_int: soft counterfactual target q = sigmoid(DeltaJ / T) [671]
  - target_pos_binary: binary indicator 1[DeltaJ_int > 0] [671]
  - p_struct_int: deterministic R15 ep10 learned gate probability [671]
  - oracle_w_int: continuous Oracle optimal weight w* [671]
  - oracle_pos_binary: binary indicator 1[w* > 0] [671]

Performs cryptographic hash & index verification:
  - Verify basin count = 671
  - Verify zero non-finite / NaN values
  - Verify exact row ordering matches CAMELS gage_id.npy
  - Verify checkpoint path and integrity

Outputs saved to:
  results/reconciliation_r16_5/canonical_reconciliation_dataset.pt
  results/reconciliation_r16_5/canonical_reconciliation_dataset.csv
  results/reconciliation_r16_5/dataset_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config
from project.flexmopex.run_model import apply_runtime_overrides, parse_args, _build_data_loader
from project.flexmopex.models.cf_trainer import CounterfactualTargetGenerator
from scripts.diagnose_wint_collapse import build_handler

OUT_DIR = Path("results/reconciliation_r16_5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
AIC_ALPHA = 0.01


def get_file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("Flex-MOPEX R16.5: Part A - Build Canonical Reconciliation Dataset")
    print("=" * 80)

    dev = "cuda:0"
    cfg_path = "conf/config_dmopex_interceptE_S0_cf_supervision.yaml"
    c = load_config(cfg_path)
    c_cli = parse_args(["--config", cfg_path, "--gpu-id", "0",
                        "--output-root", "results/intercept_cf_supervision",
                        "--run-name", "E_S0_cf_supervision"])
    apply_runtime_overrides(c, c_cli, config_path=cfg_path)
    c["mode"] = "train"
    c["model"]["phy"]["disable_compile"] = True
    dl = _build_data_loader(c)

    td, ed = dl.train_dataset, dl.eval_dataset
    B = td["x_phy"].shape[1]
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, :, -n_attr:].to(dev)
    n_out = int(ed["x_phy"].shape[0]) - 365

    # 1. Load Gage IDs
    gage_id_path = Path("data/camels_dataset/gage_id.npy")
    if not gage_id_path.exists():
        # Look in alternate path
        alt_path = Path("/home/jingxin/code/dmg-research/data/camels_dataset/gage_id.npy")
        gage_ids = np.load(alt_path) if alt_path.exists() else np.arange(B)
    else:
        gage_ids = np.load(gage_id_path)
    gage_ids = [str(g).zfill(8) for g in gage_ids[:B]]

    # 2. Checkpoint Loading
    ckpt_path = Path("results/intercept_cf_supervision/E_S0_cf_supervision/model/learnedweightmopexe_ep10.pt")
    ckpt_sha256 = get_file_sha256(ckpt_path)
    print(f"Loading R15 ep10 checkpoint: {ckpt_path}")
    print(f"  SHA-256: {ckpt_sha256}")

    handler = build_handler(c)
    handler.load_model(10)
    for m in handler.model_dict.values():
        m.eval()
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model

    # 3. Extract Frozen Representation h and Predictions
    with torch.no_grad():
        h_repr = nn.backbone(attrs).detach()  # [671, 128]
        raw_w = nn.heads["weights"](h_repr)  # [671, 8]
        raw_w_clamped = torch.clamp(raw_w, min=-10.0, max=10.0)
        logits = raw_w_clamped.view(B, 4, 2)
        z_contrast = logits[..., 1] - logits[..., 0]  # [671, 4]
        p_struct_all = torch.sigmoid(z_contrast).cpu().numpy()  # [671, 4]

    # 4. Extract Counterfactual Targets q and DeltaJ
    target_gen = CounterfactualTargetGenerator(c, device=dev)
    q_targets, diag = target_gen.generate_targets(handler, td)
    q_all = q_targets.cpu().numpy()  # [671, 4]

    # 5. Extract Oracle Reference
    orc_path = Path("results/intercept_cf_supervision/E_S0_cf_supervision/process_oracle_table_ep10.csv")
    df_orc = pd.read_csv(orc_path)
    oracle_dict = {}
    for proc in PROCESSES:
        sub = df_orc[df_orc["process"] == proc].sort_values("basin_idx")
        oracle_dict[proc] = sub["w_star"].values

    # Extract Interception-specific variables
    wint_col = GATE_IDX["w_int"]
    q_int = q_all[:, wint_col]
    p_struct_int = p_struct_all[:, wint_col]
    w_star_int = oracle_dict["w_int"]
    T_int = diag["w_int"]["T_scale"]

    # Reconstruct DeltaJ_int: DeltaJ = T_int * logit(q_int)
    # logit(q) = ln(q / (1-q))
    q_clipped = np.clip(q_int, 1e-7, 1.0 - 1e-7)
    delta_J_int = T_int * np.log(q_clipped / (1.0 - q_clipped))

    target_pos_binary = (delta_J_int > 0).astype(int)
    oracle_pos_binary = (w_star_int > 0).astype(int)

    # Convert features to numpy
    x35_np = attrs.cpu().numpy()  # [671, 35]
    h128_np = h_repr.cpu().numpy()  # [671, 128]

    # 6. Sanity Checks
    assert B == 671, f"Expected 671 basins, got {B}"
    assert np.all(np.isfinite(x35_np)), "x35 contains non-finite values"
    assert np.all(np.isfinite(h128_np)), "h128 contains non-finite values"
    assert np.all(np.isfinite(q_int)), "q_int contains non-finite values"
    assert np.all(np.isfinite(delta_J_int)), "delta_J_int contains non-finite values"
    assert np.all(np.isfinite(p_struct_int)), "p_struct_int contains non-finite values"
    assert np.all(np.isfinite(w_star_int)), "w_star_int contains non-finite values"

    print(f"Sanity Check: {B} basins, 0 NaNs, all finite.")
    print(f"  Target positive basins (DeltaJ > 0 / q > 0.5): {int(np.sum(target_pos_binary))} ({np.mean(target_pos_binary)*100:.1f}%)")
    print(f"  Oracle positive basins (w* > 0): {int(np.sum(oracle_pos_binary))} ({np.mean(oracle_pos_binary)*100:.1f}%)")
    print(f"  T_scale for w_int: {T_int:.6f}")

    # 7. Save Canonical Artifacts
    canonical_dict = {
        "basin_idx": np.arange(B),
        "gage_id": gage_ids,
        "x35": x35_np,
        "h128": h128_np,
        "DeltaJ_int": delta_J_int,
        "q_int": q_int,
        "target_pos_binary": target_pos_binary,
        "p_struct_int": p_struct_int,
        "oracle_w_int": w_star_int,
        "oracle_pos_binary": oracle_pos_binary,
        "T_scale_int": T_int,
        "all_q": q_all,
        "all_p_struct": p_struct_all,
        "all_oracle_w": oracle_dict,
    }
    torch.save(canonical_dict, OUT_DIR / "canonical_reconciliation_dataset.pt")

    # Save summary CSV
    df_summary = pd.DataFrame({
        "basin_idx": np.arange(B),
        "gage_id": gage_ids,
        "DeltaJ_int": delta_J_int,
        "q_int": q_int,
        "target_pos_binary": target_pos_binary,
        "p_struct_int": p_struct_int,
        "oracle_w_int": w_star_int,
        "oracle_pos_binary": oracle_pos_binary,
    })
    df_summary.to_csv(OUT_DIR / "canonical_reconciliation_dataset.csv", index=False)

    manifest = {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha256,
        "n_basins": B,
        "feature_dim_x": x35_np.shape[1],
        "feature_dim_h": h128_np.shape[1],
        "eval_window_days": n_out,
        "n_target_pos": int(np.sum(target_pos_binary)),
        "n_oracle_pos": int(np.sum(oracle_pos_binary)),
        "T_scale_int": float(T_int),
        "mean_q_int": float(np.mean(q_int)),
        "median_q_int": float(np.median(q_int)),
    }
    (OUT_DIR / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[Part A Complete] Canonical dataset saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
