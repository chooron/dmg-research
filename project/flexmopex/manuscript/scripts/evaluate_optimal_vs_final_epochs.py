#!/usr/bin/env python3
"""Comprehensive evaluation & standardized registry of Optimal (Min Train Loss) and Final (Ep 100) Epochs.
"""
import os, json, torch, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
for p in (REPO_ROOT, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config
from project.flexmopex.models.learned_weight_mopex_candidates import (
    LearnedWeightMopexE, LearnedStructureNetPureAttrEncoder
)
from project.flexmopex.models.fixed_weight_mopex import FixedWeightMopex
from project.flexmopex.models.parameter_nets import ParamRoutingNet
from project.flexmopex.run_model import _build_data_loader

base_dir = "project/flexmopex/results/formal_531_parallel"

models_info = [
    {
        "id": "base",
        "name": "Base (Fixed w=0)",
        "lambda": 0.0,
        "config": "conf/config_formal_531_base.yaml",
        "dir": base_dir + "/config_formal_531_base/base/seed_42",
        "log": "project/flexmopex/logs/formal_531_parallel/formal_531_base.log",
        "type": "fixed",
        "fixed_w": 0.0,
    },
    {
        "id": "flex_0003",
        "name": "Flex (λ=0.003)",
        "lambda": 0.003,
        "config": "conf/config_formal_531_flex_lambda0003.yaml",
        "dir": base_dir + "/config_formal_531_flex_lambda0003/flex_alpha_config/seed_42",
        "log": "project/flexmopex/logs/formal_531_lambda_sweep/formal_531_flex_lambda0003.log",
        "type": "flex",
    },
    {
        "id": "flex_0005",
        "name": "Flex (λ=0.005)",
        "lambda": 0.005,
        "config": "conf/config_formal_531_flex_lambda0005.yaml",
        "dir": base_dir + "/config_formal_531_flex_lambda0005/flex_alpha_config/seed_42",
        "log": "project/flexmopex/logs/formal_531_parallel/formal_531_flex_lambda0005.log",
        "type": "flex",
    },
    {
        "id": "flex_0007",
        "name": "Flex (λ=0.007)",
        "lambda": 0.007,
        "config": "conf/config_formal_531_flex_lambda0007.yaml",
        "dir": base_dir + "/config_formal_531_flex_lambda0007/flex_alpha_config/seed_42",
        "log": "project/flexmopex/logs/formal_531_parallel/formal_531_flex_lambda0007.log",
        "type": "flex",
    },
    {
        "id": "flex_0010",
        "name": "Flex (λ=0.010)",
        "lambda": 0.010,
        "config": "conf/config_formal_531_flex_lambda0010.yaml",
        "dir": base_dir + "/config_formal_531_flex_lambda0010/flex_alpha_config/seed_42",
        "log": "project/flexmopex/logs/formal_531_parallel/formal_531_flex_lambda0010.log",
        "type": "flex",
    },
    {
        "id": "flex_0015",
        "name": "Flex (λ=0.015)",
        "lambda": 0.015,
        "config": "conf/config_formal_531_flex_lambda0015.yaml",
        "dir": base_dir + "/config_formal_531_flex_lambda0015/flex_alpha_config/seed_42",
        "log": "project/flexmopex/logs/formal_531_lambda_sweep/formal_531_flex_lambda0015.log",
        "type": "flex",
    },
    {
        "id": "flex_0020",
        "name": "Flex (λ=0.020)",
        "lambda": 0.020,
        "config": "conf/config_formal_531_flex_lambda0020.yaml",
        "dir": base_dir + "/config_formal_531_flex_lambda0020/flex_alpha_config/seed_42",
        "log": "project/flexmopex/logs/formal_531_lambda_sweep/formal_531_flex_lambda0020.log",
        "type": "flex",
    },
    {
        "id": "flex_0030",
        "name": "Flex (λ=0.030)",
        "lambda": 0.030,
        "config": "conf/config_formal_531_flex_lambda0030.yaml",
        "dir": base_dir + "/config_formal_531_flex_lambda0030/flex_alpha_config/seed_42",
        "log": "project/flexmopex/logs/formal_531_lambda_sweep/formal_531_flex_lambda0030.log",
        "type": "flex",
    },
    {
        "id": "flex_0050",
        "name": "Flex (λ=0.050)",
        "lambda": 0.050,
        "config": "conf/config_formal_531_flex_lambda0050.yaml",
        "dir": base_dir + "/config_formal_531_flex_lambda0050/flex_alpha_config/seed_42",
        "log": "project/flexmopex/logs/formal_531_lambda_sweep/formal_531_flex_lambda0050.log",
        "type": "flex",
    },
    {
        "id": "flex_0100",
        "name": "Flex (λ=0.100)",
        "lambda": 0.100,
        "config": "conf/config_formal_531_flex_lambda0100.yaml",
        "dir": base_dir + "/config_formal_531_flex_lambda0100/flex_alpha_config/seed_42",
        "log": "project/flexmopex/logs/formal_531_lambda_sweep/formal_531_flex_lambda0100.log",
        "type": "flex",
    },
    {
        "id": "full",
        "name": "Full (Fixed w=1)",
        "lambda": 0.0,
        "config": "conf/config_formal_531_full.yaml",
        "dir": base_dir + "/config_formal_531_full/full/seed_42",
        "log": "project/flexmopex/logs/formal_531_parallel/formal_531_full.log",
        "type": "fixed",
        "fixed_w": 1.0,
    },
]

for m in models_info:
    content = open(m["log"]).read()
    records = []
    for line in content.splitlines():
        m_bf = re.search(r"\[Epoch\s+(\d+)/100\]\s+loss=([0-9\.]+)", line)
        if m_bf:
            records.append({"epoch": int(m_bf.group(1)), "loss": float(m_bf.group(2))})
            continue
        m_fl = re.search(r"\[R15 Epoch\s+(\d+)/100\].*Loss_total=([0-9\.]+)", line)
        if m_fl:
            records.append({"epoch": int(m_fl.group(1)), "loss": float(m_fl.group(2))})
            continue
    df_rec = pd.DataFrame(records)
    min_row = df_rec.loc[df_rec["loss"].idxmin()]
    opt_ep = int(min_row["epoch"])
    m["opt_epoch"] = opt_ep
    m["opt_train_loss"] = float(min_row["loss"])
    m["final_train_loss"] = float(df_rec.iloc[-1]["loss"])
    m["ep1_train_loss"] = float(df_rec.iloc[0]["loss"])
    
    prefix = "fixedweightmopex" if m["type"] == "fixed" else "learnedweightmopexe"
    m["opt_ckpt"] = os.path.join(m["dir"], "model", f"{prefix}_ep{opt_ep}.pt")
    m["final_ckpt"] = os.path.join(m["dir"], "model", f"{prefix}_ep100.pt")

import yaml
from dmg.core.data.loaders.hydro_loader import HydroLoader

cfg_dict = yaml.safe_load(open("project/flexmopex/conf/config_formal_531_flex_lambda0007.yaml"))
obs_cfg = yaml.safe_load(open("project/flexmopex/conf/observations/camels_531.yaml"))
cfg_dict["observations"] = obs_cfg
cfg_dict["device"] = torch.device("cpu")
cfg_dict["dtype"] = torch.float32
cfg_dict["train_time"] = [cfg_dict["train"]["start_time"], cfg_dict["train"]["end_time"]]
cfg_dict["test_time"] = [cfg_dict["test"]["start_time"], cfg_dict["test"]["end_time"]]
cfg_dict["sim_time"] = [cfg_dict["simulation"]["start_time"], cfg_dict["simulation"]["end_time"]]
cfg_dict["batch_size"] = cfg_dict["test"]["batch_size"]
cfg_dict["all_time"] = [obs_cfg["start_time"], obs_cfg["end_time"]]
cfg_dict["model_dir"] = "project/flexmopex/results/formal_531_parallel/config_formal_531_flex_lambda0007/flex_alpha_config/seed_42/model"
cfg_dict["model"] = {"phy": cfg_dict["delta_model"]["phy_model"], "nn": cfg_dict["delta_model"]["nn_model"]}
cfg_dict["model"]["phy"]["name"] = cfg_dict["delta_model"]["phy_model"]["model"]
cfg_dict["model"]["nn"]["name"] = cfg_dict["delta_model"]["nn_model"]["model"]

from project.flexmopex.run_model import _attach_doy

loader = HydroLoader(cfg_dict, test_split=True)
_attach_doy(loader.eval_dataset, cfg_dict["test"])
cfg_sample = cfg_dict
ed = loader.eval_dataset
y_obs = ed["target"][365:365+5110, :, 0].cpu().numpy()
n_attr = ed["xc_nn_norm"].shape[-1] - 3
attrs = ed["xc_nn_norm"][0, :, -n_attr:].cpu()
B = attrs.shape[0]

def calc_kge(sim, obs):
    v = ~np.isnan(obs)
    if v.sum() < 30: return np.nan
    s, o = sim[v], obs[v]
    mean_s, mean_o = np.mean(s), np.mean(o)
    std_s, std_o = np.std(s), np.std(o)
    if std_s == 0 or std_o == 0: return np.nan
    r = np.corrcoef(s, o)[0, 1]
    alpha = std_s / std_o
    beta = mean_s / (mean_o + 1e-12)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def calc_nse(sim, obs):
    v = ~np.isnan(obs)
    if v.sum() < 30: return np.nan
    s, o = sim[v], obs[v]
    ss_res = np.sum((s - o)**2)
    ss_tot = np.sum((o - np.mean(o))**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

def evaluate_checkpoint(m_info, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if m_info["type"] == "fixed":
        phy_cfg = dict(cfg_sample["delta_model"]["phy_model"])
        phy_cfg["fixed_weights"] = {"w_phen": m_info["fixed_w"], "w_int": m_info["fixed_w"], "w_snow": m_info["fixed_w"], "w_sub": m_info["fixed_w"]}
        phy = FixedWeightMopex(phy_cfg, device="cpu")
        nn = ParamRoutingNet(input_dim=35, hidden_dim=128, nmul=16, device="cpu")
    else:
        phy = LearnedWeightMopexE(cfg_sample["delta_model"]["phy_model"], device="cpu")
        nn = LearnedStructureNetPureAttrEncoder(input_dim=35, hidden_dim=128, nmul=16, device="cpu")
        
    sd = {k.replace("nn_model.", ""): v for k, v in ckpt.items() if k.startswith("nn_model.")}
    nn.load_state_dict(sd, strict=False)
    phy.eval()
    nn.eval()
    
    with torch.no_grad():
        nn_out = nn({"c_nn_norm": attrs})
        mopex_params = phy._descale_mopex_params(nn_out["params"])
        routing = phy._descale_routing_params(nn_out["gamma_uh"])
        
        if m_info["type"] == "fixed":
            val = m_info["fixed_w"]
            w_probs = np.full((B, 4), val)
            weights_on = phy.fixed_weight_values.unsqueeze(0).expand(B, -1)
        else:
            logits = nn_out["weights"].view(B, 4, 2).clamp(-10.0, 10.0)
            weights_on = F.softmax(logits, dim=-1)[..., 1]
            w_probs = weights_on.numpy()
            
        predictions_chunks = []
        for chunk_idx in range(14):
            t0 = chunk_idx * 365
            t1 = t0 + 730
            sample_chunk = {
                "x_phy": ed["x_phy"][t0:t1].cpu(),
                "doy": ed["doy"][t0:t1].cpu(),
                "c_nn_norm": attrs,
            }
            out_phy = phy(sample_chunk, nn_out)
            q_val = out_phy["streamflow"].numpy()[:, :, 0]
            predictions_chunks.append(q_val)
            
        q_sim = np.concatenate(predictions_chunks, axis=0)
        
    nses = [calc_nse(q_sim[:, b], y_obs[:, b]) for b in range(B)]
    kges = [calc_kge(q_sim[:, b], y_obs[:, b]) for b in range(B)]
    
    res = {
        "median_nse": float(np.nanmedian(nses)),
        "mean_nse": float(np.nanmean(nses)),
        "median_kge": float(np.nanmedian(kges)),
        "mean_kge": float(np.nanmean(kges)),
        "weights": {
            "w_phen": {
                "mean": float(np.mean(w_probs[:, 0])),
                "median": float(np.median(w_probs[:, 0])),
                "std": float(np.std(w_probs[:, 0])),
                "act_010": float(np.mean(w_probs[:, 0] > 0.10) * 100),
                "act_050": float(np.mean(w_probs[:, 0] > 0.50) * 100),
            },
            "w_int": {
                "mean": float(np.mean(w_probs[:, 1])),
                "median": float(np.median(w_probs[:, 1])),
                "std": float(np.std(w_probs[:, 1])),
                "act_010": float(np.mean(w_probs[:, 1] > 0.10) * 100),
                "act_050": float(np.mean(w_probs[:, 1] > 0.50) * 100),
            },
            "w_snow": {
                "mean": float(np.mean(w_probs[:, 2])),
                "median": float(np.median(w_probs[:, 2])),
                "std": float(np.std(w_probs[:, 2])),
                "act_010": float(np.mean(w_probs[:, 2] > 0.10) * 100),
                "act_050": float(np.mean(w_probs[:, 2] > 0.50) * 100),
            },
            "w_sub": {
                "mean": float(np.mean(w_probs[:, 3])),
                "median": float(np.median(w_probs[:, 3])),
                "std": float(np.std(w_probs[:, 3])),
                "act_010": float(np.mean(w_probs[:, 3] > 0.10) * 100),
                "act_050": float(np.mean(w_probs[:, 3] > 0.50) * 100),
            },
            "total_mean": float(np.mean(np.sum(w_probs, axis=-1))),
            "total_median": float(np.median(np.sum(w_probs, axis=-1))),
        }
    }
    return res

print("="*95)
print("EVALUATING OPTIMAL EPOCHS vs FINAL EPOCHS FOR ALL 11 MODELS")
print("="*95)

comparison_rows = []
registry_data = {}

for m in models_info:
    print(f"Evaluating {m['name']} (Opt Ep {m['opt_epoch']} vs Final Ep 100)...")
    res_opt = evaluate_checkpoint(m, m["opt_ckpt"])
    res_final = evaluate_checkpoint(m, m["final_ckpt"])
    
    row = {
        "Model": m["name"],
        "lambda": m["lambda"],
        "Opt_Epoch": m["opt_epoch"],
        "Opt_Train_Loss": m["opt_train_loss"],
        "Final_Train_Loss": m["final_train_loss"],
        "Opt_Test_Median_NSE": res_opt["median_nse"],
        "Final_Test_Median_NSE": res_final["median_nse"],
        "Opt_Test_Mean_NSE": res_opt["mean_nse"],
        "Final_Test_Mean_NSE": res_final["mean_nse"],
        "Opt_Test_Median_KGE": res_opt["median_kge"],
        "Final_Test_Median_KGE": res_final["median_kge"],
        "Opt_w_int_act050": res_opt["weights"]["w_int"]["act_050"],
        "Final_w_int_act050": res_final["weights"]["w_int"]["act_050"],
        "Opt_Total_Weight": res_opt["weights"]["total_mean"],
        "Final_Total_Weight": res_final["weights"]["total_mean"],
    }
    comparison_rows.append(row)
    
    registry_data[m["id"]] = {
        "name": m["name"],
        "lambda": m["lambda"],
        "type": m["type"],
        "config_file": m["config"],
        "result_dir": m["dir"],
        "optimal_epoch": {
            "epoch": m["opt_epoch"],
            "train_loss": m["opt_train_loss"],
            "checkpoint_file": os.path.basename(m["opt_ckpt"]),
            "metrics": {
                "test_median_nse": res_opt["median_nse"],
                "test_mean_nse": res_opt["mean_nse"],
                "test_median_kge": res_opt["median_kge"],
                "test_mean_kge": res_opt["mean_kge"],
            },
            "weights": res_opt["weights"],
        },
        "final_epoch": {
            "epoch": 100,
            "train_loss": m["final_train_loss"],
            "checkpoint_file": os.path.basename(m["final_ckpt"]),
            "metrics": {
                "test_median_nse": res_final["median_nse"],
                "test_mean_nse": res_final["mean_nse"],
                "test_median_kge": res_final["median_kge"],
                "test_mean_kge": res_final["mean_kge"],
            },
            "weights": res_final["weights"],
        }
    }

df_comp = pd.DataFrame(comparison_rows)
csv_out = os.path.join(base_dir, "optimal_vs_final_epoch_audit.csv")
df_comp.to_csv(csv_out, index=False)
print(f"Saved: {csv_out}")

json_out = os.path.join(base_dir, "CHECKPOINT_REGISTRY.json")
with open(json_out, "w") as f:
    json.dump(registry_data, f, indent=2)
print(f"Saved: {json_out}")

print("\n" + "="*110)
print("OPTIMAL EPOCH vs FINAL EPOCH COMPARISON TABLE")
print("="*110)
cols_disp = ["Model", "Opt_Epoch", "Opt_Train_Loss", "Final_Train_Loss", "Opt_Test_Median_NSE", "Final_Test_Median_NSE", "Opt_Test_Median_KGE", "Final_Test_Median_KGE", "Opt_w_int_act050", "Final_w_int_act050"]
print(df_comp[cols_disp].to_string(index=False))
