import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Set paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = Path("/root/outputs/full_model_series_calibration/v1")
TASKS_DIR = OUTPUTS_DIR / "tasks"
DPL_RESULTS_DIR = PROJECT_ROOT / "results" / "dpl_camels_531_lite_v2"

SEEDS = [101, 202, 303, 404, 505]


def load_531_basins():
    for p in [
        "/autodl-fs/data/531sub_id.txt",
        str(PROJECT_ROOT.parent / "data" / "531sub_id.txt"),
        str(PROJECT_ROOT / "data" / "531sub_id.txt"),
    ]:
        if os.path.exists(p):
            with open(p) as f:
                text = f.read().strip()
                try:
                    return [str(x).zfill(8) for x in json.loads(text)]
                except Exception:
                    return [x.strip().zfill(8) for x in text.splitlines() if x.strip()]
    raise FileNotFoundError("Could not find 531sub_id.txt")


def get_dpl_data(model_name):
    dfs = []
    for s in [42, 123, 2026]:
        p = DPL_RESULTS_DIR / model_name / f"seed_{s}" / "train_test_kge_by_basin.csv"
        if not p.exists():
            p = (
                SCRIPT_DIR.parent
                / "results"
                / "dpl_camels_531_lite_v2"
                / model_name
                / f"seed_{s}"
                / "train_test_kge_by_basin.csv"
            )
        df = pd.read_csv(p)
        df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
        dfs.append(df.set_index("basin_id"))

    tr_med = pd.concat([d["train_kge"] for d in dfs], axis=1).median(axis=1)
    te_med = pd.concat([d["test_kge"] for d in dfs], axis=1).median(axis=1)
    return pd.DataFrame({"dpl_tr_med": tr_med, "dpl_te_med": te_med})


def load_frac_snow(all_531):
    try:
        from ablation.ic_core.data_adapter import load_531_bundle

        foundation_cfg_path = (
            PROJECT_ROOT / "ablation" / "configs" / "ic_foundation_531_v1.json"
        )
        with open(foundation_cfg_path) as f:
            cfg = json.load(f)
        cfg["device"] = "cpu"
        if os.path.exists("/autodl-fs/data/531sub_id.txt"):
            cfg["basin_list_path"] = "/autodl-fs/data/531sub_id.txt"
            cfg["gage_ids_path"] = "/autodl-fs/data/gage_id.npy"
            cfg["dates_path"] = "/autodl-fs/data/camels_dates.npy"
            cfg["dataset_path"] = "/autodl-fs/data/camels_dataset"
        bundle = load_531_bundle(cfg)
        fs_array = bundle.attributes[:, 3]
        return {b: float(fs) for b, fs in zip(bundle.basin_ids, fs_array)}
    except Exception as e:
        print(
            f"Warning: could not load frac_snow from bundle ({e}), attempting fallback..."
        )
        return {b: 0.0 for b in all_531}


def load_optimizer_results(
    model_key, opt_name, all_531, seeds=SEEDS, starts_per_seed=3, cap_gen=200
):
    model_dir = TASKS_DIR / model_key / opt_name
    records = {b: [] for b in all_531}
    seed101_records = {b: {} for b in all_531}

    for b_id in all_531:
        for seed in seeds:
            for s in range(starts_per_seed):
                rf = model_dir / b_id / f"seed_{seed}" / f"start_{s}" / "result.json"
                tf = model_dir / b_id / f"seed_{seed}" / f"start_{s}" / "trace.json"

                if not rf.exists():
                    alt_files = glob.glob(
                        str(
                            model_dir
                            / b_id
                            / f"seed_{seed}"
                            / f"*start*{s}*"
                            / "result.json"
                        )
                    )
                    if alt_files:
                        rf = Path(alt_files[0])
                        tf = rf.parent / "trace.json"

                if rf.exists():
                    try:
                        d = json.load(open(rf))
                        tr_kge = d.get("best_train_kge")
                        te_kge = d.get("best_test_kge")
                        theta = d.get("best_theta_normalized")
                        tot_gen = d.get("total_generations", cap_gen)

                        plateau_gen = None
                        if tf.exists():
                            try:
                                t_data = json.load(open(tf))
                                best_fits = [x["best_fitness"] for x in t_data]
                                final_fit = best_fits[-1]
                                for g_idx, f_val in enumerate(best_fits):
                                    if (final_fit - f_val) < 1e-4:
                                        plateau_gen = g_idx + 1
                                        break
                            except Exception:
                                pass
                        if plateau_gen is None:
                            plateau_gen = tot_gen

                        rec = {
                            "seed": seed,
                            "start": s,
                            "train_kge": tr_kge,
                            "test_kge": te_kge,
                            "theta": theta,
                            "total_gen": tot_gen,
                            "plateau_gen": plateau_gen,
                        }
                        records[b_id].append(rec)
                        if seed == 101:
                            seed101_records[b_id][s] = rec
                    except Exception:
                        pass
    return records, seed101_records


def analyze_model(model_key, cap_gen=200):
    all_531 = load_531_basins()
    frac_snow_map = load_frac_snow(all_531)

    # Load CMA-ES results (5 seeds x 3 starts)
    cmaes_all, cmaes_s101 = load_optimizer_results(
        model_key, "CMAES", all_531, seeds=SEEDS, starts_per_seed=3, cap_gen=cap_gen
    )
    # Load XNES results (seed 101, 3 starts)
    xnes_all, xnes_s101 = load_optimizer_results(
        model_key, "XNES", all_531, seeds=[101], starts_per_seed=3, cap_gen=300
    )

    rows = []
    for b_id in all_531:
        c_recs = cmaes_all[b_id]
        c_101 = cmaes_s101[b_id]
        x_101 = xnes_s101[b_id]

        c_tr_all = [r["train_kge"] for r in c_recs if r["train_kge"] is not None]
        c_te_all = [r["test_kge"] for r in c_recs if r["test_kge"] is not None]

        c_tr_3 = [
            c_101[s]["train_kge"]
            for s in range(3)
            if s in c_101 and c_101[s]["train_kge"] is not None
        ]
        c_te_3 = [
            c_101[s]["test_kge"]
            for s in range(3)
            if s in c_101 and c_101[s]["test_kge"] is not None
        ]

        x_tr_3 = [
            x_101[s]["train_kge"]
            for s in range(3)
            if s in x_101 and x_101[s]["train_kge"] is not None
        ]
        x_te_3 = [
            x_101[s]["test_kge"]
            for s in range(3)
            if s in x_101 and x_101[s]["test_kge"] is not None
        ]

        # Medians
        cma_tr_med3 = float(np.median(c_tr_3)) if len(c_tr_3) >= 3 else np.nan
        cma_te_med3 = float(np.median(c_te_3)) if len(c_te_3) >= 3 else np.nan
        cma_tr_med_all = float(np.median(c_tr_all)) if len(c_tr_all) > 0 else np.nan
        cma_te_med_all = float(np.median(c_te_all)) if len(c_te_all) > 0 else np.nan

        xnes_tr_med3 = float(np.median(x_tr_3)) if len(x_tr_3) >= 3 else np.nan
        xnes_te_med3 = float(np.median(x_te_3)) if len(x_te_3) >= 3 else np.nan

        # Spreads
        cma_spread3 = max(c_tr_3) - min(c_tr_3) if len(c_tr_3) >= 3 else np.nan
        cma_spread_all = max(c_tr_all) - min(c_tr_all) if len(c_tr_all) >= 2 else np.nan
        xnes_spread3 = max(x_tr_3) - min(x_tr_3) if len(x_tr_3) >= 3 else np.nan

        # Plateau gen & termination reason
        plateaus = [r["plateau_gen"] for r in c_recs if r["plateau_gen"] is not None]
        avg_plateau = float(np.mean(plateaus)) if len(plateaus) > 0 else np.nan
        term_reason = "converged" if avg_plateau < cap_gen else "hit_generation_cap"

        # Pairwise parameter distance
        thetas = [np.array(r["theta"]) for r in c_recs if r["theta"] is not None]
        p_dists = []
        if len(thetas) >= 2:
            for i in range(len(thetas)):
                for j in range(i + 1, len(thetas)):
                    p_dists.append(np.linalg.norm(thetas[i] - thetas[j]))
        param_dist_mean = float(np.mean(p_dists)) if len(p_dists) > 0 else np.nan

        row = {
            "basin_id": b_id,
            "frac_snow": frac_snow_map.get(b_id, 0.0),
            "cmaes_tr_med3": cma_tr_med3,
            "cmaes_te_med3": cma_te_med3,
            "cmaes_tr_med_all": cma_tr_med_all,
            "cmaes_te_med_all": cma_te_med_all,
            "xnes_tr_med3": xnes_tr_med3,
            "xnes_te_med3": xnes_te_med3,
            "cma_spread3": cma_spread3,
            "cma_spread_all": cma_spread_all,
            "xnes_spread3": xnes_spread3,
            "avg_plateau_gen": avg_plateau,
            "term_reason": term_reason,
            "param_dist_mean": param_dist_mean,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def run_all_audits_and_comparisons():
    all_531 = load_531_basins()

    print("=== Analyzing XAJ_base ===")
    df_xaj = analyze_model("XAJ", cap_gen=200)
    df_xaj.to_csv(OUTPUTS_DIR / "IC_XAJ_base_CMAES_531_results.csv", index=False)

    print("=== Analyzing XAJ_CN ===")
    df_xaj_cn = analyze_model("XAJ_CN", cap_gen=200)
    df_xaj_cn.to_csv(OUTPUTS_DIR / "IC_XAJ_CN_CMAES_531_results.csv", index=False)

    # Load dPL data
    dpl_xaj = get_dpl_data("XAJ")
    dpl_xaj_cn = get_dpl_data("XAJ_CN")

    # Merge for final analysis
    df_xaj = df_xaj.join(dpl_xaj, on="basin_id")
    df_xaj_cn = df_xaj_cn.join(dpl_xaj_cn, on="basin_id")

    print("\n=======================================================")
    print("=== CMA-ES vs XNES RE-CALIBRATION AUDIT & COMPARISON ===")
    print("=======================================================\n")

    def compute_summary_stats(df, model_label):
        diff_tr = df["cmaes_tr_med3"] - df["xnes_tr_med3"]
        diff_te = df["cmaes_te_med3"] - df["xnes_te_med3"]

        w_tr_stat, w_tr_p = wilcoxon(diff_tr.dropna())
        w_te_stat, w_te_p = wilcoxon(diff_te.dropna())

        tr_win = (diff_tr > 0.0001).mean()
        te_win = (diff_te > 0.0001).mean()

        stuck_basins = df[diff_tr > 0.05]

        print(f"--- {model_label} ---")
        print(
            f"  Train KGE Paired Diff (CMAES - XNES): Median = {diff_tr.median():.4f} [IQR = {diff_tr.quantile(0.75) - diff_tr.quantile(0.25):.4f}]"
        )
        print(
            f"    Wilcoxon p = {w_tr_p:.2e} | Frac CMAES > XNES = {tr_win * 100:.1f}% ({int(tr_win * len(df))}/{len(df)})"
        )
        print(
            f"  Test KGE Paired Diff (CMAES - XNES):  Median = {diff_te.median():.4f} [IQR = {diff_te.quantile(0.75) - diff_te.quantile(0.25):.4f}]"
        )
        print(
            f"    Wilcoxon p = {w_te_p:.2e} | Frac CMAES > XNES = {te_win * 100:.1f}% ({int(te_win * len(df))}/{len(df)})"
        )
        print(
            f"  Stuck Basins under XNES (Train gain > 0.05): Count = {len(stuck_basins)} ({len(stuck_basins) / len(df) * 100:.1f}%)"
        )
        if len(stuck_basins) > 0:
            print(
                f"    Stuck basins frac_snow distribution: min={stuck_basins['frac_snow'].min():.3f}, med={stuck_basins['frac_snow'].median():.3f}, max={stuck_basins['frac_snow'].max():.3f}"
            )

        print(
            f"  Across-Start Spread (3-start): CMAES Med = {df['cma_spread3'].median():.6f} vs XNES Med = {df['xnes_spread3'].median():.6f}"
        )
        print(f"  Convergence Audit:")
        print(
            f"    Spread < 0.01: {(df['cma_spread3'] < 0.01).mean() * 100:.1f}% | Spread < 0.001: {(df['cma_spread3'] < 0.001).mean() * 100:.1f}%"
        )
        print(
            f"    Avg Plateau Generation: Median = {df['avg_plateau_gen'].median():.1f}"
        )
        print(
            f"    Termination Reason: {(df['term_reason'] == 'converged').mean() * 100:.1f}% converged before cap"
        )
        print(
            f"    Across-Start Parameter Distance: Median = {df['param_dist_mean'].median():.4f}\n"
        )

    compute_summary_stats(df_xaj, "XAJ_base")
    compute_summary_stats(df_xaj_cn, "XAJ_CN")


if __name__ == "__main__":
    run_all_audits_and_comparisons()
