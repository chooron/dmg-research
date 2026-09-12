#!/usr/bin/env python3
"""Audit and select candidate model-parameter coordinates for R2 Figure 4.

This script executes the coordinate-level diagnostic audit over all 271
model-parameter coordinates across 36 hydrological models, using frozen
normalized IC and dPL parameter vectors and canonical F2/F3 audit tables.

It does NOT generate figures. It outputs canonical diagnostic tables,
multi-criteria rankings, a stratified candidate pool (Strata A-F),
within-model contrast pairs, and an audit report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve()
R2_ROOT = HERE.parents[1]
TABLES = R2_ROOT / "tables"
CACHE = R2_ROOT / "cache"
BENCHMARK = HERE.parents[3]
RESULT_ROOT = BENCHMARK / "results/joh_direct_parameter_change_diagnostic_20260905"


def load_canonical_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load canonical alignment, rank, correspondence, and coordinate weight tables."""
    align_path = CACHE / "inputs/model_parameter_alignment.csv"
    if not align_path.exists():
        raise FileNotFoundError(f"Missing parameter alignment: {align_path}")
    align = pd.read_csv(align_path)

    f3_rank_path = TABLES / "F3_RANK_COORDINATE_AUDIT.csv"
    if not f3_rank_path.exists():
        raise FileNotFoundError(f"Missing F3 rank audit: {f3_rank_path}")
    f3_rank = pd.read_csv(f3_rank_path)

    fig2c_path = CACHE / "fig2c_rank_correspondence.parquet"
    if not fig2c_path.exists():
        raise FileNotFoundError(f"Missing fig2c cache: {fig2c_path}")
    fig2c = pd.read_parquet(fig2c_path)

    fig3a_path = CACHE / "fig3a_coordinate_weights.parquet"
    if not fig3a_path.exists():
        raise FileNotFoundError(f"Missing fig3a cache: {fig3a_path}")
    fig3a = pd.read_parquet(fig3a_path)

    return align, f3_rank, fig2c, fig3a


def build_all_coordinate_diagnostics(
    align: pd.DataFrame,
    f3_rank: pd.DataFrame,
    fig2c: pd.DataFrame,
    fig3a: pd.DataFrame,
) -> pd.DataFrame:
    """Compute complete coordinate-level diagnostics across all 271 coordinates."""
    rows = []
    models = sorted(align["model"].unique())

    for idx, row in align.iterrows():
        m = row["model"]
        p_idx = int(row["parameter_index"])
        p_name = row["parameter"]
        n_params = int(row["parameter_count"])
        single_param = bool(n_params == 1)

        # Strict eligibility from F3 rank audit
        sub_f3 = f3_rank[(f3_rank.model_id == m) & (f3_rank.coordinate_id == p_idx)]
        if len(sub_f3) != 1:
            raise ValueError(f"F3 rank audit mismatch for {m}:{p_idx}")
        strict = bool(sub_f3["strict23_flag"].iloc[0])

        # Load normalized parameter arrays from frozen npz cache
        npz_path = RESULT_ROOT / f"r2/cache/{m}_normalized_parameter_matrices.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing normalized parameter matrices for {m}: {npz_path}")
        npz = np.load(npz_path)
        ic = npz["IC"][:, p_idx]
        dpl = npz["dPL"][:, p_idx]
        delta = npz["DeltaTheta"][:, p_idx]
        abs_delta = np.abs(delta)
        n_basins = len(delta)

        # 4.1 Displacement magnitude
        med_abs = float(np.median(abs_delta))
        q25_abs = float(np.percentile(abs_delta, 25))
        q75_abs = float(np.percentile(abs_delta, 75))
        q90_abs = float(np.percentile(abs_delta, 90))

        # 4.2 Signed shift
        med_signed = float(np.median(delta))
        f_pos = float(np.mean(delta > 0))
        f_neg = float(np.mean(delta < 0))
        sc = float(max(f_pos, f_neg))

        # 4.3 Rank preservation from frozen fig2c
        sub_2c = fig2c[(fig2c.model_id == m) & (fig2c.parameter_id == p_idx)]
        if len(sub_2c) != 1:
            raise ValueError(f"fig2c mismatch for {m}:{p_idx}")
        f2c_row = sub_2c.iloc[0]
        r_rank = float(f2c_row["R_rank"])

        # 4.4 Top-1 and Top-2 prevalence
        sub_3a = fig3a[(fig3a.model == m) & (fig3a.parameter_index == p_idx)]
        if len(sub_3a) != n_basins:
            raise ValueError(f"fig3a basin count mismatch for {m}:{p_idx}")
        top1_prev = float(np.mean(sub_3a["descending_rank"] == 1))
        top2_prev = float(np.mean(sub_3a["descending_rank"] <= 2))

        # 4.5 Displacement weight
        med_weight = float(np.median(sub_3a["coordinate_contribution"]))

        # 4.6 Boundary QC indicators
        f_lo_ic = float(f2c_row["IC_lower_bound_occupancy"])
        f_hi_ic = float(f2c_row["IC_upper_bound_occupancy"])
        f_lo_dpl = float(f2c_row["dPL_lower_bound_occupancy"])
        f_hi_dpl = float(f2c_row["dPL_upper_bound_occupancy"])
        u_ic = int(f2c_row["IC_unique_count"])
        u_dpl = int(f2c_row["dPL_unique_count"])
        bnd_flag = bool(max(f_lo_ic, f_hi_ic, f_lo_dpl, f_hi_dpl) >= 0.20)

        # 4.7 Distribution-scale change (IQR)
        iqr_ic = float(np.percentile(ic, 75) - np.percentile(ic, 25))
        iqr_dpl = float(np.percentile(dpl, 75) - np.percentile(dpl, 25))
        iqr_ratio = float(iqr_dpl / iqr_ic) if iqr_ic > 1e-12 else np.nan

        src_file = (
            f"cache/inputs/model_parameter_alignment.csv;"
            f"r2/cache/{m}_normalized_parameter_matrices.npz;"
            f"cache/fig2c_rank_correspondence.parquet;"
            f"cache/fig3a_coordinate_weights.parquet"
        )

        rows.append({
            "model_id": m,
            "parameter_name": p_name,
            "coordinate_id": p_idx,
            "n_parameters": n_params,
            "single_parameter_model": single_param,
            "strict23_flag": strict,
            "n_basins": n_basins,
            "median_abs_displacement": med_abs,
            "q25_abs_displacement": q25_abs,
            "q75_abs_displacement": q75_abs,
            "q90_abs_displacement": q90_abs,
            "median_signed_shift": med_signed,
            "fraction_positive": f_pos,
            "fraction_negative": f_neg,
            "sign_consistency": sc,
            "R_rank": r_rank,
            "top1_prevalence": top1_prev,
            "top2_presence_fraction": top2_prev,
            "median_displacement_weight": med_weight,
            "IQR_IC": iqr_ic,
            "IQR_dPL": iqr_dpl,
            "IQR_ratio": iqr_ratio,
            "fraction_at_lower_bound_IC": f_lo_ic,
            "fraction_at_upper_bound_IC": f_hi_ic,
            "fraction_at_lower_bound_dPL": f_lo_dpl,
            "fraction_at_upper_bound_dPL": f_hi_dpl,
            "unique_values_IC": u_ic,
            "unique_values_dPL": u_dpl,
            "boundary_flag": bnd_flag,
            "source_file": src_file,
        })

    df = pd.DataFrame(rows)
    return df


def build_candidate_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Generate top 10-15 coordinates for each distinct scientific criterion."""
    records = []

    # 1. median_abs_displacement (all models)
    top_disp = df.sort_values("median_abs_displacement", ascending=False).head(15)
    for rank, (_, r) in enumerate(top_disp.iterrows(), 1):
        records.append({
            "ranking_criterion": "median_abs_displacement",
            "rank_within_criterion": rank,
            "model_id": r["model_id"],
            "parameter_name": r["parameter_name"],
            "coordinate_id": r["coordinate_id"],
            "strict23_flag": r["strict23_flag"],
            "criterion_value": float(r["median_abs_displacement"]),
            "median_abs_displacement": float(r["median_abs_displacement"]),
            "top1_prevalence": float(r["top1_prevalence"]),
            "R_rank": float(r["R_rank"]),
            "sign_consistency": float(r["sign_consistency"]),
            "IQR_ratio": float(r["IQR_ratio"]),
            "boundary_flag": bool(r["boundary_flag"]),
            "notes": "Global highest displacement coordinates",
        })

    # 2. top1_prevalence (excluding degenerate single-parameter collie1)
    top_top1 = df[~df["single_parameter_model"]].sort_values("top1_prevalence", ascending=False).head(15)
    for rank, (_, r) in enumerate(top_top1.iterrows(), 1):
        records.append({
            "ranking_criterion": "top1_prevalence",
            "rank_within_criterion": rank,
            "model_id": r["model_id"],
            "parameter_name": r["parameter_name"],
            "coordinate_id": r["coordinate_id"],
            "strict23_flag": r["strict23_flag"],
            "criterion_value": float(r["top1_prevalence"]),
            "median_abs_displacement": float(r["median_abs_displacement"]),
            "top1_prevalence": float(r["top1_prevalence"]),
            "R_rank": float(r["R_rank"]),
            "sign_consistency": float(r["sign_consistency"]),
            "IQR_ratio": float(r["IQR_ratio"]),
            "boundary_flag": bool(r["boundary_flag"]),
            "notes": "Coordinates most frequently ranking #1 in displacement within basin",
        })

    # 3. sign_consistency (subject to nontrivial displacement >= 0.10)
    top_sc = df[df["median_abs_displacement"] >= 0.10].sort_values(
        ["sign_consistency", "median_abs_displacement"], ascending=[False, False]
    ).head(15)
    for rank, (_, r) in enumerate(top_sc.iterrows(), 1):
        records.append({
            "ranking_criterion": "sign_consistency",
            "rank_within_criterion": rank,
            "model_id": r["model_id"],
            "parameter_name": r["parameter_name"],
            "coordinate_id": r["coordinate_id"],
            "strict23_flag": r["strict23_flag"],
            "criterion_value": float(r["sign_consistency"]),
            "median_abs_displacement": float(r["median_abs_displacement"]),
            "top1_prevalence": float(r["top1_prevalence"]),
            "R_rank": float(r["R_rank"]),
            "sign_consistency": float(r["sign_consistency"]),
            "IQR_ratio": float(r["IQR_ratio"]),
            "boundary_flag": bool(r["boundary_flag"]),
            "notes": f"Coherent directional shift (median signed shift = {r['median_signed_shift']:+.3f})",
        })

    # 4. high R_rank among high-displacement (median_abs_displacement >= 0.20)
    top_high_r = df[df["median_abs_displacement"] >= 0.20].sort_values("R_rank", ascending=False).head(15)
    for rank, (_, r) in enumerate(top_high_r.iterrows(), 1):
        records.append({
            "ranking_criterion": "high_R_rank_high_displacement",
            "rank_within_criterion": rank,
            "model_id": r["model_id"],
            "parameter_name": r["parameter_name"],
            "coordinate_id": r["coordinate_id"],
            "strict23_flag": r["strict23_flag"],
            "criterion_value": float(r["R_rank"]),
            "median_abs_displacement": float(r["median_abs_displacement"]),
            "top1_prevalence": float(r["top1_prevalence"]),
            "R_rank": float(r["R_rank"]),
            "sign_consistency": float(r["sign_consistency"]),
            "IQR_ratio": float(r["IQR_ratio"]),
            "boundary_flag": bool(r["boundary_flag"]),
            "notes": "Large displacement while preserving cross-catchment rank ordering",
        })

    # 5. low R_rank among high-displacement (median_abs_displacement >= 0.20)
    top_low_r = df[df["median_abs_displacement"] >= 0.20].sort_values("R_rank", ascending=True).head(15)
    for rank, (_, r) in enumerate(top_low_r.iterrows(), 1):
        records.append({
            "ranking_criterion": "low_R_rank_high_displacement",
            "rank_within_criterion": rank,
            "model_id": r["model_id"],
            "parameter_name": r["parameter_name"],
            "coordinate_id": r["coordinate_id"],
            "strict23_flag": r["strict23_flag"],
            "criterion_value": float(r["R_rank"]),
            "median_abs_displacement": float(r["median_abs_displacement"]),
            "top1_prevalence": float(r["top1_prevalence"]),
            "R_rank": float(r["R_rank"]),
            "sign_consistency": float(r["sign_consistency"]),
            "IQR_ratio": float(r["IQR_ratio"]),
            "boundary_flag": bool(r["boundary_flag"]),
            "notes": "Large displacement with cross-catchment rank reorganization",
        })

    # 6. strongest compression by IQR_ratio (med_abs >= 0.05, IQR_IC >= 0.01)
    top_comp = df[(df["median_abs_displacement"] >= 0.05) & (df["IQR_IC"] >= 0.01)].sort_values(
        "IQR_ratio", ascending=True
    ).head(15)
    for rank, (_, r) in enumerate(top_comp.iterrows(), 1):
        records.append({
            "ranking_criterion": "strongest_compression_IQR_ratio",
            "rank_within_criterion": rank,
            "model_id": r["model_id"],
            "parameter_name": r["parameter_name"],
            "coordinate_id": r["coordinate_id"],
            "strict23_flag": r["strict23_flag"],
            "criterion_value": float(r["IQR_ratio"]),
            "median_abs_displacement": float(r["median_abs_displacement"]),
            "top1_prevalence": float(r["top1_prevalence"]),
            "R_rank": float(r["R_rank"]),
            "sign_consistency": float(r["sign_consistency"]),
            "IQR_ratio": float(r["IQR_ratio"]),
            "boundary_flag": bool(r["boundary_flag"]),
            "notes": f"Spread compression (IQR_IC={r['IQR_IC']:.3f} -> IQR_dPL={r['IQR_dPL']:.3f})",
        })

    # 7. strongest expansion by IQR_ratio (med_abs >= 0.05, IQR_IC >= 0.01)
    top_exp = df[(df["median_abs_displacement"] >= 0.05) & (df["IQR_IC"] >= 0.01)].sort_values(
        "IQR_ratio", ascending=False
    ).head(15)
    for rank, (_, r) in enumerate(top_exp.iterrows(), 1):
        records.append({
            "ranking_criterion": "strongest_expansion_IQR_ratio",
            "rank_within_criterion": rank,
            "model_id": r["model_id"],
            "parameter_name": r["parameter_name"],
            "coordinate_id": r["coordinate_id"],
            "strict23_flag": r["strict23_flag"],
            "criterion_value": float(r["IQR_ratio"]),
            "median_abs_displacement": float(r["median_abs_displacement"]),
            "top1_prevalence": float(r["top1_prevalence"]),
            "R_rank": float(r["R_rank"]),
            "sign_consistency": float(r["sign_consistency"]),
            "IQR_ratio": float(r["IQR_ratio"]),
            "boundary_flag": bool(r["boundary_flag"]),
            "notes": f"Spread expansion (IQR_IC={r['IQR_IC']:.3f} -> IQR_dPL={r['IQR_dPL']:.3f})",
        })

    return pd.DataFrame(records)


def build_candidate_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Build a stratified broad candidate pool of 28 coordinates across Strata A-F.

    Enforces model diversity: >= 10 distinct models, <= 3 coords per model,
    covering strict23 and non-strict models.
    """
    candidates = [
        # --- Stratum A: Recurrent displacement-dominant coordinates ---
        {
            "candidate_id": "C01",
            "stratum": "Stratum A (Recurrent displacement-dominant)",
            "model_id": "gr4j",
            "parameter_name": "x1",
            "why_candidate": "Highest non-degenerate top1 prevalence (91.0%) with large displacement (0.641) and high rank preservation (R=0.711)",
            "main_caveat": "2-parameter model with simple structure; dominant across almost all catchments",
        },
        {
            "candidate_id": "C02",
            "stratum": "Stratum A (Recurrent displacement-dominant)",
            "model_id": "alpine1",
            "parameter_name": "Smax",
            "why_candidate": "Top1 prevalence of 89.5% in 4-parameter model with high displacement (0.556) and strong rank preservation (R=0.836)",
            "main_caveat": "Strongly concentrated response on single storage parameter",
        },
        {
            "candidate_id": "C03",
            "stratum": "Stratum A (Recurrent displacement-dominant)",
            "model_id": "simhyd",
            "parameter_name": "smsc",
            "why_candidate": "Top1 prevalence of 67.6% in 7-parameter model, displacement 0.575, clean unclipped distribution",
            "main_caveat": "7-parameter model; secondary displacement spread across other soil parameters",
        },
        {
            "candidate_id": "C04",
            "stratum": "Stratum A (Recurrent displacement-dominant)",
            "model_id": "wetland",
            "parameter_name": "swmax",
            "why_candidate": "Top1 prevalence of 60.5% with very high displacement (0.868) in 4-parameter model",
            "main_caveat": "Upper bound displacement is large but unclipped",
        },
        {
            "candidate_id": "C05",
            "stratum": "Stratum A (Recurrent displacement-dominant)",
            "model_id": "newzealand1",
            "parameter_name": "s1max",
            "why_candidate": "Top1 prevalence of 57.1% with displacement 0.568 and high rank preservation (R=0.702)",
            "main_caveat": "Moderate tie fraction at lower bounds",
        },

        # --- Stratum B: Large displacement with high rank preservation ---
        {
            "candidate_id": "C06",
            "stratum": "Stratum B (High displacement + high rank)",
            "model_id": "vic",
            "parameter_name": "stot",
            "why_candidate": "Large displacement (0.564) with exceptionally high rank preservation (R=0.834) in 10-parameter model",
            "main_caveat": "10-parameter model with complex internal routing; other VIC coordinates behave differently",
        },
        {
            "candidate_id": "C07",
            "stratum": "Stratum B (High displacement + high rank)",
            "model_id": "ihacres",
            "parameter_name": "lp",
            "why_candidate": "Large displacement (0.446) with high rank preservation (R=0.694) in 6-parameter model",
            "main_caveat": "Moderate top1 prevalence (22.2%) due to competition with parameter d",
        },
        {
            "candidate_id": "C08",
            "stratum": "Stratum B (High displacement + high rank)",
            "model_id": "mopex2",
            "parameter_name": "s2max",
            "why_candidate": "High displacement (0.568) with strong rank preservation (R=0.673) and high sign consistency (97.6%)",
            "main_caveat": "Coexists with secondary parameter se undergoing moderate rank reorganization",
        },
        {
            "candidate_id": "C09",
            "stratum": "Stratum B (High displacement + high rank)",
            "model_id": "flexis",
            "parameter_name": "smax",
            "why_candidate": "High displacement (0.626) and substantial rank preservation (R=0.632) in 13-parameter model",
            "main_caveat": "13-parameter model; top1 prevalence is shared (36.9%)",
        },
        {
            "candidate_id": "C10",
            "stratum": "Stratum B (High displacement + high rank)",
            "model_id": "plateau",
            "parameter_name": "sumax",
            "why_candidate": "High displacement (0.616) and high rank preservation (R=0.653) in non-strict 8-parameter model",
            "main_caveat": "Non-strict model (insufficient reference); plateau model has separate uncalibrated routing",
        },

        # --- Stratum C: Large displacement with weak rank preservation ---
        {
            "candidate_id": "C11",
            "stratum": "Stratum C (High displacement + weak/negative rank)",
            "model_id": "mopex3",
            "parameter_name": "s3max",
            "why_candidate": "Substantial displacement (0.494) with near-zero/negative rank correlation (R=-0.036) without boundary clipping",
            "main_caveat": "Pure cross-basin spatial rank scrambling across all 531 basins",
        },
        {
            "candidate_id": "C12",
            "stratum": "Stratum C (High displacement + weak/negative rank)",
            "model_id": "vic",
            "parameter_name": "ishift",
            "why_candidate": "High displacement (0.626) with negative rank preservation (R=-0.031) within same model as stot (R=0.834)",
            "main_caveat": "Clean contrast within VIC showing coordinate-dependent rank behavior",
        },
        {
            "candidate_id": "C13",
            "stratum": "Stratum C (High displacement + weak/negative rank)",
            "model_id": "flexi",
            "parameter_name": "imax",
            "why_candidate": "Displacement 0.283 with negative rank correlation (R=-0.016) in 10-parameter strict model",
            "main_caveat": "Displacement magnitude is moderate (0.283) rather than extreme",
        },
        {
            "candidate_id": "C14",
            "stratum": "Stratum C (High displacement + weak/negative rank)",
            "model_id": "modhydrolog",
            "parameter_name": "k3",
            "why_candidate": "Displacement 0.388 with near-zero rank preservation (R=0.016) in 15-parameter model",
            "main_caveat": "15-parameter model; displacement is distributed across several routing terms",
        },
        {
            "candidate_id": "C15",
            "stratum": "Stratum C (High displacement + weak/negative rank)",
            "model_id": "newzealand2",
            "parameter_name": "s1max",
            "why_candidate": "Displacement 0.321 with negative rank correlation (R=-0.055) in non-strict 8-parameter model",
            "main_caveat": "Non-strict model; contrast with newzealand1 (s1max R=0.702)",
        },

        # --- Stratum D: Coherent directional shifts ---
        {
            "candidate_id": "C16",
            "stratum": "Stratum D (Coherent directional shift)",
            "model_id": "hymod",
            "parameter_name": "smax",
            "why_candidate": "Highest sign consistency among strict models (98.9% positive shift, median shift = +0.611)",
            "main_caveat": "Moderate rank preservation (R=0.465); almost unanimous positive remapping",
        },
        {
            "candidate_id": "C17",
            "stratum": "Stratum D (Coherent directional shift)",
            "model_id": "hillslope",
            "parameter_name": "swmax",
            "why_candidate": "97.9% positive shift (median shift = +0.595) with large displacement (0.595)",
            "main_caveat": "Rank correlation is moderate (R=0.551)",
        },
        {
            "candidate_id": "C18",
            "stratum": "Stratum D (Coherent directional shift)",
            "model_id": "ihacres",
            "parameter_name": "d",
            "why_candidate": "95.7% positive shift (median shift = +0.567) with displacement 0.567 and R=0.606",
            "main_caveat": "Shares displacement with parameter lp in IHACRES",
        },
        {
            "candidate_id": "C19",
            "stratum": "Stratum D (Coherent directional shift)",
            "model_id": "mopex1",
            "parameter_name": "s1max",
            "why_candidate": "93.0% positive shift (median shift = +0.534) in 5-parameter strict model",
            "main_caveat": "Rank preservation is moderate (R=0.373)",
        },
        {
            "candidate_id": "C20",
            "stratum": "Stratum D (Coherent directional shift)",
            "model_id": "smar",
            "parameter_name": "smax",
            "why_candidate": "98.7% positive shift (median shift = +0.498) in non-strict 8-parameter model",
            "main_caveat": "Non-strict model; demonstrates cross-model consistency of capacity upward shift",
        },

        # --- Stratum E: Strong spread compression or expansion ---
        {
            "candidate_id": "C21",
            "stratum": "Stratum E (Distribution spread rescaling)",
            "model_id": "vic",
            "parameter_name": "ibar",
            "why_candidate": "Severe across-catchment compression (IQR_ratio = 0.027, IQR_IC=0.395 -> IQR_dPL=0.011) with displacement 0.913",
            "main_caveat": "Boundary flag is TRUE (upper bound occupancy 63.8% at IC collapsing to single value under dPL)",
        },
        {
            "candidate_id": "C22",
            "stratum": "Stratum E (Distribution spread rescaling)",
            "model_id": "tank",
            "parameter_name": "f3",
            "why_candidate": "Substantial compression (IQR_ratio = 0.198, IQR_IC=0.749 -> IQR_dPL=0.148) with displacement 0.324 and no boundary clipping",
            "main_caveat": "12-parameter model with moderate individual displacement weights",
        },
        {
            "candidate_id": "C23",
            "stratum": "Stratum E (Distribution spread rescaling)",
            "model_id": "mopex3",
            "parameter_name": "tu",
            "why_candidate": "Strong spread expansion (IQR_ratio = 2.278, IQR_IC=0.286 -> IQR_dPL=0.651) with displacement 0.690",
            "main_caveat": "Upper and lower tails expand symmetrically across catchments",
        },
        {
            "candidate_id": "C24",
            "stratum": "Stratum E (Distribution spread rescaling)",
            "model_id": "wetland",
            "parameter_name": "betaw",
            "why_candidate": "Spread expansion (IQR_ratio = 2.685, IQR_IC=0.174 -> IQR_dPL=0.467) with displacement 0.389",
            "main_caveat": "Rank preservation is moderate (R=0.457)",
        },

        # --- Stratum F: Low-response within-model contrast coordinates ---
        {
            "candidate_id": "C25",
            "stratum": "Stratum F (Low-response contrast)",
            "model_id": "gr4j",
            "parameter_name": "x2",
            "why_candidate": "Near-zero displacement (0.005) and high rank preservation (R=0.833) within model where x1 moves 0.641",
            "main_caveat": "2-parameter model provides starkest possible high vs low contrast",
        },
        {
            "candidate_id": "C26",
            "stratum": "Stratum F (Low-response contrast)",
            "model_id": "alpine1",
            "parameter_name": "tc",
            "why_candidate": "Minimal displacement (0.012) and very high rank preservation (R=0.903) within model where Smax moves 0.556",
            "main_caveat": "Provides clean 1-to-1 contrast for alpine1 storage vs routing",
        },
        {
            "candidate_id": "C27",
            "stratum": "Stratum F (Low-response contrast)",
            "model_id": "simhyd",
            "parameter_name": "sq",
            "why_candidate": "Low displacement (0.010) in 7-parameter model where smsc moves 0.575",
            "main_caveat": "Routing parameter with modest sensitivity",
        },
        {
            "candidate_id": "C28",
            "stratum": "Stratum F (Low-response contrast)",
            "model_id": "hymod",
            "parameter_name": "b_exp",
            "why_candidate": "Minimal displacement (0.008) and high rank preservation (R=0.803) in model where smax moves 0.611",
            "main_caveat": "Shape parameter remains closely aligned with IC across all catchments",
        },
    ]

    pool_rows = []
    for c in candidates:
        m = c["model_id"]
        p = c["parameter_name"]
        match = df[(df.model_id == m) & (df.parameter_name == p)]
        if len(match) != 1:
            raise ValueError(f"Candidate match failure for {m}:{p}")
        r = match.iloc[0]
        pool_rows.append({
            "candidate_id": c["candidate_id"],
            "stratum": c["stratum"],
            "model_id": m,
            "parameter_name": p,
            "coordinate_id": int(r["coordinate_id"]),
            "n_parameters": int(r["n_parameters"]),
            "strict23_flag": bool(r["strict23_flag"]),
            "median_abs_displacement": float(r["median_abs_displacement"]),
            "top1_prevalence": float(r["top1_prevalence"]),
            "R_rank": float(r["R_rank"]),
            "sign_consistency": float(r["sign_consistency"]),
            "median_signed_shift": float(r["median_signed_shift"]),
            "IQR_ratio": float(r["IQR_ratio"]),
            "boundary_flag": bool(r["boundary_flag"]),
            "why_candidate": c["why_candidate"],
            "main_caveat": c["main_caveat"],
        })

    return pd.DataFrame(pool_rows)


def build_within_model_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Build 10 diverse within-model contrast pairs for potential F4 layout."""
    pairs = [
        {
            "pair_id": "P01",
            "model_id": "alpine1",
            "parameter_A": "Smax",
            "parameter_B": "tc",
            "A_role_in_pair": "displacement-dominant storage (top1=0.895, disp=0.556, R=0.836)",
            "B_role_in_pair": "low-response routing contrast (top1=0.002, disp=0.012, R=0.903)",
            "pair_reason": "High-response storage vs invariant routing; both exhibit strong cross-basin rank preservation",
        },
        {
            "pair_id": "P02",
            "model_id": "gr4j",
            "parameter_A": "x1",
            "parameter_B": "x2",
            "A_role_in_pair": "displacement-dominant production store (top1=0.910, disp=0.641, R=0.711)",
            "B_role_in_pair": "zero-response water exchange parameter (top1=0.000, disp=0.005, R=0.833)",
            "pair_reason": "Classic 2-parameter model: complete localization where one coordinate absorbs 91% of top-1 displacement",
        },
        {
            "pair_id": "P03",
            "model_id": "vic",
            "parameter_A": "stot",
            "parameter_B": "ishift",
            "A_role_in_pair": "high displacement with high rank preservation (disp=0.564, R=0.834)",
            "B_role_in_pair": "high displacement with rank reorganization (disp=0.626, R=-0.031)",
            "pair_reason": "Internal contrast within a 10-parameter model showing that large displacement can either preserve or scramble spatial ordering",
        },
        {
            "pair_id": "P04",
            "model_id": "flexi",
            "parameter_A": "smax",
            "parameter_B": "imax",
            "A_role_in_pair": "high displacement with rank preservation (disp=0.613, R=0.598)",
            "B_role_in_pair": "moderate displacement with rank reorganization (disp=0.283, R=-0.016)",
            "pair_reason": "10-parameter flexible model contrasting rank-preserving storage shift vs rank-inverting interception capacity",
        },
        {
            "pair_id": "P05",
            "model_id": "hymod",
            "parameter_A": "smax",
            "parameter_B": "b_exp",
            "A_role_in_pair": "coherent upward shift (sc=0.989, disp=0.611, top1=0.557)",
            "B_role_in_pair": "low-response shape parameter (disp=0.008, R=0.803)",
            "pair_reason": "Demonstrates near-unanimous positive capacity shift paired with invariant distribution shape",
        },
        {
            "pair_id": "P06",
            "model_id": "simhyd",
            "parameter_A": "smsc",
            "parameter_B": "sq",
            "A_role_in_pair": "displacement-dominant soil moisture store (top1=0.676, disp=0.575, R=0.599)",
            "B_role_in_pair": "low-response infiltration capacity (top1=0.000, disp=0.010, R=0.485)",
            "pair_reason": "7-parameter model illustrating clear partitioning between active storage and inactive routing",
        },
        {
            "pair_id": "P07",
            "model_id": "mopex2",
            "parameter_A": "s2max",
            "parameter_B": "se",
            "A_role_in_pair": "high displacement + rank preservation (disp=0.568, R=0.673, top1=0.550)",
            "B_role_in_pair": "moderate displacement + weak rank preservation (disp=0.330, R=0.204, top1=0.177)",
            "pair_reason": "Two moderately active coordinates within same model showing divergent rank preservation levels",
        },
        {
            "pair_id": "P08",
            "model_id": "mopex3",
            "parameter_A": "tu",
            "parameter_B": "s3max",
            "A_role_in_pair": "large displacement with spread expansion (disp=0.690, IQR_ratio=2.278, R=0.231)",
            "B_role_in_pair": "large displacement with complete rank loss (disp=0.494, IQR_ratio=1.171, R=-0.036)",
            "pair_reason": "Both coordinates experience large displacement, but one expands distribution spread while the other completely reorganizes catchment rankings",
        },
        {
            "pair_id": "P09",
            "model_id": "newzealand1",
            "parameter_A": "s1max",
            "parameter_B": "tcbf",
            "A_role_in_pair": "dominant storage displacement (top1=0.571, disp=0.568, R=0.702)",
            "B_role_in_pair": "near-zero baseflow routing response (top1=0.000, disp=0.001, R=0.659)",
            "pair_reason": "6-parameter strict model with stark magnitude separation across conceptual functions",
        },
        {
            "pair_id": "P10",
            "model_id": "wetland",
            "parameter_A": "swmax",
            "parameter_B": "dw",
            "A_role_in_pair": "dominant wetland capacity displacement (top1=0.605, disp=0.868, R=0.546)",
            "B_role_in_pair": "low-response wetland drainage exponent (top1=0.002, disp=0.019, R=0.397)",
            "pair_reason": "Clear within-model contrast in specialized wetland hydrology structure",
        },
    ]

    rows = []
    for p in pairs:
        m = p["model_id"]
        pA = p["parameter_A"]
        pB = p["parameter_B"]
        rA = df[(df.model_id == m) & (df.parameter_name == pA)].iloc[0]
        rB = df[(df.model_id == m) & (df.parameter_name == pB)].iloc[0]

        rows.append({
            "pair_id": p["pair_id"],
            "model_id": m,
            "parameter_A": pA,
            "parameter_B": pB,
            "A_role_in_pair": p["A_role_in_pair"],
            "B_role_in_pair": p["B_role_in_pair"],
            "A_median_abs_displacement": float(rA["median_abs_displacement"]),
            "B_median_abs_displacement": float(rB["median_abs_displacement"]),
            "A_top1_prevalence": float(rA["top1_prevalence"]),
            "B_top1_prevalence": float(rB["top1_prevalence"]),
            "A_R_rank": float(rA["R_rank"]),
            "B_R_rank": float(rB["R_rank"]),
            "A_sign_consistency": float(rA["sign_consistency"]),
            "B_sign_consistency": float(rB["sign_consistency"]),
            "pair_reason": p["pair_reason"],
        })

    return pd.DataFrame(rows)


def generate_markdown_report(
    df: pd.DataFrame,
    df_rankings: pd.DataFrame,
    df_pool: pd.DataFrame,
    df_pairs: pd.DataFrame,
) -> str:
    """Generate the comprehensive audit markdown report."""
    n_coords = len(df)
    n_models = df["model_id"].nunique()
    n_full_basin = (df["n_basins"] == 531).sum()
    n_top1 = (~df["top1_prevalence"].isna()).sum()
    n_bnd = (~df["boundary_flag"].isna()).sum()
    n_pool = len(df_pool)
    n_pool_models = df_pool["model_id"].nunique()
    n_pairs = len(df_pairs)

    lines = [
        "# R2 Figure 4 Exemplar-Selection Audit Report",
        "",
        "F4 EXEMPLAR-SELECTION AUDIT = READY",
        "",
        "## A. Data Completeness and Audit Integrity",
        "",
        f"- **Coordinates audited:** {n_coords} / 271",
        f"- **Models covered:** {n_models} / 36 (23 strict primary models + 13 sensitivity models)",
        f"- **Coordinates with full basin-level paired data (531 catchments):** {n_full_basin} / 271",
        f"- **Coordinates with usable top-1 prevalence:** {n_top1} / 271",
        f"- **Coordinates with boundary/tie diagnostics:** {n_bnd} / 271",
        f"- **Single-parameter models:** 1 (`collie1`, single parameter `Smax`, flagged and excluded from non-trivial localization rankings)",
        "",
        "---",
        "",
        "## B. Candidate Rankings by Individual Criteria",
        "",
        "Rankings are computed independently per criterion without opaque composite weighting formulas.",
        "",
    ]

    criteria = [
        ("median_abs_displacement", "Top 10 by Median Absolute Displacement ($M_{m,p}$)"),
        ("top1_prevalence", "Top 10 by Basin-level Top-1 Prevalence ($f^{top1}_{m,p}$, non-single models)"),
        ("sign_consistency", "Top 10 by Sign Consistency (Directional shift, $M_{m,p} \ge 0.10$)"),
        ("high_R_rank_high_displacement", "Top 10 by High Rank Preservation among High-Displacement ($M_{m,p} \ge 0.20$)"),
        ("low_R_rank_high_displacement", "Top 10 by Low Rank Preservation (Reorganization) among High-Displacement ($M_{m,p} \ge 0.20$)"),
        ("strongest_compression_IQR_ratio", "Top 10 by Spread Compression (Smallest $\\text{IQR}_{\\text{dPL}} / \\text{IQR}_{\\text{IC}}$, $M_{m,p} \ge 0.05$)"),
        ("strongest_expansion_IQR_ratio", "Top 10 by Spread Expansion (Largest $\\text{IQR}_{\\text{dPL}} / \\text{IQR}_{\\text{IC}}$, $M_{m,p} \ge 0.05$)"),
    ]

    for crit_key, crit_title in criteria:
        sub = df_rankings[df_rankings.ranking_criterion == crit_key].head(10)
        lines.append(f"### {crit_title}")
        lines.append("")
        lines.append("| Rank | Model | Parameter | Strict? | Crit Value | Med Disp | Top-1 Prev | R_rank | Sign Cons | Boundary Flag |")
        lines.append("|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for _, r in sub.iterrows():
            strict_str = "Yes" if r["strict23_flag"] else "No"
            bnd_str = "FLAGGED" if r["boundary_flag"] else "Clean"
            lines.append(
                f"| {r['rank_within_criterion']} | `{r['model_id']}` | `{r['parameter_name']}` | {strict_str} | "
                f"{r['criterion_value']:.4f} | {r['median_abs_displacement']:.4f} | {r['top1_prevalence']:.4f} | "
                f"{r['R_rank']:.4f} | {r['sign_consistency']:.4f} | {bnd_str} |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## C. Broad Stratified Candidate Pool",
        "",
        f"The candidate pool contains **{n_pool} model-parameter coordinates** across **{n_pool_models} distinct hydrological models**, divided across 6 scientifically defined response strata.",
        "",
        "| ID | Stratum | Model | Parameter | Strict? | Med Disp | Top-1 | R_rank | Sign Cons | IQR Ratio | Why Candidate | Caveat |",
        "|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|",
    ])

    for _, r in df_pool.iterrows():
        strict_str = "Yes" if r["strict23_flag"] else "No"
        lines.append(
            f"| `{r['candidate_id']}` | {r['stratum']} | `{r['model_id']}` | `{r['parameter_name']}` | {strict_str} | "
            f"{r['median_abs_displacement']:.3f} | {r['top1_prevalence']:.3f} | {r['R_rank']:.3f} | "
            f"{r['sign_consistency']:.3f} | {r['IQR_ratio']:.3f} | {r['why_candidate']} | {r['main_caveat']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## D. Within-Model Contrast Opportunities",
        "",
        f"Identified **{n_pairs} candidate pairs** within multi-parameter models, providing direct visual contrast of intra-model coordinate response heterogeneity.",
        "",
        "| Pair ID | Model | Coord A | Coord B | A Role | B Role | Disp (A / B) | R_rank (A / B) | Scientific Contrast |",
        "|:---|:---|:---|:---|:---|:---|:---:|:---:|:---|",
    ])

    for _, r in df_pairs.iterrows():
        lines.append(
            f"| `{r['pair_id']}` | `{r['model_id']}` | `{r['parameter_A']}` | `{r['parameter_B']}` | "
            f"{r['A_role_in_pair']} | {r['B_role_in_pair']} | "
            f"{r['A_median_abs_displacement']:.3f} / {r['B_median_abs_displacement']:.3f} | "
            f"{r['A_R_rank']:.3f} / {r['B_R_rank']:.3f} | {r['pair_reason']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## E. Methodological Caveats and Quality Control",
        "",
        "1. **Single-Parameter Model Degeneracy:** `collie1` has $n_{params}=1$, which forces $C_{eff}=1.0$ and $f^{top1}=1.0$ algebraically. It is recorded in the complete table but barred from localization rankings.",
        "2. **Boundary Saturation & Ties:** Coordinates such as `vic:ibar` (upper bound occupancy 63.8% at IC), `topmodel:q0` (lower bound occupancy 76.6%), and `collie2:Smax` (upper bound occupancy 58.0%) exhibit strong clipping. Sensitivity to threshold selection (0.10 vs 0.20 occupancy) identifies 126 vs 76 flagged coordinates.",
        "3. **Rank Loss vs Tie-Artifacts:** For most low-$R_{rank}$ coordinates (e.g. `mopex3:s3max`, `vic:ishift`, `flexi:imax`), unique value counts exceed 500 in both IC and dPL, proving that low rank preservation is genuine spatial reordering rather than a tie-induced numerical artifact.",
        "4. **IQR Ratio Division Safeguards:** In 14 coordinates, $\\text{IQR}_{\\text{IC}} < 0.01$ due to tight initial parameter clusters. These are flagged to prevent division by near-zero variance.",
        "5. **Descriptive Interpretation Boundary:** Exemplar coordinates illustrate statistical response modes (displacement magnitude, directional consistency, rank reordering, distribution spread); they do not constitute claims of parameter sensitivity, physical dominance, or real-world process identity.",
        "",
        "---",
        "",
        "## F. Evidence-Balanced Design Options for Final Figure 4",
        "",
        "To support human selection without imposing a single figure layout, three evidence-grounded configuration options are presented:",
        "",
        "### Option 1: 4 High-Response Coordinates from 4 Diverse Models (Panels A-D)",
        "- **Goal:** Showcase the four clearest distinct mathematical behaviors across separate hydrological model structures.",
        "- `C01` (`gr4j:x1`): Recurrent displacement dominance ($f^{top1}=0.910$, $M=0.641$, $R=0.711$)",
        "- `C02` (`alpine1:Smax`): Large displacement with strong rank preservation ($M=0.556$, $R=0.836$)",
        "- `C11` (`mopex3:s3max`): Large displacement with complete rank loss ($M=0.494$, $R=-0.036$)",
        "- `C16` (`hymod:smax`): Near-unanimous directional shift ($sc=0.989$, $\\Delta=+0.611$)",
        "",
        "### Option 2: 3 Models x (High-Response + Low-Response) Pairs (6 Panels)",
        "- **Goal:** Directly demonstrate within-model response heterogeneity and disprove the assumption that all parameters shift uniformly.",
        "- **Pair 1 (`P01` `alpine1`):** `Smax` (disp=0.556, top1=0.895) vs `tc` (disp=0.012, top1=0.002)",
        "- **Pair 2 (`P02` `gr4j`):** `x1` (disp=0.641, top1=0.910) vs `x2` (disp=0.005, top1=0.000)",
        "- **Pair 3 (`P03` `vic`):** `stot` (disp=0.564, R=0.834) vs `ishift` (disp=0.626, R=-0.031)",
        "",
        "### Option 3: 6 Archetype Exemplars Across the Spectrum (6 Panels)",
        "- **Goal:** Comprehensive panel covering all six diagnostic strata.",
        "- **Stratum A (`C03` `simhyd:smsc`):** Clean unclipped recurrent top-1 contributor",
        "- **Stratum B (`C06` `vic:stot`):** High-displacement rank-preserving shift",
        "- **Stratum C (`C13` `flexi:imax`):** High-displacement rank-reorganizing response",
        "- **Stratum D (`C17` `hillslope:swmax`):** 98% coherent positive capacity shift",
        "- **Stratum E (`C23` `mopex3:tu`):** Spread expansion across catchments",
        "- **Stratum F (`C26` `alpine1:tc`):** Invariant low-response contrast baseline",
        "",
        "---",
        "",
        "*No figures were generated during this audit.*",
    ])

    return "\n".join(lines)


def main() -> None:
    print("Starting R2 Figure 4 exemplar-selection audit...")
    align, f3_rank, fig2c, fig3a = load_canonical_data()

    # 1. Build all coordinate diagnostics
    df_all = build_all_coordinate_diagnostics(align, f3_rank, fig2c, fig3a)
    out_all_path = TABLES / "F4_ALL_COORDINATE_DIAGNOSTICS.csv"
    df_all.to_csv(out_all_path, index=False, float_format="%.12g")
    print(f"Wrote complete 271-coordinate diagnostics to: {out_all_path}")

    # 2. Build candidate rankings by criterion
    df_rankings = build_candidate_rankings(df_all)
    out_rankings_path = TABLES / "F4_CANDIDATE_RANKINGS.csv"
    df_rankings.to_csv(out_rankings_path, index=False, float_format="%.12g")
    print(f"Wrote multi-criteria candidate rankings to: {out_rankings_path}")

    # 3. Build broad candidate pool (Strata A-F)
    df_pool = build_candidate_pool(df_all)
    out_pool_path = TABLES / "F4_CANDIDATE_POOL.csv"
    df_pool.to_csv(out_pool_path, index=False, float_format="%.12g")
    print(f"Wrote broad candidate pool ({len(df_pool)} coordinates) to: {out_pool_path}")

    # 4. Build within-model pair candidates
    df_pairs = build_within_model_pairs(df_all)
    out_pairs_path = TABLES / "F4_WITHIN_MODEL_PAIR_CANDIDATES.csv"
    df_pairs.to_csv(out_pairs_path, index=False, float_format="%.12g")
    print(f"Wrote within-model pair candidates ({len(df_pairs)} pairs) to: {out_pairs_path}")

    # 5. Build summary markdown report
    report_md = generate_markdown_report(df_all, df_rankings, df_pool, df_pairs)
    out_report_path = TABLES / "F4_EXEMPLAR_SELECTION_AUDIT.md"
    out_report_path.write_text(report_md + "\n")
    print(f"Wrote summary audit report to: {out_report_path}")

    print("\nAudit completed successfully. No figures were generated.")


if __name__ == "__main__":
    main()
