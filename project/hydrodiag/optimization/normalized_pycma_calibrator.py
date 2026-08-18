"""Normalized-space CMA-ES calibrator with GPU batch evaluation.

Manages independent CMA-ES instances per basin in normalized [0,1] space,
with composite early stopping, checkpoint/resume, and numerical failure handling.
"""

from __future__ import annotations

import csv
import os
import pickle
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cma
import numpy as np
import torch
from models.base import BaseHydrologicalModel


def compute_kge(sim: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim) & (obs >= 0) & (sim >= 0)
    if mask.sum() < 30:
        return -999.0
    s, o = sim[mask].astype(np.float64), obs[mask].astype(np.float64)
    o_std = o.std()
    if o_std < 1e-10:
        return -999.0
    r = np.corrcoef(s, o)[0, 1]
    alpha = s.std() / o_std
    beta = s.mean() / o.mean()
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


STOP_COMPOSITE_CONVERGED = "COMPOSITE_CONVERGED"
STOP_MAX_GENERATIONS = "MAX_GENERATIONS_UNCONVERGED"
STOP_INVALID_FITNESS = "INVALID_FITNESS"
STOP_NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
STOP_PYCMA_INTERNAL = "PYCMA_INTERNAL_STOP"
STOP_INTERRUPTED = "INTERRUPTED"
STOP_UNKNOWN = "UNKNOWN_FAILURE"


@dataclass
class BasinTrace:
    basin_id: str
    basin_index: int
    snow_class: str
    restart_type: str
    restart_id: int
    generation: int
    population_size: int
    cumulative_evaluations: int
    generation_best_train_kge: float
    global_best_train_kge: float
    generation_mean_train_kge: float
    generation_worst_train_kge: float
    generation_kge_range: float
    history_kge_range: float
    sigma: float
    max_coordinate_std: float
    boundary_candidate_count: int
    invalid_candidate_count: int
    stop_trigger_history: bool
    stop_trigger_std: bool
    stop_reason: str


@dataclass
class RestartSummary:
    basin_id: str
    basin_index: int
    snow_class: str
    restart_type: str
    restart_id: int
    best_train_kge: float
    best_params_norm: list
    best_params_physical: list
    best_generation: int
    stop_reason: str
    total_evaluations: int
    total_generations: int
    total_invalid_candidates: int
    final_sigma: float
    final_max_coordinate_std: float
    final_history_kge_range: float
    sobol_seed: int
    cma_seed: int
    initial_center: list
    initial_sigma: float


class NormalizedCMAESCalibrator:
    def __init__(
        self,
        model_cls: type[BaseHydrologicalModel],
        param_specs: dict[str, dict[str, Any]],
        forcing_data: dict[str, np.ndarray],
        obs_train: np.ndarray,
        obs_val: np.ndarray,
        basin_ids: list[str],
        frac_snow: np.ndarray,
        config: dict,
        model_config: dict,
        output_dir: Path,
        device: torch.device,
    ):
        self.model_cls = model_cls
        self.param_specs = param_specs
        self.forcing_data = forcing_data
        self.obs_train = obs_train
        self.obs_val = obs_val
        self.basin_ids = basin_ids
        self.frac_snow = frac_snow
        self.config = config
        self.model_config = model_config
        self.output_dir = output_dir
        self.device = device

        self.param_names = list(param_specs.keys())
        self.D = len(self.param_names)
        self.N = len(basin_ids)
        self.P = model_config["population"]

        self.lower = np.array(
            [param_specs[n]["lower"] for n in self.param_names], dtype=np.float64
        )
        self.upper = np.array(
            [param_specs[n]["upper"] for n in self.param_names], dtype=np.float64
        )
        self.param_range = self.upper - self.lower

        self.warmup_days = config["time_periods"]["warmup_days"]

        self.cma_config = config["cmaes"]
        self.model_offset = model_config["model_offset"]
        self.model = None

        self.snow_class = np.where(
            frac_snow >= config["data"]["snow_threshold"], "snow", "non_snow"
        )

    def _ensure_model(self):
        if self.model is None:
            self.model = self.model_cls().to(device=self.device, dtype=torch.float32)
            self.model.eval()

    def _normalize(self, x_phys: np.ndarray) -> np.ndarray:
        return (x_phys - self.lower) / self.param_range

    def _denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        return self.lower + x_norm * self.param_range

    def _params_to_dict(
        self, theta: np.ndarray, batch: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        return {
            n: torch.from_numpy(theta[:, i]).float().to(device)
            for i, n in enumerate(self.param_names)
        }

    def _make_cma_opts(self, seed: int, sigma0: float) -> dict:
        disabled = dict(self.cma_config["disabled_stop_conditions"])
        return {
            "popsize": self.P,
            "seed": seed,
            "verbose": -9,
            "bounds": [0.0, 1.0],
            **disabled,
        }

    def _make_sobol_center(self, sobol_seed: int) -> np.ndarray:
        engine = torch.quasirandom.SobolEngine(
            dimension=self.D, scramble=True, seed=sobol_seed
        )
        pts = engine.draw(self.N).cpu().numpy().astype(np.float64)
        eps = self.cma_config["boundary_near_threshold"]
        pts = np.clip(pts, eps, 1.0 - eps)
        return pts

    def _check_early_stop(
        self,
        best_kge_history: list,
        es,
        gen: int,
        min_generations: int,
        tol_history_kge: float,
        tol_coord_std: float,
        history_window: int,
    ) -> tuple:
        info = {"history_kge_range": np.nan, "max_coordinate_std": np.nan}
        if gen < min_generations:
            return False, "", info

        if len(best_kge_history) >= history_window:
            recent = best_kge_history[-history_window:]
            info["history_kge_range"] = float(max(recent) - min(recent))

        try:
            stds = np.asarray(es.stds, dtype=np.float64)
            info["max_coordinate_std"] = float(np.max(stds))
        except Exception:
            info["max_coordinate_std"] = np.nan

        hist_ok = (
            not np.isnan(info["history_kge_range"])
            and info["history_kge_range"] <= tol_history_kge
        )
        std_ok = (
            not np.isnan(info["max_coordinate_std"])
            and info["max_coordinate_std"] <= tol_coord_std
        )

        if hist_ok and std_ok:
            return True, STOP_COMPOSITE_CONVERGED, info
        return False, "", info

    def _gpu_batch_eval(
        self,
        theta_norm: np.ndarray,
        fc_rep: dict[str, np.ndarray],
        active_indices: list[int],
    ) -> np.ndarray:
        """Single GPU forward pass for all candidates. Returns KGE array."""
        self._ensure_model()
        n_total = len(theta_norm)
        theta_phys = self._denormalize(theta_norm.astype(np.float64)).astype(np.float32)

        fc = {}
        for key in ["precip", "pet", "temp"]:
            fc[key] = torch.from_numpy(fc_rep[key].astype(np.float32)).to(self.device)
        pdict = self._params_to_dict(theta_phys, n_total, self.device)

        with torch.no_grad():
            q_all, _ = self.model(forcings=fc, params=pdict)
        q_all = q_all.cpu().numpy()

        kge_vals = np.full(n_total, -999.0, dtype=np.float32)
        for i, bi in enumerate(active_indices):
            obs = self.obs_train[bi]
            for p in range(self.P):
                idx = i * self.P + p
                sim = q_all[idx, self.warmup_days :]
                kge_vals[idx] = compute_kge(sim, obs)

        del fc, pdict, q_all
        return kge_vals

    def run_restart(
        self,
        restart_id: int,
        basin_indices: list[int],
        restart_type: str,
        sobol_seed: int,
        sigma0: float,
        max_generations: int,
        min_generations: int,
        tol_history_kge: float,
        tol_coord_std: float,
        history_window: int,
        checkpoint_dir: Path,
        trace_dir: Path,
    ) -> dict:
        """Run one CMA-ES restart for specified basins."""
        ba = basin_indices
        if len(ba) == 0:
            return {}

        sobol_centers = self._make_sobol_center(sobol_seed)

        es_dict = {}
        best_kge = {bi: -999.0 for bi in ba}
        best_x_norm = {bi: np.zeros(self.D) for bi in ba}
        best_gen = {bi: 0 for bi in ba}
        kge_hist = {bi: [] for bi in ba}
        active = {bi: True for bi in ba}
        stop_reason = {bi: "" for bi in ba}
        total_evals = {bi: 0 for bi in ba}
        invalid_count = {bi: 0 for bi in ba}
        traces = {bi: [] for bi in ba}
        final_sigma = {bi: sigma0 for bi in ba}
        final_max_std = {bi: np.nan for bi in ba}

        for bi in ba:
            x0 = sobol_centers[bi].tolist()
            seed = self.model_offset * 1_000_000 + restart_id * 10_000 + bi
            opts = self._make_cma_opts(seed, sigma0)
            es_dict[bi] = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            best_x_norm[bi] = np.array(x0, dtype=np.float64)

        self._ensure_model()
        chunk_size = None

        gen_range = range(max_generations)
        done_early = False
        for gen in gen_range:
            cur = [bi for bi in ba if active[bi]]
            if not cur:
                done_early = True
                break

            n_cur = len(cur)
            theta_flat = np.zeros((n_cur * self.P, self.D), dtype=np.float64)

            for i, bi in enumerate(cur):
                sols = np.array(es_dict[bi].ask(), dtype=np.float64)
                theta_flat[i * self.P : (i + 1) * self.P] = sols

            fc_rep = {}
            for key in ["precip", "pet", "temp"]:
                fc_rep[key] = np.concatenate(
                    [np.tile(self.forcing_data[key][bi], (self.P, 1)) for bi in cur],
                    axis=0,
                ).astype(np.float32)

            kge_vals = None
            try:
                kge_vals = self._gpu_batch_eval(theta_flat, fc_rep, cur)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                chunk_size = max(1, (chunk_size or n_cur) // 2)

            if kge_vals is None and chunk_size is not None and chunk_size < n_cur:
                kge_vals = np.full(n_cur * self.P, -999.0, dtype=np.float32)
                for cs in range(0, n_cur, chunk_size):
                    ce = min(cs + chunk_size, n_cur)
                    cb = cur[cs:ce]
                    sl = slice(cs * self.P, ce * self.P)
                    fc_chunk = {}
                    for key in ["precip", "pet", "temp"]:
                        fc_chunk[key] = np.concatenate(
                            [
                                np.tile(self.forcing_data[key][bi], (self.P, 1))
                                for bi in cb
                            ],
                            axis=0,
                        ).astype(np.float32)
                    try:
                        kge_vals[sl] = self._gpu_batch_eval(
                            theta_flat[sl], fc_chunk, cb
                        )
                    except torch.cuda.OutOfMemoryError:
                        chunk_size = max(1, chunk_size // 2)
                        torch.cuda.empty_cache()
            elif kge_vals is None:
                kge_vals = np.full(n_cur * self.P, -999.0, dtype=np.float32)

            kge_matrix = kge_vals.reshape(n_cur, self.P)
            eps = self.cma_config["boundary_near_threshold"]
            boundary_count = int(
                np.sum((theta_flat <= eps) | (theta_flat >= 1.0 - eps))
            )

            ctrl_info = {}
            for i, bi in enumerate(cur):
                if not active[bi]:
                    continue

                basin_kges = kge_matrix[i]
                invalid_mask = ~np.isfinite(basin_kges) | (basin_kges <= -998.0)
                n_invalid = int(invalid_mask.sum())
                invalid_count[bi] += n_invalid

                valid_kges = basin_kges[~invalid_mask]
                gen_best = float(np.max(valid_kges)) if len(valid_kges) > 0 else -999.0

                if n_invalid == self.P:
                    active[bi] = False
                    stop_reason[bi] = STOP_INVALID_FITNESS
                    total_evals[bi] += self.P
                    continue

                if gen_best > best_kge[bi]:
                    best_kge[bi] = gen_best
                    best_idx = int(np.argmax(basin_kges))
                    best_x_norm[bi] = theta_flat[i * self.P + best_idx].copy()
                    best_gen[bi] = gen

                kge_hist[bi].append(best_kge[bi])

                penalty = self.cma_config["failure_penalty"]
                fitness = np.where(
                    invalid_mask, penalty, -basin_kges.astype(np.float64)
                )

                try:
                    sols_tell = [
                        (float(theta_flat[i * self.P + p, d]) for d in range(self.D))
                        for p in range(self.P)
                    ]
                    es_dict[bi].tell(sols_tell, fitness.tolist())
                except Exception:
                    active[bi] = False
                    stop_reason[bi] = STOP_NUMERICAL_FAILURE
                    total_evals[bi] += self.P
                    continue

                total_evals[bi] += self.P

                gen_mean = float(np.mean(valid_kges)) if len(valid_kges) > 0 else np.nan
                gen_worst = float(np.min(valid_kges)) if len(valid_kges) > 0 else np.nan
                gen_range = (
                    float(np.max(valid_kges) - np.min(valid_kges))
                    if len(valid_kges) > 1
                    else 0.0
                )

                try:
                    sigma_val = float(es_dict[bi].sigma)
                    final_sigma[bi] = sigma_val
                    max_std = float(
                        np.max(np.asarray(es_dict[bi].stds, dtype=np.float64))
                    )
                    final_max_std[bi] = max_std
                except Exception:
                    sigma_val = np.nan
                    max_std = np.nan

                ctrl_stop, ctrl_reason, stop_info = self._check_early_stop(
                    kge_hist[bi],
                    es_dict[bi],
                    gen + 1,
                    min_generations,
                    tol_history_kge,
                    tol_coord_std,
                    history_window,
                )
                ctrl_info[bi] = stop_info

                if ctrl_stop:
                    active[bi] = False
                    stop_reason[bi] = STOP_COMPOSITE_CONVERGED
                elif gen + 1 >= max_generations:
                    active[bi] = False
                    stop_reason[bi] = STOP_MAX_GENERATIONS
                else:
                    es_stop = es_dict[bi].stop()
                    if es_stop:
                        active[bi] = False
                        stop_reason[bi] = STOP_PYCMA_INTERNAL + ":" + str(es_stop)

                si = ctrl_info.get(bi, stop_info)
                trace = BasinTrace(
                    basin_id=self.basin_ids[bi],
                    basin_index=bi,
                    snow_class=self.snow_class[bi],
                    restart_type=restart_type,
                    restart_id=restart_id,
                    generation=gen + 1,
                    population_size=self.P,
                    cumulative_evaluations=total_evals[bi],
                    generation_best_train_kge=gen_best,
                    global_best_train_kge=best_kge[bi],
                    generation_mean_train_kge=gen_mean,
                    generation_worst_train_kge=gen_worst,
                    generation_kge_range=gen_range,
                    history_kge_range=si.get("history_kge_range", np.nan),
                    sigma=sigma_val,
                    max_coordinate_std=si.get("max_coordinate_std", np.nan),
                    boundary_candidate_count=boundary_count,
                    invalid_candidate_count=n_invalid,
                    stop_trigger_history=(
                        not np.isnan(si.get("history_kge_range", np.nan))
                        and si.get("history_kge_range", np.nan) <= tol_history_kge
                    ),
                    stop_trigger_std=(
                        not np.isnan(si.get("max_coordinate_std", np.nan))
                        and si.get("max_coordinate_std", np.nan) <= tol_coord_std
                    ),
                    stop_reason=stop_reason.get(bi, ""),
                )
                traces[bi].append(trace)

            done = sum(1 for bi in ba if not active[bi])
            if gen % 20 == 0 or gen + 1 >= max_generations:
                act_kges = [best_kge[bi] for bi in ba if active[bi]]
                med = np.median(act_kges) if act_kges else 0.0
                print(
                    f"    gen {gen + 1:4d}: active={len(ba) - done}/{len(ba)} done={done} "
                    f"best_med={med:.4f}",
                    flush=True,
                )

            if (gen + 1) % self.cma_config["checkpoint_interval_generations"] == 0:
                self._save_checkpoint(
                    restart_id,
                    gen + 1,
                    ba,
                    es_dict,
                    best_kge,
                    best_x_norm,
                    best_gen,
                    kge_hist,
                    active,
                    stop_reason,
                    total_evals,
                    invalid_count,
                    checkpoint_dir,
                )

        summaries = {}
        for bi in ba:
            ht = len(traces[bi])
            summaries[bi] = RestartSummary(
                basin_id=self.basin_ids[bi],
                basin_index=bi,
                snow_class=self.snow_class[bi],
                restart_type=restart_type,
                restart_id=restart_id,
                best_train_kge=best_kge[bi],
                best_params_norm=best_x_norm[bi].tolist(),
                best_params_physical=self._denormalize(best_x_norm[bi]).tolist(),
                best_generation=best_gen[bi],
                stop_reason=stop_reason[bi],
                total_evaluations=total_evals[bi],
                total_generations=ht,
                total_invalid_candidates=invalid_count[bi],
                final_sigma=final_sigma[bi],
                final_max_coordinate_std=final_max_std.get(bi, np.nan),
                final_history_kge_range=(
                    traces[bi][-1].history_kge_range if ht > 0 else np.nan
                ),
                sobol_seed=sobol_seed,
                cma_seed=self.model_offset * 1_000_000 + restart_id * 10_000 + bi,
                initial_center=sobol_centers[bi].tolist(),
                initial_sigma=sigma0,
            )

        self._save_traces(traces, trace_dir, restart_id)
        self._save_summaries(summaries, checkpoint_dir, restart_id)

        return summaries

    def _save_checkpoint(
        self,
        restart_id,
        gen,
        basin_indices,
        es_dict,
        best_kge,
        best_x_norm,
        best_gen,
        kge_hist,
        active,
        stop_reason,
        total_evals,
        invalid_count,
        checkpoint_dir,
    ):
        ckpt = {
            "restart_id": restart_id,
            "generation": gen,
            "basin_indices": basin_indices,
            "cma_states": {},
            "best_kge": {str(k): v for k, v in best_kge.items()},
            "best_x_norm": {str(k): v.tolist() for k, v in best_x_norm.items()},
            "best_gen": best_gen,
            "kge_hist": {str(k): v for k, v in kge_hist.items()},
            "active": {str(k): v for k, v in active.items()},
            "stop_reason": stop_reason,
            "total_evals": total_evals,
            "invalid_count": invalid_count,
        }
        for bi in basin_indices:
            try:
                es = es_dict[bi]
                ckpt["cma_states"][str(bi)] = {
                    "mean": es.mean.tolist(),
                    "sigma": float(es.sigma),
                    "C": es.C.tolist(),
                    "countiter": int(es.countiter),
                }
            except Exception:
                ckpt["cma_states"][str(bi)] = None

        ckpt_path = checkpoint_dir / f"ckpt_r{restart_id:02d}_g{gen:04d}.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump(ckpt, f)

    def _save_traces(self, traces, trace_dir, restart_id):
        all_t = []
        for bi in sorted(traces.keys()):
            all_t.extend(traces[bi])
        if not all_t:
            return
        arrays = {}
        float_fields = [
            "generation_best_train_kge",
            "global_best_train_kge",
            "generation_mean_train_kge",
            "generation_worst_train_kge",
            "generation_kge_range",
            "history_kge_range",
            "sigma",
            "max_coordinate_std",
        ]
        int_fields = [
            "generation",
            "population_size",
            "cumulative_evaluations",
            "boundary_candidate_count",
            "invalid_candidate_count",
        ]
        bool_fields = ["stop_trigger_history", "stop_trigger_std"]
        for fn in float_fields:
            arrays[fn] = np.array([getattr(t, fn) for t in all_t], dtype=np.float32)
        for fn in int_fields:
            arrays[fn] = np.array([getattr(t, fn) for t in all_t], dtype=np.int32)
        for fn in bool_fields:
            arrays[fn] = np.array([getattr(t, fn) for t in all_t], dtype=np.bool_)
        np.savez_compressed(trace_dir / f"traces_r{restart_id:02d}.npz", **arrays)

    def _save_summaries(self, summaries, checkpoint_dir, restart_id):
        if not summaries:
            return
        p = checkpoint_dir / f"summary_r{restart_id:02d}.csv"
        fields = [f.name for f in RestartSummary.__dataclass_fields__.values()]
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for s in summaries.values():
                row = {k: getattr(s, k) for k in fields}
                w.writerow(row)
