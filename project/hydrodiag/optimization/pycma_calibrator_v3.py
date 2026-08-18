"""pycma CMA-ES calibrator v3 — FP32 forward, FP64 metric, pycma native.

Independent per-basin pycma instances. GPU batches candidate evaluation.
Supports KGE(Q) and KGE(1/Q) objectives with proper epsilon handling.
"""

from __future__ import annotations

import csv
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cma
import numpy as np
import torch


def compute_kge_fp64(
    sim_fp32: np.ndarray, obs_fp32: np.ndarray, min_samples: int = 30
) -> tuple[float, dict]:
    """KGE with FP64 accumulation. Returns (kge, components_dict)."""
    mask = (
        np.isfinite(obs_fp32)
        & np.isfinite(sim_fp32)
        & (obs_fp32 >= 0)
        & (sim_fp32 >= 0)
    )
    n_valid = int(mask.sum())
    if n_valid < min_samples:
        return -999.0, {
            "r": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "n_valid": n_valid,
        }

    s = sim_fp32[mask].astype(np.float64)
    o = obs_fp32[mask].astype(np.float64)

    o_std = o.std()
    if o_std < 1e-10:
        return -999.0, {
            "r": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "n_valid": n_valid,
        }

    r = np.corrcoef(s, o)[0, 1]
    alpha = s.std() / o_std
    beta = s.mean() / o.mean()
    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    return kge, {
        "r": float(r),
        "alpha": float(alpha),
        "beta": float(beta),
        "n_valid": n_valid,
    }


def compute_kge_invq(
    sim_fp32: np.ndarray, obs_fp32: np.ndarray, epsilon: float, min_samples: int = 30
) -> tuple[float, dict]:
    """KGE on 1/Q with basin-specific epsilon. FP64 accumulation."""
    mask = (
        np.isfinite(obs_fp32)
        & np.isfinite(sim_fp32)
        & (obs_fp32 >= 0)
        & (sim_fp32 >= 0)
    )
    n_valid = int(mask.sum())
    if n_valid < min_samples:
        return -999.0, {
            "r": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "n_valid": n_valid,
        }

    eps = max(epsilon, 1e-8)
    y_o = 1.0 / (obs_fp32[mask].astype(np.float64) + eps)
    y_s = 1.0 / (sim_fp32[mask].astype(np.float64) + eps)

    o_std = y_o.std()
    if o_std < 1e-10:
        return -999.0, {
            "r": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "n_valid": n_valid,
        }

    r = np.corrcoef(y_s, y_o)[0, 1]
    alpha = y_s.std() / o_std
    beta = y_s.mean() / y_o.mean()
    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    return kge, {
        "r": float(r),
        "alpha": float(alpha),
        "beta": float(beta),
        "n_valid": n_valid,
    }


STOP_COMPOSITE_CONVERGED = "COMPOSITE_CONVERGED"
STOP_MAX_GENERATIONS = "MAX_GENERATIONS_UNCONVERGED"
STOP_INVALID_FITNESS = "INVALID_FITNESS"
STOP_NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
STOP_PYCMA_INTERNAL = "PYCMA_INTERNAL_STOP"
STOP_SIGMA_CAP = "SIGMA_CAP"
STOP_COVARIANCE_CAP = "COVARIANCE_CONDITION_CAP"
STOP_INTERRUPTED = "INTERRUPTED"


@dataclass
class RestartSummary:
    basin_id: str
    basin_index: int
    snow_class: str
    restart_type: str
    restart_id: int
    best_train_objective: float
    best_train_objective_raw: float
    best_params_norm: list
    best_params_physical: list
    best_generation: int
    search_termination_label: str
    total_evaluations: int
    total_generations: int
    total_invalid_candidates: int
    final_sigma: float
    final_max_coordinate_std: float
    sigma_resets: int
    sigma_cap_triggers: int
    sobol_seed: int
    cma_seed: int
    initial_center: list
    initial_sigma: float
    profiler_summary: str


class PycmaCalibratorV3:
    """Independent pycma instances + GPU batched evaluation.

    Operates in normalized [0,1] space with BoundTransform.
    FP32 model forward, FP64 metric accumulation.
    """

    def __init__(
        self,
        model_cls,
        param_specs,
        forcing,
        obs_cal,
        obs_eval,
        basin_ids,
        frac_snow,
        config,
        model_cfg,
        obj_cfg,
        out_dir,
        device,
    ):
        self.model_cls = model_cls
        self.param_specs = param_specs
        self.forcing = forcing
        self.obs_cal = obs_cal
        self.obs_eval = obs_eval
        self.basin_ids = basin_ids
        self.frac_snow = frac_snow
        self.config = config
        self.model_cfg = model_cfg
        self.obj_cfg = obj_cfg
        self.out_dir = out_dir
        self.device = device

        self.param_names = list(param_specs.keys())
        self.D = len(self.param_names)
        self.N = len(basin_ids)
        self.P = model_cfg["population"]

        self.lower = np.array(
            [param_specs[n]["lower"] for n in self.param_names], dtype=np.float64
        )
        self.upper = np.array(
            [param_specs[n]["upper"] for n in self.param_names], dtype=np.float64
        )
        self.param_range = self.upper - self.lower

        self.warmup_days = config["time_periods"]["warmup"]["days"]

        self.cma_cfg = config["cmaes"]
        self.model_offset = model_cfg["model_offset"]
        self.obj_offset = obj_cfg["objective_offset"]
        self.model = None

        self.is_invq = obj_cfg["key"] == "invq"
        self.epsilons = self._compute_epsilons() if self.is_invq else None

        self.snow_class = np.where(frac_snow >= 0.1, "snow", "non_snow")

        # CMA-ES is deliberately defined on a dimensionless unit cube.  Keep
        # this contract explicit and fail early if a malformed parameter spec
        # would create a degenerate physical-to-normalized mapping.
        if np.any(~np.isfinite(self.param_range)) or np.any(self.param_range <= 0):
            raise ValueError(
                "All parameter ranges must be finite and strictly positive"
            )
        self.boundary_handler_name = str(
            self.cma_cfg.get("boundary_handler", "BoundTransform")
        )
        if self.boundary_handler_name != "BoundTransform":
            raise ValueError(
                "The repaired calibrator requires BoundaryHandler=BoundTransform; "
                f"got {self.boundary_handler_name!r}"
            )
        self.objective_floor = float(self.cma_cfg.get("optimization_kge_floor", -1.0))
        self.objective_tie_break = float(
            self.cma_cfg.get("optimization_tie_break_scale", 0.0)
        )
        self.sigma_cap = float(self.cma_cfg.get("sigma_cap", 1.0))
        self.max_sigma_resets = int(self.cma_cfg.get("max_sigma_resets", 0))
        self.condition_cap = float(self.cma_cfg.get("condition_cap", 1e8))

    def _compute_epsilons(self):
        eps = np.zeros(self.N, dtype=np.float64)
        for b in range(self.N):
            vals = self.obs_cal[b]
            valid = vals[np.isfinite(vals) & (vals >= 0)]
            eps[b] = float(valid.mean()) if len(valid) > 0 else 1e-8
        return eps

    def _denormalize(self, xn):
        return self.lower + xn * self.param_range

    def _make_es(self, x0, seed):
        opts = {
            "popsize": self.P,
            "seed": seed,
            "verbose": -9,
            "bounds": [0.0, 1.0],
            "BoundaryHandler": self.boundary_handler_name,
        }
        if "tolconditioncov" in self.cma_cfg:
            opts["tolconditioncov"] = float(self.cma_cfg["tolconditioncov"])
        if "tolupsigma" in self.cma_cfg:
            opts["tolupsigma"] = float(self.cma_cfg["tolupsigma"])
        for key in (
            "CMA_active",
            "CMA_diagonal",
            "CMA_const_trace",
            "CMA_on",
            "CSA_dampfac",
        ):
            if key in self.cma_cfg:
                opts[key] = self.cma_cfg[key]
        disabled = dict(self.cma_cfg["disabled_stops"])
        opts.update(disabled)
        return cma.CMAEvolutionStrategy(x0, self.cma_cfg["sigma0"], opts)

    def _sobol_centers(self, sobol_seed):
        e = torch.quasirandom.SobolEngine(self.D, scramble=True, seed=sobol_seed)
        pts = e.draw(self.N).cpu().numpy().astype(np.float64)
        eps = 1e-4
        return np.clip(pts, eps, 1.0 - eps)

    def _compute_kge_for_model(self, qsim, obs, bi=None):
        if self.is_invq:
            return compute_kge_invq(qsim, obs, self.epsilons[bi])
        return compute_kge_fp64(qsim, obs)

    def _protect_objective(self, value: float) -> float:
        """Protect optimization ranking; final validation remains untrimmed/raw.

        The reported protected value is floored at ``objective_floor``.  When
        enabled, the tiny logarithmic tie-break is used only by CMA-ES so a
        population of catastrophically bad candidates does not become exactly
        flat and inflate sigma.
        """
        if not np.isfinite(value):
            return value
        value = float(value)
        if value >= self.objective_floor or self.objective_tie_break <= 0:
            return max(value, self.objective_floor)
        return self.objective_floor - self.objective_tie_break * np.log1p(
            self.objective_floor - value
        )

    def _pycma_stop_detail(self, es):
        """Return the actual pycma stop dictionary for auditable reports."""
        try:
            stop = dict(es.stop())
        except Exception as exc:
            return {"stop_error": f"{type(exc).__name__}:{exc}"}
        return {
            str(k): (float(v) if isinstance(v, (np.floating, float, int)) else str(v))
            for k, v in stop.items()
        }

    @staticmethod
    def _distribution_diagnostics(es):
        try:
            sigma = float(es.sigma)
            stds = np.asarray(es.stds, dtype=np.float64)
            coord_std = float(np.max(stds))
            d = np.asarray(es.D, dtype=np.float64)
            condition = float((d[-1] / d[0]) ** 2) if d[0] > 0 else np.inf
            return sigma, coord_std, condition
        except Exception:
            return np.nan, np.nan, np.inf

    def _check_stop(self, kge_hist, es, gen, min_gens, tol_kge, tol_std, hw):
        info = {"hist_range": np.nan, "max_std": np.nan}
        if gen < min_gens:
            return False, "", info
        if len(kge_hist) >= hw:
            recent = kge_hist[-hw:]
            info["hist_range"] = float(max(recent) - min(recent))
        try:
            stds = np.asarray(es.stds, dtype=np.float64)
            info["max_std"] = float(np.max(stds))
        except Exception:
            info["max_std"] = np.nan
        h_ok = (not np.isnan(info["hist_range"])) and info["hist_range"] <= tol_kge
        s_ok = (not np.isnan(info["max_std"])) and info["max_std"] <= tol_std
        if h_ok and s_ok:
            return True, STOP_COMPOSITE_CONVERGED, info
        return False, "", info

    def _gpu_eval(self, theta_norm, fc_rep, active_indices):
        if self.model is None:
            self.model = self.model_cls().to(self.device, torch.float32)
            self.model.eval()
            # Pre-warm: trigger torch.compile with a tiny batch
            dummy_fc = {
                k: torch.zeros(2, 100, device=self.device, dtype=torch.float32)
                for k in ["precip", "pet", "temp"]
            }
            dummy_p = {
                n: torch.ones(2, device=self.device, dtype=torch.float32)
                for n in self.param_names
            }
            with torch.no_grad():
                _ = self.model(forcings=dummy_fc, params=dummy_p)

        n_total = len(theta_norm)
        theta_phys = self._denormalize(theta_norm.astype(np.float64)).astype(np.float32)

        fc = {}
        for k in ["precip", "pet", "temp"]:
            fc[k] = torch.from_numpy(fc_rep[k].astype(np.float32)).to(self.device)
        pdict = {
            n: torch.from_numpy(theta_phys[:, i]).float().to(self.device)
            for i, n in enumerate(self.param_names)
        }

        with torch.no_grad():
            q_all, _ = self.model(forcings=fc, params=pdict)
        q_all = q_all.cpu().numpy()

        kge_vals = np.full(n_total, -999.0, dtype=np.float32)
        raw_vals = np.full(n_total, -999.0, dtype=np.float32)
        for i, bi in enumerate(active_indices):
            obs = self.obs_cal[bi]
            for p in range(self.P):
                idx = i * self.P + p
                sim = q_all[idx, self.warmup_days :]
                kge, _ = self._compute_kge_for_model(sim, obs, bi)
                raw_vals[idx] = kge
                kge_vals[idx] = self._protect_objective(kge)

        del fc, pdict, q_all
        return kge_vals, raw_vals

    def run_restart(
        self,
        restart_id,
        basin_indices,
        restart_type,
        sobol_seed,
        sigma0,
        max_gens,
        min_gens,
        tol_kge,
        tol_std,
        hw,
        ckpt_dir,
        trace_dir,
        custom_centers=None,
    ):
        ba = list(basin_indices)
        if not ba:
            return {}

        # For local refinement: use best global params as starting point
        if custom_centers is not None:
            centers = np.full((self.N, self.D), 0.5, dtype=np.float64)
            for bi in ba:
                if bi in custom_centers:
                    centers[bi] = np.array(custom_centers[bi], dtype=np.float64)
        else:
            centers = self._sobol_centers(sobol_seed)
        es_d = {}
        best_kge = {}
        best_kge_raw = {}
        best_x = {}
        best_g = {}
        kge_hist = {}
        active = {}
        stop_label = {}
        total_evals = {}
        invalid_cnt = {}
        traces = {}
        final_sig = {}
        final_mstd = {}
        sigma_resets = {}
        sigma_cap_triggers = {}
        event_label = {}

        for bi in ba:
            x0 = centers[bi].tolist()
            seed = int(
                self.model_offset * 1e6 + self.obj_offset * 1e5 + restart_id * 1e4 + bi
            )
            es_d[bi] = self._make_es(x0, seed)
            best_kge[bi] = -999.0
            best_kge_raw[bi] = -999.0
            best_x[bi] = np.array(x0, dtype=np.float64)
            best_g[bi] = 0
            kge_hist[bi] = []
            active[bi] = True
            stop_label[bi] = ""
            total_evals[bi] = 0
            invalid_cnt[bi] = 0
            traces[bi] = []
            final_sig[bi] = sigma0
            final_mstd[bi] = np.nan
            sigma_resets[bi] = 0
            sigma_cap_triggers[bi] = 0
            event_label[bi] = ""

        chunk_size = None

        for gen in range(max_gens):
            cur = [bi for bi in ba if active[bi]]
            if not cur:
                break

            n_cur = len(cur)
            theta_flat = np.zeros((n_cur * self.P, self.D), dtype=np.float64)
            for i, bi in enumerate(cur):
                sols = np.array(es_d[bi].ask(), dtype=np.float64)
                theta_flat[i * self.P : (i + 1) * self.P] = sols

            fc_rep = {}
            for key in ["precip", "pet", "temp"]:
                fc_rep[key] = np.concatenate(
                    [np.tile(self.forcing[key][bi], (self.P, 1)) for bi in cur], axis=0
                ).astype(np.float32)

            try:
                kge_vals, raw_vals = self._gpu_eval(theta_flat, fc_rep, cur)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                chunk_size = max(1, (chunk_size or n_cur) // 2)
                kge_vals = np.full(n_cur * self.P, -999.0, dtype=np.float32)
                raw_vals = np.full(n_cur * self.P, -999.0, dtype=np.float32)
                for cs in range(0, n_cur, chunk_size):
                    ce = min(cs + chunk_size, n_cur)
                    cb = cur[cs:ce]
                    sl = slice(cs * self.P, ce * self.P)
                    fc_c = {}
                    for key in ["precip", "pet", "temp"]:
                        fc_c[key] = np.concatenate(
                            [np.tile(self.forcing[key][bi], (self.P, 1)) for bi in cb],
                            axis=0,
                        ).astype(np.float32)
                    try:
                        kge_vals[sl], raw_vals[sl] = self._gpu_eval(
                            theta_flat[sl], fc_c, cb
                        )
                    except torch.cuda.OutOfMemoryError:
                        chunk_size = max(1, chunk_size // 2)
                        torch.cuda.empty_cache()

            kge_mat = kge_vals.reshape(n_cur, self.P)
            raw_mat = raw_vals.reshape(n_cur, self.P)
            eps_b = self.cma_cfg["boundary_near_threshold"]
            bdy = int(np.sum((theta_flat <= eps_b) | (theta_flat >= 1.0 - eps_b)))

            for i, bi in enumerate(cur):
                if not active[bi]:
                    continue
                bk = kge_mat[i]
                invalid_mask = ~np.isfinite(bk) | (bk <= -998.0)
                n_inv = int(invalid_mask.sum())
                invalid_cnt[bi] += n_inv

                valid_k = bk[~invalid_mask]
                gen_best = float(np.max(valid_k)) if len(valid_k) > 0 else -999.0

                if n_inv == self.P:
                    active[bi] = False
                    stop_label[bi] = STOP_INVALID_FITNESS
                    total_evals[bi] += self.P
                    continue

                if gen_best > best_kge[bi]:
                    best_kge[bi] = gen_best
                    bi_idx = int(np.argmax(bk))
                    best_x[bi] = theta_flat[i * self.P + bi_idx].copy()
                    best_kge_raw[bi] = float(raw_mat[i, bi_idx])
                    best_g[bi] = gen

                kge_hist[bi].append(best_kge[bi])

                penalty = self.cma_cfg["failure_penalty"]
                fit = np.where(invalid_mask, penalty, -bk.astype(np.float64))
                st = [
                    (float(theta_flat[i * self.P + p, d]) for d in range(self.D))
                    for p in range(self.P)
                ]
                try:
                    es_d[bi].tell(st, fit.tolist())
                except Exception:
                    active[bi] = False
                    stop_label[bi] = STOP_NUMERICAL_FAILURE
                    total_evals[bi] += self.P
                    continue

                total_evals[bi] += self.P

                sigma_val, mstd, condition = self._distribution_diagnostics(es_d[bi])
                final_sig[bi] = sigma_val
                final_mstd[bi] = mstd

                cstop, creason, cinfo = self._check_stop(
                    kge_hist[bi], es_d[bi], gen + 1, min_gens, tol_kge, tol_std, hw
                )

                if cstop:
                    active[bi] = False
                    stop_label[bi] = STOP_COMPOSITE_CONVERGED
                elif np.isfinite(sigma_val) and sigma_val > self.sigma_cap:
                    sigma_cap_triggers[bi] += 1
                    if sigma_resets[bi] < self.max_sigma_resets:
                        sigma_resets[bi] += 1
                        reset_seed = int(
                            self.model_offset * 1e6
                            + self.obj_offset * 1e5
                            + restart_id * 1e4
                            + bi
                            + sigma_resets[bi] * 1000003
                        )
                        es_d[bi] = self._make_es(best_x[bi].tolist(), reset_seed)
                        event_label[bi] = (
                            f"{STOP_SIGMA_CAP}_RESET#{sigma_resets[bi]}"
                            f":sigma={sigma_val:.6g}:cap={self.sigma_cap:.6g}"
                        )
                    else:
                        active[bi] = False
                        stop_label[bi] = (
                            f"{STOP_SIGMA_CAP}:sigma={sigma_val:.6g}:cap={self.sigma_cap:.6g}"
                        )
                elif np.isfinite(condition) and condition > self.condition_cap:
                    active[bi] = False
                    stop_label[bi] = (
                        f"{STOP_COVARIANCE_CAP}:condition={condition:.6g}:cap={self.condition_cap:.6g}"
                    )
                elif gen + 1 >= max_gens:
                    active[bi] = False
                    stop_label[bi] = STOP_MAX_GENERATIONS
                else:
                    estop = es_d[bi].stop()
                    if estop:
                        active[bi] = False
                        detail = self._pycma_stop_detail(es_d[bi])
                        detail_text = ",".join(f"{k}={v}" for k, v in detail.items())
                        stop_label[bi] = f"{STOP_PYCMA_INTERNAL}:{detail_text}"

                traces[bi].append(
                    {
                        "gen": gen + 1,
                        "gen_best": gen_best,
                        "global_best": best_kge[bi],
                        "sigma": sigma_val,
                        "max_std": mstd,
                        "condition": condition,
                        "hist_range": cinfo.get("hist_range", np.nan),
                        "stop_label": event_label[bi] or stop_label[bi],
                        "bdy_count": bdy,
                        "invalid": n_inv,
                        "evals": total_evals[bi],
                    }
                )
                event_label[bi] = ""

            done = sum(1 for bi in ba if not active[bi])
            if gen % 20 == 0 or gen + 1 >= max_gens:
                ak = [best_kge[bi] for bi in ba if active[bi]]
                med = np.median(ak) if ak else 0.0
                print(
                    f"    gen {gen + 1:4d}: active={len(ba) - done}/{len(ba)} done={done} "
                    f"best_med={med:.4f}",
                    flush=True,
                )

            if (gen + 1) % self.cma_cfg["checkpoint_interval"] == 0:
                self._save_ckpt(
                    restart_id,
                    gen + 1,
                    ba,
                    es_d,
                    best_kge,
                    best_x,
                    best_g,
                    kge_hist,
                    active,
                    stop_label,
                    total_evals,
                    ckpt_dir,
                )

        summaries = {}
        for bi in ba:
            summaries[bi] = RestartSummary(
                basin_id=self.basin_ids[bi],
                basin_index=bi,
                snow_class=self.snow_class[bi],
                restart_type=restart_type,
                restart_id=restart_id,
                best_train_objective=best_kge[bi],
                best_train_objective_raw=best_kge_raw[bi],
                best_params_norm=best_x[bi].tolist(),
                best_params_physical=self._denormalize(best_x[bi]).tolist(),
                best_generation=best_g[bi],
                search_termination_label=stop_label[bi],
                total_evaluations=total_evals[bi],
                total_generations=len(traces[bi]),
                total_invalid_candidates=invalid_cnt[bi],
                final_sigma=final_sig[bi],
                final_max_coordinate_std=final_mstd.get(bi, np.nan),
                sigma_resets=sigma_resets[bi],
                sigma_cap_triggers=sigma_cap_triggers[bi],
                sobol_seed=sobol_seed,
                cma_seed=0,
                initial_center=centers[bi].tolist(),
                initial_sigma=sigma0,
                profiler_summary="",
            )

        self._save_traces(traces, trace_dir, restart_id)
        self._save_summaries(summaries, ckpt_dir, restart_id)
        return summaries

    def _save_ckpt(self, rid, gen, ba, es_d, bk, bx, bg, kh, active, sl, te, d):
        ckpt = {
            "restart_id": rid,
            "generation": gen,
            "basins": ba,
            "best_kge": {str(k): v for k, v in bk.items()},
            "best_x": {str(k): v.tolist() for k, v in bx.items()},
            "best_gen": bg,
            "kge_hist": {str(k): v for k, v in kh.items()},
            "active": {str(k): v for k, v in active.items()},
            "stop_label": sl,
            "total_evals": te,
            "cma_states": {},
        }
        for bi in ba:
            try:
                es = es_d[bi]
                ckpt["cma_states"][str(bi)] = {
                    "mean": es.mean.tolist(),
                    "sigma": float(es.sigma),
                    "C": es.C.tolist(),
                    "countiter": int(es.countiter),
                }
            except Exception:
                ckpt["cma_states"][str(bi)] = None
        with open(d / f"ckpt_r{rid:02d}_g{gen:04d}.pkl", "wb") as f:
            pickle.dump(ckpt, f)

    def _save_traces(self, traces, d, rid):
        all_t = []
        for bi in sorted(traces.keys()):
            all_t.extend(traces[bi])
        if not all_t:
            return
        a = {}
        for fn in [
            "gen",
            "gen_best",
            "global_best",
            "sigma",
            "max_std",
            "condition",
            "hist_range",
            "bdy_count",
            "invalid",
            "evals",
        ]:
            a[fn] = np.array([t[fn] for t in all_t], dtype=np.float32)
        np.savez_compressed(d / f"traces_r{rid:02d}.npz", **a)

    def _save_summaries(self, summaries, d, rid):
        if not summaries:
            return
        fields = [f.name for f in RestartSummary.__dataclass_fields__.values()]
        with open(d / f"summary_r{rid:02d}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for s in summaries.values():
                w.writerow({k: getattr(s, k) for k in fields})
