from __future__ import annotations

import os
import time
import json
import numpy as np
import torch
from typing import List
from dataclasses import asdict

from .protocol import Phase1Task
from ablation.ic_core.data_adapter import load_531_bundle, read_basin_ids
from ablation.ic_core.runtime import ICObjectiveRuntime
from ablation.optimizers.registry import get_optimizer_class

def run_batched_tasks_for_group_process(args):
    """Worker function for ProcessPoolExecutor to avoid PyTorch multithreading lazy wrapper conflicts."""
    opt_name, population, start_idx, seed, tasks_dicts, config = args
    
    # Cap CPU threads per worker process
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    
    # Reconstruct Phase1Task objects
    tasks_for_group = [Phase1Task(**d) for d in tasks_dicts]
    
    # Filter pending tasks
    pending_tasks = []
    for task in tasks_for_group:
        os.makedirs(task.output_dir, exist_ok=True)
        if not os.path.exists(os.path.join(task.output_dir, "done.txt")):
            pending_tasks.append(task)

    if not pending_tasks:
        print(f"Group ({opt_name}, pop {population}, start {start_idx}, seed {seed}): All {len(tasks_for_group)} tasks already completed.")
        return

    # Fail-loud hyperparameter validation from Task object (using ValueError so python -O cannot strip)
    first_task = pending_tasks[0]
    task_pop = first_task.population
    task_gen = first_task.generations
    task_stdev = first_task.stdev_init
    task_opt = first_task.optimizer_name

    if task_pop is None or task_pop <= 0:
        raise ValueError(f"Fail-loud: Invalid population {task_pop}")
    if task_gen is None or task_gen <= 0:
        raise ValueError(f"Fail-loud: Invalid generations {task_gen}")
    if task_stdev is None or task_stdev <= 0.0:
        raise ValueError(f"Fail-loud: Invalid stdev_init {task_stdev}")
    if task_opt != opt_name:
        raise ValueError(f"Fail-loud: Task optimizer mismatch {task_opt} vs {opt_name}")

    for t in pending_tasks:
        if t.population != task_pop:
            raise ValueError(f"Fail-loud: Population mismatch within group {t.population} vs {task_pop}")
        if t.generations != task_gen:
            raise ValueError(f"Fail-loud: Generations mismatch within group {t.generations} vs {task_gen}")
        if t.stdev_init != task_stdev:
            raise ValueError(f"Fail-loud: stdev_init mismatch within group {t.stdev_init} vs {task_stdev}")

    device = config.get("device", "cuda")
    generations = task_gen
    stdev_init = task_stdev


    print(f"Running Group ({opt_name}, pop {task_pop}, gen {task_gen}, stdev {task_stdev}, start {start_idx}, seed {seed}): {len(pending_tasks)} pending basins on GPU...")

    bundle = load_531_bundle(config)
    rt = ICObjectiveRuntime(bundle, config, first_task.model_key)
    
    warmup_length = bundle.periods.warmup.days
    train_length = bundle.periods.train.days
    input_length = bundle.periods.train_forcing_end_index - bundle.periods.train_forcing_start_index
    objective_length = train_length
    
    if warmup_length <= 0 or train_length <= 0 or input_length <= 0 or objective_length <= 0:
        raise ValueError(f"Fail-loud: Invalid period lengths warmup={warmup_length}, train={train_length}, input={input_length}, obj={objective_length}")

    
    opt_cls = get_optimizer_class(opt_name)

    # Prepare searchers & runtime indices
    searchers = []
    basin_indices = []
    all_basin_ids = read_basin_ids(config["basin_list_path"])

    for t in pending_tasks:
        s = opt_cls()
        dim = len(t.center_init)
        s.initialize(
            dimension=dim,
            population=t.population,
            center_init=np.array(t.center_init),
            stdev_init=t.stdev_init,
            seed=t.seed,
            device="cpu",
            dtype=config.get("model_dtype", "float32"),
            config=config
        )
        searchers.append(s)
        b_idx = all_basin_ids.index(t.basin_id)
        basin_indices.append(b_idx)

    # Run generation loop
    t0_group = time.perf_counter()

    traces = [[] for _ in pending_tasks]

    for gen in range(generations):
        gen_t0 = time.perf_counter()
        
        cands_list = None
        cands_np = None
        eval_result = None
        try:
            # 1. Ask candidates from searchers
            cands_list = [s.ask() for s in searchers]  # List of [P, D] arrays
            cands_np = np.stack(cands_list, axis=0)     # [B, P, D]
            
            # 2. Evaluate candidates in a single GPU batched matrix call per generation
            eval_result = rt.evaluate_candidates(cands_np, basin_indices=basin_indices, split="train")
            
            # 3. Tell fitness back to searchers
            for b_i, s in enumerate(searchers):
                s.tell(eval_result.fitness[b_i])
                best_cand, best_fit = s.get_best()
                traces[b_i].append({
                    "generation": gen + 1,
                    "best_fitness": float(best_fit),
                    "mean_fitness": float(np.mean(eval_result.fitness[b_i])),
                    "step_time": time.perf_counter() - gen_t0
                })
        except Exception as err:
            err_str = str(err)
            if "out of memory" in err_str.lower():
                print(f"[CUDA OOM ERROR DETECTED] in Group ({opt_name}, pop{population}, start{start_idx}, seed{seed}): {err_str}", flush=True)
                with open("/autodl-fs/data/dmg_hydro_structure_diagnosis/oom_error.log", "a") as f_oom:
                    f_oom.write(f"[{time.ctime()}] CUDA OOM in group ({opt_name}, pop{population}, start{start_idx}, seed{seed}): {err_str}\n")
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            raise err
        finally:
            del cands_list, cands_np, eval_result
            if (gen + 1) % 20 == 0:
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

        if (gen + 1) % 50 == 0 or gen == 0:
            elapsed_gen = time.perf_counter() - gen_t0
            med_kge = np.median([t_list[-1]["best_fitness"] for t_list in traces])
            print(f"  [{opt_name} pop{population} start{start_idx} seed{seed}] gen {gen+1}/{generations}: median_kge={med_kge:.4f}, step_time={elapsed_gen:.2f}s", flush=True)


    elapsed_group = time.perf_counter() - t0_group

    # Post-calibration out-of-sample evaluation on TEST split for best candidates
    best_thetas = np.stack([s.get_best()[0] for s in searchers], axis=0)
    thetas_3d = torch.tensor(best_thetas, dtype=torch.float32, device=device).unsqueeze(1)
    test_eval_res = rt.evaluate_candidates(thetas_3d, basin_indices=basin_indices, split="test")
    test_kges = np.asarray(test_eval_res.fitness).flatten()

    # Save results per task
    for b_i, t in enumerate(pending_tasks):
        best_cand, best_fit = searchers[b_i].get_best()
        opt = searchers[b_i]
        
        res = {
            "basin_id": t.basin_id,
            "optimizer_name": opt_name,
            "population": t.population,
            "model_key": t.model_key,
            "optimizer_seed": t.seed,
            "start_idx": t.start_idx,
            "best_train_kge": float(best_fit),
            "best_test_kge": float(test_kges[b_i]),
            "best_theta_normalized": best_cand.tolist(),
            "total_generations": generations,
            "total_evaluations": generations * t.population,
            "runtime_seconds": elapsed_group / len(pending_tasks),
            "period_protocol": {
                "warmup": "1980-10-01 to 1981-09-30",
                "train": "1981-10-01 to 1995-09-30",
                "test": "1995-10-01 to 2010-09-30"
            }
        }
        
        with open(os.path.join(t.output_dir, "result.json"), "w") as f:
            json.dump(res, f, indent=2)
            
        state = opt.state_dict()
        torch.save(state, os.path.join(t.output_dir, "checkpoint.pt"))
        
        period_meta = {
            "warmup_length": warmup_length,
            "train_length": train_length,
            "input_length": input_length,
            "objective_length": objective_length
        }
        with open(os.path.join(t.output_dir, "period_metadata.json"), "w") as f:
            json.dump(period_meta, f)
            
        with open(os.path.join(t.output_dir, "trace.json"), "w") as f:
            json.dump(traces[b_i], f)

        with open(os.path.join(t.output_dir, "done.txt"), "w") as f:
            f.write("DONE")

    # Clean up CUDA memory cache to prevent OOM across worker processes
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


import concurrent.futures

def run_tasks(tasks: List[Phase1Task], config: dict):
    # Set single CPU thread
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    
    # Group tasks by (optimizer_name, population, start_idx, seed)
    groups = {}
    for task in tasks:
        key = (task.optimizer_name, task.population, task.start_idx, task.seed)
        if key not in groups:
            groups[key] = []
        groups[key].append(task)

    print(f"Total task groups: {len(groups)} (each group runs 32 basins in a single GPU batched matrix call per generation).")
    
    max_concurrent_groups = config.get("max_concurrent_groups", 6)
    print(f"Running up to {max_concurrent_groups} task groups concurrently using ProcessPoolExecutor (spawn context) on GPU...")

    # Prepare worker args using dict representations for serialization
    worker_args_list = []
    for i, (key, group_tasks) in enumerate(groups.items()):
        opt_name, pop, s_idx, s_seed = key
        tasks_dicts = [asdict(t) for t in group_tasks]
        worker_args_list.append((opt_name, pop, s_idx, s_seed, tasks_dicts, config))

    # Use ProcessPoolExecutor with spawn context for PyTorch CUDA multiprocessing isolation
    mp_ctx = torch.multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_groups, mp_context=mp_ctx) as executor:
        futures = [executor.submit(run_batched_tasks_for_group_process, args) for args in worker_args_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[FAIL-LOUD ERROR] Task group execution failed: {e}", flush=True)
                raise e

    # Force complete CUDA VRAM cache release and python GC at the end of task batch execution
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


