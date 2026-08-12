from __future__ import annotations
from _common import EXPERIMENT, settings
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
import torch
from src.gpu_batch_planner import measure
from src.model_adapter import BatchedModelAdapter

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(EXPERIMENT / "benchmarks/torchinductor_cache"))


def adapter_for(model: str, dim: int, basins: int, device: str, *, backend: str = "eager", objective_mode: str = "eager") -> BatchedModelAdapter:
    g=torch.Generator(device="cpu").manual_seed(17)
    x=torch.stack((torch.rand(42,basins,generator=g)*10,torch.randn(42,basins,generator=g)*4,torch.rand(42,basins,generator=g)*3),-1).to(device)
    if model in {"mopex4","mopex5"}: x=torch.cat((x,torch.arange(1,43,device=device).view(-1,1,1).expand(-1,basins,-1)), -1)
    obs=torch.rand(42,basins,generator=g).to(device)
    return BatchedModelAdapter(model,x,obs,warm_up=5,device=device,backend=backend,objective_compile_mode=objective_mode)


def run_case(model: str, dim: int, pop: int, mode: str) -> dict:
    device="cuda" if torch.cuda.is_available() else "cpu"; sample=torch.zeros((2,2,pop,dim),dtype=torch.float64,device=device)
    eager_adapter=adapter_for(model,dim,2,device)
    baseline=eager_adapter.evaluate(sample).kge; torch.cuda.synchronize() if device=="cuda" else None
    started=time.perf_counter()
    try:
        adapter=adapter_for(model,dim,2,device,backend="eager" if mode=="eager" else "compile",objective_mode=mode)
        adapter.evaluate(sample); torch.cuda.synchronize() if device=="cuda" else None
        compile_seconds=time.perf_counter()-started
    except Exception as exc:
        return {"model":model,"mode":mode,"compile_seconds":time.perf_counter()-started,"stable_seconds":"","candidates_per_second":"","max_abs_error":"","peak_allocated":0,"peak_reserved":0,"error":f"{type(exc).__name__}: {exc}"}
    row={"model":model,"mode":mode,"compile_seconds":compile_seconds,"stable_seconds":"","candidates_per_second":"","max_abs_error":"","peak_allocated":0,"peak_reserved":0,"error":""}
    try:
        if device=="cuda": torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter()
        for _ in range(10): out=adapter.evaluate(sample).kge
        if device=="cuda": torch.cuda.synchronize()
        elapsed=time.perf_counter()-t
        row.update(stable_seconds=elapsed,candidates_per_second=2*2*pop*10/elapsed,max_abs_error=float((out-baseline).abs().max()),peak_allocated=torch.cuda.max_memory_allocated() if device=="cuda" else 0,peak_reserved=torch.cuda.max_memory_reserved() if device=="cuda" else 0)
    except Exception as exc: row["error"]=repr(exc)
    return row


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--case", nargs=4); parser.add_argument("--try-compile", action="store_true"); args=parser.parse_args()
    if args.case:
        model, dim, pop, mode=args.case; print(json.dumps(run_case(model,int(dim),int(pop),mode))); return 0
    cfg=settings(); device="cuda" if torch.cuda.is_available() else "cpu"; rows=[]; plans={}
    reps=[("collie1",1,8),("gr4j",4,12),("xinanjiang",12,20),("hbv96",15,20)]
    for model,dim,pop in reps:
        # conservative dry-run plan using the production compile boundary.
        best=None; cap=min(cfg["memory_limit_gib"]*2**30, cfg["memory_fraction_limit"]*(torch.cuda.get_device_properties(0).total_memory if device=="cuda" else 2**63))
        for b in (2,4):
            trial=adapter_for(model,dim,b,device,backend="compile",objective_mode="reduce-overhead"); m=measure(trial,basins=b,starts=3,population=pop,iterations=10)
            if not m.oom and m.peak_allocated <= cap: best=m
            else: break
        plans[model]=best.__dict__ if best else {"oom":True}
        modes = ("eager", "default", "reduce-overhead")
        for mode in modes:
            proc=subprocess.run([sys.executable, str(Path(__file__)), "--case", model, str(dim), str(pop), mode], cwd=EXPERIMENT.parents[1], text=True, capture_output=True)
            try: rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))
            except Exception: rows.append({"model":model,"mode":mode,"compile_seconds":"","stable_seconds":"","candidates_per_second":"","max_abs_error":"","peak_allocated":0,"peak_reserved":0,"error":proc.stderr[-1000:] or proc.stdout[-1000:]})
    with (EXPERIMENT/"benchmarks/compile_benchmark.csv").open("w",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    (EXPERIMENT/"benchmarks/gpu_batch_plan.json").write_text(json.dumps(plans,indent=2))
    report=["# GPU memory and compile benchmark",f"device: {device}",f"portable cap GiB: {cfg['memory_limit_gib']}","", "Results are 10 stable iterations; compilation warm-up excluded."]
    report += [str(r) for r in rows]; report += ["", "## Batch plans", json.dumps(plans,indent=2)]
    (EXPERIMENT/"reports/gpu_memory_report.md").write_text("\n".join(report))
    return 0

if __name__ == "__main__": raise SystemExit(main())
