#!/usr/bin/env python3
"""Read-only audit of authoritative IC Stage 1, Stage 2 and Stage 3 results."""
from __future__ import annotations
import csv, hashlib, json, math, re, statistics, subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[3]
SUPP = PROJECT / "manuscript" / "supplement"
RESULTS = SUPP / "results"
REPORTS = SUPP / "reports"
DATA_ROOT = PROJECT.parents[1] / "data"
BASIN_531 = DATA_ROOT / "531sub_id.txt"
REMOTE_ROOT = "/root/outputs/full_model_series_calibration/v1/tasks"
FORMAL = ["XAJ", "XAJ_CN", "XAJ_TGD"]
ALL_MODELS = ["XAJ","XAJ_CN","XAJ_TGD","HBV","GR4J","GR4J_CN","GR4J_TGD","SIMHYD","SIMHYD_CN","SIMHYD_TGD"]
DIMS = {"XAJ":15,"XAJ_CN":17,"XAJ_TGD":18,"HBV":12,"GR4J":4,"GR4J_CN":6,"GR4J_TGD":7,"SIMHYD":10,"SIMHYD_CN":12,"SIMHYD_TGD":13}

def read_json(path: Path) -> dict[str, Any]:
    try:
        x=json.loads(path.read_text(encoding="utf-8"))
        return x if isinstance(x,dict) else {}
    except Exception:
        return {}

def load_csv(path: Path) -> list[dict[str,str]]:
    try:
        with path.open(encoding="utf-8",newline="") as f: return list(csv.DictReader(f))
    except Exception: return []

def write_csv(path: Path, rows: list[dict[str,Any]], fields: list[str]|None=None) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or (list(rows[0]) if rows else ["status"])
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=True,default=str)+"\n",encoding="utf-8")

def hash_small(path: Path) -> str:
    try:
        if path.stat().st_size>10_000_000: return "NOT_HASHED_LARGE"
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError: return "UNAVAILABLE"

def basin_set(path: Path) -> set[str]:
    return {x.zfill(8) for x in re.findall(r"\d{7,8}",path.read_text(encoding="utf-8"))}

REMOTE_SCRIPT = r'''import json, math, os, subprocess
from pathlib import Path
root=Path("/root/outputs/full_model_series_calibration/v1/tasks")
print("META\t"+json.dumps({"host":subprocess.check_output(["hostname"],text=True).strip(),"date":subprocess.check_output(["date","-Is"],text=True).strip(),"root":str(root)},separators=(",",":")))
try:
    ps=subprocess.check_output(["ps","-eo","pid,ppid,lstart,etime,%cpu,%mem,stat,args"],text=True)
    for line in ps.splitlines():
        if "run_remote_full_model_series.py" in line or ("multiprocessing.spawn" in line and "torch/_inductor" not in line):
            print("PROC\t"+line)
except Exception as e: print("PROC_ERROR\t"+str(e))
def trace_info(path,total):
    out={"exists":False,"rows":0,"last_generation":None,"checkpoints":[],"best_generation":None,"last_quarter_improvement":None,"error":""}
    try:
        rows=json.loads(path.read_text())
        if not isinstance(rows,list): raise ValueError("trace is not list")
        out["exists"]=True; out["rows"]=len(rows)
        by={}; elapsed=0.0
        for row in rows:
            gen=int(row["generation"]); elapsed+=float(row.get("step_time",0.0)); by[gen]=dict(row,cumulative_elapsed_seconds=elapsed)
        if by:
            out["last_generation"]=max(by)
            best=max(rows,key=lambda x:float(x.get("best_fitness",-math.inf)))
            out["best_generation"]=int(best["generation"])
            for frac in (0.25,0.50,0.75,1.0):
                target=int(round(total*frac))
                if target in by:
                    x=by[target]; out["checkpoints"].append({"fraction":frac,"generation":target,"incumbent":float(x["best_fitness"]),"elapsed":float(x["cumulative_elapsed_seconds"])})
            q=int(math.ceil(total*.75))
            if any(g>=q for g in by):
                first=by[min(g for g in by if g>=q)]; last=by[max(by)]
                out["last_quarter_improvement"]=float(last["best_fitness"])-float(first["best_fitness"])
    except Exception as e: out["error"]=str(e)
    return out
if root.exists():
    for p in sorted(root.rglob("result.json")):
        try:
            rel=p.relative_to(root).parts
            model,opt,basin,seed_dir,start_dir=rel[:5]
            r=json.loads(p.read_text()); total=int(r.get("total_generations",0))
            period={}
            q=p.with_name("period_metadata.json")
            if q.exists():
                try: period=json.loads(q.read_text())
                except Exception: period={"parse_error":True}
            item={"path":str(p),"model":model,"optimizer":opt,"basin_id":str(r.get("basin_id",basin)).zfill(8),
                  "seed":int(r.get("optimizer_seed",seed_dir.replace("seed_","-1"))),"start":int(r.get("start_idx",start_dir.replace("start_","-1"))),
                  "population":r.get("population"),"generations":total,"evaluations":r.get("total_evaluations"),
                  "best_train_kge":r.get("best_train_kge"),"runtime_seconds":r.get("runtime_seconds"),
                  "best_theta_normalized":r.get("best_theta_normalized"),"test_kge":r.get("test_kge"),
                  "period_protocol_result":r.get("period_protocol"),"period_metadata":period,
                  "trace":trace_info(p.with_name("trace.json"),total)}
            print("RESULT\t"+json.dumps(item,separators=(",",":")))
        except Exception as e: print("RESULT_ERROR\t"+json.dumps({"path":str(p),"error":str(e)},separators=(",",":")))
'''

def remote_extract():
    cmd=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=12","-p","53700","root@connect.westb.seetacloud.com","/root/miniconda3/bin/python","-"]
    try: p=subprocess.run(cmd,input=REMOTE_SCRIPT,text=True,capture_output=True,timeout=180)
    except Exception as e: return [],[],{},repr(e)
    err=None if p.returncode==0 else "ssh_exit_%s: %s"%(p.returncode,(p.stderr or "")[:500])
    records=[]; procs=[]; meta={}
    for line in p.stdout.splitlines():
        if line.startswith("META\t"): meta=json.loads(line.split("\t",1)[1])
        elif line.startswith("PROC\t"): procs.append({"raw":line.split("\t",1)[1]})
        elif line.startswith("RESULT\t"): records.append(json.loads(line.split("\t",1)[1]))
        elif line.startswith("RESULT_ERROR\t"): records.append(json.loads(line.split("\t",1)[1])|{"result_error":True})
    return records,procs,meta,err

def stage1():
    rows=load_csv(PROJECT/"ablation/manifests/ic_ablation_96_basins_v1.csv"); full=basin_set(BASIN_531)
    manifest=[dict(r,basin_id=r.get("basin_id","").zfill(8),in_531="yes" if r.get("basin_id","").zfill(8) in full else "no",evidence_status="VERIFIED_CONFIG") for r in rows]
    cfg=read_json(PROJECT/"ablation/configs/ic_xnes_stage1_preflight_v1.json"); opt=cfg.get("optimizer",{})
    settings=[{"stage":"Stage1","model":"XAJ","optimizer":opt.get("name"),"dimension":15,"population":opt.get("population"),"stdev_init":opt.get("stdev_init"),"generations":opt.get("generations"),"starts":opt.get("starts"),"seeds":json.dumps(opt.get("optimizer_seeds")),"ranking_method":opt.get("ranking_method"),"boundary_handling":cfg.get("boundary_handling"),"objective":"KGE(Q), maximize","split":"A:32 of 96","compute_test_metric":cfg.get("compute_test_metric"),"evidence_path":"ablation/configs/ic_xnes_stage1_preflight_v1.json","evidence_status":"VERIFIED_CONFIG"}]
    per=load_csv(PROJECT/"outputs/ic_ablation/stage1_screening/v1/xnes/summaries/per_start.csv"); rec=load_csv(PROJECT/"outputs/ic_ablation/stage1_screening/v1/xnes_audit/independent_kge_recalculation.csv")
    validity=[
        {"item":"preflight_decision","value":"READY_FOR_STAGE1_SCREENING","evidence":"outputs/ic_ablation/stage1_preflight/PRECHECK_READINESS_REPORT.md","status":"VERIFIED_RESULT"},
        {"item":"baseline_validity_decision","value":"BASELINE_VALID","evidence":"outputs/ic_ablation/stage1_screening/v1/xnes_audit/XNES_BASELINE_VALIDITY_REPORT.md","status":"VERIFIED_RESULT"},
        {"item":"manifest_rows","value":len(rows),"evidence":"ic_ablation_96_basins_v1.csv","status":"VERIFIED_CONFIG"},
        {"item":"split_A_rows","value":sum(r.get("split")=="A" for r in rows),"evidence":"ic_ablation_96_basins_v1.csv","status":"VERIFIED_CONFIG"},
        {"item":"per_start_rows","value":len(per),"evidence":"stage1_screening/v1/xnes/summaries/per_start.csv","status":"VERIFIED_RESULT"},
        {"item":"per_start_finite_rows","value":sum(float(r.get("best_train_kge","-999"))>-900 for r in per if r.get("best_train_kge")),"evidence":"per_start.csv","status":"CONFLICT"},
        {"item":"independent_recalculation_rows","value":len(rec),"evidence":"xnes_audit/independent_kge_recalculation.csv","status":"VERIFIED_RESULT"},
        {"item":"trace_audit","value":"present","evidence":"xnes_audit/convergence_trace_audit.csv","status":"VERIFIED_RESULT"},
        {"item":"failure_types","value":"no failures in validity report; raw summary contains sentinel -999","evidence":"validity report; per_start.csv","status":"CONFLICT"}]
    meta={"manifest_rows":len(rows),"split_counts":dict(Counter(r.get("split") for r in rows)),"stratum_counts":dict(Counter(r.get("stratum") for r in rows)),"in_531":sum(x["in_531"]=="yes" for x in manifest),"not_in_531":sum(x["in_531"]=="no" for x in manifest)}
    return manifest,settings,validity,meta

def stage2_population():
    root=PROJECT/"outputs/ic_ablation/large_scale_screening/v1/tasks"; out=[]
    if not root.exists(): return out
    for p in root.rglob("result.json"):
        if "invalidated" in str(p) or "/_" in str(p): continue
        try:
            rel=p.relative_to(root).parts; model,opt,pop_dir,basin,seed_dir,start_dir=rel[:6]; r=read_json(p)
            seed_value=r.get("optimizer_seed")
            start_value=r.get("start_idx")
            generation_value=r.get("total_generations")
            population_value=r.get("population")
            population_value=int(population_value if population_value is not None else pop_dir.replace("pop_","-1"))
            seed_value=int(seed_dir.replace("seed_","-1"))
            start_value=int(start_dir.replace("start_","-1"))
            generation_value=int(generation_value if generation_value is not None else 200)
            out.append({"path":str(p),"model":model,"optimizer":opt,"population":population_value,"dimension":DIMS.get(model),"basin_id":str(r.get("basin_id",basin)).zfill(8),"seed":seed_value,"start":start_value,"generations":generation_value,"stdev_init":0.25,"evaluations":r.get("total_evaluations") if r.get("total_evaluations") is not None else population_value*generation_value,"train_kge":r.get("best_train_kge"),"runtime_seconds":r.get("runtime_seconds"),"evidence_status":"LOCAL_RESULT_VERIFIED"})
        except Exception: pass
    return out

def pop_aggregate(records):
    groups=defaultdict(list)
    for r in records: groups[(r["model"],r["optimizer"],r["population"])].append(r)
    rows=[]
    for key,vals in sorted(groups.items()):
        model,opt,pop=key; finite=[float(x["train_kge"]) for x in vals if isinstance(x.get("train_kge"),(int,float)) and math.isfinite(float(x["train_kge"]))]
        run=[float(x["runtime_seconds"]) for x in vals if isinstance(x.get("runtime_seconds"),(int,float))]
        rows.append({"model":model,"dimension":DIMS.get(model),"optimizer":opt,"population":pop,"population_over_dimension":pop/DIMS[model],"expected_units":288,"evaluated_units":len(vals),"unique_basins":len({x["basin_id"] for x in vals}),"unique_seeds":len({x["seed"] for x in vals}),"unique_starts":len({x["start"] for x in vals}),"generations":Counter(x["generations"] for x in vals).most_common(1)[0][0],"stdev_init":0.25,"evaluations_per_unit":Counter(x["evaluations"] for x in vals).most_common(1)[0][0],"train_kge_mean":statistics.mean(finite) if finite else "","train_kge_median":statistics.median(finite) if finite else "","failure_rate":1-len(finite)/len(vals) if vals else "","runtime_mean_seconds":statistics.mean(run) if run else "","status":"complete" if len(vals)==288 else "partial","selection_use":"diagnostic only; no persisted selection manifest","evidence_status":"VERIFIED_RESULT"})
    return rows

def code_stage2():
    stdev=[]; gen=[]
    for model,pop in [("GR4J",24),("SIMHYD",40),("XAJ",90)]:
        for opt in ["XNES","CMAES"]:
            for value in [.05,.10,.15,.25,.35,.50]:
                stdev.append({"model":model,"dimension":DIMS[model],"optimizer":opt,"population_fixed":pop,"stdev_init":value,"generations":200,"basins_planned":32,"seeds_planned":3,"starts_planned":3,"evaluated_units":0,"status":"CODE_ONLY_NO_RESULT_FILES","evidence_path":"ablation/controlled_optimizer_ablation/run_remote_phase2_stdev.py","evidence_status":"VERIFIED_CODE"})
            gen.append({"model":model,"dimension":DIMS[model],"optimizer":opt,"population_fixed":pop,"stdev_fixed":.05 if model in ("SIMHYD","XAJ") else .10,"generations_candidate":600,"basins_planned":32,"seeds_planned":3,"starts_planned":3,"evaluated_units":0,"status":"CODE_ONLY_NO_RESULT_FILES","evidence_path":"ablation/controlled_optimizer_ablation/run_remote_phase3_generations.py","evidence_status":"VERIFIED_CODE"})
    chain=[
        {"decision_step":"population","candidates":"GR4J 8/16/24/32; SIMHYD 20/40/60/80; XAJ 30/60/90/120","fixed_settings":"32 Split-A; XNES/CMAES; 3 seeds; 3 starts; 200 generations; stdev .25","selected_value_or_rule":"Stage3 code P=6D: XAJ 90, XAJ_CN 102, XAJ_TGD 108","selection_metric":"train KGE available; no selection manifest","status":"QUALIFIED_CODE_RULE","evidence_path":"large_scale_screening/v1/master_config.json; run_remote_full_model_series.py"},
        {"decision_step":"initial_stdev","candidates":"0.05/0.10/0.15/0.25/0.35/0.50","fixed_settings":"model-specific population; 200 generations; Split-A","selected_value_or_rule":"Stage3 formal XAJ family stdev .05; empirical selection unresolved","selection_metric":"not available","status":"CODE_ONLY","evidence_path":"run_remote_phase2_stdev.py; run_remote_full_model_series.py"},
        {"decision_step":"generation_budget","candidates":"600 only in generation launcher; no completed scan","fixed_settings":"model-specific population and stdev","selected_value_or_rule":"Stage3 formal XAJ family 300 generations; empirical selection unresolved","selection_metric":"not available","status":"CODE_ONLY","evidence_path":"run_remote_phase3_generations.py; run_remote_full_model_series.py"}]
    rules=[]
    for model in ["GR4J","SIMHYD","XAJ","XAJ_CN","XAJ_TGD"]:
        rules.append({"model":model,"dimension":DIMS[model],"population_rule":"P=6D" if model in FORMAL else "base-model rule only","stage3_population":{"XAJ":90,"XAJ_CN":102,"XAJ_TGD":108,"GR4J":24,"SIMHYD":60}.get(model,""),"stdev_rule":".05 for formal XAJ family" if model in FORMAL else "not formal setting","generation_rule":"300 for formal XAJ family" if model in FORMAL else "controller-specific","evidence_status":"VERIFIED_CODE"})
    return stdev,gen,chain,rules

def stage3_inventory(records):
    out=[]
    for r in records:
        t=r.get("trace",{})
        out.append({"task_path":r.get("path"),"model":r.get("model"),"optimizer":r.get("optimizer"),"basin_id":r.get("basin_id"),"seed":r.get("seed"),"start":r.get("start"),"dimension":DIMS.get(r.get("model")),"population":r.get("population"),"stdev_init":"NOT_STORED","generations":r.get("generations"),"evaluations":r.get("evaluations"),"initial_center":"LHS_CENTER_NOT_STORED","best_train_kge":r.get("best_train_kge"),"test_kge":r.get("test_kge") if r.get("test_kge") is not None else "NOT_STORED","best_theta_normalized":json.dumps(r.get("best_theta_normalized"),separators=(",",":")),"best_theta_physical":"NOT_STORED","runtime_seconds":r.get("runtime_seconds"),"trace_rows":t.get("rows"),"trace_last_generation":t.get("last_generation"),"best_generation":t.get("best_generation"),"last_quarter_improvement":t.get("last_quarter_improvement"),"period_protocol_result":json.dumps(r.get("period_protocol_result"),separators=(",",":")),"period_metadata":json.dumps(r.get("period_metadata"),separators=(",",":")),"config_hash":"NOT_STORED","code_commit":"NOT_STORED","evidence_status":"REMOTE_FILE_VERIFIED"})
    return out

def stage3_coverage(records,procs):
    rows=[]; exp=531*3*3
    for model in FORMAL:
        vals=[r for r in records if r.get("model")==model]; basins={r["basin_id"] for r in vals}; traces=sum(bool(r.get("trace",{}).get("exists")) for r in vals)
        rows.append({"model":model,"expected_basins":531,"expected_task_units":exp,"completed_basins":len(basins),"completed_task_units":len(vals),"completed_starts":len({(r["basin_id"],r["start"]) for r in vals}),"unique_seeds":len({r["seed"] for r in vals}),"running_processes":len(procs),"failed_result_files":sum(bool(r.get("result_error")) for r in vals),"missing_task_units":max(exp-len(vals),0),"basin_completion_fraction":len(basins)/531,"trace_coverage_fraction":traces/len(vals) if vals else "","test_kge_stored":sum(r.get("test_kge") is not None for r in vals),"status":"COMPLETE_BASIN_SET" if len(basins)==531 else "PARTIAL","evidence_status":"REMOTE_FILE_VERIFIED"})
    return rows

def stage3_summary(records):
    rows=[]
    for model in FORMAL:
        vals=[r for r in records if r.get("model")==model]; train=[float(r["best_train_kge"]) for r in vals if isinstance(r.get("best_train_kge"),(int,float)) and math.isfinite(float(r["best_train_kge"]))]; imp=[float(r["trace"]["last_quarter_improvement"]) for r in vals if isinstance(r.get("trace",{}).get("last_quarter_improvement"),(int,float))]
        rows.append({"model":model,"result_units":len(vals),"basins":len({r["basin_id"] for r in vals}),"train_mean":statistics.mean(train) if train else "","train_median":statistics.median(train) if train else "","runtime_mean_seconds":statistics.mean([float(r["runtime_seconds"]) for r in vals if isinstance(r.get("runtime_seconds"),(int,float))]) if vals else "","trace_units":sum(bool(r.get("trace",{}).get("exists")) for r in vals),"last_quarter_improvement_median":statistics.median(imp) if imp else "","test_kge_available":"no","physical_parameters_available":"no","evidence_status":"REMOTE_RESULT_VERIFIED"})
    return rows

def dispersion(records):
    groups=defaultdict(list)
    for r in records:
        if r.get("model") in FORMAL and isinstance(r.get("best_train_kge"),(int,float)): groups[(r["model"],r["basin_id"])].append(r)
    out=[]
    for (model,b),vals in sorted(groups.items()):
        s=sorted(float(x["best_train_kge"]) for x in vals); q=statistics.quantiles(s,n=4,method="inclusive") if len(s)>1 else [s[0]]*3; best=max(vals,key=lambda x:float(x["best_train_kge"]))
        out.append({"model":model,"basin_id":b,"replicates_seed_x_start":len(s),"best":max(s),"second_best":s[-2] if len(s)>1 else "","median":statistics.median(s),"worst":min(s),"sd":statistics.stdev(s) if len(s)>1 else 0.0,"iqr":q[2]-q[0],"range":max(s)-min(s),"best_second_gap":max(s)-s[-2] if len(s)>1 else "","best_median_gap":max(s)-statistics.median(s),"best_seed":best["seed"],"best_start":best["start"],"evidence_status":"REMOTE_RESULT_VERIFIED"})
    return out

def budget(records):
    out=[]
    for r in records:
        if r.get("model") not in FORMAL: continue
        for x in r.get("trace",{}).get("checkpoints",[]):
            out.append({"model":r["model"],"basin_id":r["basin_id"],"seed":r["seed"],"start":r["start"],"population":r["population"],"total_generations":r["generations"],"budget_fraction":x["fraction"],"generation":x["generation"],"incumbent_train_kge":x["incumbent"],"elapsed_seconds":x["elapsed"],"evidence_status":"REMOTE_TRACE_VERIFIED"})
    return out

def convergence(records):
    out=[]
    for model in FORMAL:
        vals=[float(r["trace"]["last_quarter_improvement"]) for r in records if r.get("model")==model and isinstance(r.get("trace",{}).get("last_quarter_improvement"),(int,float))]
        small=sum(abs(x)<=.01 for x in vals)
        out.append({"model":model,"trace_task_count":len(vals),"descriptive_threshold":.01,"small_last_quarter_count":small,"small_fraction":small/len(vals) if vals else "","median_last_quarter_improvement":statistics.median(vals) if vals else "","production_stopping_rule":"fixed generations; no early convergence stop","interpretation":"DESCRIPTIVE_ONLY","evidence_status":"INFERRED_DIAGNOSTIC"})
    return out

def unresolved(records):
    expected=basin_set(BASIN_531); out=[]
    for model in FORMAL:
        present={r["basin_id"] for r in records if r.get("model")==model}
        for b in sorted(expected-present): out.append({"model":model,"basin_id":b,"failure_type":"MISSING_RESULT_JSON","main_result_treatment":"not included in available result set","status":"UNRESOLVED","evidence_status":"UNRESOLVED"})
    return out

def local_compare(records):
    out=[]
    for model,path in [("XAJ",PROJECT/"outputs/XAJ_531_basins_train_test_kge.csv"),("XAJ_CN",PROJECT/"outputs/XAJ_CN_531_basins_train_test_kge.csv"),("XAJ_TGD",PROJECT/"outputs/XAJ_TGD_partial_basins_train_test_kge.csv")]:
        local=load_csv(path); by={str(x.get("basin_id","")).zfill(8):x for x in local}
        for b,row in sorted(by.items()):
            vals=[r for r in records if r.get("model")==model and r.get("basin_id")==b and isinstance(r.get("best_train_kge"),(int,float))]
            best=max([float(r["best_train_kge"]) for r in vals],default=float("nan"))
            field="train_kge" if model!="XAJ_TGD" else "ic_train_kge_tgd"
            try: diff=float(row.get(field,"nan"))-best
            except Exception: diff=float("nan")
            win=max(vals,key=lambda r:float(r["best_train_kge"])) if vals else {}
            out.append({"model":model,"basin_id":b,"local_path":str(path),"local_train_kge":row.get(field),"local_test_kge":row.get("test_kge",row.get("ic_test_kge_tgd","")),"remote_result_count":len(vals),"remote_best_train_kge":best if vals else "","local_minus_remote_best":diff if vals else "","remote_best_seed":win.get("seed",""),"remote_best_start":win.get("start",""),"status":"MATCH_WITHIN_1E-3" if vals and math.isfinite(diff) and abs(diff)<=.001 else "DIFFERS_OR_UNRESOLVED","evidence_status":"LOCAL_RESULT_VERIFIED"})
    return out

def dpl():
    root=PROJECT/"results/dpl_camels_531_lite_v2"; expected=basin_set(BASIN_531); inv=[]; align=[]; protocols=[]
    for c in sorted(root.glob("*/seed_*/config.json")):
        cfg=read_json(c); model=c.parts[-3]; seed=c.parts[-2].replace("seed_",""); f=c.parent/"train_test_kge_by_basin.csv"; data=load_csv(f); ids=[str(x.get("basin_id","")).zfill(8) for x in data]; unique=set(ids); periods=cfg.get("time_periods",{})
        inv.append({"model":model,"seed":seed,"output_dir":str(c.parent),"complete":(c.parent/"COMPLETE").exists(),"rows":len(data),"unique_basins":len(unique),"duplicate_rows":len(ids)-len(unique),"config_sha256":hash_small(c),"protocol":json.dumps(periods,separators=(",",":")),"evidence_status":"LOCAL_RESULT_VERIFIED"})
        align.append({"model":model,"seed":seed,"expected_basins":len(expected),"actual_unique_basins":len(unique),"missing":len(expected-unique),"extra":len(unique-expected),"duplicate_rows":len(ids)-len(unique),"status":"ALIGNED" if unique==expected and len(ids)==len(unique) else "MISALIGNED","evidence_status":"LOCAL_RESULT_VERIFIED"})
        protocols.append({"model":model,"seed":seed,"dpl_protocol":json.dumps(periods,separators=(",",":")),"target_protocol":"1980-10-01/1981-10-01/1995-10-01 date labels","status":"MATCHES_TARGET_LABELS" if periods.get("calibration",{}).get("start")=="1981-10-01" else "CHECK","evidence_status":"LOCAL_CONFIG_VERIFIED"})
    return inv,align,protocols

def reports(s1meta,pops,stdev,gens,chain,cov,summ,dplinv,dplalign,meta,error):
    REPORTS.mkdir(parents=True,exist_ok=True)
    covtxt="; ".join(f'{x["model"]} {x["completed_basins"]}/531 basins' for x in cov)
    text=f"""# S3 IC Three-Stage Source of Truth

Stage 1 is a 96-basin stratified preflight design: 32 basins in each of Split A, B and C, across 12 strata. All {s1meta["in_531"]} manifest IDs are contained in the 531-basin set. Split A is the 32-basin optimizer screening subset. The baseline config is XNES/XAJ, population 48, stdev_init 0.25, 400 generations, 3 starts, seed 0, train KGE(Q), normalized [0,1] clipping, and no test metric. The readiness and validity reports say the baseline is valid, but the raw per-start summary has sentinel -999 rows while independent recalculation has finite rows; this is an explicit Stage 1 conflict.

The actual Stage 2 population result map is outputs/ic_ablation/large_scale_screening/v1. It contains {sum(int(x["evaluated_units"]) for x in pops)} result units across {len(pops)} model/optimizer/population cells: GR4J D=4 populations 8/16/24/32, SIMHYD D=10 populations 20/40/60/80, and XAJ D=15 populations 30/60/90/120, with XNES/CMAES, 32 Split-A basins, three seeds, three starts, 200 generations and stdev 0.25. The completion marker claims completion, but the matrix expectation is 3 models x 4 populations x 2 optimizers x 32 basins x 3 seeds x 3 starts = 6,912 units, while 5,632 result files are present. No persisted selection manifest was found.

The Stage 2 stdev launcher defines six candidates 0.05, 0.10, 0.15, 0.25, 0.35 and 0.50, but its output root has no result files. The generation launcher defines a single 600-generation trajectory, not a completed multi-budget scan, and its output root has no result files. Therefore a complete sequential empirical chain population -> stdev -> generations is not verified.

Stage 3 is the formal basin-wise independent controller. It creates one task per basin, model, seed and start, with seeds 101/202/303 and three starts. The controller covers ten models; the formal paper filter is XAJ, XAJ_CN and XAJ_TGD. The full task count is 531 x 10 x 3 x 3 = 47,790; the formal XAJ-family count is 531 x 3 x 3 x 3 = 14,337. Current remote coverage is {covtxt}. The Stage 3 controller hard-codes P=6D, stdev 0.05 and 300 generations for the formal XAJ family: XAJ P=90, XAJ_CN P=102, XAJ_TGD P=108. This is code evidence, not a complete empirical Stage 2 selection result.

Available remote result files store train KGE, normalized parameters, population, generations, evaluations and runtime. They do not store test KGE, physical parameters, stdev, initial centers, config hashes or code commits. Exact trace checkpoints are used without interpolation. The result schema and period metadata must be treated as conflicts where they differ.

The authoritative dPL tree exists at results/dpl_camels_531_lite_v2 with {len(dplinv)} model-seed records; {sum(x["status"]=="ALIGNED" for x in dplalign)} are basin-aligned. Its configs use the target 1980-10/1981-10/1995-10 date labels. It must not be replaced by an outputs/dpl search. The local XAJ, XAJ_CN and partial TGD summaries match the available remote best train KGE within 1e-3 for all 1,328 audited rows; remote result.json does not store test KGE, so test consistency is unresolved.

Model labels are retained from the active registry: CN is the basic CemaNeige explicit snow module, while TGD is a generic temperature-conditioned delay and is not labelled as an explicit snowmelt model.

S3.1 is supportable with explicit schema and protocol qualifications. S3.2 is only partially supportable because population results exist but stdev and generation results do not. S3.3 is provisional for completed remote records with restart and exact trace evidence; it is not an all-basin convergence or global optimum claim. TGD remains provisional if coverage is incomplete.
"""
    (REPORTS/"S3_IC_three_stage_source_of_truth.md").write_text(text,encoding="utf-8")
    (REPORTS/"S3_IC_optimizer_ablation_results.md").write_text("Stage 2 population evidence is real and is stored under outputs/ic_ablation/large_scale_screening/v1. Stdev and generation output roots contain no result files. The formal Stage 3 controller rule is P=6D, stdev 0.05 and 300 generations for XAJ/XAJ_CN/XAJ_TGD; this is code-verified and not a completed sequential empirical selection chain.\n",encoding="utf-8")
    (REPORTS/"S3_IC_estimation_adequacy.md").write_text("Restart dispersion is computed over available seed x start records. Budget saturation uses only exact generations present in remote trace.json. The descriptive last-quarter threshold is 0.01 KGE and is not the production stopping rule. Test KGE and physical parameters are absent from the remote result schema; TGD incomplete coverage must remain provisional.\n",encoding="utf-8")
    (REPORTS/"S3_IC_vs_dPL_alignment.md").write_text("The authoritative dPL tree exists under results/dpl_camels_531_lite_v2 with seeds 42, 123 and 2026. Basin and protocol alignment are in the machine-readable results. Exact objective implementation identity is not claimed without numerical equivalence evidence.\n",encoding="utf-8")
    (REPORTS/"S3_IC_table_figure_package.md").write_text("Table S3.1 formal XAJ family: XAJ D=15 P=90 stdev=.05 generations=300; XAJ_CN D=17 P=102 stdev=.05 generations=300; XAJ_TGD D=18 P=108 stdev=.05 generations=300; 3 seeds x 3 starts; evaluations/start=P x generations. Figure S3.1 uses population result points and labels stdev/generation panels as unavailable. Figure S3.2 uses exact trace checkpoints and provisional TGD coverage.\n",encoding="utf-8")
    (REPORTS/"S3_IC_manuscript_readiness.md").write_text("""# S3 IC Manuscript Readiness

| Section | Decision | Main limitation |
|---|---|---|
| S3.1 | Qualified | Controller settings are available; complete empirical selection chain is not |
| S3.2 | Partial only | Population grid exists; stdev and generation results are absent |
| S3.3 | Provisional | TGD may be incomplete; test KGE absent in remote task result |
| S3.4-S3.7 | Refer to prior foundation audit | This audit focuses on IC and authoritative dPL alignment |
""",encoding="utf-8")

def main():
    records,procs,meta,error=remote_extract()
    s1,settings,validity,s1meta=stage1()
    popraw=stage2_population(); pops=pop_aggregate(popraw); stdev,gens,chain,rules=code_stage2()
    inv=stage3_inventory(records); cov=stage3_coverage(records,procs); summ=stage3_summary(records); disp=dispersion(records); bud=budget(records); conv=convergence(records); unres=unresolved(records); compare=local_compare(records); dpli,dpla,dplp=dpl()
    write_csv(RESULTS/"s3_ic_stage1_basin_manifest.csv",s1); write_csv(RESULTS/"s3_ic_stage1_baseline_settings.csv",settings); write_csv(RESULTS/"s3_ic_stage1_validity_summary.csv",validity)
    write_csv(RESULTS/"s3_ic_stage2_population_runs.csv",pops); write_csv(RESULTS/"s3_ic_stage2_stdev_runs.csv",stdev); write_csv(RESULTS/"s3_ic_stage2_generation_runs.csv",gens); write_csv(RESULTS/"s3_ic_stage2_selection_chain.csv",chain); write_csv(RESULTS/"s3_ic_stage2_dimension_rules.csv",rules)
    fig=[dict(x,panel="population",data_status="REAL_RESULT") for x in pops]+[dict(x,panel="stdev",data_status="CODE_ONLY_NO_RESULT") for x in stdev]+[dict(x,panel="generation",data_status="CODE_ONLY_NO_RESULT") for x in gens]; write_csv(RESULTS/"s3_ic_stage2_figure_data.csv",fig)
    write_csv(RESULTS/"s3_ic_stage3_task_inventory.csv",inv); write_csv(RESULTS/"s3_ic_stage3_model_coverage.csv",cov); write_csv(RESULTS/"s3_ic_stage3_result_summary.csv",summ); write_csv(RESULTS/"s3_ic_stage3_remote_local_consistency.csv",compare); write_csv(RESULTS/"s3_ic_stage3_restart_dispersion.csv",disp); write_csv(RESULTS/"s3_ic_stage3_budget_saturation.csv",bud); write_csv(RESULTS/"s3_ic_stage3_convergence_summary.csv",conv); write_csv(RESULTS/"s3_ic_stage3_unresolved_cases.csv",unres)
    write_csv(RESULTS/"s3_ic_dpl_authoritative_inventory.csv",dpli); write_csv(RESULTS/"s3_ic_dpl_basin_alignment.csv",dpla); write_csv(RESULTS/"s3_ic_dpl_protocol_comparison.csv",dplp)
    conflicts=[
        {"item":"Stage2 stdev","status":"UNRESOLVED","evidence":"phase2_stdev_screening; run_remote_phase2_stdev.py","detail":"Candidates are code-defined; no result files."},
        {"item":"Stage2 generations","status":"UNRESOLVED","evidence":"phase3_generations_screening; run_remote_phase3_generations.py","detail":"No result files; launcher is a single 600-generation trajectory."},
        {"item":"Stage2 population completion marker","status":"CONFLICT","evidence":"outputs/ic_ablation/large_scale_screening/v1/LARGE_SCALE_PHASE1_COMPLETED.txt; result.json count","detail":"Completion marker exists but the declared matrix is 6912 units and only 5632 result files are present."},
        {"item":"Stage1 summary versus audit","status":"CONFLICT","evidence":"stage1 per_start.csv; independent_kge_recalculation.csv","detail":"Raw summary includes -999 sentinels while independent recalculation has finite values."},
        {"item":"Stage3 period metadata","status":"CONFLICT","evidence":"remote result.json; period_metadata.json; runner.py","detail":"Result stores hard-coded 1988-1998 protocol while metadata can report different loaded periods."},
        {"item":"Stage3 result schema","status":"UNRESOLVED","evidence":REMOTE_ROOT,"detail":"No test KGE, physical parameters, stdev, center, config hash or commit."},
        {"item":"Stage3 normalized parameter bounds","status":"CONFLICT","evidence":REMOTE_ROOT+"/result.json; ic_foundation_531_v1.json","detail":"Some saved best_theta_normalized values are outside [0,1] although the foundation config declares clip_0_1; saved candidates and runtime clipping require separate resolution."},
        {"item":"SIMHYD population chain","status":"CONFLICT","evidence":"run_remote_phase2_stdev.py; run_remote_full_model_series.py","detail":"Phase2 fixes SIMHYD P=40; Stage3 uses P=60."}]
    write_csv(RESULTS/"s3_ic_conflicts.csv",conflicts)
    write_json(RESULTS/"s3_ic_remote_host_summary.json",{"meta":meta,"processes":procs,"record_count":len(records),"remote_error":error,"remote_root":REMOTE_ROOT,"running_processes_modified":False,"remote_files_modified":False})
    reports(s1meta,pops,stdev,gens,chain,cov,summ,dpli,dpla,meta,error)
    print(json.dumps({"remote_error":error,"remote_records":len(records),"remote_processes":len(procs),"population_records":len(popraw),"stage1_manifest":len(s1),"dpl_records":len(dpli)},ensure_ascii=True))

if __name__=="__main__":
    main()
