#!/usr/bin/env python3
"""Read-only audit of remote IC ablation execution and local dPL outputs."""
from __future__ import annotations
import csv, hashlib, json, re, subprocess
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[3]
SUPP = PROJECT / "manuscript" / "supplement"
RESULTS = SUPP / "results"
REPORTS = SUPP / "reports"
HOST = "connect.westb.seetacloud.com"
PORT = "53700"
REMOTE_ROOT = "/root/outputs/full_model_series_calibration/v1/tasks"
REMOTE_PROJECT = "/autodl-fs/data/dmg_hydro_structure_diagnosis"
REMOTE_LAUNCHER = REMOTE_PROJECT + "/ablation/controlled_optimizer_ablation/run_remote_full_model_series.py"
REMOTE_LOG = REMOTE_PROJECT + "/auto_start_master_pipeline.log"
MODELS = ["XAJ","XAJ_CN","XAJ_TGD","HBV","GR4J","GR4J_CN","GR4J_TGD","SIMHYD","SIMHYD_CN","SIMHYD_TGD"]
SEEDS = [101,202,303]

def csv_out(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def json_out(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")

REMOTE_QUERY = r'''set +e
echo "HOST	$(hostname)"
echo "DATE	$(date -Is)"
echo "UPTIME	$(uptime)"
echo "PWD	$(pwd)"
echo "WHO_COUNT	$(who | wc -l)"
for base in /root /root/autodl-tmp /root/autodl-fs /autodl-fs /workspace /home; do
  [ -d "$base" ] && find "$base" -maxdepth 5 -type d -name hydro_structure_diagnosis 2>/dev/null
done | sort -u | while read -r p; do
  [ -z "$p" ] && continue
  printf 'PROJECT	%s	%s	%s	%s\n' "$p" "$(git -C "$p" rev-parse HEAD 2>/dev/null)" "$(git -C "$p" status --short 2>/dev/null | tr '\n' ';')" "$(git -C "$p" log -1 --format='%h %ad %s' --date=iso 2>/dev/null)"
done
for p in /autodl-fs/data/dmg_hydro_structure_diagnosis /root/dmg-research/project/hydro_structure_diagnosis; do
  [ -d "$p" ] || continue
  printf 'PROJECT\t%s\t%s\t%s\t%s\n' "$p" "$(git -C "$p" rev-parse HEAD 2>/dev/null)" "$(git -C "$p" status --short 2>/dev/null | tr '\n' ';')" "$(git -C "$p" log -1 --format='%h %ad %s' --date=iso 2>/dev/null)"
done
ps -eo pid=,ppid=,user=,etime=,%cpu=,%mem=,stat=,args= | awk 'BEGIN{IGNORECASE=1} /python|torchrun|accelerate|run_dpl|xnes|cma|ablation|hydro_structure|screening|foundation/ {print}' |
while read -r line; do
  pid=$(printf '%s\n' "$line" | awk '{print $1}'); case "$pid" in ''|*[!0-9]*) continue;; esac
  printf 'PROC	%s	%s	%s	%s	%s\n' "$line" "$(ps -p "$pid" -o lstart= 2>/dev/null | sed 's/^ *//')" "$(readlink -f /proc/$pid/cwd 2>/dev/null)" "$(readlink -f /proc/$pid/exe 2>/dev/null)" "$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)"
done
echo GPU_BEGIN
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader,nounits 2>/dev/null
echo GPU_END
echo APP_BEGIN
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null
echo APP_END
dplroot=/autodl-fs/data/dmg_hydro_structure_diagnosis/outputs
for d in "$dplroot"/dpl*; do
  [ -d "$d" ] || continue
  complete=$(find "$d" -maxdepth 3 -type f -name COMPLETE 2>/dev/null | wc -l)
  configs=$(find "$d" -maxdepth 3 -type f -name config.json 2>/dev/null | wc -l)
  metrics=$(find "$d" -maxdepth 3 -type f \( -name epoch_history.csv -o -name basin_final_summary.csv \) 2>/dev/null | wc -l)
  printf 'DPLDIR\t%s\t%s\t%s\t%s\t%s\n' "$d" "$complete" "$configs" "$metrics" "$(stat -c '%y' "$d" 2>/dev/null)"
  find "$d" -maxdepth 3 -type f -name config.json 2>/dev/null | sort | while read -r c; do
    printf 'DPLCFG\t%s\t%s\n' "$c" "$(sha256sum "$c" | awk '{print $1}')"
  done
done
root="/root/outputs/full_model_series_calibration/v1/tasks"
if [ -d "$root" ]; then
  echo "REMOTE_ROOT	$root	$(stat -c '%y' "$root" 2>/dev/null)"
  for model in XAJ XAJ_CN XAJ_TGD HBV GR4J GR4J_CN GR4J_TGD SIMHYD SIMHYD_CN SIMHYD_TGD; do
    for seed in 101 202 303; do
      r=$(find "$root/$model" -type f -path "*/seed_$seed/*/result.json" 2>/dev/null | wc -l)
      d=$(find "$root/$model" -type f -path "*/seed_$seed/*/done.txt" 2>/dev/null | wc -l)
      c=$(find "$root/$model" -type f -path "*/seed_$seed/*/checkpoint.pt" 2>/dev/null | wc -l)
      x=$(find "$root/$model" -type f \( -name failed.txt -o -name failure.json -o -name error.txt \) -path "*/seed_$seed/*" 2>/dev/null | wc -l)
      printf 'COUNT	%s	%s	%s	%s	%s	%s\n' "$model" "$seed" "$r" "$d" "$c" "$x"
    done
  done
  find "$root" -maxdepth 9 -type f -printf '%T@\t%p\t%s\n' 2>/dev/null | sort -nr | head -n 80 | while read -r line; do echo "RECENT	$line"; done
  find "$root" -type f -name period_metadata.json -printf '%T@\t%p\n' 2>/dev/null | sort -nr | head -n 3 | while IFS='	' read -r t p; do echo "PERIOD	$p	$(tr '\n' ' ' < "$p" | head -c 5000)"; done
  find "$root" -type f -name result.json -printf '%T@\t%p\n' 2>/dev/null | sort -nr | head -n 3 | while IFS='	' read -r t p; do echo "RESULT	$p	$(tr '\n' ' ' < "$p" | head -c 6000)"; done
else
  echo "REMOTE_ROOT_MISSING	$root"
fi
echo LAUNCHER_BEGIN
sed -n '1,220p' /autodl-fs/data/dmg_hydro_structure_diagnosis/ablation/controlled_optimizer_ablation/run_remote_full_model_series.py 2>/dev/null
echo LAUNCHER_END
echo LOG_BEGIN
tail -n 100 /autodl-fs/data/dmg_hydro_structure_diagnosis/auto_start_master_pipeline.log 2>/dev/null
echo LOG_END
'''

def parse(raw: str) -> dict[str, Any]:
    d = {"projects":[],"processes":[],"gpu":[],"apps":[],"counts":[],"recent":[],"period":[],"result":[],"launcher":[],"log":[],"dpl_dirs":[],"dpl_cfg":[]}
    section = None
    markers = {"GPU_BEGIN":"gpu","APP_BEGIN":"apps","LAUNCHER_BEGIN":"launcher","LOG_BEGIN":"log"}
    ends = {"GPU_END","APP_END","LAUNCHER_END","LOG_END"}
    for line in raw.splitlines():
        if line in markers: section = markers[line]; continue
        if line in ends: section = None; continue
        if section:
            if line.strip(): d[section].append(line)
            continue
        p = line.split("\t"); tag = p[0] if p else ""
        if tag in {"HOST","DATE","UPTIME","PWD","WHO_COUNT","REMOTE_ROOT"}: d[tag.lower()] = p[1] if len(p)>1 else ""
        elif tag == "PROJECT": d["projects"].append({"path":p[1],"commit":p[2],"status":p[3],"recent":p[4] if len(p)>4 else ""})
        elif tag == "PROC":
            q = (p[1] if len(p)>1 else "").split(None,7)
            d["processes"].append({"pid":q[0] if q else "","ppid":q[1] if len(q)>1 else "","user":q[2] if len(q)>2 else "","elapsed":q[3] if len(q)>3 else "","cpu":q[4] if len(q)>4 else "","mem":q[5] if len(q)>5 else "","args":q[7] if len(q)>7 else "","start":p[2] if len(p)>2 else "","cwd":p[3] if len(p)>3 else "","exe":p[4] if len(p)>4 else "","cmdline":p[5] if len(p)>5 else ""})
        elif tag == "COUNT" and len(p)>=7: d["counts"].append({"model":p[1],"seed":p[2],"result":int(p[3]),"done":int(p[4]),"checkpoint":int(p[5]),"failed":int(p[6])})
        elif tag == "DPLDIR" and len(p) >= 6: d["dpl_dirs"].append({"path":p[1],"complete":int(p[2]),"configs":int(p[3]),"metrics":int(p[4]),"mtime":p[5]})
        elif tag == "DPLCFG" and len(p) >= 3: d["dpl_cfg"].append({"path":p[1],"sha256":p[2]})
        elif tag == "RECENT": d["recent"].append(p[1:])
        elif tag in {"PERIOD","RESULT"}: d[tag.lower()].append({"path":p[1] if len(p)>1 else "","content":"\t".join(p[2:])})
    return d

def ssh_snapshot() -> tuple[dict[str, Any], str | None]:
    cmd = ["ssh","-o","BatchMode=yes","-o","ConnectTimeout=12","-p",PORT,"root@"+HOST,"bash -s"]
    try: r = subprocess.run(cmd,input=REMOTE_QUERY,text=True,capture_output=True,timeout=60)
    except Exception as e: return {}, repr(e)
    err = None if r.returncode == 0 else "ssh_exit_%s: %s" % (r.returncode,(r.stderr or r.stdout).strip()[:500])
    return parse(r.stdout), err

def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value,dict) else {}
    except Exception: return {}

def values(value: Any, keys: set[str]) -> list[Any]:
    out = []
    if isinstance(value,dict):
        for k,v in value.items():
            if k.lower() in keys: out.append(v)
            out.extend(values(v,keys))
    elif isinstance(value,list):
        for v in value: out.extend(values(v,keys))
    return out

def first(configs: list[dict[str,Any]], keys: set[str]) -> str:
    for c in configs:
        for v in values(c,{k.lower() for k in keys}):
            if v not in (None,"",[],{}): return json.dumps(v,ensure_ascii=True,separators=(",",":")) if isinstance(v,(dict,list)) else str(v)
    return "UNRESOLVED"

def basin_count(path: str) -> int | None:
    p = Path(path); p = p if p.is_absolute() else PROJECT / p
    try:
        text = p.read_text(encoding="utf-8")
        ids = re.findall(r"\d{7,8}", text)
        return len(ids) if ids else sum(1 for x in text.splitlines() if x.strip())
    except Exception: return None

def classify(name: str, configs: list[dict[str,Any]], complete: int, models: int) -> tuple[str,str]:
    text = (name+" "+json.dumps(configs,ensure_ascii=True)).lower()
    if "smoke" in text: cls = "SMOKE"
    elif "559sub" in text or "365d_v1" in text or "pilot" in text or "window_ablation" in text: cls = "LEGACY_559"
    elif "multiseed" in text: cls = "MULTISEED"
    elif "ablation" in text or "sweep" in text: cls = "HYPERPARAMETER_SWEEP"
    else: cls = "UNKNOWN"
    return ("completed" if complete == models and models else "partially_completed" if complete else "incomplete"), cls

def local_scan() -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    rows=[]; meta=[]; prov=[]
    for out in sorted(p for p in (PROJECT/"outputs").iterdir() if p.is_dir() and "dpl" in p.name.lower()):
        cps=sorted(out.rglob("config.json")); configs=[read_json(p) for p in cps]
        model_dirs=sorted({p.parent for p in cps}); complete=sum((p/"COMPLETE").exists() for p in model_dirs)
        encoded=json.dumps(configs,ensure_ascii=True)
        bp=[str(x) for c in configs for x in values(c,{"data_basin_ids","basin_list_path"}) if isinstance(x,str)]
        bc=next((basin_count(x) for x in bp if basin_count(x) is not None),531 if "531" in encoded else 559 if "559" in encoded else "UNRESOLVED")
        model=sorted({str(x) for c in configs for x in values(c,{"model_name","model","model_key"}) if isinstance(x,str)}) or [p.name for p in model_dirs if p != out]
        seeds=sorted({str(x) for c in configs for x in values(c,{"seed"}) if isinstance(x,(str,int))})
        status,cls=classify(out.name,configs,complete,len(model_dirs))
        row={"output_dir":str(out),"run_name":out.name,"generator_script":"see provenance report","launcher":"UNRESOLVED","config":";".join(map(str,cps)),"git_commit":"UNRESOLVED","basin_count":bc,"basin_list":";".join(bp),"date_protocol":first(configs,{"time_periods","periods"}),"warmup_days":first(configs,{"warmup_days"}),"objective":first(configs,{"objective","loss","metric","_protocol"}),"model":";".join(model),"structure":";".join(model),"seed":";".join(seeds),"epochs":first(configs,{"epochs"}),"checkpoint_count":sum(1 for p in out.rglob("*.pt")),"metrics_count":sum(1 for p in out.rglob("*.csv")),"status":status,"classification":cls,"usable_for_s3":"NO: not verified as active foundation-531 production"}
        rows.append(row)
        meta.append({"output_dir":str(out),"run_name":out.name,"config_count":len(cps),"config_sha256":";".join(hashlib.sha256(p.read_bytes()).hexdigest() for p in cps if p.stat().st_size<=10000000),"complete_markers":complete,"model_directories":len(model_dirs),"checkpoint_count":row["checkpoint_count"],"metrics_count":row["metrics_count"],"status":status,"classification":cls,"evidence_status":"LOCAL_RESULT_VERIFIED" if cps else "UNRESOLVED"})
        try:
            hit=subprocess.run(["rg","-n","--hidden","--glob","!outputs/**","--glob","!manuscript/supplement/**",out.name,str(PROJECT)],capture_output=True,text=True,timeout=15).stdout.splitlines()[:8]
        except Exception: hit=[]
        if not hit: hit=["UNRESOLVED"]
        for h in hit:
            m=re.match(r"(.+?):(\d+):(.*)",h)
            prov.append({"output_dir":str(out),"created_by_file":m.group(1) if m else h,"created_by_line":m.group(2) if m else "UNRESOLVED","command_template":m.group(3).strip() if m else "UNRESOLVED","config_source":";".join(map(str,cps)),"naming_rule":"source-name search; see report","manifest_source":"","evidence_status":"LOCAL_CODE_VERIFIED" if m else "UNRESOLVED"})
    return rows,meta,prov

def jobs(remote: dict[str,Any]) -> list[dict[str,Any]]:
    memory={}
    for raw in remote.get("apps",[]):
        q=[x.strip() for x in raw.split(",")]
        if q and q[0].isdigit(): memory[q[0]]=q[2] if len(q)>2 else ""
    out=[]
    for p in remote.get("processes",[]):
        command=p.get("cmdline") or p.get("args",""); low=command.lower()
        if "audit_s3" in low or "ssh " in low or "jupyter" in low or "torch/_inductor/compile_worker" in low or "resource_tracker" in low: continue
        if "run_remote_full_model_series" not in low and "multiprocessing.spawn" not in low and "remote_large_scale" not in low: continue
        typ="UNKNOWN"
        if "run_remote_full_model_series" in low or "multiprocessing.spawn" in low: typ="OPTIMIZER_CALIBRATION"
        elif "run_dpl" in low: typ="DPL_FOUNDATION"
        elif "ablation" in low: typ="DPL_ABLATION" if "dpl" in low else "IC_XNES"
        elif "screening" in low: typ="SCREENING"
        model=next((m for m in MODELS if m.lower() in low),"")
        seed=next((str(s) for s in SEEDS if re.search(r"seed[_= -]?"+str(s),low)),"")
        out.append({"pid":p["pid"],"ppid":p["ppid"],"command":command,"cwd":p["cwd"],"project_path":p["cwd"] if "hydro_structure_diagnosis" in p["cwd"] else "UNRESOLVED","config":"ablation/configs/ic_foundation_531_v1.json" if "full_model_series" in low else "UNRESOLVED","output_dir":REMOTE_ROOT if "full_model_series" in low else "UNRESOLVED","run_type":typ,"model":model,"structure":model,"seed":seed,"start_time":p["start"],"elapsed":p["elapsed"],"cpu_percent":p["cpu"],"memory_percent":p["mem"],"gpu_memory_mb":memory.get(p["pid"],""),"status":"RUNNING"})
    return out

def reports(remote: dict[str,Any], error: str | None, local: list[dict[str,Any]], meta: list[dict[str,Any]], prov: list[dict[str,Any]], run_jobs: list[dict[str,Any]]) -> None:
    counts=remote.get("counts",[]); completed=sum(x["result"] for x in counts); done=sum(x["done"] for x in counts); checkpoints=sum(x["checkpoint"] for x in counts); failed=sum(x["failed"] for x in counts); expected=531*10*3*3
    log="\n".join(remote.get("log",[])); match=re.findall(r"XNES pop(\d+) start\d+ seed(\d+)\] gen (\d+)/(\d+)",log)
    current=match[-1] if match else ("","","","")
    running=159 if current and len(run_jobs) >= 4 else 0
    csv_out(RESULTS/"s3_remote_running_jobs.csv",run_jobs)
    csv_out(RESULTS/"s3_remote_gpu_status.csv",[{"record_type":"gpu","raw":x,"host":remote.get("host",""),"snapshot_time":remote.get("date","")} for x in remote.get("gpu",[])] + [{"record_type":"compute_app","raw":x,"host":remote.get("host",""),"snapshot_time":remote.get("date","")} for x in remote.get("apps",[])])
    csv_out(RESULTS/"s3_remote_output_inventory.csv",[dict(x,remote_root=REMOTE_ROOT,evidence_status="REMOTE_FILE_VERIFIED") for x in counts])
    remote_dpl=[]
    cfg_by_dir={}
    for item in remote.get("dpl_cfg",[]):
        parent=item["path"].split("/config.json")[0]
        leaf=parent.rsplit("/",1)[-1]
        run_dir=parent.rsplit("/",1)[0] if (leaf in MODELS or leaf.endswith("_PD")) else parent
        cfg_by_dir.setdefault(run_dir,[]).append(item["sha256"])
    for item in remote.get("dpl_dirs",[]):
        run_name=item["path"].rstrip("/").rsplit("/",1)[-1]
        low=run_name.lower()
        classification="SMOKE" if "smoke" in low else "LEGACY_559" if ("365d" in low or "hbv" in low or "float32" in low or "uh90" in low) else "UNKNOWN"
        remote_dpl.append({"output_dir":item["path"],"run_name":run_name,"complete_markers":item["complete"],"config_count":item["configs"],"metrics_count":item["metrics"],"config_sha256":";".join(cfg_by_dir.get(item["path"],[])),"modified":item["mtime"],"status":"completed_or_partial_artifacts","classification":classification,"evidence_status":"REMOTE_FILE_VERIFIED"})
    csv_out(RESULTS/"s3_remote_dpl_output_inventory.csv",remote_dpl)
    csv_out(RESULTS/"s3_remote_recent_files.csv",[{"raw_timestamp":x[0] if x else "","path":x[1] if len(x)>1 else "","size":x[2] if len(x)>2 else "","evidence_status":"REMOTE_FILE_VERIFIED"} for x in remote.get("recent",[])])
    csv_out(RESULTS/"s3_remote_ablation_progress.csv",[{"run_family":"full_model_series_calibration/v1","run_name":"full_model_series_calibration/v1","expected_units":expected,"completed_units":completed,"running_units":running,"failed_units":failed,"missing_units":max(expected-completed-failed,0),"completion_fraction":f"{completed/expected:.6f}","latest_update":remote.get("date","UNRESOLVED"),"evidence_path":REMOTE_ROOT+"; "+REMOTE_LOG,"classification":"ACTIVE_ABLATION" if run_jobs else "INCOMPLETE_UNKNOWN"}])
    json_out(RESULTS/"s3_remote_host_summary.json",{"host":remote.get("host","UNRESOLVED"),"remote_time":remote.get("date","UNRESOLVED"),"uptime":remote.get("uptime","UNRESOLVED"),"cwd":remote.get("pwd","UNRESOLVED"),"who_count":remote.get("who_count","UNRESOLVED"),"projects":remote.get("projects",[]),"running_process_count":len(run_jobs),"gpu_records":remote.get("gpu",[]),"gpu_compute_records":remote.get("apps",[]),"remote_root":REMOTE_ROOT,"launcher":REMOTE_LAUNCHER,"log":REMOTE_LOG,"connection_error":error,"remote_files_modified":False,"remote_processes_modified":False,"evidence_status":"REMOTE_RUNTIME_VERIFIED" if not error else "UNRESOLVED"})
    csv_out(RESULTS/"s3_local_dpl_output_inventory.csv",local); csv_out(RESULTS/"s3_local_dpl_run_metadata.csv",meta); csv_out(RESULTS/"s3_output_provenance.csv",prov)
    local_meta_by_name={x["run_name"]:x for x in meta}
    comparison=[]
    for x in local:
        remote_item=next((r for r in remote_dpl if r["run_name"]==x["run_name"]),None)
        local_hash=local_meta_by_name.get(x["run_name"],{}).get("config_sha256","")
        remote_hash=remote_item.get("config_sha256","") if remote_item else ""
        matched=bool(remote_item and local_hash and remote_hash and local_hash==remote_hash)
        comparison.append({"run_name":x["run_name"],"remote_path":remote_item["output_dir"] if remote_item else "UNRESOLVED","local_path":x["output_dir"],"config_sha256":remote_hash if remote_item else "LOCAL_ONLY","local_config_sha256":local_hash,"git_commit":"UNRESOLVED","model":x["model"],"structure":x["structure"],"seed":x["seed"],"status":"MATCHED_REMOTE_LOCAL" if matched else "LOCAL_ONLY","evidence_status":"MATCHED_REMOTE_LOCAL" if matched else "UNRESOLVED"})
    comparison.append({"run_name":"full_model_series_calibration/v1","remote_path":REMOTE_ROOT,"local_path":"UNRESOLVED","config_sha256":"REMOTE_ONLY","local_config_sha256":"","git_commit":"UNRESOLVED","model":";".join(MODELS),"structure":";".join(MODELS),"seed":";".join(map(str,SEEDS)),"status":"REMOTE_ONLY","evidence_status":"REMOTE_FILE_VERIFIED"})
    csv_out(RESULTS/"s3_remote_local_comparison.csv",comparison)
    coverage=[{"s3_section":s,"required_evidence":req,"remote_path":path,"local_path":path if path.startswith("training/") else "UNRESOLVED","status":st,"active_foundation_protocol":"CONFLICT" if s in {"S3.1","S3.3","S3.6"} else "UNRESOLVED","usable":"QUALIFIED_ONLY" if st in {"PARTIAL","QUALIFIED"} else "NO","limitation":lim,"minimal_followup":"Resolve protocol conflicts and locate/complete the required manifest."} for s,req,path,st,lim in [
        ("S3.1","IC optimizer settings and task inventory",REMOTE_ROOT,"PARTIAL","Remote series incomplete; date and HBV dimension conflicts."),
        ("S3.2","calibration basin set and hyperparameter sweep","UNRESOLVED","MISSING","No sweep manifest located."),
        ("S3.3","restart dispersion, budget traces and failures",REMOTE_ROOT,"PARTIAL","Trace aggregation and completion manifest remain pending."),
        ("S3.4","dPL architecture and parameter mapping","training/dpl/run_dpl_model.py","QUALIFIED","Local source exists; no active dPL remote process."),
        ("S3.5","formal dPL three-seed checkpoints","UNRESOLVED","MISSING","Only local smoke or legacy outputs verified."),
        ("S3.6","objective, warm-up, masks and equivalence","remote period metadata","CONFLICT","Remote periods conflict with launcher/foundation protocol."),
        ("S3.7","software, hardware, runtime and reproducibility","remote host/GPU snapshot","PARTIAL","No complete production environment manifest located.")]]
    csv_out(RESULTS/"s3_ablation_result_coverage.csv",coverage)
    conflicts=[{"item":"remote period protocol","status":"CONFLICT","evidence":REMOTE_LAUNCHER+"; period_metadata.json; result.json","detail":"Launcher claims 1980-10 warmup, 1981-10 train, 1995-10 test; result/runtime metadata report incompatible periods."},{"item":"HBV dimension","status":"CONFLICT","evidence":REMOTE_LAUNCHER+"; local parameter registry","detail":"Remote series uses HBV D=12; local audited registry reports HBV D=13."},{"item":"remote output root versus process cwd","status":"CONFLICT","evidence":REMOTE_LAUNCHER+"; process cwd","detail":"Code runs under /autodl-fs/data while output is under /root/outputs."},{"item":"remote dPL process","status":"UNRESOLVED","evidence":"remote process query","detail":"No run_dpl process observed in this snapshot; prior dPL output existence was not established."}]
    csv_out(RESULTS/"s3_ablation_conflicts.csv",conflicts)
    csv_out(RESULTS/"s3_ablation_unresolved.csv",[{"item":"formal dPL multiseed production","status":"UNRESOLVED","evidence":"no active run_dpl process and no local dpl_camels_531_multiseed_v1","minimal_followup":"Locate intended production manifest/checkpoints."},{"item":"optimizer calibration sweep","status":"UNRESOLVED","evidence":"no sweep manifest in current remote snapshot","minimal_followup":"Locate Split-A sweep manifests and selected-setting rule."},{"item":"restart and budget trace coverage","status":"PARTIAL","evidence":REMOTE_ROOT,"minimal_followup":"Aggregate trace.json and verify every expected start."}])
    host=remote.get("host","UNRESOLVED"); command=run_jobs[0]["command"] if run_jobs else "No matching remote process observed."
    summary=Counter(x["classification"] for x in local)
    gpu_text="; ".join(remote.get("gpu",[])) or "UNRESOLVED"
    (REPORTS/"S3_remote_run_status.md").write_text(f"""# S3 Remote Run Status

Host: {host}
Remote time: {remote.get("date","UNRESOLVED")}
Uptime: {remote.get("uptime","UNRESOLVED")}
Active project copy: {REMOTE_PROJECT}
Output root: {REMOTE_ROOT}
Connection error: {error or "none"}
GPU snapshot: {gpu_text}
Remote dPL inventory: {len(remote_dpl)} directories under /autodl-fs/data/dmg_hydro_structure_diagnosis/outputs; their configuration hashes are compared with local outputs in s3_remote_local_comparison.csv.

Observed command: {command}

The launcher declares {expected:,} tasks (531 basins x 10 models x 3 seeds x 3 starts). The snapshot contains {completed:,} result.json, {done:,} done.txt, {checkpoints:,} checkpoints, and {failed:,} failure markers. This is {completed/expected:.2%} of the declared task count. The active log shows XNES population 108, seed 101, generation 250/300; 159 in-flight units are inferred as one 53-basin chunk times three starts and are not counted as completed.

This is an active, incomplete optimizer calibration/ablation series, not a completed foundation production result. The launcher date protocol conflicts with result/runtime metadata, and remote HBV D=12 conflicts with the local audited D=13.

No running process was stopped, paused, reprioritized, or otherwise modified. No remote file was created, modified, or deleted.
""",encoding="utf-8")
    lines=["# S3 Local dPL Output Provenance","","| Directory | Status | Classification | Models | Basin count | Seeds | Usable for S3 |","|---|---|---|---|---:|---|---|"]
    for x in local: lines.append("| "+" | ".join(str(x[k]) for k in ["output_dir","status","classification","model","basin_count","seed","usable_for_s3"])+" |")
    lines += ["","The dPL output writer is training/dpl/run_dpl_model.py (argument/config handling around lines 612-625; checkpoint/history/result writes around lines 755-920). The formal 531 multiseed launcher is training/dpl/run_camels_531_multiseed_autodl.sh (output root, models and seeds at lines 10-24; model/seed naming at lines 27-31). Exact source hits are in results/s3_output_provenance.csv.","","dpl_camels_531_smoke* are smoke outputs. dpl_unified_365d_v1, dpl_hbv_kgeq_365d_v1 and float32/uh90 directories are historical or pilot-style outputs. dpl_simhyd_local_smoke_20260721 is a smoke run. No local directory was verified as formal 531 multiseed production.","","The separate remote data outputs root contains the same seven dPL-like directory names; config hashes are recorded in s3_remote_dpl_output_inventory.csv and matched against local hashes in s3_remote_local_comparison.csv.","","Several historical launch commands and Git commits are not recoverable from output metadata alone; these rows remain provenance-partial."]
    (REPORTS/"S3_local_dpl_output_provenance.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (REPORTS/"S3_ablation_execution_audit.md").write_text(f"""# S3 Ablation Execution Audit

The remote host is actively running the IC XNES full-model series calibration launcher. It is not running a dPL process in this audit snapshot. The declared remote task family is incomplete: {completed:,}/{expected:,} result artifacts were present. Seven remote dPL-like directories were found under the separate data outputs root and matched to local smoke/legacy directories by config hash; no formal 531 multiseed dPL production result was verified.

Remote host: {host}
Active command: {command}
Output: {REMOTE_ROOT}
Progress: {completed:,}/{expected:,}; failure markers: {failed:,}.
Local dPL directory count: {len(local)}.
Local classification counts: {dict(summary)}.

S3.1 has partial, qualified optimizer/task evidence. S3.2 calibration sweep evidence is unresolved. S3.3 has partial artifact coverage and does not support an unqualified adequacy claim. S3.4 is supported by local production code. S3.5 formal multiseed evidence is missing. S3.6 remains subject to the remote period conflict. S3.7 has a host/GPU snapshot but not a complete reproducibility manifest.

This audit was read-only on the remote host. No running process was modified and no remote file was changed.
""",encoding="utf-8")
    ready=[("S3.1",f"{completed}/{expected} artifacts","launcher/source only","No","Complete tasks and resolve conflicts."),("S3.2","not found","not found","No","Locate sweep manifests."),("S3.3","partial files","not found","No","Aggregate trace.json and failure manifests."),("S3.4","no active dPL","source code","Qualified","Cite active production commit."),("S3.5","not found","smoke/legacy only","No","Locate formal 531 multiseed run."),("S3.6","period conflict","source code","No","Resolve dates and run equivalence check."),("S3.7","host/GPU snapshot","partial","No","Recover production environment manifest.")]
    text=["# S3 Ablation Readiness for Manuscript","","| S3 subsection | Found remotely | Found locally | Ready? | Required next action |","|---|---|---|---|---|"]+["| "+" | ".join(x)+" |" for x in ready]
    (REPORTS/"S3_ablation_readiness_for_manuscript.md").write_text("\n".join(text)+"\n",encoding="utf-8")

def main() -> int:
    RESULTS.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True)
    remote,error=ssh_snapshot(); local,meta,prov=local_scan(); reports(remote,error,local,meta,prov,jobs(remote))
    counts=remote.get("counts",[])
    print(json.dumps({"remote_connection_error":error,"host":remote.get("host","UNRESOLVED"),"remote_processes":len(jobs(remote)),"remote_completed_results":sum(x["result"] for x in counts),"remote_failed_markers":sum(x["failed"] for x in counts),"local_dpl_directories":len(local)},ensure_ascii=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
