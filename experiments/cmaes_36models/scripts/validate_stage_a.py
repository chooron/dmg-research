from __future__ import annotations
from _common import EXPERIMENT, ROOT
import csv, json, subprocess, sys


def rows(path):
    with open(path,newline="") as handle: return list(csv.DictReader(handle))


def main() -> int:
    wb=rows(ROOT/"dmotpy/validation_results/core_water_balance/core_water_balance_summary.csv")
    grad=rows(ROOT/"dmotpy/validation_results/model_gradcheck_water_balance_tests/model_gradient_end_to_end_summary.csv")
    wb_ok=len(wb) >= 36 and all(str(r.get("pass_fail", "")).lower()=="true" for r in wb)
    grad_ok=len(grad)==36 and all(r.get("status") in {"passed","expected_skip"} for r in grad)
    env={**__import__("os").environ,"PYTHONPATH":str(ROOT/"dmotpy")}
    focused=[str(ROOT/"dmotpy/tests/test_flex_saturation3_parameter_bound_fix.py"),str(ROOT/"dmotpy/tests/test_hbv96_gradient_activation.py"),str(ROOT/"dmotpy/tests/test_modhydrolog_fixes.py")]
    result=subprocess.run([sys.executable,"-m","pytest","-q",*focused],cwd=ROOT,env=env,text=True,capture_output=True)
    payload={"passed":bool(wb_ok and grad_ok and result.returncode==0),"water_balance_cases":len(wb),"water_balance_failures":sum(str(r.get("pass_fail","")).lower()!="true" for r in wb),"gradient_models":len(grad),"gradient_failures":sum(r.get("status") not in {"passed","expected_skip"} for r in grad),"boundary_gradient_pytest_returncode":result.returncode,"boundary_gradient_pytest_tail":result.stdout[-1000:]}
    (EXPERIMENT/"results/stage_a_validation.json").write_text(json.dumps(payload,indent=2))
    (EXPERIMENT/"reports/stage_a_validation.md").write_text("# Stage A existing-validation certification\n\n"+"\n".join(f"- {k}: {v}" for k,v in payload.items()))
    return 0 if payload["passed"] else 1

if __name__ == "__main__": raise SystemExit(main())
