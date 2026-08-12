from __future__ import annotations
from _common import EXPERIMENT
import json


def main() -> int:
    smoke=json.loads((EXPERIMENT/"results/smoke_gate.json").read_text())
    compile_csv=(EXPERIMENT/"benchmarks/compile_benchmark.csv").read_text()
    compile_ok=("BackendCompilerFailed" not in compile_csv and "PythonDispatcher internal assertion" not in compile_csv)
    report=["# Pilot summary",f"smoke_gate: {smoke['passed']}",f"compile_gate: {compile_ok}"]
    if not (smoke["passed"] and compile_ok):
        reasons=[]
        if not smoke["passed"]: reasons.append("Stage A water-balance/boundary/gradient smoke gate is not certified")
        if not compile_ok: reasons.append("the measured torch.compile attempt failed in the installed PyTorch runtime")
        report += ["", "Pilot was not started.", "Reason: " + "; ".join(reasons) + ". No test period, pilot unit, or full-calibration result was used or fabricated."]
        (EXPERIMENT/"reports/pilot_summary.md").write_text("\n".join(report))
        (EXPERIMENT/"results/pilot_gate.json").write_text(json.dumps({"passed":False,"reason":"torch.compile validation failed"},indent=2))
        return 2
    report += ["", "Pilot is eligible to start: Stage A and compiled-kernel gates passed.", "The current script does not yet contain the 36-model production queue; no pilot results have been written."]
    (EXPERIMENT/"reports/pilot_summary.md").write_text("\n".join(report))
    (EXPERIMENT/"results/pilot_gate.json").write_text(json.dumps({"passed":True,"status":"eligible_not_started"},indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
