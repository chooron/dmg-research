import re

with open(
    "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/ablation/diagnostics/xnes_baseline_audit/audit_main.py",
    "r",
) as f:
    content = f.read()

content = content.replace(
    "obs = target_mm[b_idx].astype(np.float64)[warmup_days:]",
    "obs = target_mm[b_idx].astype(np.float64)",
)

with open(
    "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/ablation/diagnostics/xnes_baseline_audit/audit_main.py",
    "w",
) as f:
    f.write(content)
