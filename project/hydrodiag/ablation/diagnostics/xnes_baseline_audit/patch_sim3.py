import re

with open(
    "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/ablation/diagnostics/xnes_baseline_audit/audit_main.py",
    "r",
) as f:
    content = f.read()

# Fix obs masking
mask_patch = """
            obs = target_mm[b_idx].astype(np.float64)[warmup_days:]
            # find valid by looking at non-negative values or finite values? 
            # In KGE calculation calc_kge_cpu we already filter by np.isfinite(obs)!
            # We can just replace valid mask logic. Let's see if there are invalid fitnesses -999.0
            invalid_val = -999.0
            obs[obs < -100] = np.nan
"""

content = re.sub(
    r"obs = target_mm\[b_idx\].astype\(np.float64\)\[warmup_days:\]\s*valid = bundle.valid_target_mask\[b_idx\]\[warmup_days:\]\s*obs\[~valid\] = np.nan",
    mask_patch,
    content,
    flags=re.DOTALL,
)

with open(
    "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/ablation/diagnostics/xnes_baseline_audit/audit_main.py",
    "w",
) as f:
    f.write(content)
