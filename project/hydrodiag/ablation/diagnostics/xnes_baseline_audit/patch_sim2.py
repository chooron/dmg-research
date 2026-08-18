import re

with open(
    "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/ablation/diagnostics/xnes_baseline_audit/audit_main.py",
    "r",
) as f:
    content = f.read()

# find and replace the args
old_args = """                forcing_tensor,
                physical_tensor
"""
new_args = """                forcing_tensor,
                physical_tensor,
                forcing_names=bundle.forcing_names,
                temp_mean_train=torch.from_numpy(bundle.temp_mean_train[[b_idx]]).to(device=runtime.device, dtype=torch.float32),
                temp_std_train=torch.from_numpy(bundle.temp_std_train[[b_idx]]).to(device=runtime.device, dtype=torch.float32)
"""

content = content.replace(old_args, new_args)

with open(
    "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/ablation/diagnostics/xnes_baseline_audit/audit_main.py",
    "w",
) as f:
    f.write(content)
