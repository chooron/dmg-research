import re

with open(
    "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/ablation/diagnostics/xnes_baseline_audit/audit_main.py",
    "r",
) as f:
    content = f.read()

# Fix evaluate_candidates calls
content = re.sub(
    r'evals = runtime\.evaluate_candidates\(\s*\[b_idx\],\s*cand_tensor\.unsqueeze\(0\),\s*#\s*"train"\s*\)',
    'evals = runtime.evaluate_candidates(cand_tensor.unsqueeze(0), basin_indices=[b_idx], split="train")',
    content,
)

content = re.sub(
    r'ev = runtime\.evaluate_candidates\(\[b_idx\], c_tensor, "train"\)',
    'ev = runtime.evaluate_candidates(c_tensor, basin_indices=[b_idx], split="train")',
    content,
)

content = re.sub(
    r'evals = runtime\.evaluate_candidates\(cand_tensor, basin_indices=\[b_idx\], split="train"\) #\[b_idx\], cands_tensor, "train"\)',
    'evals = runtime.evaluate_candidates(cands_tensor, basin_indices=[b_idx], split="train")',
    content,
)

# Fix sims extraction and CPU KGE calculation
sim_patch = """
            gpu_kge = evals.fitness[0, 0].item()
            
            # Forward pass via model directly
            physical = normalized_to_physical(runtime.model_key, cand_tensor, clip=True).squeeze(0) # (1, D)
            forcing, target_mm, warmup_days = runtime._split_arrays("train")
            forcing_sub = forcing[[b_idx]]
            
            forcing_tensor = torch.from_numpy(forcing_sub).to(device=runtime.device, dtype=torch.float32)
            
            sim_tensor = runtime.model_adapter.model.forward_from_3d_forcing(
                forcing_tensor,
                physical.to(device=runtime.device, dtype=torch.float32)
            )
            sim_cpu = sim_tensor[0].cpu().numpy().astype(np.float64)
            sim_cpu = sim_cpu[warmup_days:]
            
            obs = target_mm[b_idx].astype(np.float64)[warmup_days:]
            valid = bundle.valid_target_mask[b_idx][warmup_days:]
"""

content = re.sub(
    r"gpu_kge = evals\[0\].fitness\[0\].item\(\).*?valid = valid_mask\[b_idx\].numpy\(\)",
    sim_patch,
    content,
    flags=re.DOTALL,
)

content = content.replace("ev[0].fitness[0].item()", "ev.fitness[0, 0].item()")

with open(
    "/home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/ablation/diagnostics/xnes_baseline_audit/audit_main.py",
    "w",
) as f:
    f.write(content)
