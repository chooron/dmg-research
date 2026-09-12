# S3 Local dPL Output Provenance

| Directory | Status | Classification | Models | Basin count | Seeds | Usable for S3 |
|---|---|---|---|---:|---|---|
| /home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/outputs/dpl_camels_531_smoke | completed | SMOKE | GR4J | 531 | 42 | NO: not verified as active foundation-531 production |
| /home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/outputs/dpl_camels_531_smoke_fullperiod | partially_completed | SMOKE | GR4J;GR4J_CN;GR4J_PD;SIMHYD;SIMHYD_CN;SIMHYD_PD | 531 | 42 | NO: not verified as active foundation-531 production |
| /home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/outputs/dpl_hbv_kgeq_365d_v1 | completed | LEGACY_559 |  | 559 | 42 | NO: not verified as active foundation-531 production |
| /home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/outputs/dpl_simhyd_local_smoke_20260721 | completed | SMOKE | SIMHYD | 531 | 42 | NO: not verified as active foundation-531 production |
| /home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/outputs/dpl_unified_365d_v1 | partially_completed | LEGACY_559 | GR4J;GR4J_CN;SIMHYD;XAJ;XAJ_CN | 559 | 42 | NO: not verified as active foundation-531 production |
| /home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/outputs/dpl_xaj_cn_float32_fix_uh90_full | completed | LEGACY_559 | XAJ_CN | 559 | 42 | NO: not verified as active foundation-531 production |
| /home/jingxin/code/dmg-research/project/hydro_structure_diagnosis/outputs/dpl_xaj_float32_fix_uh90_full | completed | LEGACY_559 | XAJ | 559 | 42 | NO: not verified as active foundation-531 production |

The dPL output writer is training/dpl/run_dpl_model.py (argument/config handling around lines 612-625; checkpoint/history/result writes around lines 755-920). The formal 531 multiseed launcher is training/dpl/run_camels_531_multiseed_autodl.sh (output root, models and seeds at lines 10-24; model/seed naming at lines 27-31). Exact source hits are in results/s3_output_provenance.csv.

dpl_camels_531_smoke* are smoke outputs. dpl_unified_365d_v1, dpl_hbv_kgeq_365d_v1 and float32/uh90 directories are historical or pilot-style outputs. dpl_simhyd_local_smoke_20260721 is a smoke run. No local directory was verified as formal 531 multiseed production.

The separate remote data outputs root contains the same seven dPL-like directory names; config hashes are recorded in s3_remote_dpl_output_inventory.csv and matched against local hashes in s3_remote_local_comparison.csv.

Several historical launch commands and Git commits are not recoverable from output metadata alone; these rows remain provenance-partial.
