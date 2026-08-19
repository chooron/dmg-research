# R4 远端同步清单（remote sync manifest）

生成日期：2026-08-16 ｜ 依据：对 `connect.westb.seetacloud.com` 节点的只读调研（未修改远端任何文件）
机器可读版本：`r4/sync_manifest.json`（含全部期望 sha256）

## 同步原则

1. **不重训**：以下产物同步后即可做 R4 正式 post-hoc forward + snow-state export。
2. **保留可审计性**：IC 同步完整 `per_start.csv`（含每 basin × restart 的 theta_normalized、fitness history、train/test KGE），不预先提炼参数；best-restart 选择在本地按 R1 canonical 规则执行。
3. **dPL 同步完整 seed 目录**：checkpoint（best + 周期 epoch）+ config + attribute_normalization + epoch_history，缺一不可。
4. 本清单**不含** SSH host/port 凭据；端口由用户另行提供（AutoDL 节点重启后端口会变）。

## 必同步项（R4 正式 Base/CN 轨道）

| # | 远端路径（`/autodl-fs/data/dmg_hydro_structure_diagnosis/` 下） | 本地目标 | 内容 | 规模 |
|---|---|---|---|---|
| 1 | `dpl_camels_531_lite_v2/XAJ/seed_{42,123,2026}/` | `results/dpl_camels_531_lite_v2/XAJ/seed_*/` | **dPL Base**：`best_checkpoint.pt` + `checkpoint_epoch_010..100.pt` + `config.json` + `epoch_history.csv` + `attribute_normalization.npz` + `best_parameters_{physical,normalized}.npz` + `basin_final_summary.csv` + `COMPLETE` | ~57 MB |
| 2 | `dpl_camels_531_lite_v2/XAJ_CN/seed_{42,123,2026}/` | `results/dpl_camels_531_lite_v2/XAJ_CN/seed_*/` | **dPL CN**（同上结构） | ~57 MB |
| 3 | `results/ic_cmaes_recalibration_p20_p25_20260728_fused/XAJ/` | `results/ic_cmaes_recalibration_p20_p25_20260728_fused/XAJ/` | **IC Base（fused，5 starts × 200 gens，Q_obs）**：`per_basin.csv`、`per_start.csv`（全 restart 元数据）、`runtime.json`、`convergence_audit.json`、`snow_stratified_summary.csv`、`COMPLETE` | ~12 MB |
| 4 | `results/ic_cmaes_recalibration_p20_p25_20260728_fused/XAJ_CN/` | 同左 | **IC CN（fused）**（同上） | ~11 MB |
| 5 | `results/dpl_camels_531_lite_v2/{XAJ,XAJ_CN}/seed_{42,123}/train_test_kge_by_basin.csv` | `results/dpl_camels_531_lite_v2/{XAJ,XAJ_CN}/seed_{42,123}/` | **R1 统计表复现输入**（3 列：basin_id,train_kge,test_kge） | <1 MB |

合计约 **140 MB**。

## 可选同步项

| 远端路径 | 说明 |
|---|---|
| `results/ic_cmaes_recalibration_p20_p25_20260729_tgd_fused/XAJ_TGD/` | IC legacy-TGD（`tgd_a/tgd_k_slow`，**非** canonical TGD2）— 仅当需要旧 TGD 补充对比 |
| `/autodl-fs/data/phase0_controlled_531_v1/<MODEL>/`（N/D_E/G_E/D_R/G_R，各 5310 raw JSON） | phase0 受控模型（10 starts × 100 gens，canonical batched 格式）— R2/R5 相关 |
| `dpl_camels_531_lite_v2/` 其他模型族（GR4J/GR4J_CN/SIMHYD/SIMHYD_CN/XAJ_TGD/HBV…） | R5 跨结构需要时再同步 |

## 已知缺口（本节点不存在，需另寻来源）

1. **IC canonical 10-start × 300-gen 三件套**：`xaj_base_cmaes_531_batched_paired_v2`、`xaj_cn_cmaes_531_batched_paired_v2`、`xaj_tgd2_cmaes_531_batched_v1`（本地 canonical 名）。本节点只有 fused（5×200）版本——**observation-trained、可用于 R4**，但 R1 正式数字的逐字节复现需要原 paired_v2 产物。
2. **TGD2 一切产物**（IC `xaj_tgd2_cmaes_531_batched_v1` + dPL `dpl_camels_531_lite_v3_tgd2_dpl_audited`）：本节点只有旧 TGD。**R4 TGD2 轨道阻塞**，Base/CN 轨道不受影响。
3. **seed_2026 的 `train_test_kge_by_basin.csv`**：远端仅有 seed 42/123 的 KGE CSV；seed_2026 完整 checkpoint 存在，可在同步后由 checkpoint 直接后处理生成（无需重训）。
4. **R1/R2 统计表**（`results/R1`、`results/R2`、`manuscript/results/R1`）：原机计算，本节点没有；同步 1–5 后用 `manuscript/scripts/build_r1_statistics.py` 重建。

## 同步后完整性校验（每条目的 `integrity.checks`）

```bash
# 1) sha256 逐文件比对（期望值在 sync_manifest.json 的 expected_sha256）
# 2) dPL：config.json 中 model_name == XAJ / XAJ_CN；
#    best_checkpoint.pt 键含 model_name/lite_mode/state_dict/epoch/parameter_names/parameter_specs
#    epoch_history.csv 100 行；basin_final_summary.csv 531 行；
#    attribute_normalization.npz 含 median(35)/scale(35)
# 3) IC fused：per_basin.csv 532 行；per_start.csv 2656 行（531×5 + header）；
#    p_* 物理参数列与模型参数个数一致（XAJ=15，XAJ_CN=17）；COMPLETE 存在
# 4) KGE CSV：531 行，列为 basin_id,train_kge,test_kge
# 5) 本地重跑 r4 smoke 数值校验（见 README）
```

## 建议同步命令形态（端口由用户提供后执行；本清单不实际执行）

```bash
# 从远端拉取（rsync over ssh；远端路径前缀 /autodl-fs/data/dmg_hydro_structure_diagnosis/）
rsync -av --checksum \
  -e "ssh -p <PORT>" \
  root@connect.westb.seetacloud.com:/autodl-fs/data/dmg_hydro_structure_diagnosis/dpl_camels_531_lite_v2/XAJ/ \
  <本地>/results/dpl_camels_531_lite_v2/XAJ/
# ... 其余条目同理；随后按 sync_manifest.json 的 expected_sha256 做校验
```
