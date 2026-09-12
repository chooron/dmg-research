# 3.6 其他水文过程结构差异扩展诊断——训练与数据状态核准报告

> **审计性质**：只读现状审计与数据准备。未训练、未重新率定、未增加 seed、未生成 replay、未生成正式 figure、未覆盖任何 canonical artifact、未提交 git commit。
>
> **审计结论先行**：当前仓库只保存了 ET/Response 受控模型的代码、参数规格和 dPL seed 42 配置；没有发现这五个受控模型的 IC/dPL 训练输出、replay、过程诊断、P5 校正输出或 3.6 freeze registry。因此下文对 ET/Response 的数值结果全部标为 **NOT AVAILABLE / PENDING**，不采用旧 handoff 中的候选数字。

## 1. Executive verdict

当前**不能开始写有数值结论的 3.6**；需要先完成五个受控结构在 IC 和 dPL 下的实际训练/结果落盘，再做既定的 replay 与诊断闭环。当前 cross-process 状态为 **`NOT_FROZEN`**，存在 blocker：受控训练结果和一级诊断证据缺失，而不是仅差图件排版。没有在本轮启动任何训练。

可以先写的只有方法性骨架：受控模型确实按单因素设计实现，N→D/G 不是复杂度连续谱；但“ET 近似等效”“dry-down 非等效”“Response targeted signature 重组”等科学判断目前均不能从本地仓库核准。

## 2. Repository / git / freeze status

### 2.1 仓库与 worktree

- Repository：`/home/jingxin/code/dmg-research`
- Chapter 3 工程：`project/hydrodiag/`
- Branch：`master`；upstream：`origin/master`
- HEAD：`3caca37a4243ae0a95ebe9cc4f22998672ddf464`
- 审计前 worktree：16 个已修改路径、24 个未跟踪路径。已有修改主要位于 `dmotpy/`、`project/benchmark/` 以及 `project/hydrodiag` 的既有 audit/supplement 文件；本报告未触碰这些文件。

### 2.2 用户指定历史路径核对

以下路径在当前 worktree 均不存在：

- `project/hydrodiag/results/CHAPTER3_AUDIT_REPORT.md`
- `project/hydrodiag/results/JOINT_DPL_IC_MEMORY.md`
- `project/hydrodiag/results/CHAPTER3_ANALYSIS_PROTOCOL.md`
- `project/hydrodiag/results/DATA_AUDIT_REPORT.md`
- `project/hydrodiag/results/EXECUTION_REPORT.md`
- `project/hydrodiag/results/ch3_analysis/`

实际可用的相邻证据包括：

- 模型实现：`project/hydrodiag/models/controlled_composed.py`、`xaj_variants.py`、`xaj.py`、`structure_evaporation.py`、`structure_response.py`
- 参数与映射：`project/hydrodiag/models/parameter_specs.py`、`ablation/ic_core/parameter_adapter.py`
- IC 运行器：`project/hydrodiag/training/ic/run_tgd2_batched_cmaes_531.py`
- dPL 运行器和 registry：`project/hydrodiag/training/dpl/run_dpl_model.py`
- 受控 dPL 配置：`project/hydrodiag/training/dpl/generated_configs/dpl_controlled_531_v1_XAJ_{D_E,G_E,D_R,G_R}_CN_seed_42.json`
- 受控 dPL launcher：`training/dpl/launch_dpl_controlled_531_v1.sh`、`resume_dpl_controlled_531_v1.sh`
- 代码级冻结测试：`tests/test_phase0_single_factor_differential.py`、`tests/test_structure_diagnosis_modules.py`
- Snow 已有核准材料：`CH3_3_3_PARAMETER_COMPENSATION_AUDIT.md`、`CH3_3_4_CONTROLLED_RECOVERY_AUDIT.md`、`CH3_3_5_INTERNAL_STATE_PROCESS_AUDIT.md`

仓库内当前 manuscript-facing draft 仍以 3.5 为跨模型 XAJ/GR4J/SIMHYD 输出扩展，未包含用户所述 ET/Response 3.6 内容。用户指定的最新编号在本报告中统一按 **3.6** 使用；不把历史 3.5 编号迁移为新结论。

### 2.3 Freeze 判定

**唯一判定：`NOT_FROZEN`。**

判定依据：

1. 未找到 `CHAPTER3_ANALYSIS_FREEZE*`、cross-process freeze、ET/Response evidence registry 或 claim registry；
2. 未找到 ET `P < 0.5 mm/day` sensitivity 的任何 machine-readable 输出；
3. 未找到 corrected process-only P5 输出；
4. 没有受控 ET/Response 的训练结果目录可供 freeze。

`manuscript/cache/results_freeze_R1_R5/` 是 R1–R5 manuscript freeze 审计缓存，不是 ET/Response cross-process freeze；其自身报告仍把 R1/R2/R4 部分标为 `PARTIAL`，不能作为本任务的 freeze 证明。

### 2.4 关键源码/manifest checksum

以下 checksum 是本次审计识别 load-bearing 文件的版本锚点；不是 ET/Response 结果 checksum：

| 文件 | SHA-256（当前） |
|---|---|
| `models/controlled_composed.py` | `2317a5b372c9d5fee1965b83ba6227e2d7326af5e1b5dde95c48d9966f0f4925` |
| `models/structure_evaporation.py` | `402b5871d83ff53dd24009f71fdf9aee3befddeee8182e82003bd81d18179` |
| `models/structure_response.py` | `8edc0a7ac2567916a3f793a8cfc7bc66484095a3889f2a8e46b1ae021efecf06` |
| `models/parameter_specs.py` | `2901655a53b7cad0bcb92ee8cd5b6837ec8a9cfb2042f245f565aeacc392ef5d` |
| `ablation/ic_core/model_adapter.py` | `e0325f27cd05b81288c2acbbf52acbdd94449b8b9b24fc81cc6cd08d827fe31e` |
| `training/ic/run_tgd2_batched_cmaes_531.py` | `4d6221d9f703101b07ac7c7bd49c59058610d74f8e5f668fda845adc784ef4ec` |
| `training/dpl/generated_configs/dpl_controlled_531_v1_XAJ_D_E_CN_seed_42.json` | `024db3b3706bb4e969f2f703aaaa6417ac46f522d3ba75588edf1459faa2c5b8` |
| `training/dpl/generated_configs/dpl_controlled_531_v1_XAJ_G_E_CN_seed_42.json` | `d57454f0109685badc1d3ef2a6bdaa47299feec39f423e8428c4ab1c46fbc12c` |
| `training/dpl/generated_configs/dpl_controlled_531_v1_XAJ_D_R_CN_seed_42.json` | `762d18b3fa300858a66f46f5467b8f6dfa763abdd410c4b6237cabd996c68426` |
| `training/dpl/generated_configs/dpl_controlled_531_v1_XAJ_G_R_CN_seed_42.json` | `ec3b490d419dc80c9ecbb9e3841882765203262476cea42384ee63a44e098402` |
| `results/xaj_base_cmaes_531_batched_paired_v2/manifest.json` | `47e72ddc29861736ca05a9f43b15bc650f3e2e35e07c54bd3a0d7a07f0f8f042` |
| `results/xaj_cn_cmaes_531_batched_paired_v2/manifest.json` | `decb80c2bbe3eb254fc037318f4cd501cc432c0a0b6ab97765670ce87394985b` |
| `results/r4_replay_dpl_XAJ_TGD2_seed42/reconstruction_manifest.json` | `fedab803ba2d92f8702cc7a7805ab897d826a0ad1a7ebe0eaccb1ad1af49f8a5` |

## 3. ET / Response model definitions

模型定义来自源码和 `test_phase0_single_factor_differential.py` 的冻结设计，不根据字母名称猜测。

| Variant | ET 结构 | Subsurface response-path 结构 | 保持不变的部分 | 参数增删与总数（XAJ+CN） |
|---|---|---|---|---|
| N | 原生三层 sequential/threshold ET，`xaj_c` active | 原生 RI/RG 两路径，`xaj_ki,xaj_kg,xaj_ci,xaj_cg` | CN、张力水产流生成、自由水分离、Gamma-UH | 17：CN 2 + XAJ 15 |
| D_E | WL/WD 使用同一时刻的 pre-extraction 状态，parallel linear lower/deep ET；等价于 `gamma=1` | 原生 RI/RG response 不变 | CN、XAJ runoff generation、RI/RG、routing | 删除 `xaj_c`；16 |
| G_E | 与 D_E 同一 parallel ET kernel，但使用一个 basin-specific `xaj_gamma` power-law stress，`gamma∈[0.2,5]` | 原生 RI/RG response 不变 | 除 ET 外的部分均不变 | 删除 `xaj_c`、加入 `xaj_gamma`；17 |
| D_R | 原生三层 ET，`xaj_c` active | 单一 linear latent storage `z`，参数 `xaj_kss,xaj_tau0`；不创建 RI/RG identities | CN、张力水容量/产流、自由水分离、surface UH | 删除 `ki,kg,ci,cg`；加入 `kss,tau0`；15 |
| G_R | 原生三层 ET | 单一 power-law `z` response，另加 `xaj_beta∈[0.5,2]` | 与 D_R 相同 | D_R 加 `beta`；16 |

D_R/G_R 的受控路径在 `xaj.py` 中将总 subsurface input 组织为 `xaj_kss * s * fr * (1-im)`，随后进入单一 `z` reservoir；surface runoff 和 Gamma-UH 没有替换。Response 应优先称为“地下水/次表层响应路径结构差异”或 “subsurface response-path organization”，不称为简单的“产流模块替换”。

### 3.1 跨结构可比性

- ET 对照中可直接比较的参数是 16 个共同参数：两项 CN 参数，加上 XAJ 中除 `xaj_c`/`xaj_gamma` 外的共同参数。`xaj_c` 与 `xaj_gamma` 不直接配对解释。
- Response 对照中 N 与 D_R/G_R 只有 13 个共同参数；原生 `ki/kg/ci/cg` 与受控 `kss/tau0/beta` 不可直接当作同一参数。D_R 与 G_R 共有 15 个参数，`beta` 是额外参数。
- N 的内部 response 有 `qi,qg`；有限端点下可构造 `Z_i=CI/(1-CI)·QI`、`Z_g=CG/(1-CG)·QG`、`Z_N=Z_i+Z_g`。D_R/G_R 只有单一 `z`。因此内部状态必须先做明确 aggregation/standardization；不能比较未经处理的 `qi`、`qg` 与 `z` 的绝对量。
- N→D/G 是不同的受控结构替换，不是从简单到复杂的严格连续梯度。

## 4. Training completion matrix

“缺失”表示当前 worktree 没有该受控结果目录，不等于已观测到 531 个训练失败。IC 的 10 starts/300 generations 和 dPL 的 seed 42/100 epochs 是运行器/配置中的**计划协议**，不是完成证据。

### 表 A：训练与参数结果状态

| Regime | Model | Target basins | Completed basins | Failed/missing | Restarts/seeds | Train config | Final epoch/gen | Canonical parameter artifact | Status |
|---|---:|---:|---:|---:|---|---|---|---|---|
| IC | N | 531 | 0 | 531 missing | 10 starts planned | `run_tgd2_batched_cmaes_531.py`; dim 17, pop 25, train-only KGE | — | — | **INCOMPLETE—未执行/无输出** |
| IC | D_E | 531 | 0 | 531 missing | 10 starts planned | same; dim 16, pop 24, 300 generations | — | — | **INCOMPLETE—未执行/无输出** |
| IC | G_E | 531 | 0 | 531 missing | 10 starts planned | same; dim 17, pop 25, 300 generations | — | — | **INCOMPLETE—未执行/无输出** |
| IC | D_R | 531 | 0 | 531 missing | 10 starts planned | same; dim 15, pop 22, 300 generations | — | — | **INCOMPLETE—未执行/无输出** |
| IC | G_R | 531 | 0 | 531 missing | 10 starts planned | same; dim 16, pop 24, 300 generations | — | — | **INCOMPLETE—未执行/无输出** |
| dPL | N（controlled N） | 531 | 0 | 531 missing | controlled seed 42 not observed | registry key `XAJ_CONTROLLED_N_CN` exists；无独立生成配置 | — | — | **INCOMPLETE—未执行/无输出** |
| dPL | D_E | 531 | 0 | 531 missing | seed 42 configured, 0 observed | 100 epochs；AdamW 1e-3；batch 128；cosine；FP32 forward/FP64 metric | — | — | **INCOMPLETE—无输出** |
| dPL | G_E | 531 | 0 | 531 missing | seed 42 configured, 0 observed | same controlled dPL config | — | — | **INCOMPLETE—无输出** |
| dPL | D_R | 531 | 0 | 531 missing | seed 42 configured, 0 observed | same controlled dPL config | — | — | **INCOMPLETE—无输出** |
| dPL | G_R | 531 | 0 | 531 missing | seed 42 configured, 0 observed | same controlled dPL config | — | — | **INCOMPLETE—无输出** |

### 4.1 不可替代的现有参考结果

当前确实存在、但不能替代受控 ET/Response 的结果：

- `results/xaj_base_cmaes_531_batched_paired_v2/`：5310 raw JSON records = 531 basins × 10 starts；全部 `status=complete`，全部 300 generations；`DONE.json` status complete。
- `results/xaj_cn_cmaes_531_batched_paired_v2/`：同样 5310 records、10 starts、300 generations；这是 legacy XAJ+CN，不是 controlled-N。
- `results/xaj_tgd2_cmaes_531_batched_v1/`：5310 records、10 starts、300 generations；这是 TGD2，不是 G_R。
- `results/dpl_camels_531_lite_v2/XAJ_CN/seed_{42,123,2026}/`：每个 seed 均有 `COMPLETE`、531 行 `basin_final_summary.csv`、epoch 100 checkpoint；这是 legacy `XAJ_CN`，不是 controlled-N/D_E/G_E/D_R/G_R。
- `results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2/seed_{123,2026}/`：有完整结果；seed 42 只有部分 checkpoint/physical parameter，缺少 `COMPLETE` 和 final summary。它是 TGD2，不是 controlled Response。

受控 dPL 配置和 launcher 指向 `/autodl-fs/data/dmg_hydro_structure_diagnosis/dpl_controlled_531_v1`；该路径在当前 WSL 中不存在。当前实际不存在 `results/dpl_controlled_531_v1/` 及 `results/{n,d_e,g_e,d_r,g_r}_cmaes_531_batched_v1/` 等对应输出根目录。

## 5. Replay / forcing integrity

### 5.1 协议锚点

已有 XAJ/TGD2 manifests 一致给出：warm-up 1980-10-01–1981-09-30（365 d）、calibration/train 1981-10-01–1995-09-30（5113 d）、test/evaluation 1995-10-01–2010-09-30（5479 d），full axis 1980-10-01–2014-09-30（12418 d），basin N=531，forcing order `P,T,PET`。

### 表 B：Replay / forcing inventory

| Artifact | Path | Regime/model | Basin N | Time length | Variables | Size | Replay consistency | Canonical? |
|---|---|---|---:|---:|---|---:|---|---|
| IC XAJ full arrays | `results/r4_ic_fused_XAJ/ic_fused_XAJ_full_arrays.npz` | IC legacy XAJ | 531 | 12418 | IDs/dates, `q_full,qi,qg,rs_instant,s,fr,wu,wl,wd` | 118,292,825 B | 未发现 ET/Response 专用 replay check | R4 XAJ reference only |
| IC XAJ+CN full arrays | `results/r4_ic_fused_XAJ_CN/ic_fused_XAJ_CN_full_arrays.npz` | IC legacy XAJ+CN | 531 | 12418 | 上述 runoff/state + `evap,effective_precip,G,eTG,rain,melt,sca` | 231,120,665 B | 仅 legacy XAJ+CN asset；非受控 | Legacy reference only |
| dPL XAJ+CN full arrays | `results/r4_official_dpl_XAJ_CN_seed42/official_dpl_XAJ_CN_seed42_full_arrays.npz` | dPL legacy XAJ_CN seed42 | 531 | 12418 | 上述 runoff/state + ET/snow diagnostics | 250,255,839 B | 无受控 ET replay check | Legacy reference only |
| dPL TGD2 reconstructed replay | `results/r4_replay_dpl_XAJ_TGD2_seed42/reconstructed_dpl_XAJ_TGD2_seed42_full_arrays.npz` | dPL TGD2 seed42 | 531 | 12418 | `q_full,qi,qg,rs_instant,s,fr,wu,wl,wd,tgd_*` | 191,959,434 B | manifest 中 7 项 validation，mismatch rows=0；仅 TGD2/F7 | F7-only accepted |
| R1 cached daily simulations | `manuscript/cache/R1_ic_complete_rebuild/`；`manuscript/cache/r1_rebuild_audit_staged/daily_dpl_gpu_compile/` | legacy XAJ/XAJ_CN/TGD2 IC/dPL | 531 per source | test-oriented | parquet 中为 q/obs/metadata；不是 controlled ET/Response state export | 单文件约 62 MB | R1 evaluator scope | Not 3.6 canonical |
| Forcing source | `data/camels_dataset`、`data/camels_dates.npy`、`data/gage_id.npy`、`data/531sub_id.txt` | common CAMELS source | 531 selected / 671 source | 12418 axis | `P,T,PET` + target/attributes | 未在本次报告重新计量 | manifests record P/T/PET and 531 selection | Input source |
| Controlled ET/Response replay | 应为新的 `results/` 或 audit 子目录 | IC/dPL N,D_E,G_E,D_R,G_R | 531 expected | 12418 expected | q + ET/state/PET as protocol requires | **不存在** | **无法核验** | **不存在** |

`results/` 与 manuscript cache 下本次扫描到的所有 `.npz/.parquet` 候选资产合计约 **3,329,926,136 B（3.33 GB）**，但包含 Snow/R3/R4/legacy 资产，不能写成 ET/Response 的 time-series storage size。旧记录中的“约 534 MB”在当前 worktree 中无法复现。现有 arrays 通常不含 PET；PET 在 forcing source 中存在，不能把 `evap` 当作 PET。

## 6. Superseded / dangerous artifacts

### 表 C：Known limitations / dangerous artifacts

| Issue | Current status | Affected analysis | Severity | Safe usage | Unsafe usage |
|---|---|---|---|---|---|
| Controlled ET/Response output absent | 受控目录、参数 artifact、summary、replay 均未发现 | 所有 3.6 数值层 | **Blocker** | 只报告 code/config readiness | 用旧数字当作当前结果 |
| D_E dPL `eval_latest` epoch 90 vs canonical epoch 100 | 当前仓库未找到 `eval_latest`、epoch-90 文件或 D_E 输出；无法核准 exact file/actual analysis source | D_E dPL 全部结果 | High | 标记“未实例化/不可核验” | 把旧 handoff 的 epoch 90/100 当作已冻结事实 |
| dPL N legacy domain | 当前可见 `dpl_camels_531_lite_v2/XAJ_CN` config 是 legacy：`ci∈[0.1,1.0]`、`cg∈[0.9,1.0]`；controlled code 是 `ci∈[0.1,0.9]`、`cg∈[0.9,0.998]` | dPL N 与 D_R/G_R 的参数、tau、response comparison | **Blocker for cross-variant use** | 仅称 legacy XAJ_CN，不能作 controlled-N | 称其为 controlled N 或与 D_R/G_R 直接配对 |
| tau range provenance conflict | code-authoritative `tau0∈[0.4342944819,499.4998332]`；`docs/pre_phase0_subsurface_freeze.md` 仍写 `[0.4342944819,15478.1439023]`，两者不一致 | 所有 Response tau displacement/overrange | **High** | 先以代码为当前 authority，并在训练前修订/确认 protocol | 使用 15478 或旧“约31 basins”而不注明冲突 |
| native tau incompatibility | 本次审计以 `results/xaj_base_cmaes_531_batched_paired_v2/raw/xaj/*.json` 的 5310 条 raw record，按 `max train_objective, tie min start` 选 531 个 basin vector 派生统计；当前 worktree 没有保存 endpoint summary artifact，因此 99/108 是 **audit-derived、未独立落盘核准**。派生值：CI=1 有41、CG=1 有60、union=99；按 code upper 499.4998，finite tau 超上界的 basin=10，union outside/singular=108 | D_R/G_R tau comparison、P4 aggregation | High | 把端点视为 singular，不 epsilon-clamp；将这些数字作为待存档 audit anchor，重新确认 controlled support 后再比较 | 把 audit-derived 计数当 canonical、把 singular endpoint 当有限 tau，或直接引用旧“31” |
| dPL `xaj_tau0` transform conflict | Active `training/dpl/run_dpl_model.py` 的 `physical_parameters()` 实际只对 `DPL_LOG_RESIDENCE_PARAMETERS={tgd_tau_warm,tgd_delta_tau_cold}` 做 log 映射，因此 `xaj_tau0` 当前走 **linear** mapping；`generated_configs/...D_R/G_R...json` 也写 `sigmoid_to_physical_range_linear_v1`。但 `structure_response.py` 的 helper、`parameter_specs.py` 注释和 `docs/dissertation_controlled_response_domain.md` 声明 `tau0` 保留 log mapping | dPL D_R/G_R tau magnitude and boundary interpretation | **Blocker before training** | 训练前明确唯一 transform 并写入 manifest | 将未来结果称为 log-mapped tau 而未验证，或将 helper 的声明当作实际训练路径 |
| IC `xaj_tau0` transform conflict | Active IC `ablation/ic_core/parameter_adapter.py` 的 `LOG_SCALED_PARAMETERS` 只列 TGD2 参数，`xaj_tau0` 当前走 **linear** mapping；但 `structure_response.py` helper、`parameter_specs.py` 注释和 `docs/dissertation_controlled_response_domain.md` 声明 `tau0` 保留 log mapping | IC D_R/G_R tau magnitude | **Blocker before training** | 训练前明确唯一 adapter convention，并写入 manifest | 混用 model helper 与 optimizer adapter 的坐标，或声称实际使用了 log mapping |
| registry/documentation drift | `docs/dissertation_controlled_response_domain.md` 和 `parameter_specs.py` 注释仍说 controlled variants 仅供 forward/tests、未加入 registry/config；但 active `model_adapter.py`、`run_dpl_model.py` 已注册 D_E/G_E/D_R/G_R，且已有四个 controlled dPL 配置 | 训练可执行性、provenance interpretation | Medium | 以 active registry + config + actual output markers 的三方核对为准 | 把旧“未注册”注释当成当前状态，或把 registry/config 当作训练完成证据 |
| boundary pinning reference | legacy XAJ IC selected vector 中：`xaj_c` lower/upper=115/292；`xaj_ci` upper=41；`xaj_cg` upper=60；`xaj_ki` lower/upper=123/89；`xaj_kg` lower/upper=187/91（531 basins） | ET/Response parameter magnitude interpretation | High as reference warning | 只作为 legacy reference，未来 controlled artifact 需独立 audit | 把 legacy pinning 当作 controlled D_E/G_E/D_R/G_R 结果 |
| IC D_R `kss` missingness | 没有 IC D_R parameter/replay/summary，因而 exact missing N 和原因不可计算 | IC D_R support evidence | High | 记为 `NOT ASSESSABLE` | 从旧记录填入 missing N |
| P4 semantic gate | 未找到名为 `COMPARABLE_AFTER_EXPLICIT_AGGREGATION` 的 registry/gate；只有 pre-Phase-0 文档给出 latent storage/Z0 数值语义 | storage–release evidence | High | 只保留为待确认的 semantic requirement | 把未核准的 storage metric 当正文证据 |
| P5 old distance | 未找到 corrected P5；也未找到本地 old P5 artifact | performance–process decoupling | **Blocker for decoupling claim** | 等 corrected process-only vector | 使用含 `ΔKGE` 的旧 distance |
| R3 synthetic and R4 TGD2 assets | R3 `q_star/x_star` 是 Snow synthetic truth；R4 replay 是 TGD2/F7 | 3.6 ET/Response | Medium | 只作 Snow参照或 replay方法参照 | 把 Snow/TGD2 artifact 充当 ET/Response output |
| historical `ic_xnes_production_v1.json` | 旧 config 使用 XNES、1988–2009 period，且未列受控模型；当前 raw XAJ manifests 是 CMA-ES、1980–2010 | IC protocol identification | Medium | 仅作 historical config | 称其为当前 controlled IC config |

## 7. ET output-level evidence

### 表 D：ET output-level differences

当前没有 D_E/G_E 的 IC 或 dPL test/evaluation KGE 结果，以下不是零值，而是缺失值。

| Regime | Contrast | ΔKGE median | 95% CI | median \|ΔKGE\| | ET partition effect | Seasonal timing effect | Interpretation |
|---|---|---:|---|---:|---:|---|---|
| IC | D_E–N | N/A | N/A | N/A | N/A | N/A | **不可核准** |
| IC | G_E–N | N/A | N/A | N/A | N/A | N/A | **不可核准** |
| dPL | D_E–N | N/A | N/A | N/A | N/A | N/A | **不可核准** |
| dPL | G_E–N | N/A | N/A | N/A | N/A | N/A | **不可核准** |

因此不能从当前仓库支持 `aggregate-output-equivalent / near-equivalent`，也不能支持 ET/P、seasonal ET timing 或 Q-equivalent proportion 的任何方向/幅度结论。现有 legacy XAJ+CN `evap` array 不是 ET structural contrast。

## 8. ET dry-period process evidence

### 表 E：ET dry-down process differences

当前没有 dry-spell event catalog、dry-down metric CSV/JSON 或 D_E/G_E replay。旧候选 `49,295 events`、`34,434 usable metrics`、`+0.0052/+0.0082` 等均未在本地 artifact 中核准。

| Regime | Contrast | N basin | N events | ΔAUC median | 95% CI | Δdecay | 95% CI | Sign fraction |
|---|---|---:|---:|---:|---|---:|---|---:|
| IC | D_E–N | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| IC | G_E–N | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| dPL | D_E–N | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| dPL | G_E–N | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

当前不能判断 dry-down 是否系统变化、IC/dPL 是否同向、D_E/G_E 是否都成立，也不能使用 `aggregate-output-equivalent but dry-down-process-non-equivalent` 作为论文结论。

## 9. ET threshold-sensitivity status

**`PENDING — DO NOT CLAIM THRESHOLD ROBUSTNESS`**。

- 主阈值 `P < 1.0 mm/day` 的事件/metric 输出未发现；
- sensitivity `P < 0.5 mm/day` 的 machine-readable 输出未发现；
- 未进入 cross-process freeze registry；
- 本轮不启动该分析，不覆盖任何旧文件。

因此不能分类为 `ROBUST TO 0.5 MM/D THRESHOLD`、`DIRECTIONALLY ROBUST, MAGNITUDE SENSITIVE` 或 `THRESHOLD SENSITIVE`；当前只能是 **PENDING**。

## 10. ET parameter / parameter–process evidence

### 表 F：ET parameter evidence

受控参数值、restart noise reference、boundary mass、dry-down process metric 和 association output 均缺失。源码层面仅可确认 `xaj_c` 被 D_E/G_E 删除、`xaj_gamma` 仅在 G_E 增加；不能推导 shift magnitude 或 correspondence。

| Regime | Contrast | Parameter | Shift magnitude | Process metric | Spearman rho | 95% CI | Boundary concern | Strength |
|---|---|---|---:|---|---:|---|---|---|
| IC | D_E–N | `xaj_c` removed | N/A | N/A | N/A | N/A | 未有 controlled fit | Not assessable |
| IC | G_E–N | `xaj_gamma` added / `xaj_c` removed | N/A | N/A | N/A | N/A | 未有 controlled fit | Not assessable |
| dPL | D_E–N | `xaj_c` removed | N/A | N/A | N/A | N/A | 未有 controlled fit | Not assessable |
| dPL | G_E–N | `xaj_gamma` added / `xaj_c` removed | N/A | N/A | N/A | N/A | 未有 controlled fit | Not assessable |

必须保留的解释边界：未来即使 correspondence 显著，也要将 `small parameter displacement magnitude` 与 `strong k–process correspondence under IC` 分开；目前两者都没有数值证据。不能写成“ET 强参数补偿”。

## 11. Response output and targeted-signature evidence

### 表 G：Response output / signature differences

当前没有 D_R/G_R 的 IC/dPL discharge summary 或 BFI、recession、low-flow/FDC canonical CSV。

| Regime | Contrast | ΔKGE | 95% CI | ΔBFI | Δrecession | Δlow-flow/FDC | Valid N |
|---|---|---:|---|---:|---:|---:|---:|
| IC | D_R–N | N/A | N/A | N/A | N/A | N/A | N/A |
| IC | G_R–N | N/A | N/A | N/A | N/A | N/A | N/A |
| dPL | D_R–N | N/A | N/A | N/A | N/A | N/A | N/A |
| dPL | G_R–N | N/A | N/A | N/A | N/A | N/A | N/A |

不能判断总体 KGE 是否近似不变，也不能判断 targeted response organization 是否改变。现有 `qi/qg` 仅来自 legacy XAJ/XAJ+CN/TGD2 arrays，不能充当 D_R/G_R signature。

## 12. IC G_R parameter-footprint evidence

### 表 H：IC G_R parameter–response association

G_R IC 没有训练、参数或 response metric artifact。因此旧候选 `normalized tau shift≈+0.247`、`median |d_struct|≈1.21`、`rho≈0.398`、`slope≈0.92`、`sign concordance≈0.75` 等全部未核准。

| Quantity | Estimate | 95% CI / spread | Valid N | Robustness | Source |
|---|---:|---|---:|---|---|
| normalized `tau` shift | N/A | N/A | 0 | 无 G_R IC output | missing |
| median `d_struct` (absolute) | N/A | N/A | 0 | 无 parameter artifact | missing |
| `Δlog(recession)` | N/A | N/A | 0 | 无 response CSV | missing |
| Spearman `Δz_tau ↔ Δlog(recession)` | N/A | N/A | 0 | 无 association output | missing |
| robust slope / sign concordance | N/A | N/A | 0 | 无 scatter/robustness output | missing |
| controlled `abs(ΔKGE)` + BFI association | N/A | N/A | 0 | 无 covariate table | missing |

若未来得到该结果，最强安全表述仍只能是 **`parameter footprint associated with response-behavior change`**；禁止写成 `tau causally determines recession`。

## 13. IC D_R and dPL Response evidence

### 表 I：Response regime-dependent expression

| Regime | Contrast | Parameter evidence | Response signature evidence | IC–dPL correspondence | Strength | Limitation |
|---|---|---|---|---|---|---|
| IC | D_R–N | `kss` displacement/missingness 未有 artifact | 未有 BFI/recession/low-flow output | N/A | None | 未训练，不能升级为 supporting evidence |
| IC | G_R–N | G_R parameter footprint 未有 artifact | 未有 targeted signature | N/A | None | 未训练 |
| dPL | D_R–N | 未有 controlled dPL parameter artifact | 未有 targeted signature | N/A | None | seed42 仅有配置；legacy N domain不能替代 |
| dPL | G_R–N | 未有 controlled dPL parameter artifact | 未有 targeted signature | N/A | None | seed42 仅有配置；tau mapping 未定 |

因此当前不能写 `directional evidence of response reorganization under dPL`。该措辞只有在 controlled dPL 至少有可核验 response outputs 且绑定 single-seed/domain limitation 后才可使用。

## 14. Response storage–release evidence

当前未找到 P4 命名 gate，也未找到 D_R/G_R storage–release metric。可以从源码确认语义，但不能宣称已完成内部状态比较：

- N：双路径 `Z_i + Z_g` 必须显式 aggregation；`CI=1`/`CG=1` 为 singular endpoint，不应 epsilon-clamp 成有限 storage；
- D_R/G_R：单一 `z`，带固定 `Z0=3.1553493591016335` 的数值语义；
- 可比较的候选应是显式标准化后的 percentile / relative metric，而不是未经转换的绝对量。

### 表 J：Response internal-state evidence

| Regime | Contrast | Storage metric | Effect | Association with tau | Association with recession | Strength |
|---|---|---|---:|---:|---:|---|
| IC | D_R–N | N/A | N/A | N/A | N/A | Not assessable |
| IC | G_R–N | N/A | N/A | N/A | N/A | Not assessable |
| dPL | D_R–N | N/A | N/A | N/A | N/A | Not assessable |
| dPL | G_R–N | N/A | N/A | N/A | N/A | Not assessable |

即使未来 gate 通过，storage–release 也只能作为 **complementary internal-state evidence**，不得写成 `storage–release explains recession`。正文是否保留应在真正的 figure planning 阶段决定；当前更适合预留附录位置。

## 15. Corrected P5 process-only distance status

**`P5 CORRECTION PENDING`**。

当前没有 corrected output，也没有可登记的 old P5 machine-readable source。对未来 audit 的硬约束为：

- ET vector 只能使用已冻结的 ET/P（或 canonical partition）、dry-persistence、dry-down AUC 和 canonical decay；
- Response vector 只能使用 BFI、recession、low-flow/FDC primary signature；storage–release 只有在 P4 gate 已通过且已冻结时才能加入；
- process-only vector 不得包含 `ΔKGE`。

### 表 K：Corrected performance–process decoupling

| Process | Regime | Contrast | Process-only distance | rho with \|ΔKGE\| | 95% CI | Final strength |
|---|---|---|---:|---:|---|---|
| ET | IC | D_E–N / G_E–N | N/A | N/A | N/A | **PENDING** |
| ET | dPL | D_E–N / G_E–N | N/A | N/A | N/A | **PENDING** |
| Response | IC | D_R–N / G_R–N | N/A | N/A | N/A | **PENDING** |
| Response | dPL | D_R–N / G_R–N | N/A | N/A | N/A | **PENDING** |

P5 correction 不会改变 ET dry-down 或 Response targeted-signature 的一级结果定义；但在当前仓库连一级结果都不存在时，不能把 P5 当作唯一 blocker。任何“performance–process decoupled”的综合统计主张必须等 P5 校正完成后再写。

## 16. Cross-process comparison

Snow 仅作为已完成主案例的 qualitative reference，不在本轮重新分析其数字。

### 表 L：Snow / ET / Response 跨过程证据层级

| Process | Structural intervention | Aggregate output visibility | Parameter footprint | Targeted process/response visibility | Internal-state evidence | IC/dPL regime dependence | Strongest safe claim |
|---|---|---|---|---|---|---|---|
| Snow | Base/CN/TGD：融雪结构缺失、正确 CN 与 generic temperature control | Snow 已在高过程活跃度/高 snow 条件下表现为 output-visible；low-snow 条件弱 | Snow 主案例已有 organized parameter compensation/parameter-space footprint | runoff timing、active-melt/spring response 可见 | R3 synthetic truth-relative state/flux 与 R4 external consistency 已有分层证据 | IC/dPL 方向总体可比较，但不可排名；Snow 结论以既有 freeze/audit 为准 | **在过程活跃条件下，出口差异、参数重组和内部残差可同时出现；出口恢复不等于内部恢复** |
| ET | N vs D_E/G_E：只替换 lower/deep ET organization | **未训练，未知** | **未有 parameter shift/correspondence** | **未有 dry-down catalog/metric** | **未有 ET state evidence** | **未评估** | 只能说“代码已定义受控 ET 对照”；不能给科学效果结论 |
| Response | N vs D_R/G_R：只替换 subsurface response-path organization | **未训练，未知** | **未有 G_R tau footprint 或 D_R kss evidence** | **未有 BFI/recession/low-flow/FDC** | **P4 gate 未核准，未有 storage–release output** | **未评估** | 只能说“代码已定义受控 Response 对照”；不能给科学效果结论 |

当前不能从表 L 推导“ET/Response 复制 Snow 的 parameter compensation pattern”。process-dependent structural identifiability 目前是 3.6 的**待检验上位命题**，不是已完成的 cross-process statistic。

## 17. 3.6 是否需要补训练

### 17.1 分模型 verdict

- **ET IC：`INCOMPLETE`**。D_E/G_E/N controlled IC 均无输出。
- **ET dPL：`INCOMPLETE`**。只有 seed42 配置和 launcher，没有一个受控 ET seed 的完成标记、summary 或 checkpoint。
- **Response IC：`INCOMPLETE`**。D_R/G_R/N controlled IC 均无输出。
- **Response dPL：`INCOMPLETE`**。没有 controlled response seed42 输出；legacy XAJ_CN 不能替代 controlled N，TGD2 不能替代 G_R。

### 17.2 对用户问题的逐项回答

1. **为了写博士论文 3.6 是否需要补训练？** 需要，至少要完成受控 N/D_E/G_E/D_R/G_R 的实际训练并生成 canonical parameter/result artifacts；当前不是“只差核准”。
2. **是否需要补 dPL seeds？** 目前首先缺的是预声明的 seed42 受控训练，不能以“需要更多 seed”作为当前判断。补 seed 123/2026 不是本轮自动 blocker；在 seed42 完成后可把 single-seed 作为限定，再决定是否需要稳健性 seed。
3. **是否需要重新率定 IC？** 需要运行新的受控 IC variants；不是重新率定已有 Snow/XAJ reference。当前本轮不执行。
4. **是否只需核准已有机器可读结果？** 不可以。已有机器可读结果主要是 legacy XAJ/XAJ_CN/TGD2、Snow R3/R4，不能支撑 ET/Response 3.6。
5. **两个 closure checks 是否足够？** 不足以解决当前状态。P5 corrected distance 和 ET 0.5-mm/day sensitivity 是训练/replay/一级诊断完成后的 closure checks；当前更大的缺口是受控训练和结果资产不存在。
6. **如果不补训练，claim strength 到什么程度？** 仅能给出源码级 Strong/Moderate 方法性陈述；对 ET/Response 的 aggregate、parameter、targeted process 和 internal-state 科学结论均为 Weak 或 Unsupported。不得用旧候选数字补叙事。

本轮遵守 hard constraint：没有启动任何训练、重率定、seed 增加或 replay 生成。

## 18. 3.6 推荐三级标题

推荐沿用用户确定的最新结构：

- **3.6.1 蒸散发过程结构差异的诊断**
- **3.6.2 地下水响应路径结构差异的诊断**
- **3.6.3 不同水文过程结构差异的综合比较**

标题和科学组织是合理的；但当前不能将其标为“ready to write results”。建议先保留 3.6.3 的比较框架，待 ET/Response 一级结果和 closure 完成后填入。

## 19. 正文主图规划

本轮不生成正式图。以下是**条件式**图件方案，不能把不存在的 source path 写成已冻结 source。

### Figure A（拟 Figure 3-13）：蒸散发结构差异的多层诊断

- (a) Output：D_E/G_E–N paired ΔKGE；machine-readable source 当前为 **PENDING—无受控 KGE summary**。
- (b) Long-term/seasonal ET：ET/P 或已冻结 partition metric + seasonal timing；source 当前为 **PENDING—无 ET replay/metric file**。
- (c) Dry-period process：AUC/decay paired effect；source 当前为 **PENDING—无 dry-spell catalog/CSV**。
- (d) Parameter–process：IC `Δk` vs dry-down metric，dPL 仅作对照；source 当前为 **PENDING—无 parameter/process association output**。
- 若未来只有弱或 null correspondence，不为填 panel 强行加入；可把 panel (d) 删除或移入附录。

核心叙事只能在数据核准后决定：候选为“outlet nearly equivalent, dry-period process different”，目前不能写成结果。

### Figure B（拟 Figure 3-14）：地下水响应路径结构差异的多层诊断

- (a) ΔKGE 与 BFI/recession/low-flow-FDC；source 当前为 **PENDING—无 targeted-signature table**。
- (b) IC G_R tau shift/parameter footprint；source 当前为 **PENDING—无 G_R IC artifact**。
- (c) `Δtau`–`Δrecession` scatter/binned summary；source 当前为 **PENDING—无 paired basin table**。
- (d) IC vs dPL response direction；source 当前为 **PENDING—无两 regime 的 controlled outputs**。
- P4 storage–release 如未来信息过载，优先放附录；当前不作为主图 panel。

### 综合表

需要 1 张 qualitative + quantity-scale 的综合表（拟 Table 3-X），但在当前状态只能写 Snow 已核准、ET/Response 未评估，不能做“谁最大”的排名。

## 20. 建议进入正文的关键数字

当前不建议把任何 ET/Response effect 数字放入正文：**可核准的 ET/Response 科学效应数字为 0 组**。以下仅是下一轮数据核准可安全引用的 15 组**审计锚点**，不是 3.6 科学结果；其中标注“planned/config”者不能写成 completed：

1. 531 个目标流域；
2. full time axis 12,418 d；
3. test/evaluation 5,479 d；
4. warm-up 365 d；
5. train/calibration 5,113 d；
6. IC 受控 runner default 10 starts（planned）；
7. IC 受控 runner default 300 generations（planned）；
8. dPL 受控配置只有 seed 42（configured, not completed）；
9. dPL 配置 100 epochs（configured, not completed）；
10. controlled parameter counts N/D_E/G_E/D_R/G_R = 17/16/17/15/16；
11. current code controlled CI range `[0.1,0.9]`；
12. current code controlled CG range `[0.9,0.998]`；
13. current code tau0 range `[0.4342944819,499.4998332]`；
14. **审计派生、未独立落盘核准**：legacy native selected XAJ IC 的 CI=1/CG=1 singular endpoint union=99；
15. **审计派生、未独立落盘核准**：按 current code tau upper，legacy selected XAJ 的 outside/singular union=108 个 basin。

第 6–9、11–15 组必须带 protocol/legacy qualifier；不能把它们包装成 3.6 的 ET/Response 实验结果。

## 21. 建议进入脚注与附录的数据

### 脚注

- dPL controlled seed42 是配置状态，不是完成状态；未来若仍 single seed，正文需明确。
- dPL N 的 legacy `XAJ_CN` 与 controlled-N domain 不同；不能混用。
- D_E epoch-90 `eval_latest` 当前未找到，不能引用。
- native `CI/CG=1` 是 singular endpoint；不能转换成有限 tau。
- code tau upper 与 pre-Phase-0 文档的 15478.1439 存在冲突，需在训练前解决。
- `xaj_tau0` 当前 active IC/dPL adapter transform 未与文档 helper 完全对齐，需写入 manifest。
- parameter bounds/boundary pinning 仅在 controlled fit 完成后解释；legacy reference 不可直接迁移。

### 附录

- 完整 N/D_E/G_E/D_R/G_R parameter tables 与 bounds audit；
- all dry-down event/metric tables及 `P<1.0`/`P<0.5` full sensitivity；
- BFI、recession、low-flow/FDC 全 signature 表；
- P4 aggregation、storage–release、percentile/standardized relative metrics；
- IC restart noise、dPL seed spread、boundary mass、outlier robustness；
- corrected P5 process-only distance、低 `|ΔKGE|`/Q-equivalent sensitivity；
- missingness manifest、checkpoint-to-summary reconciliation；
- replay variable inventory、dates/forcing checksum；
- legacy artifact supersession map。

## 22. 当前仍需 closure 的事项

只列用户预先明确的两个 closure checks；不自行增加新的分析任务：

1. **ET `P < 0.5 mm/day` threshold sensitivity：`PENDING`**。主阈值 `P < 1.0 mm/day` 及 sensitivity 输出均尚未在当前仓库出现。
2. **P5 corrected process-only distance：`PENDING`**。必须从不含 `ΔKGE` 的冻结 process vector 重新计算；旧含 `ΔKGE` 的 distance 不得继续使用。

注意：上述 closure 不替代当前更大的受控训练/replay 缺口。

## 23. Final claim hierarchy

| Claim | Classification | Exact evidence | Limitation | Recommended thesis wording |
|---|---|---|---|---|
| N/D_E/G_E/D_R/G_R 是单因素受控结构设计 | **Strong（源码级；runtime test pending）** | `controlled_composed.py`、`xaj_variants.py`、`parameter_specs.py`、Phase-0 differential test assertions | 本轮 pytest 因环境缺少 `numpy`/`hydrodl2` 未能执行；设计结论来自源码静态核对，实际 forward/registry runtime 仍需补验证 | “本研究在共同 XAJ+CN host 上分别替换 ET 或 subsurface response-path，保持另一过程结构不变。” |
| N→D/G 不是简单到复杂的连续梯度 | **Strong**（定义级） | variant mapping 与参数增删规则 | 不等价于复杂度/性能排序 | “这些是不同的受控结构对照，而非严格复杂度谱。” |
| 当前 cross-process 结果已冻结 | **Strong: Unsupported / false** | freeze registry、P5、ET threshold 和 controlled outputs 均缺失 | `results_freeze_R1_R5` 仅为其他 manuscript scope | 不写 freeze；状态记为 `NOT_FROZEN` |
| ET 的 aggregate output 近似等效 | **Unsupported** | 表 D 所有 controlled ΔKGE 缺失 | 无 paired test/evaluation KGE | 暂不写；待 canonical ΔKGE 后再定 |
| ET dry-down process 非等效 | **Unsupported** | 表 E 无 event/metric artifact | 无 protocol execution、AUC、decay | 暂不写；不得采用旧候选数字 |
| IC `k` 与 ET dry-down 有强 correspondence | **Unsupported** | 表 F 无 parameter/process association | 无 rho/CI/slope/FDR | 暂不写 |
| Response targeted recession/low-flow/FDC 被系统重组 | **Unsupported** | 表 G 无 signature output | 无 D_R/G_R replay | 暂不写 |
| IC G_R 存在 parameter footprint | **Unsupported** | 表 H 无 G_R IC artifact | 无 tau shift/recession association | 未来最多写“associated with”，不能写因果 |
| dPL 下存在 response reorganization | **Unsupported** | 表 I 无 controlled dPL output | legacy XAJ_CN/TGD2 不能替代 controlled response | 未来如证据足够，只能写“directional evidence … under dPL” |
| storage–release explains recession | **Unsupported** | 表 J 无 P4 gate/metric | 且语义只能是 complementary evidence | 禁止该表述 |
| process distance 与 performance decoupled | **Unsupported pending P5** | 表 K corrected P5 缺失 | 旧 vector 若含 ΔKGE 会造成 part–whole coupling | P5 完成前不写 decoupling 综合统计 |
| Snow/ET/Response structural identifiability 都沿同一路径显现 | **Unsupported** | 只有 Snow 主案例有 frozen evidence | ET/Response 未训练/未诊断 | “需要检验不同水文过程是否呈现 process-dependent expression。” |

以下写法在当前证据下均明确为 **UNSUPPORTED**：

1. “ET 结构差异造成了强参数补偿。”
2. “tau 的变化因果决定 recession。”
3. “dPL 的共享参数约束导致了 response 方向反转。”
4. “dPL 通过 sm/ex 完成了补偿。”
5. “N→D/G 是从简单到复杂的结构梯度。”
6. “Response 的 storage–release 差异解释了 recession 差异。”
7. “很小的 ΔKGE 说明结构完全等效。”
8. “所有水文过程都会重复 Snow 的参数补偿模式。”
9. 使用旧的含 ΔKGE P5 distance 支撑 performance–process decoupling。
10. 在 0.5 mm/day sensitivity 未完成时声称 ET dry-down 对阈值完全稳健。

## 24. Final readiness verdict

# `NOT_READY`

当前受控 ET/Response 仅达到 **code/config readiness**，没有达到论文 3.6 的 data/result readiness。必须先解决受控训练输出缺失、tau domain/transform provenance 冲突，并在训练/replay 后执行用户预先定义的两个 closure checks。当前最稳妥的下一步不是继续找故事或画图，而是恢复/取得受控训练输出及其 canonical manifest，然后再进行数字核准。
