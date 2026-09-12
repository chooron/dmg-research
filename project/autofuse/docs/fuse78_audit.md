# FUSE 78 第一阶段审计与复现准备

## 状态

**当前 reference/fidelity 阶段基本完成，论文科学运行仍受可复现数据细节限制。** 已生成 rootless Fortran oracle、恢复 544 basin manifest、完成 TOPMODEL 公式修正、四母配置对照、solver scan 和 78 结构回归；没有启动 544×78 SCE 或正式 dPL 训练。

## 1. Provenance

本地只读参考仓库位于 `vendor/upstream/`，目录内容被 `.gitignore` 忽略；逐仓库的 remote、ref、SHA、clone date 和 dirty state 见 `vendor/provenance.json`。

| reference | requested ref | exact HEAD |
|---|---|---|
| `CyrilThebault/fuse` | `v1.0_MMpaper`（detached） | `e6e23a4fc4ff4019bcab55f14537ea43b9525967` |
| `CyrilThebault/FUSE-MMComparison-paper` | `main` | `f1f94a3e9662672d01861bee53eee74a34b8942a` |
| `NCAR/FUSE` | `master` | `dbcd0bcc6b3774888a51701146e016ac68f6a2ce` |
| `fathom-global/fuse` | `master` | `ebf4fa9d687ef47157adcca01b3e6aa2104fcbbd` |
| `cvitolo/fuse`（1248 结构二级参考） | `master` | `b44cb98699a62bb17dd4279a71afd9124496e936` |

所有记录的仓库在固定后均为 clean。上游代码没有复制进 `dfuse/`。


环境审计：`.venv/bin/python` 为 Python 3.10.20，Torch 为 `2.9.1+cu128`、NumPy 为 `2.2.6`、pytest 为 `9.1.1`；运行环境为 WSL2。系统无 `gfortran`/R/nvcc，`sudo -n` 不可用；本轮使用 `/tmp/autofuse-reference-toolchain` 中 rootless 提取的 Debian gfortran 14.2.0 + NetCDF/HDF5 依赖，没有修改系统安装。
## 2. 论文实际 78 结构

来源是：

- `vendor/upstream/fuse-mmcomparison-paper/01_FUSEscripts/fuse_template/settings/list_decision_78.txt`：实际 78 行、字段顺序为 `ID;RFERR;ARCH1;ARCH2;QSURF;QPERC;ESOIL;QINTF;Q_TDH;SNOWM`；
- 论文脚本 `01_FromCamelsToFUSE_Cluster.R`、`03_SimArrayFUSE.R`、`04_EvalFUSE.R`、`02_FUSE_Lumped_CAMELS_Cluster.slurm`；
- 上游 Fortran 的 `uniquemodl.f90` / `selectmodl.f90` / `model_defnames.f90`。

完整的 ID↔decision 行已提取到 `dfuse/specs/structures_78.json`；`dfuse.spec.get_structure(id)` 是唯一反向解析入口。实际 ID 顺序为：

`2, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 146, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216`。

决策频数与固定项：

- `RFERR=multiplc_e`：78/78；`ESOIL=sequential`、`QINTF=intflwnone`、`Q_TDH=rout_gamma`、`SNOWM=temp_index`：各 78/78；
- 开放项为 `ARCH1 × ARCH2 × QPERC × QSURF`；
- `ARCH1`：`tension1_1` 24、`tension2_1` 18、`onestate_1` 36；
- `ARCH2`：`tens2pll_2` 15、`unlimfrc_2` 21、`unlimpow_2` 21、`fixedsiz_2` 21；
- `QPERC`：`perc_f2sat` 33、`perc_w2sat` 12、`perc_lower` 33；
- `QSURF`：三种各 26。

理论 108 个开放组合是 `3 ARCH1 × 4 ARCH2 × 3 QSURF × 3 QPERC`。上游 `uniquemodl.f90` / `selectmodl.f90` 实际排除规则为：

1. `ARCH1=tension2_1` 且 `ARCH2=tens2pll_2`：9 个（3×3）；
2. `ARCH1!=onestate_1` 且 `QPERC=perc_w2sat`：24 个（2×4×3）；
3. 两规则交集为 3 个，因此排除总数 `9+24-3=30`，有效组合 `108-30=78`。

`validate_catalog()` 自动验证 78 个 ID、78 个 decision vectors、108/30 计数和反向映射；不通过则抛出异常。

## 3. 状态、参数、通量和拓扑

### 状态超集

`dfuse.spec.STATE_NAMES` 的 9 个坐标为：

`TENS_1A, TENS_1B, TENS_1, FREE_1, WATR_1, TENS_2, FREE_2A, FREE_2B, WATR_2`。

这与 `assign_stt.f90` 一致；`FREE_2` 是上游派生量，不是独立状态。每个结构的 active state mask 和顺序由 `ARCH1/ARCH2` 唯一推导。active state 数分布：2 个 27、3 个 18、4 个 27、5 个 6。

### 参数并集

`dfuse/specs/parameter_catalog.json` 从论文模板的 `fuse_zConstraints_snow.txt` 实际提取，含 37 个 union 坐标的 fit/default/lower/upper/transform 元数据。完整 bounds/defaults 以该 JSON 为准；主要参数分组如下：

- rainfall/hyper：`RFERR_ADD 0 [-10,10]`、`RFERR_MLT 1 [1,1.001]`、`RFH1_MEAN 0.003 [-.5,.5]`、`RFH2_SDEV .361 [.05,.7]` 及 4 个先验坐标；
- storage：`MAXWATR_1 100 [25,500]`、`MAXWATR_2 1000 [50,5000]`、`FRACTEN .5 [.05,.95]`、`FRCHZNE .5 [.05,.95]`、`FPRIMQB .5 [.05,.95]`、`RTFRAC1 .75 [.05,.95]`；
- percolation：`PERCRTE 100 [.1,1000]`、`PERCEXP 5 [1,20]`、`SACPMLT 10 [1,250]`、`SACPEXP 5 [1,5]`、`PERCFRAC .5 [.05,.95]`、`FRACLOWZ .5 [.05,.95]`；
- baseflow/surface：`BASERTE 50 [.001,1000]`、`QB_POWR 5 [1,10]`、`QB_PRMS .01 [.001,.25]`、`QBRATE_2A .025 [.001,.25]`、`QBRATE_2B .01 [.001,.25]`、`SAREAMAX .25 [.001,1]`、`AXV_BEXP .3 [.001,3]`、`LOGLAMB 7.5 [5,10]`、`TISHAPE 3 [2,5]`；
- routing/Snow-17：`TIMEDELAY .9 [.01,5]`、`MBASE 1 [0,3]`、`MFMAX 4.2 [2,8]`、`MFMIN 2.4 [.4,6]`、`PXTEMP 1 [-1,5]`、`OPG .5 [0,2.5]`、`LAPSE -5 [-7,-4]`。

`assign_par.f90` 的条件参数选择已在 `dfuse.spec._parameter_names()` 单点实现；重复参数名只占一个 union 坐标。每个结构的 parameter mask、flux selection、state-flux incidence/topology 和 decision integer codes 都由同一个 `StructureSpec` 返回。

### 数值配置

实际读取 `fuse_zNumerix.txt`：`solution_method=2`（implicit Euler）、fixed time steps、old-state Newton initial guess、line-search error trap、look-ahead step-end processing、`err_trunc_abs=1e-2`、`err_trunc_rel=1e-2`、Newton tolerances `1e-12`、max iterations 1000、min/max step `0.01/1440` minutes。对应值保存在 `dfuse.spec.SOLVER_CONFIG`。

TOPMODEL 逐项定义：`POWLAMB`/`MAXPOW` 按 `MEAN_TIPOW` 的 offset=3、shape=`TISHAPE`、`CHI=(LOGLAMB-3)/TISHAPE`、log-index upper=50 和 2,000 midpoint bins 计算；饱和面积按 `TI_SAT=POWLAMB/(WATR_2/MAXWATR_2+1e-8)`、`TI_LOG=log(TI_SAT**QB_POWR)`、`SATAREA=1-GAMMP(TISHAPE,max(TI_LOG-3,0)/CHI)`；TOPMODEL power baseflow 的 `QBSAT`/`QBASE` 直接对应 `qbsaturatn.f90`/`q_baseflow.f90`，单位转换 `MAXWATR_2/1000` 保留。

## 4. CAMELS/SCE/评价协议

从论文脚本实际读取到：

- forcing：CAMELS `daymet` basin-mean forcing，`P[mm/d]`、平均温度和 Oudin PET；Snow-17/temperature-index 输入还使用 elevation-band 文件；
- 原始论文流域文件名为 `liste_BV_CAMELS_559.txt`，论文脚本和 Slurm 数组均按 559；HydroShare 配套资源明确提供 `Shp/CatchmentBoundaries_544.*`；本轮没有按缺测规则猜测，而是直接以该 544 边界产品的 `gauge_id` 恢复 manifest；
- `SimStart=1989-01-01`、`SimEnd=2009-12-31`；`WU=2`，forcing 文件从 `1987-01-01` 开始，warm-up 到 `1988-12-31`；
- calibration：`1989-01-01..1998-12-31`；evaluation：`1999-01-01..2009-12-31`；
- 目标为 `KGECOMP`；`Metrics.R` 的 KGE 使用 R `sd`（sample SD），`KGEcomp` 由 `KGE(Q)` 和 inverse-flow KGE 的均值构成；inverse 项使用 calibration mean/100 的 epsilon；
- `02_FUSE_Lumped_CAMELS_Cluster.slurm` 从 fm 模板执行 `calib_sce` 后 `run_best`；模板的 SCE `KSTOP=3`、`PCENTO=.001`，`01_FromCamelsToFUSE_Cluster.R` 将 `MAXN` 覆盖为 10000；
- 流程读取 output 的 `q_routed`，再由 `04_EvalFUSE.R` 对 calibration/evaluation 分期评价。

544 manifest 状态：**已唯一恢复**。权威文件为 `project/autofuse/manifests/camels_544.json`，544 个 `basin_id`/`hru_id`/坐标/面积/平均高程，稳定 canonical-row hash=`5b3e30d34799921c44e0a3c6bca2c713f89ca83f115ad1a46262defd5d745db3`; `load_basin_manifest()` 会校验 hash、数量和重复 ID。它是 HydroShare `CatchmentBoundaries_544` 的属性表，而不是由缺测规则重算。559 清单与 544 `gauge_id` 的集合差为 15 个：`USA_02427250, USA_04056500, USA_05062500, USA_05412500, USA_06354000, USA_06360500, USA_06441500, USA_06447000, USA_06452000, USA_06468250, USA_07263295, USA_07362587, USA_09484600, USA_11151300, USA_12141300`。公开代码没有逐 basin 排除原因；这些 basin 只能审计为“不在明确的 544 边界产品中”，不推断额外科学筛选。证据和源文件 hash 见 `camels_544.json`。

`project/autofuse/protocol.py`、`data.py`、`metrics.py`、`sce.py`、`dpl.py` 和 `configs/phase1.yaml` 已将这些边界显式化。SCE 和 dPL 都只通过 `UnifiedEvaluator -> dfuse.simulate`，没有复制水文方程。

## 5. 当前实现和验证

已生成/更新：

- `dfuse/spec.py`：单一 model-ID → decision codes/state mask/parameter mask/flux/topology/solver 解析；
- `dfuse/kernel.py`：把 `MEAN_TIPOW` 改为上游 2,000-bin shifted-Gamma midpoint quadrature；增加 forward-exact、可反传的 regularized-gamma wrapper，并将 Picard 替换为 reference-style old-state/line-search differentiable Newton implicit-Euler；默认 16 iterations；
- `project/autofuse/reference_oracle.py`：只在临时目录生成 NetCDF3/file-manager/parameter inputs，调用 `run_pre`，读取 q/state/flux；reference 固定 `fracState0=0.25`，任意初始 state 作为等价初始化断言，不硬编码 reference 结果；
- `project/autofuse/build_reference.sh`、`docs/reference_build_provenance.json`：rootless 构建 recipe 和 executable/toolchain provenance；
- `project/autofuse/fidelity.py`、`run_fidelity.py`：四母配置、78-structure regression、implicit iteration scan、explicit fixed-substep scan、KGE/KGEcomp、状态/通量/守恒/梯度指标；
- `project/autofuse/docs/fidelity_validation.json`：由 CLI/harness 实际生成的 machine-readable mother/solver/78 regression/stress 汇总，不参与 kernel 计算。
- `project/autofuse/docs/explicit_solver_scan.json`：固定子步显式 Euler 的四母模型与 78-structure 全扫描、implicit/reference 双重比较、stress 和 ranking 汇总（702 个 structure×substep rows）。
- `project/autofuse/compile_validation.py`、`docs/compile_validation.json`：仅编译 implicit Newton 内部固定 shape tensor flux/RHS，并记录 Level A/B parity、graph reuse、recompilation、gradient 和 CPU/CUDA benchmark。
- `project/autofuse/manifests/camels_544.json`：HydroShare 544 边界产品恢复的单一 manifest；`run_record.py` 记录文件 hash 和 manifest canonical-row hash。

实际检查结果：

- reference 在隔离 Debian gfortran 14.2.0 + NetCDF/HDF5 依赖下成功构建；`repro-build` 与 `repro-build-2` executable byte-identical，SHA256=`e887f4f2794117dcff7fa721211cb9949dc98c1b7d69e5ba3a4a464f85608806`；runner probe 成功返回 file-manager 缺参提示；
- 四个 canonical mother configurations（VIC=2、PRMS=108、SAC-SMA=178、TOPMODEL=210；相同 defaults、24-step warm synthetic forcing、initial fraction 0.25、Newton 16）均 finite。

| mother | model | max abs Q | RMSE Q | max state | max flux | KGE | KGEcomp | dFUSE WB | reference WB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VIC | 2 | 1.28e-7 | 6.34e-8 | 3.71e-6 | 2.38e-7 | 0.999999960 | 0.999999970 | 5.06e-14 | 9.82e-6 |
| PRMS | 108 | 1.45e-7 | 7.63e-8 | 7.04e-6 | 2.38e-7 | 0.999999907 | 0.999999902 | 5.06e-14 | 1.23e-5 |
| SAC-SMA | 178 | 1.59e-7 | 8.01e-8 | 9.60e-6 | 4.35e-6 | 0.999999959 | 0.999999948 | 4.80e-14 | 1.00e-5 |
| TOPMODEL | 210 | 2.68e-9 | 8.26e-10 | 7.43e-6 | 2.38e-7 | 0.999999968 | 0.999999970 | 4.98e-14 | 1.39e-5 |

78-structure regression（frozen Newton 16、24-step warm synthetic forcing）：78/78 finite；max Q error `2.85e-6`，max state error `4.97e-4`，max flux error `3.68e-5`，max dFUSE balance residual `1.31e-12`，max reference balance residual `5.89e-5`。最大误差结构集中在 `ARCH2=fixedsiz_2 + QPERC=perc_lower`：worst model IDs 为 164、188、212、214、166、190；无 model-specific 特例，误差已定位为共享 fixed-size implicit solve 与 float32 reference 数值残差。

implicit solver scan 覆盖 iterations=`4,8,16,32,64`：四母模型均 finite，Q/KGE 在 Newton 4 已达到平台，gradient finite-difference relative error 约 `4.1e-12` 或更低；implicit-16 仍是冻结默认 control；explicit fixed-substep scan 另见下节和 `docs/explicit_solver_scan.json`。

TOPMODEL targeted gradient check：`LOGLAMB`/`TISHAPE`/`QB_POWR`/`TIMEDELAY` 与 central finite difference 的最大相对误差 `2.94e-9`，未激活 `AXV_BEXP` gradient 为 0/None。
测试：`FUSE_REFERENCE_EXE=... .venv/bin/python -m pytest -q dfuse/tests project/autofuse/tests`：**22 passed**；manifest self-check、reference runner smoke、TOPMODEL targeted gradient/inactive-mask、leap-day 和 bound-stress tests 均通过。编译专项见 `compile_validation.py`。

### 固定子步显式 Euler 扫描（synthetic 24 steps）

实现位置：`dfuse/kernel.py::simulate_explicit`，通过 `simulate(..., solver="explicit", n_substeps=N)` 启用；每天保持 forcing，执行 N 次 `S[k+1] = project(S[k] + (dt_days/N) * f(S[k], forcing, theta))`。`project` 仅沿用现有状态域/边界守卫，不使用 Newton/Jacobian/line-search；implicit 默认路径未改变。

四母模型相对 Fortran reference 的 max Q error：

| N | VIC-2 | PRMS-108 | SAC-SMA-178 | TOPMODEL-210 |
|---:|---:|---:|---:|---:|
| 1 | 3.66e-2 | 3.59e-2 | 7.50e-2 | 1.57e-9 |
| 2 | 1.76e-2 | 1.61e-2 | 2.40e-2 | 2.10e-9 |
| 4 | 8.65e-3 | 6.95e-3 | 1.57e-2 | 2.39e-9 |
| 8 | 5.23e-3 | 2.77e-3 | 1.35e-2 | 2.53e-9 |
| 12 | 4.10e-3 | 1.86e-3 | 1.28e-2 | 2.58e-9 |
| 24 | 4.39e-3 | 1.66e-3 | 1.21e-2 | 2.63e-9 |
| 48 | 4.86e-3 | 2.35e-3 | 1.22e-2 | 2.65e-9 |
| 96 | 5.09e-3 | 2.69e-3 | 1.24e-2 | 2.66e-9 |
| 192 | 5.21e-3 | 2.86e-3 | 1.25e-2 | 2.67e-9 |

78-structure aggregate（相对 reference）：

| N | finite; WB<=1e-5 | Q max/median/P95 | state max/median/P95 | flux max/median/P95 | KGEcomp min/median/max | WB max | Spearman(explicit, implicit) | worst-10 overlap |
|---:|---:|---|---|---|---|---:|---:|---:|
| 1 | 78;72 | .305/.0411/.105 | 32.63/.529/32.57 | 32.62/.152/32.61 | -74.89/.976/1.000 | 29.55 | .243 | 7 |
| 2 | 78;72 | .0457/.0189/.0304 | 11.54/.392/11.47 | 9.55/.0735/9.54 | .803/.988/1.000 | 8.53 | .096 | 2 |
| 4 | 78;78 | .0402/.00931/.0161 | 2.94/.324/2.94 | 9.30/.0343/9.29 | .959/.991/1.000 | 1.34e-13 | .260 | 3 |
| 8 | 78;78 | .0320/.00416/.0133 | 2.76/.291/2.76 | 8.51/.0158/8.50 | .977/.993/1.000 | 1.60e-13 | .331 | 3 |
| 12 | 78;78 | .0295/.00240/.0124 | 2.67/.280/2.67 | 8.31/.0119/8.30 | .982/.995/1.000 | 2.25e-13 | .455 | 3 |
| 24 | 78;78 | .0271/.00229/.0118 | 2.57/.269/2.57 | 8.13/.0113/8.12 | .984/.995/1.000 | 5.17e-13 | .485 | 3 |
| 48 | 78;78 | .0259/.00317/.0116 | 2.52/.264/2.52 | 8.06/.0110/8.05 | .984/.994/1.000 | 6.92e-13 | .419 | 3 |
| 96 | 78;78 | .0253/.00369/.0116 | 2.49/.261/2.49 | 8.02/.0109/8.01 | .984/.994/1.000 | 1.70e-12 | .389 | 3 |
| 192 | 78;78 | .0250/.00393/.0115 | 2.48/.260/2.48 | 8.00/.0112/7.99 | .984/.993/1.000 | 2.96e-12 | .378 | 3 |

解释：四母模型中的部分 Q 误差在 N=12–24 附近达到局部最好点；78 最大 Q error 在 N=192 仍为 `2.50e-2`，而 state/flux 误差持续在 `fixedsiz_2 + perc_lower` 家族形成明显平台。最慢结构为 Q=model 164、state=model 166/190/214、flux=model 188/164/212；这些均属于已知敏感组合。

`5000 mm/day` stress：N=1,2,4,8,12,24,48,96,192 均为 78/78 finite，均无负 storage（现有 floor projection 下最小 storage 为 `5e-7` 或以上），max WB 分别为 `1.36e-12, 7.96e-13, 6.82e-13, 6.82e-13, 5.68e-13, 6.82e-13, 6.06e-13, 1.23e-12, 2.33e-12`；stress 稳定性不是主要失败点，normal synthetic 的 state/flux 离散化平台才是主要问题。
`explicit_solver_scan.json` 的每条 row/aggregate 还记录 `explicit_snow_balance_max_abs`；snow-inclusive reference balance 使用 raw `ppt` 与 `SWE_TOT` storage，cold four-day sanity residual 为 `8.76e-6`。

因此本 synthetic evidence 的结论为 **explicit fidelity insufficient**：没有一个合理固定子步数同时保持 Q、内部 state/flux 轨迹和结构误差排序稳定。当前不建议进入正式 dPL/SCE，也不建议立即开始 `torch.compile(explicit_step)`。
## 6. Implicit inner-kernel `torch.compile` 重试
本轮仅编译 Level A 的通用固定 shape tensor `flux/RHS`：输入为 9-state union、37-parameter union、tensor decision/state/parameter masks 和两个 topographic scalar；不包含时间循环、Newton 16 iterations、Jacobian/autograd、linear solve、line search、state projection 或 routing。`simulate(..., compile_inner=True, compile_inner_backend="inductor", compile_inner_fullgraph=True)` 是 opt-in，默认 eager 不变。为保留 differentiable Newton 的 double-backward，Inductor forward 通过小型 `autograd.Function` 调用，backward 使用同一公式的 eager tensor evaluation；科学 kernel 和 Jacobian 定义不变。
- Level A：78/78 inner flux/RHS finite，RHS/flux/diagnostics 最大差异均为 `0`；完整 78-structure implicit forward parity 为 Q `0`、state `8.88e-16`、flux `0`、WB `0`。Jacobian parity（全部 78 structures）最大 abs/rel 均为 `0`。active parameter gradient parity（母模型+6 sensitive models）最大 abs `3.41e-13`、rel `1.11e-14`；inactive gradient 保持 `0/None`，全部 finite。double-gradient probe（model 2/MAXWATR_1、210/TISHAPE）最大差异 `3.05e-16`，全部 finite。
- Level A compile audit：CPU float64 Inductor/fullgraph、`TORCHINDUCTOR_FX_GRAPH_CACHE=0` 下首次调用约 `5.79 s`，`compile_attempts=1`、success=`1`、fallback=`0`、graph breaks=`0`、recompilations=`0`、unique graph=`1`；跨 78 structures、2,692 RHS calls 没有 recompilation storm。
- Level B：单次 residual evaluator 独立编译，`TORCHINDUCTOR_FX_GRAPH_CACHE=0` 下 cold call 约 `2.46 s`，fullgraph 成功；residual/flux/diagnostics parity 均为 `0`。未将 residual compile 接入 Newton loop。
- 性能（Torch 2.9.1+cu128、float64、CPU 1 thread）：24-step implicit series eager median `0.200 s`，compiled-inner `0.877 s`，speedup `0.228x`；one-step `0.00829` vs `0.0361 s`，`0.229x`；isolated RHS `0.000117` vs `0.000161 s`，`0.731x`。RTX 3060、8-step series 为 `0.579` vs `1.89 s`，`0.306x`。
- reference regression 仍为 78/78 finite，implicit eager/reference 最大 Q/state/flux/WB 为 `1.65e-7 / 1.51e-5 / 4.02e-7 / 1.31e-12`；compiled-inner 与 eager parity 通过，因此没有改变 reference fidelity。完整机器结果和复现命令见 `docs/compile_validation.json` 与 `project/autofuse/compile_validation.py`。
 
结论：**compiled inner kernel usable but no meaningful speedup**。当前主要瓶颈是 differentiable Newton 所需的 eager double-backward，而非 forward flux 本身；不建议把该路径用于真实训练性能。真实 Daymet parity 可以继续使用默认 eager implicit control；后续小规模 SCE 也不应以该 compiled-inner 路径作为性能依据。
 
## 7. 未解决差异 / 阻塞


1. reference executable 已在 rootless toolchain 构建成功并通过两次 byte-identical hash 检查；但 gfortran/R/nvcc 未系统安装，executable 当前位于 `/tmp`，换机器需按 `project/autofuse/build_reference.sh` 重建。
2. 544 manifest 已从 HydroShare `CatchmentBoundaries_544.*` 唯一恢复；公开材料没有逐 basin 的科学排除理由，只能确定 559 清单中 15 个 ID 不在明确的 544 边界产品中，不能再推断缺测筛选。
3. `dfuse.kernel` 仍以 differentiable Newton implicit-Euler 作为默认 control；新增固定子步 explicit Euler (`simulate(..., solver="explicit", n_substeps=N)`) 仅用于扫描。explicit 的 78-structure 结果保存在 `docs/explicit_solver_scan.json`，当前 synthetic 扫描显示 state/flux 和结构误差排序仍有明显离散化偏差；未切换默认 solver，也未用于正式训练。
4. 78 回归的最大误差来自 `fixedsiz_2 + perc_lower` 的 6 个结构，但仍保持 finite/守恒；已定位为共享 Newton/float32 reference 数值残差，未加入 model-specific 特例。
5. `FRACLOWZ` 仍被上游 `assign_par.f90` 列为 `tension2_1` 参数但无其他使用点；`uniquemodl.f90` 的 `iSNOWM` 疑似使用了错误 decision list。本实现保留前者的参数 mask，并沿论文 list/`SELECTMODL` 路径处理后者。
6. 尚未对完整 Daymet 1987–2009 forcing 做科学 parity，也没有运行全量 SCE/dPL；因此当前只满足“可开始小规模同核 forward/SCE smoke”，不满足正式 CAMELS 科学实验启动条件。
7. 为覆盖边界高输入，dfuse 增加了共享 global overflow conservation guard；它保证 78 个结构在 `5000 mm/day` stress 下 finite 且 max WB `1.45e-12`，但尚未声称等价于 Fortran `FIX_STATES` 的逐状态 flux disaggregation，正式研究前仍需用参考可运行的边界案例核对 flux partition。
