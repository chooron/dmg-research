# Flex-MOPEX 实验脚本使用说明

## 显存参考

实测环境：RTX 3080 Ti Laptop（16GB），单次训练峰值约 **2.4 GB**，净增量约 **400 MB/job**。

| 显卡显存 | 建议最大并行数 |
|---------|-------------|
| 8 GB  | 4–5  |
| 16 GB | 8–10 |
| 24 GB | 14–16 |
| 40 GB | 24–28 |
| 80 GB | 50+  |

---

## 脚本总览

```
scripts/
├── run_block1_main.sh        # 实验块1：主模型对比（Basic / Full / Flex × 5 seeds）
├── run_block1_full_lopo.sh   # 实验块1：Full-MOPEX leave-one-process-out 消融
├── run_block1_alpha_path.sh  # 实验块1：alpha 正则化路径
├── run_block3_loro.sh        # 实验块3：Leave-one-region-out 泛化验证
├── run_analysis.sh           # 后处理分析（所有块）
└── run_parallel.sh           # 一键并行启动所有实验块
```

---

## 各脚本说明

### `run_block1_main.sh` — 主模型对比

运行 Basic-minimal / Full / Flex（alpha=0.005/0.01/0.03）× 5 seeds，共 **25 个 runs**。

对应论文 **Results 3.1**：证明 Flex-MOPEX 引入结构权重后仍保持模拟能力，alpha 能控制结构复杂度。

输出目录：`results/block1_main/`

```bash
# 用法
bash scripts/run_block1_main.sh [GPU_IDS] [MAX_PARALLEL]

# 示例
bash scripts/run_block1_main.sh 0 4          # 单 GPU，4 并行
bash scripts/run_block1_main.sh 0,1 8        # 双 GPU，8 并行
bash scripts/run_block1_main.sh 0,1,2,3 12   # 4 GPU，12 并行
```

---

### `run_block1_alpha_path.sh` — Alpha 正则化路径

对 alpha 从 0 到 0.1 的 10 个值系统扫描：
- 关键 alpha（0.005, 0.01, 0.03）× **5 seeds**
- 完整路径 alpha（0, 0.001, 0.003, 0.007, 0.05, 0.07, 0.1）× **3 seeds**

对应论文 **Results 3.1–3.2**：展示结构复杂度从 full 到 sparse 的连续路径。

输出目录：`results/block1_alpha_path/`

```bash
# 用法
bash scripts/run_block1_alpha_path.sh [MAX_JOBS] [GPU_LIST]

# 示例
bash scripts/run_block1_alpha_path.sh 4
bash scripts/run_block1_alpha_path.sh 8 "0,1"
```

---

### `run_block3_loro.sh` — Leave-one-region-out 泛化

7 个水文气候分区 × 3 seeds × 3 模型（Flex / Full / Basic-minimal），共 **63 个 runs**。

对应论文 **Results 3.5**：验证 learned structural coordinates 是否能迁移到未见水文气候区域。

> **注意**：此脚本依赖 `run_model.py` 支持 `--loro-holdout-region` 参数，以及预先生成的分区文件 `results/hydroclimatic_regions.csv`。请先完成分区（见下方"运行顺序"）。

输出目录：`results/block3_loro/`

```bash
# 用法
bash scripts/run_block3_loro.sh [GPU_IDS] [MAX_PARALLEL]

# 示例
bash scripts/run_block3_loro.sh 0 4
bash scripts/run_block3_loro.sh 0,1 8
```

---

### `run_block1_full_lopo.sh` — Full-MOPEX LOPO 消融

运行 4 个 Full-minus-one-process 消融，每个消融使用固定结构掩码并对其余参数重新训练：

- `full_minus_phenology`: `(0, 1, 1, 1)`
- `full_minus_interception`: `(1, 0, 1, 1)`
- `full_minus_snow`: `(1, 1, 0, 1)`
- `full_minus_subsurface`: `(1, 1, 1, 0)`

默认使用 `42/123/456` 三个 seeds，并发 2 个 Python 进程；完成后自动生成 basin-level `ΔNSE` 汇总和 retraining audit。

输出目录：`results/block1_full_lopo/`

```bash
# 用法
bash scripts/run_block1_full_lopo.sh [GPU_IDS] [MAX_PARALLEL]

# 示例
bash scripts/run_block1_full_lopo.sh 0 2
bash scripts/run_block1_full_lopo.sh 0,1 2
```

---

### `run_analysis.sh` — 后处理分析

对已完成的实验结果运行所有后处理分析脚本。

```bash
# 用法
bash scripts/run_analysis.sh [BLOCK]

# 示例
bash scripts/run_analysis.sh all      # 分析全部
bash scripts/run_analysis.sh block1   # 仅分析 Block 1
bash scripts/run_analysis.sh block3   # 仅分析 Block 3
```

> 需要先确保 `analysis/` 下各分析脚本已适配新的 `results/` 目录结构。

---

### `run_parallel.sh` — 一键启动（推荐）

按顺序并行执行所有实验块，分 3 个 Phase：

1. **Phase 1**：Block 1 main + alpha path（并行）
2. **Phase 2**：Block 3 LORO（等 Phase 1 完成后启动）
3. **Phase 3**：后处理分析

```bash
# 用法
bash scripts/run_parallel.sh [MAX_JOBS] [GPU_LIST]

# 示例
bash scripts/run_parallel.sh 4              # 单 GPU，4 并行
bash scripts/run_parallel.sh 8 "0,1"        # 双 GPU，8 并行
bash scripts/run_parallel.sh 12 "0,1,2,3"   # 4 GPU，12 并行
```

日志输出：`results/parallel_logs/`

---

## 推荐运行顺序

### 快速验证（先跑 1 个 seed 确认流程）

```bash
cd /workspace/autoresearch/project/flexmopex

# 1. 单 seed 冒烟测试
python run_model.py \
  --config conf/config_dmopex_v3_alpha_0_01.yaml \
  --alpha 0.01 --seed 42 --gpu-id 0 --epochs 2 \
  --output-root results/smoke_test --run-name flex_smoke

# 确认 results/smoke_test/ 下有 model/ 和训练日志，则流程正常
```

### 正式实验（推荐在 SSH 服务器上运行）

```bash
cd /workspace/autoresearch/project/flexmopex

# Block 1：主对比 + alpha 路径（同时跑）
bash scripts/run_block1_main.sh 0,1 8 &
bash scripts/run_block1_alpha_path.sh 8 "0,1" &
wait

# Block 3：LORO（需要先完成分区）
bash scripts/run_block3_loro.sh 0,1 8

# 分析
bash scripts/run_analysis.sh all
```

或者一键：

```bash
bash scripts/run_parallel.sh 8 "0,1"
```

---

## 输出目录结构

```
results/
├── block1_main/
│   ├── basic_minimal_alpha0.0_seed42/
│   │   ├── model/
│   │   └── train.log
│   ├── flex_alpha0.01_seed42/
│   └── ...
├── block1_alpha_path/
│   ├── flex_alpha0.001_seed42/
│   └── ...
├── block3_loro/
│   ├── flex_region0_seed42/
│   └── ...
└── parallel_logs/
    ├── block1_main.log
    ├── block1_alpha_path.log
    └── analysis.log
```

---

## 当前状态

- [x] 项目结构清理完成，历史结果归档至 `.tmp/`
- [x] `scripts/` 脚本已全部并行化
- [x] 单次训练显存验证：峰值 ~2.4 GB（RTX 3080 Ti 16GB 可跑 8–10 并行）
- [ ] `run_model.py` 需确认支持 `model_type` 参数（basic_minimal/full/flex）
- [ ] Block 3 需要先生成水文气候分区文件
- [ ] `analysis/` 脚本需适配新 `results/` 目录路径


### Remote run

#### Block 1 Main（远程版）— `run_remote_block1_main.sh`

将 base / full / flex×{0.005, 0.01, 0.03} × 5 seeds（共 25 个 jobs）部署到远程服务器并通过 tmux 管理。

**显存分配策略**：当前已用 6 GB，余额 32 GB，单任务约 300 MB。
脚本默认 `MAX_PARALLEL=8`（约占 2.4 GB），3 个 tmux window 各独立控制并行度，总体保守留有余量。

| tmux window  | 任务                          | 默认并行 |
|-------------|-------------------------------|---------|
| `base_full` | base×5 + full×5（10 jobs）    | 3       |
| `flex_low`  | flex 0.005×5 + flex 0.01×5（10 jobs）| 3  |
| `flex_high` | flex 0.03×5（5 jobs）          | 2       |

```bash
# 最简单用法（用 .env 里的密码，默认 GPU=0，MAX_PARALLEL=8）
cd /workspace/autoresearch
bash project/flexmopex/scripts/run_remote_block1_main.sh

# 自定义并行度（例如显存充裕时跑更多）
MAX_PARALLEL=12 bash project/flexmopex/scripts/run_remote_block1_main.sh

# 指定 GPU 和自定义 session 前缀
GPU=1 MAX_PARALLEL=8 bash project/flexmopex/scripts/run_remote_block1_main.sh
```

脚本完成后会打印 attach 命令，例如：

```bash
# 查看全部 window
ssh -p 52180 root@connect.westc.seetacloud.com "tmux attach -t block1_main_20260528_120000"

# 只看某个 window
ssh -p 52180 root@connect.westc.seetacloud.com "tmux attach -t block1_main_20260528_120000:flex_low"

# 实时跟踪日志
ssh -p 52180 root@connect.westc.seetacloud.com \
  "tail -f /root/dmg-research/project/flexmopex/results/block1_main/tmux_base_full.log"
```

结果保存在远程 `/root/dmg-research/project/flexmopex/results/block1_main/`，目录结构与本地相同。

---

#### Block 1 Alpha Path（远程版）— `run_remote_block1_alpha_path.sh`

将 flex × {0.005, 0.01, 0.03}（key, 5 seeds）+ flex × {0.0, 0.001, 0.003, 0.007, 0.05, 0.07, 0.1}（path, 3 seeds）共 **36 个 jobs** 部署到远程并通过 tmux 管理。

| tmux window   | 任务                                  | 默认并行 |
|--------------|---------------------------------------|---------|
| `key_alphas`  | flex × {0.005,0.01,0.03} × 5 seeds（15 jobs） | 13 |
| `path_alphas` | flex × 7 path alphas × 3 seeds（21 jobs）     | 12 |

```bash
# 最简单用法（默认 GPU=0，MAX_PARALLEL=24）
cd /workspace/autoresearch
bash project/flexmopex/scripts/run_remote_block1_alpha_path.sh

# 自定义并行度
MAX_PARALLEL=30 bash project/flexmopex/scripts/run_remote_block1_alpha_path.sh

# DRY_RUN 调试单个 job（不启动 tmux）
DRY_RUN=1 bash project/flexmopex/scripts/run_remote_block1_alpha_path.sh
```

脚本完成后打印 attach 命令，例如：

```bash
ssh -p 52180 root@connect.westc.seetacloud.com "tmux attach -t block1_alpha_20260528_120000"
ssh -p 52180 root@connect.westc.seetacloud.com "tmux attach -t block1_alpha_20260528_120000:key_alphas"

# 实时跟踪日志
ssh -p 52180 root@connect.westc.seetacloud.com \
  "tail -f /root/dmg-research/project/flexmopex/results/block1_alpha_path/tmux_key_alphas.log"
```

结果保存在远程 `/root/dmg-research/project/flexmopex/results/block1_alpha_path/flex/alpha{alpha}/seed{seed}/`。

---

#### Block 3 LORO（远程版）— `run_remote_block3_loro.sh`

```
cd /workspace/autoresearch && \
LORO_SEEDS="42" \
LORO_MODEL_TYPES="flex full base" \
PER_REGION_PARALLEL=3 \
GPU=0 \
SESSION="block3_loro_seed42_$(date +%Y%m%d_%H%M%S)" \
bash project/flexmopex/scripts/run_remote_block3_loro.sh
```
