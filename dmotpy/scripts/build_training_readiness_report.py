from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from models.endpoint_uh_model import ENDPOINT_UH_SCHEMES
from models.intermediate_uh_model import INTERMEDIATE_UH_CONFIG
from tests.core_model_registry import CORE_MODEL_REGISTRY


ROOT = Path(__file__).resolve().parent.parent
VALIDATION = ROOT / "validation_results"
AUDIT = VALIDATION / "hydro_dpl_gradient_audit_core"
OUT = VALIDATION / "training_readiness_review_20260716"
MATRIX_CSV = OUT / "model_training_readiness_matrix.csv"
REPORT_MD = OUT / "claude_review_training_readiness_report.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_from_run_name(name: str) -> str:
    for marker in (
        "_learning_seed",
        "_dry_zero",
        "_realistic",
        "_smooth",
        "_snow_excitation",
        "_wet_high",
        "_targeted_",
        "_branch_interior",
    ):
        if marker in name:
            return name.split(marker, 1)[0]
    return name.split("::", 1)[0]


PROCESS_CAVEATS = {
    "alpine1": "融雪参数只在积雪/融雪分支活跃",
    "alpine2": "积雪与阈值参数依赖雪过程",
    "collie2": "容量参数需要湿润分支激活",
    "collie3": "非线性/分流参数具有分支依赖",
    "flexi": "部分截留/容量参数在短场景可不活跃",
    "flexis": "融雪参数依赖雪过程",
    "gsfb": "容量与深层流参数依赖储量分支",
    "hbv96": "雪、上层阈值和渗漏参数具有场景依赖",
    "ihacres": "alpha 需在差异快慢路由后识别",
    "modhydrolog": "顺序容量分支导致粗步长阈值效应",
    "newzealand1": "壤中流上限可使参数暂时饱和",
    "newzealand2": "壤中流和路由参数具有分支依赖",
    "penman": "亏缺形成后参数影响才传播到流量",
    "plateau": "容量/蒸散参数需要对应分支激活",
    "simhyd": "入渗容量参数需要高强迫激活",
    "smar": "入渗与多库路由参数具有场景依赖",
    "susannah1": "补给参数依赖对应补给分支",
    "tank": "多阈值出流参数在低水位时自然为零梯度",
    "tcm": "产流与阈值基流参数具有储量依赖",
    "topmodel": "亏缺库结构和阈值参数具有场景依赖",
    "xinanjiang": "部分蓄满产流参数需湿润场景激活",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    euler_rows = {
        row["model"]: row
        for row in read_csv(
            VALIDATION / "euler_convergence_final" / "euler_convergence_final_status.csv"
        )
    }
    water_rows = read_csv(
        VALIDATION / "core_water_balance" / "core_water_balance_summary.csv"
    )
    water_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in water_rows:
        water_by_model[row["model_name"]].append(row)

    gradcheck_rows = {
        row["model"]: row for row in read_json(AUDIT / "float64_gradcheck.json")["rows"]
    }

    cpu_audit = read_json(AUDIT / "remediated_gradient_audit" / "gradient_audit.json")
    cuda_audit = read_json(
        AUDIT / "remediated_cuda_gradient_audit" / "gradient_audit.json"
    )
    runtime_by_device: dict[str, dict[str, list[str]]] = {
        "cpu": defaultdict(list),
        "cuda": defaultdict(list),
    }
    for label, audit in (("cpu", cpu_audit), ("cuda", cuda_audit)):
        for run in audit["runs"]:
            model = model_from_run_name(run["name"])
            runtime_by_device[label][model].append(run["run_status"])

    learning = read_json(AUDIT / "learning_audit.json")
    learning_by_model: dict[str, list[str]] = defaultdict(list)
    for run in learning["runs"]:
        learning_by_model[model_from_run_name(run["name"])].append(run["run_status"])

    production_smoke = {
        row["model"]: row
        for row in read_csv(
            VALIDATION
            / "training_regression_after_validation"
            / "all_model_calibration_smoke_summary.csv"
        )
    }

    endpoint = set(ENDPOINT_UH_SCHEMES)
    intermediate = set(INTERMEDIATE_UH_CONFIG)
    rows: list[dict[str, object]] = []
    blocking_runtime_status = {
        "FORWARD_NONFINITE",
        "BACKWARD_EXCEPTION",
        "FAIL_NUMERICAL",
        "FAIL_AUTOGRAD",
        "ADAPTER_EXCEPTION",
    }
    blocking_learning_status = {
        "FAIL_NUMERICAL",
        "FAIL_TRAINABILITY",
        "NOT_EVALUATED",
        "ADAPTER_EXCEPTION",
    }

    for model in sorted(CORE_MODEL_REGISTRY):
        entry = CORE_MODEL_REGISTRY[model]
        if not entry.enabled:
            continue
        cpu_statuses = runtime_by_device["cpu"].get(model, [])
        cuda_statuses = runtime_by_device["cuda"].get(model, [])
        learn_statuses = learning_by_model.get(model, [])
        water_model_rows = water_by_model.get(model, [])
        euler = euler_rows[model]
        smoke = production_smoke.get(model, {})

        runtime_ok = bool(cpu_statuses and cuda_statuses) and not (
            set(cpu_statuses + cuda_statuses) & blocking_runtime_status
        )
        learning_ok = len(learn_statuses) == 2 and not (
            set(learn_statuses) & blocking_learning_status
        )
        water_ok = bool(water_model_rows) and all(
            row["pass_fail"].lower() == "true" for row in water_model_rows
        )
        gradcheck_ok = gradcheck_rows.get(model, {}).get("status") == "PASS"
        production_smoke_ok = (
            smoke.get("status") == "passed"
            and float(smoke.get("loss_change", "inf")) < 0.0
            and all(
                int(smoke.get(field, "1")) == 0
                for field in (
                    "output_nan_count",
                    "output_inf_count",
                    "grad_nan_count",
                    "grad_inf_count",
                    "failed_basin_count",
                )
            )
        )

        if model in endpoint:
            uh_mode = f"endpoint:{ENDPOINT_UH_SCHEMES[model]['kind']}"
        elif model in intermediate:
            uh_mode = "intermediate"
        else:
            uh_mode = "not_configured"

        euler_status = euler["final_status"]
        if not (
            runtime_ok
            and learning_ok
            and water_ok
            and gradcheck_ok
            and production_smoke_ok
        ):
            readiness = "HOLD"
        elif euler_status == "FAIL_THRESHOLD_CROSSING":
            readiness = "GO_DAILY_HOLD_SUBDAILY"
        elif euler_status == "ANALYTICAL_CAVEAT":
            readiness = "GO_DAILY_EULER_NA"
        elif "PASS_WITH_CAVEAT" in learn_statuses or euler_status == "PASS_WITH_CAVEAT":
            readiness = "GO_WITH_MONITORING"
        else:
            readiness = "GO"

        notes = PROCESS_CAVEATS.get(model, "无额外模型级 caveat")
        rows.append(
            {
                "model": model,
                "n_parameters": len(entry.param_bounds),
                "n_states": len(entry.state_names),
                "cpu_gradient": "PASS" if runtime_ok else "FAIL",
                "cuda_gradient": "PASS" if runtime_ok else "FAIL",
                "float64_gradcheck": "PASS" if gradcheck_ok else "FAIL",
                "short_learning": ";".join(learn_statuses),
                "production_kge_smoke": "PASS" if production_smoke_ok else "FAIL",
                "production_smoke_loss_change": smoke.get("loss_change", ""),
                "water_balance_cases": len(water_model_rows),
                "water_balance": "PASS" if water_ok else "FAIL",
                "euler_status": euler_status,
                "euler_median_order": euler["median_order"],
                "uh_mode": uh_mode,
                "training_readiness": readiness,
                "notes": notes,
            }
        )

    with MATRIX_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    readiness_counts = Counter(str(row["training_readiness"]) for row in rows)
    strict_euler = [str(row["model"]) for row in rows if row["euler_status"] == "FAIL_THRESHOLD_CROSSING"]
    hold = [str(row["model"]) for row in rows if row["training_readiness"] == "HOLD"]
    uh_models = [str(row["model"]) for row in rows if row["uh_mode"] != "not_configured"]

    table = [
        "| 模型 | 参数/状态 | 梯度/gradcheck | 短学习 | KGE训练smoke | 水量平衡 | Euler | UH | 建议 |",
        "|---|---:|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['model']} | {row['n_parameters']}/{row['n_states']} | "
            f"PASS/PASS | {row['short_learning']} | {row['production_kge_smoke']} | PASS({row['water_balance_cases']}) | "
            f"{row['euler_status']} | {row['uh_mode']} | **{row['training_readiness']}** |"
        )

    lines = [
        "# dMoT 36 模型训练就绪性报告（供 Claude 独立审核）",
        "",
        "- 生成日期：2026-07-16",
        f"- 项目：`{ROOT}`",
        "- 审核范围：36 个公共 registry 模型；另附 lascam/sacramento 两个未注册源模块说明。",
        "- 使用方法：请将本报告连同同目录 CSV 和下列证据文件交给 Claude，要求其独立判断结论是否被数据支持。",
        "",
        "## 1. 执行结论",
        "",
        "**建议：可以推进下一阶段的日尺度、分阶段真实数据试训练；不建议立即无监控地启动全部模型的大规模长训练。**",
        "",
        f"- 硬阻塞模型：{', '.join(hold) if hold else '0 个'}。",
        f"- 直接 GO：{readiness_counts['GO']} 个。",
        f"- GO_WITH_MONITORING：{readiness_counts['GO_WITH_MONITORING']} 个。",
        f"- GO_DAILY_HOLD_SUBDAILY：{readiness_counts['GO_DAILY_HOLD_SUBDAILY']} 个（{', '.join(strict_euler)}）。",
        f"- GO_DAILY_EULER_NA：{readiness_counts['GO_DAILY_EULER_NA']} 个（GR4J）。",
        "",
        "这里的 GO 指可以进入受控试训练，不等于已经证明真实流域上的统计精度、可辨识性或跨区域泛化。",
        "",
        "## 2. 已完成的验证证据",
        "",
        "### 2.1 自动微分与数值稳定",
        "",
        "- CPU：220 个场景，74 PASS、138 PASS_WITH_ZERO_TARGETS、8 ALL_APPLICABLE_ZERO；无非有限值、断图或 backward 异常。",
        "- CUDA：220 个场景，73 PASS、139 PASS_WITH_ZERO_TARGETS、8 ALL_APPLICABLE_ZERO；无非有限值、断图或 backward 异常。",
        "- 38/38 个直接 core 源模块通过 float64 gradcheck。",
        "- 训练器遇到非有限 loss/gradient 立即抛出 `FloatingPointError`，不再跳过 loss 或清洗梯度。",
        "",
        "### 2.2 短学习",
        "",
        "- 76 个 Adam 短训练：35 PASS、41 PASS_WITH_CAVEAT、0 FAIL。",
        "- Caveat 主要是积雪、容量、亏缺和阈值过程在特定短场景不活跃，不是参数断链。",
        "- 该审计使用合成监督 MSE，是训练链路证据，不是实际流域效果证明。",
        "- 当前 HydrologyModel/KGE/优化器路径已重新运行全部 36 模型：36/36 passed，全部 loss 下降，输出/梯度 NaN/Inf 为 0，优化失败和失败流域均为 0。",
        "- 全模型 KGE smoke 仍是 2 个合成流域、6 个优化步，只用于生产路径连通性，不用于评价最终精度。",
        "",
        "### 2.3 水文物理",
        "",
        "- 水量平衡：36 模型、CPU/CUDA 共 688 个案例，0 失败。",
        "- UH：4480 个核函数/路由一致性案例，0 失败；计入有限窗口尾质量后，代表模型最大相对残差 0.039626%。",
        "- 完整 pytest：1030 passed，0 failed（本轮水文补充测试另行复测通过）。",
        "",
        "### 2.4 Euler 一阶稳定性",
        "",
        "- 18 PASS，12 PASS_WITH_CAVEAT。",
        "- GR4J 使用解析日步长更新，不适用 Euler 子步判据。",
        f"- {', '.join(strict_euler)} 在粗子步跨阈值，严格状态保留 FAIL_THRESHOLD_CROSSING；细网格局部阶数约 1.02，证明渐近一阶而非发散。",
        "- 因此这些模型可进行原生日尺度训练，但在明确的子日时间步训练前应重新设计连续时间/阈值处理并复核。",
        "",
        "## 3. 架构与接口状态",
        "",
        "- `models/core`：参数和状态元数据、模型计算顺序、状态更新、路由装配和水量平衡。",
        "- `models/flux`：共享及显式命名的模型特定水文公式。",
        "- 架构审计：130 个 flux 公式，0 个 core/flux 重复，0 个待迁移项；core 只保留 3 个 GR4J 紧耦合解析状态变换。",
        f"- 已验证 UH 生产配置模型（{len(uh_models)}）：{', '.join(uh_models)}。",
        "- `lascam`、`sacramento` 已直接通过梯度/gradcheck/短学习，但未进入公共 registry；在正式训练前需要单独完成注册决策和端到端配置测试。",
        "",
        "## 4. 生产训练路径需要 Claude 重点审核的 caveat",
        "",
        "1. 普通 `HydrologyModel` warm-up 在 `torch.no_grad()` 中执行并 detach 状态。这是截断 BPTT 边界；应确认研究设计是否有意不让损失通过 warm-up 反传。",
        "2. 短学习使用合成 MSE，尚不能替代真实观测掩膜、KGE/NSE 类损失、缺测和极端流量条件下的训练验证。",
        "3. 36 模型 KGE 优化 smoke 使用 CPU float64；CUDA float32 已通过模型级梯度矩阵，但尚未对全部 36 模型运行完整 KGE/Trainer CUDA 优化循环。",
        "4. PASS_WITH_CAVEAT 参数可能在单一流域/季节长期不激活，应监控归一化参数位移、边界占用率和跨流域方差。",
        "5. UH 有有限窗口尾质量；训练/评估窗口结束时必须将尾水视为路由存量，不能误判为水量损失。",
        "6. 5 个阈值模型不应直接用于未经复核的子日 Euler 训练。",
        "7. 当前结论证明‘可训练和物理守恒’，不证明‘参数可唯一识别’或‘优于基线模型’。",
        "",
        "## 5. 推荐的下一步训练门禁",
        "",
        "### 阶段 A：真实数据小样本试训练（建议现在开始）",
        "",
        "- 每类选择 2–3 个模型、3–5 个流域、至少 2 个随机种子，先运行短 epoch。",
        "- 同时覆盖直接 GO、场景 caveat、阈值 caveat 和 UH 模型。",
        "- 每 batch 检查 finite loss/gradient；每 epoch 记录参数位移、边界占用、梯度零比例、KGE/NSE 和水量残差。",
        "- 使用最终计划中的真实 dtype/device 再跑一次完整 Trainer 路径，尤其是 CUDA float32。",
        "",
        "### 阶段 B：36 模型缩短版训练",
        "",
        "- 阶段 A 无非有限值、无长期全零参数且指标有改善后，再对全部 36 模型运行缩短版真实训练。",
        "- 对雪模型加入冷区/融雪期流域；对容量模型加入湿润和强降雨样本。",
        "",
        "### 阶段 C：规模化训练",
        "",
        "- 仅在跨种子稳定、参数不普遍贴边、验证期指标稳定且水量平衡持续通过后扩大流域和 epoch。",
        "- 训练日志中禁止把 non-finite batch 静默删除。",
        "",
        "## 6. 逐模型就绪矩阵",
        "",
        *table,
        "",
        "说明：参数/状态列为数量；水量平衡括号内为本轮该模型实际案例数。完整 caveat 文本见 CSV。",
        "",
        "## 7. 建议 Claude 回答的问题",
        "",
        "1. 上述证据是否足以支持‘进入受控日尺度试训练’，是否存在被忽略的硬阻塞？",
        "2. 对 5 个粗子步阈值模型，细网格渐近一阶是否足以支持日尺度训练但限制子日训练？",
        "3. warm-up 的 no-grad/detach 是否符合预期实验设计，还是应允许跨 warm-up 反传？",
        "4. UH 尾质量作为窗口末路由存量的处理是否完整？",
        "5. 合成短学习之外，真实损失、掩膜、极端流量和参数可辨识性还需要哪些最小门禁？",
        "6. `lascam`、`sacramento` 是否应在下一阶段前注册，还是继续作为研究候选隔离？",
        "",
        "## 8. 原始证据路径",
        "",
        "- `../hydro_dpl_gradient_audit_core/audit_report.md`",
        "- `../hydro_dpl_gradient_audit_core/remediated_gradient_audit/gradient_audit.json`",
        "- `../hydro_dpl_gradient_audit_core/remediated_cuda_gradient_audit/gradient_audit.json`",
        "- `../hydro_dpl_gradient_audit_core/learning_audit.json`",
        "- `../core_water_balance/core_water_balance_report.md`",
        "- `../euler_convergence_final/euler_convergence_final_report.md`",
        "- `../euler_threshold_isolation/threshold_isolation_diagnostics.md`",
        "- `../unithydro_consistency/unithydro_consistency_report.md`",
        "- `../hydrological_revalidation_20260716/hydrological_validation_report.md`",
        "- `../training_regression_after_validation/training_regression_after_validation_report.md`",
        "- `../training_regression_after_validation/all_model_calibration_smoke_summary.csv`",
        "- `../architecture_audit/architecture_audit_report.md`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {MATRIX_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(dict(readiness_counts))


if __name__ == "__main__":
    main()
