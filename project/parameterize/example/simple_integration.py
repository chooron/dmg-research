"""
dmotpy 与 dmg 集成示例

本示例展示如何将 dmotpy 的水文模型与 dmg 框架集成使用。

核心概念：
- dmotpy 提供水文物理模型（38+ 种）
- dmg 提供数据加载、训练框架、评估指标
- 两者通过 DifferentiableModel 耦合

最小工作流程：
1. 从 dmg 加载数据
2. 从 dmotpy 创建物理模型
3. 从 dmotpy 创建神经网络
4. 组合成可微模型
5. 使用 dmotpy 训练器训练
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
from dmotpy.models import HydrologyModel
from dmotpy.neural_networks.calibrate import Calibrate


# ============================================================
# 最小示例：构建和使用可微水文模型
# ============================================================


def minimal_example():
    """
    最小示例：展示 dmotpy 核心用法。
    """
    # 1. 配置
    config = {
        "mode": "train",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "delta_model": {
            "phy_model": {
                "model_name": "hbv96",  # 选择水文模型
                "warm_up": 365,
                "variables": ["prcp", "tmean", "pet"],
            },
            "nn_model": {
                "hidden_size": 256,
                "num_layers": 3,
                "dropout_rate": 0.15,
            },
            "rho": 365,
        },
        "train": {
            "epochs": 100,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "save_epoch": 20,
            "target": ["streamflow"],
            "start_time": "1989/10/01",
            "end_time": "1998/09/30",
        },
    }

    # 2. 创建物理模型
    phy_model = HydrologyModel(
        config={
            "model_name": config["delta_model"]["phy_model"]["model_name"],
            "warm_up": config["delta_model"]["phy_model"]["warm_up"],
        },
        device=config["device"],
    )

    print(f"物理模型: {config['delta_model']['phy_model']['model_name']}")
    print(f"参数数量: {len(phy_model.parameter_bounds)}")

    # 3. 创建神经网络
    nn_model = Calibrate(
        nx=35,  # 流域属性数量
        ny=3,  # 水文变量数量
        num_basins=100,
        num_start=10,
        device=config["device"],
    )

    print(f"神经网络: Calibrate")

    # 4. 组合成可微模型
    class DifferentiableModel(torch.nn.Module):
        def __init__(self, nn_model, phy_model):
            super().__init__()
            self.nn_model = nn_model
            self.phy_model = phy_model

        def forward(self, x_dict):
            _, raw_params = self.nn_model(x_dict)
            return self.phy_model(x_dict, (None, raw_params))

    model = DifferentiableModel(nn_model, phy_model).to(config["device"])

    print(f"可微模型已创建")

    return model, config


# ============================================================
# 数据格式说明
# ============================================================


def explain_data_format():
    """
    dmotpy 期望的数据格式。

    数据字典结构：
    {
        'x_phy': tensor([batch, time, variables]),  # 水文变量
        'c_nn_norm': tensor([batch, attributes]),    # 流域属性
        'target': tensor([time, basins]),            # 观测数据
        'batch_sample': tensor([...]),               # 批次索引
    }

    使用 dmg 的 HydroLoader 加载数据：
    from dmg.core.data.loaders import HydroLoader
    data_loader = HydroLoader(config)
    train_dataset = data_loader.train_dataset
    """
    print("\n数据格式说明：")
    print("  x_phy: 水文变量（降水、温度、PET等）")
    print("  c_nn_norm: 流域属性（归一化后）")
    print("  target: 观测数据（流量等）")
    print("  batch_sample: 批次采样索引")
    print("\n使用 dmg.HydroLoader 加载数据")


# ============================================================
# 可用模型列表
# ============================================================


def list_available_models():
    """
    列出 dmotpy 支持的所有水文模型。
    """
    from dmotpy.models.core import PARAM_INFO

    print("\ndmotpy 支持的水文模型：")
    models = sorted(PARAM_INFO.keys())
    for i, model_name in enumerate(models, 1):
        params = PARAM_INFO[model_name]
        print(f"  {i:2d}. {model_name:15s} - {len(params)} 个参数")


# ============================================================
# 训练流程说明
# ============================================================


def explain_training_flow():
    """
    使用 dmotpy 训练模型的流程。
    """
    print("\n训练流程：")
    print("  1. 准备数据（使用 dmg.HydroLoader）")
    print("  2. 创建物理模型（dmotpy.HydrologyModel）")
    print("  3. 创建神经网络（dmotpy.Calibrate/Parameterize）")
    print("  4. 组合成可微模型")
    print("  5. 创建训练器（dmotpy.FasterTrainer）")
    print("  6. 调用 trainer.train() 开始训练")
    print("\n训练器自动处理：")
    print("  - 优化器初始化")
    print("  - 学习率调度")
    print("  - 模型保存")
    print("  - 训练日志")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("dmotpy 与 dmg 集成示例")
    print("=" * 60)

    # 运行最小示例
    model, config = minimal_example()

    # 显示数据格式
    explain_data_format()

    # 列出可用模型
    list_available_models()

    # 说明训练流程
    explain_training_flow()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
