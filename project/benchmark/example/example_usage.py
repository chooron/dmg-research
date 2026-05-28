"""
Example: 使用 dmotpy 构建可微水文模型

本示例展示如何使用 dmotpy 包与 dmg 框架构建和训练可微水文模型。
dmotpy 提供了 38+ 种水文模型实现，以及配套的神经网络和训练器。

主要功能：
1. 使用 HydrologyModel 构建物理模型
2. 使用 Calibrate/Parameterize 神经网络学习模型参数
3. 使用 CalTrainer/FasterTrainer 进行模型训练
4. 与 dmg 框架无缝集成

使用前准备：
1. 安装 dmotpy: pip install dmotpy
2. 安装 dmg: pip install dmg
3. 准备 CAMELS 数据集或类似格式的水文数据
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径（如果从源码运行）
sys.path.append(str(Path(__file__).parent.parent))

from dmotpy.models import HydrologyModel
from dmotpy.neural_networks.calibrate import Calibrate
from dmotpy.neural_networks.parameterize import Parameterize
from dmotpy.trainers import CalTrainer, FasterTrainer

# ============================================================
# 第一部分：模型配置
# ============================================================


def create_config():
    """
    创建模型配置字典。

    配置说明：
    - model_name: 选择水文模型（从 dmotpy.models.core 中选择）
    - warm_up: 模型预热期（天数），用于初始化模型状态
    - device: 计算设备（cuda/cpu）
    - 其他训练相关参数
    """
    config = {
        # 模式设置
        "mode": "train",  # train/test/sim
        # 设备设置
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # 随机种子
        "seed": 42,
        # 模型配置
        "delta_model": {
            "phy_model": {
                "model_name": "hbv96",  # 可选: hbv96, gr4j, xinanjiang, flexi 等
                "warm_up": 365,  # 预热期（天）
                "variables": ["prcp", "tmean", "pet"],  # 输入变量
            },
            "nn_model": {
                "hidden_size": 256,  # 神经网络隐藏层大小
                "num_layers": 3,  # 神经网络层数
                "dropout_rate": 0.15,  # Dropout 比率
            },
            "rho": 365,  # 预测长度（天）
        },
        # 训练配置
        "train": {
            "epochs": 100,
            "optimizer": "Adam",  # Adam/AdamW/Adadelta/RMSprop
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "save_epoch": 20,
            "target": ["streamflow"],  # 预测目标
            "log_interval": 1,
            "start_time": "1989/10/01",
            "end_time": "1998/09/30",
            "lr_scheduler": "CosineAnnealingLR",  # 可选学习率调度器
            "lr_scheduler_params": {"T_max": 100},
        },
        # 测试配置
        "test": {
            "test_epoch": 100,
            "batch_size": 100,
            "split_dataset": True,
            "metrics": ["nse", "kge", "rmse", "bias"],
            "start_time": "1998/10/01",
            "end_time": "2008/09/30",
        },
        # 数据配置
        "data_sampler": "HydroSampler",
        # 输出配置
        "output_dir": "./output",
        "model_path": "./output/model",
        "out_path": "./output/sim",
    }

    return config


# ============================================================
# 第二部分：构建物理模型
# ============================================================


def build_physical_model(config):
    """
    使用 dmotpy 构建水文物理模型。

    dmotpy 提供了 38+ 种水文模型，包括：
    - hbv96: HBV-96 模型
    - gr4j: GR4J 模型
    - xinanjiang: 新安江模型
    - flexi: FLEX-I 模型
    - hymod: HyMod 模型
    - 更多模型请参考 dmotpy.models.core

    每个模型都实现了可微分的前向传播，支持梯度计算。
    """
    model_config = config["delta_model"]["phy_model"]

    # 创建物理模型
    phy_model = HydrologyModel(
        config={
            "model_name": model_config["model_name"],
            "warm_up": model_config["warm_up"],
        },
        device=config["device"],
    )

    print(f"物理模型: {model_config['model_name']}")
    print(f"参数数量: {len(phy_model.parameter_bounds)}")
    print(f"状态变量数量: {phy_model.n_states}")
    print(f"参数范围:")
    for param, bounds in phy_model.parameter_bounds.items():
        print(f"  {param}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")

    return phy_model


# ============================================================
# 第三部分：构建神经网络
# ============================================================


def build_neural_network(config, phy_model):
    """
    构建神经网络用于参数学习。

    dmotpy 提供两种神经网络：
    1. Calibrate: 基于 LHS 初始化的标定网络
    2. Parameterize: MLP 参数化网络（支持 MC-Dropout）

    神经网络的输出维度等于物理模型的可学习参数数量。
    """
    nn_config = config["delta_model"]["nn_model"]

    # 获取物理模型的输入特征数量
    # 假设输入包括: 流量数据(ny) + 流域属性(nx)
    # 实际数量需要根据数据确定
    nx = 35  # 流域属性数量（示例）
    ny = 3  # 水文变量数量（prcp, tmean, pet）

    # 方法1: 使用 Calibrate 网络（适合静态参数学习）
    nn_model = Calibrate(
        nx=nx,
        ny=ny,
        num_basins=100,  # 流域数量
        num_start=10,  # 初始化样本数
        init_strategy="lhs_logit",  # 初始化策略
        device=config["device"],
    )

    # 方法2: 使用 Parameterize 网络（适合动态参数学习，支持 MC-Dropout）
    # nn_model = Parameterize(
    #     nx=nx,
    #     ny=ny,
    #     hidden_size=nn_config["hidden_size"],
    #     num_layers=nn_config["num_layers"],
    #     dropout_rate=nn_config["dropout_rate"],
    #     device=config["device"],
    # )

    print(f"\n神经网络类型: {type(nn_model).__name__}")
    print(f"输入维度: nx={nx}, ny={ny}")
    print(f"输出维度: {phy_model.n_states}")

    return nn_model


# ============================================================
# 第四部分：构建可微模型
# ============================================================


class DifferentiableModel(torch.nn.Module):
    """
    可微水文模型：将神经网络与物理模型耦合。

    工作流程：
    1. 神经网络根据输入数据预测物理模型参数
    2. 物理模型使用预测参数进行水文模拟
    3. 通过反向传播优化神经网络参数
    """

    def __init__(self, nn_model, phy_model):
        super().__init__()
        self.nn_model = nn_model
        self.phy_model = phy_model

    def forward(self, x_dict, eval=False):
        """
        前向传播。

        Args:
            x_dict: 输入数据字典，包含：
                - x_phy: 水文变量（流量、温度、降水等）
                - c_nn_norm: 流域属性（归一化后）
                - target: 观测数据（用于训练）
            eval: 是否为评估模式

        Returns:
            输出字典，包含模拟结果
        """
        # 神经网络预测参数
        _, raw_params = self.nn_model(x_dict)

        # 物理模型模拟
        output = self.phy_model(x_dict, (None, raw_params))

        return output


def build_differentiable_model(config, phy_model, nn_model):
    """
    组合神经网络和物理模型为可微模型。
    """
    model = DifferentiableModel(nn_model, phy_model)
    model = model.to(config["device"])

    print(f"\n可微模型结构:")
    print(f"  神经网络: {type(nn_model).__name__}")
    print(f"  物理模型: {config['delta_model']['phy_model']['model_name']}")
    print(f"  设备: {config['device']}")

    return model


# ============================================================
# 第五部分：训练模型
# ============================================================


def train_model(config, model, train_dataset):
    """
    使用 dmotpy 训练器训练模型。

    dmotpy 提供两种训练器：
    1. CalTrainer: 标准标定训练器
    2. FasterTrainer: 优化训练器，支持 MC-Dropout

    训练器自动处理：
    - 优化器初始化（Adam, AdamW, Adadelta, RMSprop）
    - 学习率调度（StepLR, ExponentialLR, CosineAnnealingLR 等）
    - 模型保存和检查点管理
    - 训练日志记录
    """
    # 方法1: 使用 CalTrainer
    trainer = CalTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        verbose=True,
    )

    # 方法2: 使用 FasterTrainer（推荐用于大规模训练）
    # trainer = FasterTrainer(
    #     config=config,
    #     model=model,
    #     train_dataset=train_dataset,
    #     verbose=True,
    # )

    print(f"\n开始训练...")
    print(f"训练器: {type(trainer).__name__}")
    print(f"优化器: {config['train']['optimizer']}")
    print(f"学习率: {config['train']['learning_rate']}")
    print(f"训练轮数: {config['train']['epochs']}")

    # 开始训练
    trainer.train()

    print(f"训练完成！模型保存至: {config['model_path']}")

    return trainer


# ============================================================
# 第六部分：评估模型
# ============================================================


def evaluate_model(config, model, eval_dataset):
    """
    评估模型性能。

    评估器自动计算：
    - NSE: Nash-Sutcliffe 效率系数
    - KGE: Kling-Gupta 效率系数
    - RMSE: 均方根误差
    - Bias: 偏差

    对于 FasterTrainer，还支持 MC-Dropout 不确定性量化。
    """
    # 创建评估训练器
    trainer = FasterTrainer(
        config=config,
        model=model,
        eval_dataset=eval_dataset,
        verbose=True,
    )

    print(f"\n开始评估...")

    # 标准评估
    metrics = trainer.evaluate()

    print(f"评估完成！")
    print(f"评估指标:")
    for metric, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {metric}:")
            for k, v in value.items():
                print(f"    {k}: {v:.4f}")
        else:
            print(f"  {metric}: {value:.4f}")

    # MC-Dropout 不确定性量化（仅 FasterTrainer）
    if hasattr(trainer, "evaluate_mc_dropout"):
        print(f"\n执行 MC-Dropout 不确定性量化...")
        mc_results = trainer.evaluate_mc_dropout(n_samples=100)
        print(f"MC-Dropout 完成，共 {len(mc_results)} 个样本")

    return metrics


# ============================================================
# 第七部分：完整示例
# ============================================================


def main():
    """
    完整示例：使用 dmotpy 构建、训练和评估可微水文模型。

    注意：此示例需要实际数据才能运行。
    请根据您的数据集修改配置和数据加载部分。
    """
    print("=" * 60)
    print("dmotpy 可微水文模型示例")
    print("=" * 60)

    # 1. 创建配置
    config = create_config()
    print(f"\n[1/6] 配置创建完成")
    print(f"  模式: {config['mode']}")
    print(f"  设备: {config['device']}")

    # 2. 构建物理模型
    phy_model = build_physical_model(config)
    print(f"\n[2/6] 物理模型构建完成")

    # 3. 构建神经网络
    nn_model = build_neural_network(config, phy_model)
    print(f"\n[3/6] 神经网络构建完成")

    # 4. 构建可微模型
    model = build_differentiable_model(config, phy_model, nn_model)
    print(f"\n[4/6] 可微模型构建完成")

    # 5. 加载数据（需要实际数据）
    print(f"\n[5/6] 数据加载")
    print(f"  注意：需要实际数据集才能运行训练")
    print(f"  请参考 dmg 框架的数据加载器: dmg.core.data.loaders.HydroLoader")

    # 示例：模拟数据结构
    print(f"\n  示例数据结构:")
    print(f"  train_dataset = {{")
    print(f"    'x_phy': tensor([batch, time, variables]),  # 水文变量")
    print(f"    'c_nn_norm': tensor([batch, attributes]),    # 流域属性")
    print(f"    'target': tensor([time, basins]),            # 观测数据")
    print(f"  }}")

    # 6. 训练和评估（需要实际数据）
    print(f"\n[6/6] 训练和评估")
    print(f"  训练命令:")
    print(f"    trainer = train_model(config, model, train_dataset)")
    print(f"  评估命令:")
    print(f"    metrics = evaluate_model(config, model, eval_dataset)")

    print(f"\n" + "=" * 60)
    print(f"示例完成！")
    print(f"=" * 60)

    # 打印可用模型列表
    print(f"\ndmotpy 可用水文模型:")
    from dmotpy.models.core import PARAM_INFO

    models = sorted(PARAM_INFO.keys())
    for i, model_name in enumerate(models, 1):
        print(f"  {i:2d}. {model_name}")


if __name__ == "__main__":
    main()
