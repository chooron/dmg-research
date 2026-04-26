import sys

from matplotlib.colors import TwoSlopeNorm

sys.path.append(r'E:\pycode\generic_deltamodel')
from dmg import ModelHandler
from dmg.core.utils import import_data_loader, import_trainer, set_randomseed
from dmg.models.neural_networks.hope_mlp_v1 import HopeMlpV1
from project.better_estimate import load_config
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

CONFIG_PATH = r'conf/config_dhbv_hopev1.yaml'
config = load_config(CONFIG_PATH)
config['mode'] = 'test'
config['test']['test_epoch'] = 100
set_randomseed(config['random_seed'])
model = ModelHandler(config, verbose=True)
hope_mlp_model = model.model_dict['Hbv_2'].nn_model
hope_layer = hope_mlp_model.hope_layer

s4_layers = hope_layer.s4_layers
s4_layer_kernel = s4_layers[3].kernel
kernel = s4_layer_kernel(365).detach().cpu().numpy()
norm = TwoSlopeNorm(vcenter=0, vmin=np.min(kernel), vmax=np.max(kernel))

# 绘制
plt.figure(figsize=(14, 6))
im = plt.imshow(
    kernel,
    aspect='auto',
    cmap='coolwarm',     # 冷暖色调，蓝→白→红
    norm=norm,
    interpolation='nearest',
    origin='lower'
)

# 颜色条
cbar = plt.colorbar(im, fraction=0.02, pad=0.02)
cbar.set_label('Value Intensity (negative → positive)', fontsize=12)

# 坐标轴设置
plt.xlabel('Day of Year (1–365)', fontsize=13)
plt.ylabel('Feature Index (1–256)', fontsize=13)

# 时间轴每30天一个刻度
plt.xticks(np.arange(0, 365, 30), [f'{d}' for d in range(1, 366, 30)])
plt.yticks(np.linspace(0, 255, 9).astype(int))

# 去掉边框
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# 标题
plt.title('Heatmap (Positive/Negative Deviations Highlighted)', fontsize=15, weight='bold', pad=15)

plt.tight_layout()
plt.show()


# -------------------------------------------
# 2️⃣ 时间维度的统计特征
# -------------------------------------------
mean_t = kernel.mean(axis=0)
std_t = kernel.std(axis=0)

plt.figure(figsize=(10, 4))
plt.plot(mean_t, label='Mean (all 256 kernels)')
plt.fill_between(np.arange(365), mean_t-std_t, mean_t+std_t, alpha=0.2, label='±1 std')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Mean & Variability of Kernel Weight over Time')
plt.xlabel('Lag Time (days)')
plt.ylabel('Weight')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------
# 3️⃣ 自动选择代表性 kernel 进行可视化
#    - 最大正峰
#    - 最大负峰
#    - 典型双极型（正负交替）
# -------------------------------------------
max_pos_idx = np.argmax(kernel.max(axis=1))
max_neg_idx = np.argmin(kernel.min(axis=1))
# 找一个同时有明显正负的
bipolar_idx = np.argmax(np.std(np.sign(kernel), axis=1))

plt.figure(figsize=(12, 5))
for i, idx in enumerate([max_pos_idx, max_neg_idx, bipolar_idx]):
    plt.subplot(1, 3, i+1)
    plt.plot(kernel[idx], color='tab:red' if i==0 else 'tab:blue' if i==1 else 'tab:green')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title([
        f'Positive response kernel #{idx}',
        f'Negative response kernel #{idx}',
        f'Bipolar (mixed) kernel #{idx}'
    ][i])
    plt.xlabel('Lag Time (days)')
    plt.ylabel('Weight')
plt.tight_layout()
plt.show()

# -------------------------------------------
# 4️⃣ 频谱分析（时间尺度）
# -------------------------------------------
def analyze_spectrum(k):
    N = len(k)
    yf = np.abs(fft(k))
    xf = fftfreq(N, 1)[:N//2]
    return xf, yf[:N//2]

plt.figure(figsize=(12, 4))
for idx, color, label in zip(
    [max_pos_idx, max_neg_idx, bipolar_idx],
    ['tab:red', 'tab:blue', 'tab:green'],
    ['Positive', 'Negative', 'Bipolar']
):
    xf, yf = analyze_spectrum(kernel[idx])
    plt.plot(xf, yf, label=label, color=color)
plt.title('Frequency Domain of Kernels (FFT)')
plt.xlabel('Frequency (1/day)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------
# 5️⃣ 打印简单物理解释提示
# -------------------------------------------
def interpret_kernel(k):
    pos_ratio = np.mean(k > 0)
    neg_ratio = np.mean(k < 0)
    peak_t = np.argmax(np.abs(k))
    if pos_ratio > 0.8:
        return f"→ 以正权重为主：可能对应快速产流或正向累积效应（峰值滞后 {peak_t} 天）"
    elif neg_ratio > 0.8:
        return f"→ 以负权重为主：可能对应蒸散或蓄水消耗过程（峰值滞后 {peak_t} 天）"
    else:
        return f"→ 正负交替：可能表示周期性响应或高频滤波行为（主要频率见FFT）"

print("\n===== 🧠 Kernel Physical Interpretation =====")
for name, idx in zip(['Positive kernel', 'Negative kernel', 'Bipolar kernel'],
                     [max_pos_idx, max_neg_idx, bipolar_idx]):
    print(f"{name} #{idx}: {interpret_kernel(kernel[idx])}")
