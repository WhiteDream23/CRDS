import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 1. 参数定义 ---

# --- 信号参数 ---
sequence_length = 1000  # 信号长度
I0 = 1.0                # 初始振幅
tau = 200               # 衰减时间常数
B = 0.1                 # 背景基线

# --- 干扰参数 ---
# !!! 这是关键：请务必根据您的真实情况修改采样率 !!!
fs = 10.0             # 假设数据采样率为 1000 Hz
interference_period = 8 # 从自相关图观察到的干扰周期（样本数）
interference_amplitude = 0.1 # 周期性干扰的强度
noise_level = 0.02      # 随机白噪声的强度

# --- 陷波滤波器参数 ---
# 根据采样率和周期计算干扰频率
f0 = fs / interference_period
# 品质因数，控制陷波的锐利程度
Q = 30.0


# --- 2. 生成模拟信号 ---

# 创建时间轴
t = np.linspace(0, sequence_length/fs, sequence_length)

# a. 创建一个“干净”的理想指数衰减信号
clean_signal = I0 * np.exp(-t * fs / tau) + B # 注意时间轴 t 和时间常数 tau 的单位

# b. 创建周期性干扰信号 (正弦波)
interference = interference_amplitude * np.sin(2 * np.pi * f0 * t)

# c. 创建随机白噪声
white_noise = np.random.normal(0, noise_level, sequence_length)

# d. 将它们相加，得到最终的、被污染的原始信号
original_signal = clean_signal + interference + white_noise


# --- 3. 设计并应用滤波器 ---

# a. 设计陷波滤波器
b_notch, a_notch = signal.iirnotch(f0, Q, fs)

# b. 应用零相位滤波器到原始信号上
filtered_signal = signal.filtfilt(b_notch, a_notch, original_signal)


# --- 4. 可视化对比滤波效果 ---

print(f"模拟信号已生成。")
print(f"陷波滤波器已设计并应用，目标频率: {f0:.2f} Hz。")

# 设置Matplotlib以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建一个大的图形窗口，包含3个子图
fig, axes = plt.subplots(3, 1, figsize=(14, 18))
fig.suptitle('陷波滤波器效果的直观对比', fontsize=20)


# --- 图1：时域信号对比 ---
axes[0].plot(clean_signal, '--', color='gray', label='理想无噪信号', alpha=0.8)
axes[0].plot(original_signal, label='原始信号 (带干扰)', alpha=0.7)
axes[0].plot(filtered_signal, label='滤波后信号', linewidth=2, color='red')
axes[0].set_title('图1：时域信号对比', fontsize=16)
axes[0].set_xlabel('采样点')
axes[0].set_ylabel('信号强度')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)


# --- 图2：频域信号对比 (FFT) ---
# 计算FFT
N = sequence_length
yf_original = np.fft.fft(original_signal)
yf_filtered = np.fft.fft(filtered_signal)
xf = np.fft.fftfreq(N, 1 / fs)

# 绘制频谱图
axes[1].plot(xf[:N//2], 2.0/N * np.abs(yf_original[0:N//2]), label='原始信号频谱', alpha=0.7)
axes[1].plot(xf[:N//2], 2.0/N * np.abs(yf_filtered[0:N//2]), label='滤波后信号频谱', color='red')
axes[1].axvline(f0, color='orange', linestyle='--', label=f'干扰频率: {f0:.1f} Hz')
axes[1].set_title('图2：频域对比 (FFT) - 干扰被精确消除', fontsize=16)
axes[1].set_xlabel('频率 (Hz)')
axes[1].set_ylabel('幅度')
axes[1].set_xlim(0, fs/2) # 只显示正频率部分
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)


# --- 图3：噪声自相关性对比 ---
# 计算残差（信号减去理想信号）
residual_original = original_signal - clean_signal
residual_filtered = filtered_signal - clean_signal

# 绘制自相关图
axes[2].acorr(residual_original, maxlags=100, usevlines=True, label='原始噪声自相关', normed=True)
axes[2].acorr(residual_filtered, maxlags=100, usevlines=True, label='滤波后噪声自相关', normed=True, color='red', alpha=0.7)
axes[2].set_title('图3：噪声自相关对比 - 周期性消失', fontsize=16)
axes[2].set_xlabel('延迟 (Lag)')
axes[2].set_ylabel('归一化相关系数')
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()