"""
文件名: sg_expdata_noise.py
output : processednew
用途: 对实验数据进行全面的噪声分析,包括统计特性和频域分析
功能描述:
    - 使用Savitzky-Golay滤波进行信号平滑
    - 进行指数衰减模型拟合
    - 计算残差统计特性(峰度、偏度等)
    - 进行Allan偏差、自相关等噪声分析
    - 生成详细的噪声分析报告和可视化结果
"""

# expdata_noiseana_modified.py

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, probplot
# --- MODIFICATION: Import additional libraries for enhanced analysis ---
from scipy import stats
from scipy import signal as sg
from scipy.stats import ks_2samp
from scipy.stats import t as student_t
from statsmodels.graphics.tsaplots import plot_acf
import allantools
import glob
import  math
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path  # Use pathlib for modern, cross-platform path handling
from scipy.optimize import least_squares
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
# --- Matplotlib setup for Chinese characters ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


def check_linearity_of_variance(fitted_curve, residuals, r_squared_threshold=0.8, nums_box=11):
    try:
        binner = np.linspace(np.min(fitted_curve), np.max(fitted_curve), num=nums_box + 1)
        bin_centers, bin_variances = [], []
        for i in range(len(binner) - 1):
            indices = np.where((fitted_curve >= binner[i]) & (fitted_curve < binner[i + 1]))[0]
            if len(indices) > 10:
                # bin_centers.append(np.mean(fitted_curve[indices]))
                # bin_variances.append(np.var(residuals[indices], ddof=1))
                bin_centers.append((binner[i] + binner[i + 1]) / 2)
                bin_variances.append(np.var(residuals[indices]))
        if len(bin_centers) < 5: return False
        slope, _, r_value, _, _ = linregress(bin_centers, bin_variances)
        return slope > 0 and (r_value ** 2) >= r_squared_threshold
    except:
        return False

def plot_noise_variance_comparison(all_fitted, all_residuals, qc_fitted, qc_residuals, output_path, filename):
    """
    【新增】绘制 QC 前后的噪声方差 vs 信号强度对比图。
    """
    if len(all_fitted) == 0: return

    plt.figure(figsize=(12, 7))
    
    # 辅助内部函数：计算分箱方差和拟合
    def get_binned_stats(fitted, residuals, color, label_prefix):
        fitted = np.array(fitted)
        residuals = np.array(residuals)
        binner = np.linspace(np.min(fitted), np.max(fitted), num=25)
        bin_centers, bin_variances = [], []
        
        for i in range(len(binner) - 1):
            indices = np.where((fitted >= binner[i]) & (fitted < binner[i + 1]))[0]
            if len(indices) > 10:
                bin_centers.append((binner[i] + binner[i + 1]) / 2)
                bin_variances.append(np.var(residuals[indices]))
        
        bin_centers = np.array(bin_centers)
        bin_variances = np.array(bin_variances)
        
        if len(bin_centers) > 2:
            slope, intercept, r_val, _, _ = linregress(bin_centers, bin_variances)
            # 绘制散点
            plt.plot(bin_centers, bin_variances, 'o', color=color, alpha=0.5, label=f'{label_prefix} 数据点')
            # 绘制拟合线
            plt.plot(bin_centers, bin_centers * slope + intercept, '--', color=color, linewidth=2,
                     label=f'{label_prefix} 拟合: $R^2$={r_val**2:.3f}, Slope={slope:.2e}')
            return slope, r_val**2
        return 0, 0

    # 1. 绘制所有数据 (QC前) - 使用灰色/红色
    get_binned_stats(all_fitted, all_residuals, 'gray', 'QC前(All)')

    # 2. 绘制通过QC的数据 (QC后) - 使用蓝色/绿色
    if len(qc_fitted) > 0:
        get_binned_stats(qc_fitted, qc_residuals, 'blue', 'QC后(Passed)')
    else:
        plt.text(0.5, 0.5, "No signals passed QC", transform=plt.gca().transAxes, ha='center')

    plt.xlabel("拟合信号强度 (V)")
    plt.ylabel("噪声方差 ($\sigma^2$)")
    plt.title(f"文件 {filename}: 质量控制(QC)前后噪声特性对比\n(QC过滤掉了非线性/负斜率的异常信号)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_file = output_path / f"{filename}_noise_variance_QC_comparison.png"
    plt.savefig(save_file)
    plt.close()



def decay_model_fit(t, I0, tau, B):
    """用于曲线拟合的指数衰减模型。"""
    tau = max(tau, 1e-6)
    t_clipped = np.clip(t, 0, None)
    exp_arg = -t_clipped / tau
    exp_arg = np.clip(exp_arg, -700, 700)
    with np.errstate(over='ignore'):
        result = I0 * np.exp(exp_arg) + B
    return result

def residuals_func(params, t, signal):
    """
    计算模型和信号之间的残差。
    least_squares 会最小化这个函数返回值的平方和（或L1范数等）。
    """
    # params 是一个包含 [I0, tau, B] 的数组
    return signal - decay_model_fit(t, *params)
def perform_nlf_fit(signal, t_axis):
    """
    对信号执行非线性最小二乘法拟合。
    返回拟合参数、残差标准差和成功状态。
    """
    if len(signal) == 0:
        return np.nan, np.nan, np.nan, np.nan, False


    initial_guess_I0 = signal[0]
    initial_guess_tau = 170.0
    background_slice = signal[-int(len(signal) * 0.05):]
    initial_guess_B = np.mean(background_slice) if len(background_slice) > 0 else signal[-1]
    p0 = [initial_guess_I0, initial_guess_tau, initial_guess_B]
    bounds = ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])

    try:

        # sigma_wls = np.sqrt(np.maximum(signal, 1e-9))
        # sigma_wls = np.sqrt(np.maximum(signal, 1e-3))
        # sigma_wls = np.sqrt(np.log1p(np.exp(signal)))
        # sigma_wls = np.sqrt(np.abs(signal))
        sigma_wls = np.sqrt(np.maximum(signal+0.1, 1e-9))
        params, _ = curve_fit(decay_model_fit, t_axis, signal, p0=p0, bounds=bounds,
                              sigma=sigma_wls, absolute_sigma=True, maxfev=10000)
        # params, _ = curve_fit(decay_model_fit, t_axis, signal, p0=p0, bounds=bounds, maxfev=10000)
        if not np.all(np.isfinite(params)):
            raise ValueError("拟合结果包含非有限参数。")
        fitted_signal = decay_model_fit(t_axis, *params)
        residuals = signal - fitted_signal
        residuals_std = np.std(residuals)
        # result = least_squares(
        #     residuals_func,  # <-- 使用残差函数
        #     p0,  # <-- 初始猜测
        #     args=(t_axis, signal),  # <-- 传递给残差函数的额外参数
        #     bounds=bounds,  # <-- 参数边界
        #     loss='soft_l1',  # <-- 使用 L1 (Huber) 损失函数
        #     f_scale=0.003,  # <-- L1 损失的尺度参数
        #     max_nfev=10000  # <-- 注意参数名是 max_nfev
        # )
        #
        # # 3. 从返回的 result 对象中提取信息
        # if not result.success:
        #     # result.success 是一个布尔值，表示求解器是否成功收敛
        #     raise ValueError("least_squares 拟合未成功收敛。")
        #
        # params = result.x  # 拟合后的最优参数存储在 .x 属性中
        #
        # # result.fun 中存储了最优参数下的残差值，无需重新计算
        # residuals = result.fun
        # residuals_std = np.std(residuals)
        return params[0], params[1], params[2], residuals_std, True,residuals
    except (RuntimeError, ValueError, TypeError):
        return np.nan, np.nan, np.nan, np.nan, False,np.nan


def plot_signal_fit_and_residual(t_axis, original_signal, fitted_ideal_curve, output_path, filename, index):
    """绘制原始信号、其拟合曲线以及残差。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(t_axis, original_signal, label='原始信号 (Raw Signal)', color='gray', alpha=0.7)
    ax1.plot(t_axis, fitted_ideal_curve, '--', label='指数衰减拟合曲线 (Fitted Curve)', color='red', linewidth=2)
    ax1.set_title(f"{filename} - 信号 #{index} 拟合与残差")
    ax1.legend()
    ax1.set_ylabel("信号强度")
    ax1.grid(True, linestyle='--', alpha=0.6)

    original_residuals = original_signal - fitted_ideal_curve
    ax2.plot(t_axis, original_residuals, label='拟合残差 (实际噪声)', color='gray', alpha=0.8)
    ax2.axhline(0, linestyle='--', color='black', linewidth=1)
    ax2.set_xlabel("时间轴 (Time Axis)")
    ax2.set_ylabel("残差 (Residuals)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    # plt.show()
    plt.savefig(output_path / f"{filename}_signal{index}_fit_residual.png")
    plt.close(fig)


def plot_residual_histogram(residuals, output_path, filename, index):
    """为单个信号的残差绘制直方图。"""
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=50, color='gray', alpha=0.8)
    plt.title(f"{filename} - 信号 #{index} 残差直方图")
    plt.xlabel("残差值")
    plt.ylabel("频次")
    plt.tight_layout()
    plt.savefig(output_path / f"{filename}_residual_hist{index}.png")
    plt.close()



def plot_autocorrelation(residuals, output_path, filename, index):
    """
    绘制并保存带有置信区间的噪声自相关图。
    """
    # 创建图形和子图对象，这是 statsmodels 绘图函数推荐的做法
    fig, ax = plt.subplots(figsize=(10, 5))

    # 使用 plot_acf 函数，它会自动绘制置信区间（蓝色区域）
    # lags 参数等同于之前的 maxlags
    plot_acf(residuals, lags=100, ax=ax)

    # 自定义图表标题和标签
    ax.set_title(f"{filename} - 信号 #{index} 的噪声自相关图")
    ax.set_xlabel("延迟 (Lag)")
    ax.set_ylabel("自相关系数")

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.6)

    # 保存图像
    plt.savefig(output_path / f"{filename}_signal{index}_autocorrelation.png")
    plt.close(fig)


# --- 新增分析函数 2: 噪声方差 vs. 信号强度 ---
def plot_noise_variance_vs_signal(all_fitted_in_file, all_residuals_in_file, output_path, filename):
    """
    绘制噪声方差与信号强度的关系图。
    """
    if not all_fitted_in_file or not all_residuals_in_file:
        return

    fitted_np = np.array(all_fitted_in_file)
    residuals_np = np.array(all_residuals_in_file)

    plt.figure(figsize=(10, 6))

    # 根据拟合信号的强度进行分箱
    try:
        binner = np.linspace(np.min(fitted_np), np.max(fitted_np), num=25)
        bin_centers = (binner[:-1] + binner[1:]) / 2

        # 计算每个箱内残差的方差
        bin_variances = []
        for i in range(len(binner) - 1):
            indices = np.where((fitted_np >= binner[i]) & (fitted_np < binner[i + 1]))[0]
            if len(indices) > 10:  # 只有当箱内点数足够多时才计算方差
                bin_variances.append(np.var(residuals_np[indices]))
            else:
                bin_variances.append(np.nan)

        bin_variances = np.array(bin_variances)  # ✅ 改为 numpy 数组
        valid = ~np.isnan(bin_variances)
        X = bin_centers[valid].reshape(-1, 1)
        y = np.array(bin_variances)[valid]
        model = LinearRegression().fit(X, y)
        a, b = model.coef_[0], model.intercept_
        plt.plot(X, model.predict(X), 'r--', label=f'线性拟合: σ² = {a:.3g}·I + {b:.3g}')

        # 可选：输出系数用于后续加权拟合
        with open(output_path / f"{filename}_noise_model.txt", "w", encoding="utf-8") as f:
            f.write(f"# σ² = {a:.6g} * signal + {b:.6g}\n")

        plt.plot(bin_centers, bin_variances, 'o-', label='噪声方差')
        plt.xlabel("拟合信号强度")
        plt.ylabel("噪声方差 (从残差计算)")
        plt.title(f"文件 {filename} 的噪声方差 vs. 信号强度")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(output_path / f"{filename}_noise_variance_vs_signal.png")

    except Exception as e:
        print(f"绘制噪声方差图时出错: {e}")
    finally:
        plt.close()


def plot_segmented_residuals(segmented_data, chunk_size, output_path, t_axis):
    """为分段的残差数据绘制并排的直方图，并进行高斯拟合。"""
    num_chunks = len(segmented_data)
    if num_chunks == 0: return

    ncols = 3 if num_chunks > 2 else num_chunks
    nrows = math.ceil(num_chunks / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.5), sharex=True)
    if num_chunks == 1: axes = [axes]
    axes = np.array(axes).flatten()

    fig.suptitle('分段残差（噪声）分布直方图', fontsize=18)

    for i in range(num_chunks):
        ax = axes[i]
        residuals_segment = np.array(segmented_data[i])

        if len(residuals_segment) == 0:
            ax.set_title(f"段 {i + 1}: 无数据")
            continue

        start_point = i * chunk_size
        end_point = (i + 1) * chunk_size - 1

        ax.hist(residuals_segment, bins=100, density=True, color='gray', alpha=0.8, label='实际噪声分布')

        mu, std = stats.norm.fit(residuals_segment)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'r', linewidth=2, label='高斯拟合')

        ax.set_title(f"段 {i + 1} (点 {start_point} - {end_point})")
        ax.set_xlabel("残差值")
        if i % ncols == 0: ax.set_ylabel("概率密度")
        ax.grid(True, linestyle='--', alpha=0.6)

        textstr = f'μ={mu:.4f}\nσ={std:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)

    for j in range(num_chunks, len(axes)):
        axes[j].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = output_path / "segmented_noise_distribution.png"
    plt.savefig(output_file)
    plt.close(fig)
    print(f"分段噪声分布图已保存至: {output_file}")


def calculate_parameter_stats(values):
    """计算一列数值的描述性统计量。"""
    s = pd.Series(values).dropna()
    if s.empty:
        return {'mean': np.nan, 'std': np.nan, 'min': np.nan, '25%': np.nan, '50%': np.nan, '75%': np.nan,
                'max': np.nan, 'count': 0}
    stats = s.describe().to_dict()
    stats['count'] = int(stats['count'])
    return stats


def natural_sort_key(s):
    """以自然顺序对包含数字的字符串进行排序。"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]


# --- 新增的、用于全方位对比滤波效果的绘图函数 (视觉优化版) ---
def plot_filtering_comparison(t_axis, fs, signal_original, signal_filtered, output_path, filename, index):
    """
    生成一张2x2的对比图，包含时域、频域以及带置信区间的自相关图，全方位展示滤波效果。
    """
    # --- 采用2x2的子图布局，更紧凑清晰 ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f'文件 {filename} - 信号 #{index} 滤波效果全方位对比', fontsize=20)

    # --- 1. 左上角：时域对比 ---
    ax_time = axes[0, 0]
    ax_time.plot(signal_original, label='原始信号', color='skyblue', linewidth=1.5, alpha=0.9)
    ax_time.plot(signal_filtered, label='滤波后信号', color='red', linewidth=1.5)
    ax_time.set_title('图1：时域信号对比', fontsize=16)
    ax_time.set_xlabel('采样点')
    ax_time.set_ylabel('信号强度')
    ax_time.legend()
    ax_time.grid(True, linestyle='--', alpha=0.6)

    # --- 2. 右上角：频域对比 (FFT) ---
    ax_fft = axes[0, 1]
    N = len(signal_original)
    yf_original = np.fft.fft(signal_original)
    yf_filtered = np.fft.fft(signal_filtered)
    xf = np.fft.fftfreq(N, 1 / fs)

    ax_fft.plot(xf[:N // 2], 2.0 / N * np.abs(yf_original[0:N // 2]), label='原始信号频谱', color='skyblue',
                linewidth=2)
    ax_fft.plot(xf[:N // 2], 2.0 / N * np.abs(yf_filtered[0:N // 2]), label='滤波后信号频谱', color='red',
                linewidth=1.5)
    ax_fft.set_title('图2：频域对比 (FFT)', fontsize=16)
    ax_fft.set_xlabel('频率 (Hz)')
    ax_fft.set_ylabel('幅度')
    ax_fft.set_xlim(0, fs / 2)
    ax_fft.legend()
    ax_fft.grid(True, linestyle='--', alpha=0.6)

    # --- 3. 左下角：原始噪声自相关 (带置信区间) ---
    ax_acf_before = axes[1, 0]
    # !!! 注意：修正了您代码中的一处错误。perform_nlf_fit现在返回2个值，而不是6个。
    _,_,_,_,_, residual_original = perform_nlf_fit(signal_original, t_axis)
    if residual_original is not None:
        plot_acf(residual_original, ax=ax_acf_before, lags=100, title="图3a：原始噪声自相关")
        ax_acf_before.grid(True, linestyle='--', alpha=0.6)

    # --- 4. 右下角：滤波后噪声自相关 (带置信区间) ---
    ax_acf_after = axes[1, 1]
    _, _, _, _, _, residual_filtered = perform_nlf_fit(signal_filtered, t_axis)
    if residual_filtered is not None:
        plot_acf(residual_filtered, ax=ax_acf_after, lags=100, title="图3b：滤波后噪声自相关", color='red')
        ax_acf_after.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局以适应总标题
    plt.savefig(output_path / f"{filename}_signal{index}_filtering_comparison.png")
    plt.close(fig)


def plot_allan_deviation(residuals, fs, output_path, filename_prefix):
    """
    计算并绘制艾伦偏差图，用于分析噪声特性。
    """
    # 使用 allantools.oadev 计算重叠艾伦偏差，统计效果更好
    (t_out, ad, ad_err, n) = allantools.oadev(residuals, rate=fs, data_type="freq")

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制艾伦偏差曲线
    ax.loglog(t_out, ad, label='艾伦偏差 (Allan Deviation)')

    # --- 绘制参考斜率线以供对比 ---
    # 为参考线找到一个合适的垂直位置（例如，在曲线的中点附近）
    log_t = np.log10(t_out)
    log_ad = np.log10(ad)
    mid_point_y = 10 ** np.interp((log_t[0] + log_t[-1]) / 2, log_t, log_ad)
    mid_point_x = 10 ** ((log_t[0] + log_t[-1]) / 2)

    # 斜率 -1/2 (白噪声)
    line_white = mid_point_y * (t_out / mid_point_x) ** (-0.5)
    ax.loglog(t_out, line_white, 'r--', label='参考线 (白噪声, 斜率=-1/2)')

    # 斜率 +1/2 (随机游走噪声)
    line_rw = mid_point_y * (t_out / mid_point_x) ** (0.5)
    ax.loglog(t_out, line_rw, 'g--', label='参考线 (随机游走, 斜率=+1/2)')

    ax.set_title(f"信号 {filename_prefix} 的艾伦偏差分析")
    ax.set_xlabel("平均时间 τ (秒) [对数坐标]")
    ax.set_ylabel("艾伦偏差 σ(τ) [对数坐标]")
    ax.legend()
    ax.grid(True, which="both", linestyle='--')

    output_path_full = output_path / f"{filename_prefix}_allan_deviation.png"
    plt.savefig(output_path_full)
    plt.close(fig)
    tqdm.write(f"  -> 已生成艾伦偏差图: {output_path_full.name}")


def analyze_dataset_noise(data_folder, output_folder, sequence_length=12000, t_end=1200, chunk_size=2000):
    """主分析函数，集成了多种噪声分析方法。"""
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    plots_folder = output_folder / "diagnostic_plots"
    fit_plots_path = plots_folder / "fits"
    hit_plots_path = plots_folder / "hits"
    adv_analysis_path = plots_folder / "advanced_analysis"  # 为高级分析图创建新目录
    plots_folder = output_folder / "allan_variance_plots"
    fit_plots_path.mkdir(parents=True, exist_ok=True)
    hit_plots_path.mkdir(parents=True, exist_ok=True)
    adv_analysis_path.mkdir(parents=True, exist_ok=True)
    plots_folder.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(list(data_folder.glob("*.npz")), key=natural_sort_key)
    if not npz_files:
        print(f"在 '{data_folder}' 目录中未找到任何 .npz 文件。")
        return

    print(f"在 '{data_folder}' 中找到 {len(npz_files)} 个 .npz 文件。开始处理...")
    t_axis = np.linspace(0, t_end, sequence_length, dtype=np.float32)

    all_residuals_flat = []
    num_chunks = sequence_length // chunk_size
    segmented_residuals = {i: [] for i in range(num_chunks)}
    all_files_summary_data = []
    fs = sequence_length / t_end
    for npz_file in tqdm(npz_files, desc="处理NPZ文件"):
        file_basename = npz_file.stem
        file_params = {'I0': [], 'tau': [], 'B': [], 'residuals_std': []}

        # --- 为单个文件内的“噪声方差vs信号强度”分析准备数据 ---
        file_level_residuals = []
        file_level_fitted = []

        qc_passed_residuals = []  # 仅存储通过QC的信号
        qc_passed_fitted = []

        try:
            data = np.load(npz_file)
            input_sequences = data.get('data', data.get('arr_0'))

            for i, signal in enumerate(input_sequences):
                if len(signal) < sequence_length: continue
                signal = signal[:sequence_length]
                # signal+=0.02
                # --- 核心步骤：对原始信号进行滤波 ---
                # signal_filtered = sg.filtfilt(b_notch, a_notch, signal)

                # # --- 使用滤波后的信号进行拟合和分析 ---
                # fitted_curve, residuals = perform_nlf_fit(signal_filtered, t_axis)

                I0, tau, B, residuals_std, success,_ = perform_nlf_fit(signal, t_axis)
                if success:
                    file_params['I0'].append(I0)
                    file_params['tau'].append(tau)
                    file_params['B'].append(B)
                    file_params['residuals_std'].append(residuals_std)

                    fitted = decay_model_fit(t_axis, I0, tau, B)
                    residuals = signal - fitted

                    all_residuals_flat.extend(residuals)
                    file_level_residuals.extend(residuals)
                    file_level_fitted.extend(fitted)

                    if check_linearity_of_variance(fitted, residuals, r_squared_threshold=0.8): 
                        qc_passed_residuals.extend(residuals)
                        qc_passed_fitted.extend(fitted)

                    for j in range(num_chunks):
                        start_idx, end_idx = j * chunk_size, (j + 1) * chunk_size
                        segmented_residuals[j].extend(residuals[start_idx:end_idx])

                    if i in [50, 100,150,200,250]:  # 选择一些代表性信号进行详细诊断
                        plot_allan_deviation(
                            residuals=residuals,
                            fs=fs,
                            output_path=plots_folder,
                            filename_prefix=f"{npz_file.stem}_signal{i}"
                        )
                        plot_signal_fit_and_residual(t_axis, signal, fitted, fit_plots_path, file_basename, i + 1)
                        # plot_residual_histogram(residuals, hit_plots_path, file_basename, i + 1)
                        # # --- 调用新增的自相关分析 ---
                        # plot_autocorrelation(residuals, adv_analysis_path, file_basename, i + 1)
                        # plot_filtering_comparison(
                        #     t_axis=t_axis,
                        #     fs=fs,
                        #     signal_original=signal,
                        #     signal_filtered=signal_filtered,
                        #     output_path=plots_folder,
                        #     filename=file_basename,
                        #     index=i + 1
                        # )


            if not file_params['tau']: continue

            # --- 调用新增的“噪声方差vs信号强度”分析 (处理完一个文件后) ---
            # plot_noise_variance_vs_signal(file_level_fitted, file_level_residuals, adv_analysis_path, file_basename)
            plot_noise_variance_comparison(
                file_level_fitted, file_level_residuals, 
                qc_passed_fitted, qc_passed_residuals, 
                adv_analysis_path, file_basename
            )

            summary_row = {'filename': npz_file.name, 'successful_fits': len(file_params['tau'])}
            for param_name, values in file_params.items():
                param_stats = calculate_parameter_stats(values)
                for stat_name, value in param_stats.items():
                    summary_row[f'{param_name.capitalize()}_{stat_name}'] = value
            all_files_summary_data.append(summary_row)

        except Exception as e:
            print(f"处理文件 '{npz_file.name}' 时发生错误: {e}")

    if not all_files_summary_data:
        print("所有文件均未能成功处理，无法生成摘要。")
        return

    summary_df = pd.DataFrame(all_files_summary_data)
    summary_df.to_csv(output_folder / "per_file_summary_enhanced_analysis.csv", index=False, encoding='utf-8-sig')
    print(f"分析摘要已保存至: {output_folder / 'per_file_summary_enhanced_analysis.csv'}")

    # --- 绘制分段和总体的噪声分布图 ---
    if any(segmented_residuals.values()):
        plot_segmented_residuals(segmented_residuals, chunk_size, plots_folder, t_axis)

    if all_residuals_flat:
        residuals_np = np.array(all_residuals_flat)
        mu, std = stats.norm.fit(residuals_np)

        # 1. 绘制整体残差直方图
        plt.figure(figsize=(12, 6))
        plt.title("整体残差（噪声）分布与高斯拟合对比", fontsize=16)
        plt.hist(residuals_np, bins=200, density=True, color='gray', alpha=0.8, label='实际噪声分布')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 200)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'r', linewidth=2, label='高斯拟合曲线')
        plt.xlabel("残差值 (噪声幅度)")
        plt.ylabel("概率密度")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        textstr = '\n'.join([f'观测数量: {len(residuals_np)}', f'均值 (μ): {mu:.4f}', f'标准差 (σ): {std:.4f}'])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                       bbox=props)
        plt.savefig(plots_folder / "overall_noise_distribution.png")
        plt.close()
        print(f"整体噪声分布图已保存至: {plots_folder / 'overall_noise_distribution.png'}")

        # --- 2. 启用Q-Q图进行正态性检验 ---
        plt.figure(figsize=(8, 8))
        stats.probplot(residuals_np, dist="norm", plot=plt)
        plt.title("噪声分布的正态Q-Q图", fontsize=16)
        plt.xlabel("理论分位数 (Theoretical Quantiles)")
        plt.ylabel("样本分位数 (Sample Quantiles)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(plots_folder / "noise_qq_plot.png")
        plt.close()
        print(f"噪声Q-Q图已保存至: {plots_folder / 'noise_qq_plot.png'}")

        # # --- 第二步：对总残差样本进行df值诊断 ---
        # print("\n--- 噪声分布特性分析 (DF 值诊断) ---")
        #
        # # 方法一：匹配峰度 (Kurtosis Matching)
        # print("\n[方法一：匹配峰度]")
        # real_kurtosis = kurtosis(residuals_np, fisher=False)
        # print(f"真实残差样本的峰度为: {real_kurtosis:.4f}")
        # # 添加有效性检查
        # if real_kurtosis <= 3:
        #     print("峰度 ≤ 3，无法使用峰度匹配方法")
        #     optimal_df_kurt = None
        # elif real_kurtosis > 9:  # 对应df接近4的情况
        #     print("峰度过大，可能不适合t分布建模")
        #     optimal_df_kurt = None
        # else:
        #     # 使用正确的理论公式
        #     optimal_df_kurt = 4 + 6 / (real_kurtosis - 3)
        #     # 添加合理性检查
        #     if optimal_df_kurt < 4:
        #         print("计算得到的df < 4，峰度公式不适用")
        #         optimal_df_kurt = None
        #
        # # 方法二：最小化K-S检验统计量
        # print("\n[方法二：K-S检验（更稳健，稍慢）]")
        # real_noise_normalized = (residuals_np - np.mean(residuals_np)) / np.std(
        #     residuals_np)
        #
        # best_df_ks = -1
        # min_ks_statistic = float('inf')
        # df_range = range(3, 41)  # 测试3到40的df值
        #
        # for df_test in tqdm(df_range, desc="Testing df values"):
        #     std_t = np.sqrt(df_test / (df_test - 2))
        #     simulated_t_randoms = student_t.rvs(df=df_test, size=len(real_noise_normalized)) / std_t
        #     ks_statistic, _ = ks_2samp(real_noise_normalized, simulated_t_randoms)
        #     if ks_statistic < min_ks_statistic:
        #         min_ks_statistic = ks_statistic
        #         best_df_ks = df_test
        #
        # print(f"==> 通过K-S检验，找到的最佳 df 值为: {best_df_ks}")
        # print(f"    (在该df下，仿真分布与真实分布的差异最小，K-S值为: {min_ks_statistic:.4f})")
        #
        # # --- 第三步：生成最终的Q-Q图作为视觉印证 ---
        # plt.style.use('default')  # 还原一下绘图风格
        # plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        # plt.rcParams['axes.unicode_minus'] = False
        #
        # fig, ax = plt.subplots(figsize=(8, 8))
        # probplot(residuals_np, dist="norm", plot=ax)
        # ax.set_title('净化后总残差的正态Q-Q图', fontsize=16)
        # ax.set_xlabel("理论分位数 (Theoretical Quantiles)")
        # ax.set_ylabel("样本分位数 (Sample Quantiles)")
        # ax.grid(True, linestyle='--', alpha=0.6)
        #
        # output_path = plots_folder / "final_residuals_qq_plot.png"
        # plt.savefig(output_path)
        # plt.close(fig)
        # print(f"\n最终的Q-Q图已保存至: {output_path}")
        # print("\n--- 分析完成 ---")



if __name__ == "__main__":

    BASE_DATA_FOLDER = Path(r"F:\rawdatanew\7processed_data\processed_len18000")
    # BASE_DATA_FOLDER = Path(r"C:\Users\Mingkai\Desktop\rawdata\processed_len2_18000")
    # OUTPUT_FOLDER = Path(r"processednew\noise_analysis_4data7_testwls")

    OUTPUT_FOLDER = Path(r"processednew\noise_analysis_essay")

    analyze_dataset_noise(
        data_folder=BASE_DATA_FOLDER,
        output_folder=OUTPUT_FOLDER,
        sequence_length=18000,
        t_end=1800,
        chunk_size=500  # 您指定的每段点数
    )
