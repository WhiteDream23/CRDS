"""
文件名: singlefile_size.py
用途: 对单个数据文件进行指数衰减拟合和质量评估
功能描述:
    - 使用加权最小二乘法(WLS)进行曲线拟合
    - 实现方差线性性检验等质量控制(QC)检查
    - 分析拟合效果和残差特性
    - 生成单文件级别的详细分析报告
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,least_squares
from scipy.stats import linregress
from pathlib import Path
from tqdm import tqdm
import re
from collections import defaultdict

# --- Matplotlib 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- 1. 基础函数 (拟合与QC) ---
def decay_model(t, I0, tau, B):
    return I0 * np.exp(-t / tau) + B


def check_linearity_of_variance(fitted_curve, residuals, r_squared_threshold=0.8, nums_box=11):
    try:
        binner = np.linspace(np.min(fitted_curve), np.max(fitted_curve), num=nums_box + 1)
        bin_centers, bin_variances = [], []
        for i in range(len(binner) - 1):
            indices = np.where((fitted_curve >= binner[i]) & (fitted_curve < binner[i + 1]))[0]
            if len(indices) > 10:
                bin_centers.append((binner[i] + binner[i + 1]) / 2)
                bin_variances.append(np.var(residuals[indices]))
        if len(bin_centers) < 5: return False
        slope, _, r_value, _, _ = linregress(bin_centers, bin_variances)
        return slope > 0 and (r_value ** 2) >= r_squared_threshold
    except:
        return False


def fit_wls(signal, t_axis):
    try:
        guess_B, guess_I0 = np.mean(signal[-int(len(signal) * 0.1):]), signal[0] - np.mean(
            signal[-int(len(signal) * 0.1):])
        p0, bounds = [guess_I0, 140.0, guess_B], ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        sigma_wls = np.sqrt(np.maximum(signal, 1e-9))
        params, _ = curve_fit(decay_model, t_axis, signal, p0=p0, bounds=bounds, sigma=sigma_wls, absolute_sigma=True,
                              maxfev=10000)
        return params
    except:
        return None


def fit_ols(signal, t_axis):
    try:
        guess_B, guess_I0 = np.mean(signal[-int(len(signal) * 0.1):]), signal[0] - np.mean(
            signal[-int(len(signal) * 0.1):])
        p0, bounds = [guess_I0, 140.0, guess_B], ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        params, _ = curve_fit(decay_model, t_axis, signal, p0=p0, bounds=bounds, maxfev=10000)
        return params
    except:
        return None




# --- 2. 主分析函数 ---
def analyze_subsampling_for_single_file(npz_file_path, output_folder, subsample_sizes, sequence_length, t_end,
                                        r2_threshold):
    """
    对单个NPZ文件执行子采样分析，并生成结果图和CSV。
    """
    npz_file_path = Path(npz_file_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if not npz_file_path.exists():
        print(f"错误: 文件不存在 -> {npz_file_path}")
        return

    print(f"--- 开始对文件 '{npz_file_path.name}' 进行子采样分析 ---")

    # 加载所有信号数据
    with np.load(npz_file_path) as ld:
        signals = ld.get('data', ld.get('arr_0'))
    if signals.ndim == 1: signals = [signals]
    total_signals_in_file = len(signals)

    t_axis = np.linspace(0, t_end, sequence_length)
    methods = ['WLS', 'OLS', 'WLS_QC', 'OLS_QC']

    # 用于存储所有子采样大小的结果
    analysis_results = []

    for n_signals in tqdm(subsample_sizes, desc="测试不同子采样数量"):
        if total_signals_in_file < n_signals:
            tqdm.write(f"信号总数({total_signals_in_file}) < 当前采样数({n_signals})，已停止。")
            break

        # 从中间取样
        mid_point = total_signals_in_file // 2
        start_index = mid_point - (n_signals // 2)
        start_index=60
        subsampled_signals = signals[start_index: start_index + n_signals]

        # 对当前子样本进行拟合和QC
        taus_per_method = defaultdict(list)
        for signal in subsampled_signals:
            wls_params = fit_wls(signal, t_axis)
            ols_params = fit_ols(signal, t_axis)
            if wls_params is not None:
                qc_passed_wls = check_linearity_of_variance(decay_model(t_axis, *wls_params),
                                                            signal - decay_model(t_axis, *wls_params), r2_threshold)
                if qc_passed_wls: taus_per_method['WLS_QC'].append(wls_params[1])
                taus_per_method['WLS'].append(wls_params[1])
            if ols_params is not None:
                qc_passed_ols = check_linearity_of_variance(decay_model(t_axis, *ols_params),
                                                            signal - decay_model(t_axis, *ols_params), r2_threshold)
                if qc_passed_ols: taus_per_method['OLS_QC'].append(ols_params[1])
                taus_per_method['OLS'].append(ols_params[1])

        # 计算当前子样本的统计结果
        result_row = {'num_signals': n_signals}
        for name in methods:
            taus = taus_per_method[name]
            mean_tau = np.mean(taus) if taus else np.nan
            std_tau = np.std(taus) if len(taus) > 1 else 0
            result_row[f'tau_mean_{name}'] = mean_tau
            result_row[f'tau_std_{name}'] = std_tau
        analysis_results.append(result_row)

    # --- 分析完成，开始保存和绘图 ---
    if not analysis_results:
        print("未能生成任何分析结果。")
        return

    report_df = pd.DataFrame(analysis_results)

    # 保存数据到CSV
    csv_path = output_folder / f"{npz_file_path.stem}_subsampling_report.csv"
    report_df.to_csv(csv_path, index=False, float_format='%.6f', encoding='utf-8-sig')
    print(f"\n子采样分析的数据已保存至: {csv_path}")

    # --- 绘制两张对比图 ---
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f"文件 '{npz_file_path.name}' 的子采样分析", fontsize=18)

    styles = {
        'WLS': {'color': 'blue', 'marker': 'o'},
        'OLS': {'color': 'green', 'marker': 's'},
        'WLS_QC': {'color': 'deepskyblue', 'marker': 'D'},
        'OLS_QC': {'color': 'lime', 'marker': 'X'}
    }

    # 图1：均值 vs. 信号数
    for method in methods:
        axes[0].plot(report_df['num_signals'], report_df[f'tau_mean_{method}'], label=method, **styles.get(method, {}))
    axes[0].set_title("Tau 均值 vs. 用于统计的信号数量")
    axes[0].set_ylabel("Tau (τ) 均值")
    axes[0].legend(title="方法")
    axes[0].grid(True, which='both', linestyle='--')

    # 图2：标准差 vs. 信号数
    for method in methods:
        axes[1].plot(report_df['num_signals'], report_df[f'tau_std_{method}'], label=method, **styles.get(method, {}))
    axes[1].set_title("Tau 标准差 vs. 用于统计的信号数量")
    axes[1].set_xlabel("信号数量 (Number of Signals)", fontsize=12)
    axes[1].set_ylabel("Tau (τ) 标准差")
    axes[1].legend(title="方法")
    axes[1].grid(True, which='both', linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = output_folder / f"{npz_file_path.stem}_subsampling_plots.png"
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"子采样分析的图像已保存至: {plot_path}")


# --- 3. 运行主程序 ---
if __name__ == "__main__":
    # !!! 重要：请根据您的实际情况修改以下参数 !!!

    # 1. 指定您想分析的单个NPZ文件的完整路径
    TARGET_NPZ_FILE = Path(r"C:\Users\Mingkai\Desktop\rawdata\processed_len_18000\processed_data_batch_100.npz")

    # 2. 指定输出文件夹
    OUTPUT_FOLDER = Path(r"processed/single_file_subsampling_analysis_from60")

    # 3. 定义信号和分析参数
    SEQUENCE_LENGTH = 18000
    T_END = 1800
    R_SQUARED_THRESHOLD = 0.8
    SUBSAMPLE_SIZES = list(range(10, 151, 10))  # [10, 20, 30, ..., 150]

    # 运行分析
    analyze_subsampling_for_single_file(
        npz_file_path=TARGET_NPZ_FILE,
        output_folder=OUTPUT_FOLDER,
        subsample_sizes=SUBSAMPLE_SIZES,
        sequence_length=SEQUENCE_LENGTH,
        t_end=T_END,
        r2_threshold=R_SQUARED_THRESHOLD
    )