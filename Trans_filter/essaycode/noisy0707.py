"""
文件名: noisy0707.py
output : processednew
用途: 对含噪声的信号进行指数衰减模型拟合和质量评估
功能描述:
    - 实现加权最小二乘法(WLS)进行信号拟合
    - 提取拟合参数(初始振幅I0、衰减时间常数tau、背景基线B)
    - 计算和分析拟合残差
    - 评估拟合质量和噪声水平
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from pathlib import Path
from tqdm import tqdm
import re

# --- Matplotlib 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 1. 基础函数 (与之前版本相同) ---

def decay_model_fit(t, I0, tau, B):
    """指数衰减模型。"""
    tau = max(tau, 1e-6)
    return I0 * np.exp(-t / tau) + B


def perform_nlf_fit(signal_data, t_axis):
    """执行拟合，返回包含所有结果的字典或None。"""
    if not isinstance(signal_data, np.ndarray) or signal_data.ndim != 1 or len(signal_data) < 10: return None
    try:
        bg_slice = signal_data[-int(len(signal_data) * 0.1):]
        guess_B = np.mean(bg_slice) if len(bg_slice) > 0 else signal_data[-1]
        guess_I0 = signal_data[0] - guess_B
        guess_tau = 170.0
        p0 = [guess_I0, guess_tau, guess_B]
        bounds = ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        sigma_wls = np.sqrt(np.maximum(signal_data, 1e-9))
        # sigma_wls = np.sqrt(np.log1p(np.exp(signal_data)))
        params, _ = curve_fit(decay_model_fit, t_axis, signal_data, p0=p0, bounds=bounds,
                              sigma=sigma_wls, absolute_sigma=True, maxfev=10000)
        # params, _ = curve_fit(decay_model_fit, t_axis, signal_data, p0=p0, bounds=bounds, maxfev=10000)
        if not np.all(np.isfinite(params)): return None
        fitted_curve = decay_model_fit(t_axis, *params)
        residuals = signal_data - fitted_curve
        return {
            'I0': params[0], 'tau': params[1], 'B': params[2],
            'residuals_std': np.std(residuals),
            'fitted_curve': fitted_curve,
            'residuals_array': residuals
        }
    except (RuntimeError, ValueError, TypeError):
        return None


def check_linearity_of_variance(fitted_curve, residuals, r_squared_threshold=0.7,nums_box=10):
    """检查单条信号的“噪声方差 vs 信号强度”是否具有良好的线性关系。"""
    try:
        binner = np.linspace(np.min(fitted_curve), np.max(fitted_curve), num=nums_box)
        bin_centers, bin_variances = [], []
        for i in range(len(binner) - 1):
            indices = np.where((fitted_curve >= binner[i]) & (fitted_curve < binner[i + 1]))[0]
            if len(indices) > 10:
                bin_centers.append((binner[i] + binner[i + 1]) / 2)
                bin_variances.append(np.var(residuals[indices]))
                # bin_centers.append(np.mean(fitted_curve[indices]))
                # bin_variances.append(np.var(residuals[indices], ddof=1))
        if len(bin_centers) < 5: return False, 0.0, 0.0
        slope, intercept, r_value, p_value, std_err = linregress(bin_centers, bin_variances)
        r_squared = r_value ** 2
        is_linear = slope > 0 and r_squared >= r_squared_threshold
        return is_linear, r_squared, slope
    except Exception:
        return False, 0.0, 0.0


def natural_sort_key(s):
    """自然顺序排序。"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]


def plot_signal_fit_and_residual(t_axis, original_signal, fit_results, output_path, filename_prefix):
    """【修改版】绘制原始信号、其拟合曲线以及残差。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    fitted_curve = fit_results['fitted_curve']
    residuals = fit_results['residuals_array']

    ax1.plot(t_axis, original_signal, label='原始信号 (Raw Signal)', color='gray', alpha=0.7)
    ax1.plot(t_axis, fitted_curve, '--', label='指数衰减拟合曲线 (Fitted Curve)', color='red', linewidth=2)
    ax1.set_title(f"信号诊断: {filename_prefix}")
    ax1.legend();
    ax1.set_ylabel("信号强度");
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(t_axis, residuals, label='拟合残差 (实际噪声)', color='gray', alpha=0.8)
    ax2.axhline(0, linestyle='--', color='black', linewidth=1)
    ax2.set_xlabel("时间轴 (Time Axis)");
    ax2.set_ylabel("残差 (Residuals)")
    ax2.legend();
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path / f"{filename_prefix}.png")
    plt.close(fig)
# --- 2. 主分析函数 (修改版) ---

def analyze_with_qc_filtering(data_folder, output_folder, sequence_length, t_end, r_squared_threshold=0.7,nums_box=10,num_skipped_plots_to_save=5):
    """
    主函数：对所有信号进行质量控制，并对比剔除“坏”数据前后的参数统计结果。
    """
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    skipped_plots_folder = output_folder / "skipped_signal_plots"  # 新的专用文件夹
    skipped_plots_folder.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(list(data_folder.glob("*.npz")), key=natural_sort_key)
    if not npz_files:
        print(f"错误：在 '{data_folder}' 目录中未找到任何 .npz 文件。")
        return

    print(f"--- 分析开始 (带数据质量控制) ---")
    print(f"将检查所有信号的噪声线性度，R²阈值设为: {r_squared_threshold}")
    print("-" * 40)

    master_summary_list = []


    for npz_file in tqdm(npz_files, desc="处理并审查NPZ文件"):
        all_signals = np.load(npz_file)
        input_sequences = all_signals.get('data', all_signals.get('arr_0'))
        if input_sequences.ndim == 1:
            input_sequences = [input_sequences]

        t_axis = np.linspace(0, t_end, input_sequences[0].shape[-1])
        saved_skipped_plots_count = 0  # 初始化计数器
        per_signal_results = []
        for i, signal_data in enumerate(input_sequences):
            fit_results = perform_nlf_fit(signal_data, t_axis)
            if fit_results:
                is_linear, r2, slope = check_linearity_of_variance(
                    fit_results['fitted_curve'],
                    fit_results['residuals_array'],
                    r_squared_threshold,
                    nums_box=nums_box
                )
                fit_results['is_linear'] = is_linear
                fit_results['r_squared'] = r2
                per_signal_results.append(fit_results)
                # --- 核心新增功能：如果信号不合格，且尚未达到绘图上限，则保存其图像 ---
                if not is_linear and saved_skipped_plots_count < num_skipped_plots_to_save:
                    tqdm.write(f"\n文件 {npz_file.name}, 信号 #{i}: 线性关系不佳 (R²={r2:.2f})，正在保存诊断图...")
                    plot_signal_fit_and_residual(
                        t_axis=t_axis,
                        original_signal=signal_data,
                        fit_results=fit_results,
                        output_path=skipped_plots_folder,
                        filename_prefix=f"{npz_file.stem}_signal{i}_REJECTED_R2_{r2:.2f}"
                    )
                    saved_skipped_plots_count += 1

        if not per_signal_results:
            tqdm.write(f"文件 {npz_file.name} 未能成功拟合任何信号，已跳过。")
            continue

        df_all = pd.DataFrame(per_signal_results)
        df_good = df_all[df_all['is_linear'] == True]

        # --- 核心修改点：计算详细的信号数量统计 ---
        num_in_file = len(input_sequences)
        num_fitted = len(df_all)
        num_good_qc = len(df_good)

        summary_row = {
            'filename': npz_file.name,
            'signals_in_file': num_in_file,
            'signals_fitted': num_fitted,
            'signals_passed_qc': num_good_qc,
            'signals_skipped_qc': num_fitted - num_good_qc,  # 因QC不通过而被跳过的数量
            'signals_failed_to_fit': num_in_file - num_fitted  # 因拟合失败而被跳过的数量
        }

        # “审查前”的统计
        stats_before_qc = df_all[['I0', 'tau', 'B', 'residuals_std']].describe().transpose()

        # “审查后”的统计
        if not df_good.empty:
            stats_after_qc = df_good[['I0', 'tau', 'B', 'residuals_std']].describe().transpose()
        else:
            stats_after_qc = pd.DataFrame(columns=stats_before_qc.columns, index=stats_before_qc.index)

        for param in ['I0', 'tau', 'B', 'residuals_std']:
            for stat in ['mean', 'std']:
                summary_row[f'{param}_{stat}_before_qc'] = stats_before_qc.loc[param, stat]
                summary_row[f'{param}_{stat}_after_qc'] = stats_after_qc.loc[param, stat]

        master_summary_list.append(summary_row)

    if master_summary_list:
        final_report_df = pd.DataFrame(master_summary_list)

        # 调整列顺序，让统计数据更清晰
        count_cols = ['filename', 'signals_in_file', 'signals_fitted', 'signals_passed_qc', 'signals_skipped_qc',
                      'signals_failed_to_fit']
        param_cols = [col for col in final_report_df.columns if col not in count_cols]
        final_report_df = final_report_df[count_cols + sorted(param_cols)]

        output_path = output_folder / "quality_control_comparison_reportwls.csv"
        final_report_df.to_csv(output_path, index=False, encoding='utf-8-sig', float_format='%.6f')
        print(f"\n质量控制对比分析报告已成功保存至: {output_path}")
    else:
        print("\n未能生成任何有效的分析结果。")


# --- 3. 运行主函数 ---
if __name__ == "__main__":
    # !!! 重要：请根据您的实际情况修改以下路径和参数 !!!

    BASE_DATA_FOLDER = Path(r"F:\rawdatanew\7processed_data\processed_len18000")
    OUTPUT_FOLDER = Path(r"processednew/data7_qc")

    SEQUENCE_LENGTH = 18000
    T_END = 1800

    analyze_with_qc_filtering(
        data_folder=BASE_DATA_FOLDER,
        output_folder=OUTPUT_FOLDER,
        sequence_length=SEQUENCE_LENGTH,
        t_end=T_END,
        r_squared_threshold=0.8,        #12 0.8
        nums_box=12,
        num_skipped_plots_to_save=0
    )