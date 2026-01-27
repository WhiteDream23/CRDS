import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from tqdm import tqdm
import re

# --- Matplotlib 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 1. 基础函数 ---

def decay_model_fit(t, I0, tau, B):
    """指数衰减模型。"""
    tau = max(tau, 1e-6)
    return I0 * np.exp(-t / tau) + B


def perform_nlf_fit(signal_data, t_axis, method='ols'):
    """
    【修改版】执行拟合，支持OLS和WLS两种方法。

    Args:
        signal_data (np.array): 信号数据。
        t_axis (np.array): 时间轴。
        method (str): 'ols' (普通最小二乘) 或 'wls' (加权最小二乘)。

    Returns:
        dict: 包含拟合结果的字典或None。
    """
    if not isinstance(signal_data, np.ndarray) or signal_data.ndim != 1 or len(signal_data) < 10:
        return None
    try:
        bg_slice = signal_data[-int(len(signal_data) * 0.1):]
        guess_B = np.mean(bg_slice) if len(bg_slice) > 0 else signal_data[-1]
        guess_I0 = signal_data[0] - guess_B
        guess_tau = 140.0
        p0 = [guess_I0, guess_tau, guess_B]
        bounds = ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])

        # --- 核心修改：根据方法选择不同的拟合方式 ---
        if method.lower() == 'wls':
            # WLS方法：权重与方差的倒数成正比。
            # 对于散粒噪声，Var(y) ∝ y，所以权重 ∝ 1/y。
            # curve_fit的sigma参数代表每个点的标准差，权重为1/sigma^2。
            # StdDev(y) ∝ sqrt(y)，所以我们设置 sigma = sqrt(y)。
            # 为防止信号值为0或负数导致数学错误，使用np.maximum。
            sigma_wls = np.sqrt(np.maximum(signal_data, 1e-9))
            params, _ = curve_fit(decay_model_fit, t_axis, signal_data, p0=p0, bounds=bounds,
                                  sigma=sigma_wls, absolute_sigma=True, maxfev=10000)
        else:  # 默认为 OLS
            params, _ = curve_fit(decay_model_fit, t_axis, signal_data, p0=p0, bounds=bounds, maxfev=10000)

        if not np.all(np.isfinite(params)):
            return None

        fitted_curve = decay_model_fit(t_axis, *params)
        residuals = signal_data - fitted_curve
        return {
            'I0': params[0], 'tau': params[1], 'B': params[2],
            'residuals_std': np.std(residuals)
        }
    except (RuntimeError, ValueError, TypeError):
        return None


def natural_sort_key(s):
    """自然顺序排序。"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]


# --- 2. 新的主分析函数 ---

def compare_ols_vs_wls(data_folder, output_folder, sequence_length, t_end):
    """
    主函数：对每个NPZ文件中的所有信号，分别使用OLS和WLS进行拟合，并对比统计结果。
    """
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(list(data_folder.glob("*.npz")), key=natural_sort_key)
    if not npz_files:
        print(f"错误：在 '{data_folder}' 目录中未找到任何 .npz 文件。")
        return

    print("--- 开始 OLS vs. WLS 对比分析 ---")

    master_summary_list = []

    for npz_file in tqdm(npz_files, desc="对比OLS与WLS"):
        all_signals = np.load(npz_file)
        input_sequences = all_signals.get('data', all_signals.get('arr_0'))
        if input_sequences.ndim == 1:
            input_sequences = [input_sequences]
        t_axis = np.linspace(0, t_end, input_sequences[0].shape[-1])

        results_ols = []
        results_wls = []

        for signal_data in input_sequences:
            # 执行OLS拟合
            ols_params = perform_nlf_fit(signal_data, t_axis, method='ols')
            if ols_params:
                results_ols.append(ols_params)

            # 执行WLS拟合
            wls_params = perform_nlf_fit(signal_data, t_axis, method='wls')
            if wls_params:
                results_wls.append(wls_params)

        if not results_ols or not results_wls:
            continue

        # --- 对比统计 ---
        df_ols = pd.DataFrame(results_ols)
        df_wls = pd.DataFrame(results_wls)

        stats_ols = df_ols.describe().transpose()
        stats_wls = df_wls.describe().transpose()

        summary_row = {'filename': npz_file.name}
        for param in ['I0', 'tau', 'B', 'residuals_std']:
            for stat in ['mean', 'std']:
                summary_row[f'{param}_{stat}_ols'] = stats_ols.loc[param, stat]
                summary_row[f'{param}_{stat}_wls'] = stats_wls.loc[param, stat]

        master_summary_list.append(summary_row)

    if master_summary_list:
        final_report_df = pd.DataFrame(master_summary_list)
        # 调整列顺序，让对比更清晰
        cols = ['filename'] + sorted([col for col in final_report_df.columns if col != 'filename'])
        final_report_df = final_report_df[cols]
        output_path = output_folder / "ols_vs_wls_comparison_report.csv"
        final_report_df.to_csv(output_path, index=False, encoding='utf-8-sig', float_format='%.6f')
        print(f"\nOLS与WLS对比分析报告已成功保存至: {output_path}")
    else:
        print("\n未能生成任何有效的分析结果。")


# --- 3. 运行主函数 ---
if __name__ == "__main__":
    # !!! 重要：请根据您的实际情况修改以下路径和参数 !!!

    # 建议使用您已经滤除周期性干扰的数据进行对比，这样能更清晰地看到WLS方法本身的效果
    BASE_DATA_FOLDER = Path(r"C:\Users\Mingkai\Desktop\rawdata\processed1_15000")
    OUTPUT_FOLDER = Path(r"processed/ols_vs_wls_comparison")

    SEQUENCE_LENGTH = 15000
    T_END = 1500

    compare_ols_vs_wls(
        data_folder=BASE_DATA_FOLDER,
        output_folder=OUTPUT_FOLDER,
        sequence_length=SEQUENCE_LENGTH,
        t_end=T_END
    )