"""
文件名: singlevsnoise.py
output : processednew
用途: 对比单一频率信号与噪声,进行信噪比(SNR)分析和光谱对比
功能描述:
    - 计算信号的信噪比(SNR)
    - 支持多个基线范围的动态tau0计算
    - 生成光谱图表和SNR对比分析
    - 实现多种拟合方法的结果对比
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# --- Matplotlib 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# --- 1. 核心分析与绘图函数 (修改版) ---

def calculate_and_compare_snr(report_df, c, freq_map_func, peak_range, baseline_ranges, methods_to_compare,
                              output_folder):
    """
    【修改版】读取分析报告，支持多个基线范围，并从中动态计算tau0。
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # --- 步骤1：构建频率轴并排序 ---
    report_df['frequency'] = report_df['filename'].apply(freq_map_func)
    report_df = report_df.sort_values('frequency').reset_index(drop=True)

    # --- 步骤2：构建多个基线范围的组合蒙版 ---
    baseline_masks = []
    for start_freq, end_freq in baseline_ranges:
        mask = (report_df['frequency'] >= start_freq) & (report_df['frequency'] <= end_freq)
        baseline_masks.append(mask)
    # 使用逻辑“或”将所有基线区域合并
    combined_baseline_mask = pd.concat(baseline_masks, axis=1).any(axis=1)

    # 获取所有在基线区域内的数据点
    baseline_df = report_df[combined_baseline_mask]
    if baseline_df.empty:
        print("错误：根据您设定的基线范围，没有找到任何数据点。请检查范围设置。")
        return

    # --- 步骤3：为每种方法动态计算tau0，并生成光谱和SNR ---
    spectra = {}
    results = {}

    for method in methods_to_compare:
        tau_mean_col = f'tau_mean_{method}'
        if tau_mean_col not in report_df.columns: continue

        # --- 核心改动 1：为当前方法动态计算 tau0 ---
        tau0_method = baseline_df[tau_mean_col].mean()
        print(f"方法 '{method}': 从基线区域计算得到的动态 τ₀ = {tau0_method:.4f} µs")

        # 将tau单位从μs转换为s
        tau_in_seconds = report_df[tau_mean_col] * 1e-6
        tau0_in_seconds = tau0_method * 1e-6

        # 应用CRDS公式计算吸收系数 alpha
        alpha = (1 / c) * (1 / tau_in_seconds - 1 / tau0_in_seconds)
        spectra[method] = alpha.rename(f'alpha_{method}')

        # --- 计算SNR (与之前类似，但现在使用了新的alpha和基线蒙版) ---
        peak_mask = (report_df['frequency'] >= peak_range[0]) & (report_df['frequency'] <= peak_range[1])

        # 注意：这里的基线数据点现在来自所有指定的范围
        peak_data = spectra[method][peak_mask]
        baseline_data = spectra[method][combined_baseline_mask]

        if peak_data.empty or baseline_data.empty:
            print(f"警告：方法 {method} 的峰值或基线区域没有数据点，无法计算SNR。")
            continue

        alpha_peak = peak_data.max()
        # 注意：理论上，基线的alpha均值应为0，因为我们已经减去了1/tau0
        alpha_baseline_mean = baseline_data.mean()
        signal = alpha_peak - alpha_baseline_mean
        # signal = alpha_peak
        noise_std = baseline_data.std()

        snr = signal / noise_std if noise_std > 0 else float('inf')
        results[method] = {'SNR': snr, 'Signal_Height': signal, 'Noise_Std': noise_std, 'Calculated_Tau0': tau0_method}

    # --- 步骤4：打印报告并绘图 ---
    spectra_df = pd.concat([report_df['frequency'], *spectra.values()], axis=1)
    results_df = pd.DataFrame.from_dict(results, orient='index')
    print("\n--- 光谱信噪比 (SNR) 对比报告 (使用动态τ₀) ---")
    print(results_df.to_string(float_format="%.8f"))
    results_df.to_csv(output_folder / "spectral_snr_report_dynamic_tau0.csv", float_format='%.4f')

    plt.figure(figsize=(15, 8))
    for method in methods_to_compare:
        if f'alpha_{method}' in spectra_df.columns:
            plt.plot(spectra_df['frequency'], spectra_df[f'alpha_{method}'], marker='.', markersize=4, linestyle='-',
                     label=f'{method} (SNR={results.get(method, {}).get("SNR", 0):.1f})')

    # 高亮显示所有基线区域
    for i, (start, end) in enumerate(baseline_ranges):
        plt.axvspan(start, end, color='green', alpha=0.1, label='基线区域' if i == 0 else "")
    # 高亮显示峰值区域
    plt.axvspan(*peak_range, color='red', alpha=0.1, label='峰值区域')

    plt.title("不同拟合方法生成的光谱对比 (使用动态τ₀)", fontsize=16)
    plt.xlabel("频率 / 文件批次号", fontsize=12)
    plt.ylabel("吸收系数 α (cm⁻¹)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig(output_folder / "spectral_comparison_dynamic_tau0.png", dpi=200)
    plt.close()

    print(f"\n对比图和报告已保存至: {output_folder}")


# (natural_sort_key 函数保持不变)

# --- 2. 主程序入口 ---
if __name__ == '__main__':
    # !!! 重要：请根据您的实际情况修改以下所有参数 !!!

    # 1. 指向逐个文件的统计报告CSV

    # REPORT_CSV_PATH = Path(r"processed/fit_comparison_final4data2/length_final/length_main_report_final.csv")

    # # REPORT_CSV_PATH = Path(r"processed/fit_comparison_final3/length_final/length_main_report_final.csv")
    # REPORT_CSV_PATH = Path(r"E:\pythonProject_crds\Trans_filter\processednew\fit_comparison_data2_changeols_wls\length_final\length_main_report_final.csv")

    # # 2. 输出文件夹
    # OUTPUT_FOLDER = Path(r"processednew/spectral_snr_analysis_dynamic_data2_changeols_wls")
    REPORT_CSV_PATH = Path(r"E:\pythonProject_crds\processednew\essay_pictures\fit_comarison_data12_QC\length_final\length_main_report_final.csv")

    # 2. 输出文件夹
    OUTPUT_FOLDER = Path(r"E:\pythonProject_crds\processednew\essay_pictures\fit_comarison_data12_QC\length_final")

    # 3. CRDS物理参数
    SPEED_OF_LIGHT = 2.99792458e10  # cm/s


    # 4. 定义频率映射函数
    def map_filename_to_frequency(filename):
        try:
            return float(re.findall(r'\d+', filename)[-1])
        except:
            return np.nan


    # 5. 定义峰值和基线区域
    # data2 (40 60)   (10 39)(61,90)
    # data7 (70 100)  (1 69)(101 130)
    # data9 (70 90)   (1 69)(91 110)
    # data10 (60 90)  (1 59)(91 130)
    # data11 (65 95)  (10 59)
    # data12 (70 115) (1 69)(116 130)
    PEAK_RANGE = (70, 115)

    # --- 核心改动 2：将基线范围定义为一个列表，可以包含多个范围 ---
    BASELINE_RANGES = [
        (1, 69),
        (116, 130),
    ]

    # 6. 选择要对比的方法
    METHODS_TO_COMPARE = ['WLS', 'OLS', 'robust','WLS_QC', 'OLS_QC', 'robust_QC']
    METHODS_TO_COMPARE = ['WLS_QC', 'OLS'] # 建议只选两个对比，图面更整洁
    # --- 运行分析 ---
    try:
        full_report_df = pd.read_csv(REPORT_CSV_PATH)
        # 假设我们要分析 condition == 18000 的数据
        target_condition_df = full_report_df[full_report_df['condition'] == 18000].copy()

        if target_condition_df.empty:
            print("错误：在报告中找不到指定condition的数据。")
        else:
            calculate_and_compare_snr(
                report_df=target_condition_df,
                c=SPEED_OF_LIGHT,
                freq_map_func=map_filename_to_frequency,
                peak_range=PEAK_RANGE,
                baseline_ranges=BASELINE_RANGES,  # 传入多个范围
                methods_to_compare=METHODS_TO_COMPARE,
                output_folder=OUTPUT_FOLDER
            )
    except FileNotFoundError:
        print(f"错误：报告文件不存在，请检查路径 -> {REPORT_CSV_PATH}")
    except Exception as e:
        print(f"发生错误: {e}")