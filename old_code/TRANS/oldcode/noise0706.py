import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal as sg
from pathlib import Path
from tqdm import tqdm
import re

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
# --- 1. 基础函数 (与之前版本相同) ---

def decay_model_fit(t, I0, tau, B):
    """指数衰减模型。"""
    tau = max(tau, 1e-6)
    return I0 * np.exp(-t / tau) + B


def perform_nlf_fit(signal_data, t_axis):
    """执行非线性最小二乘法拟合，返回参数字典或None。"""
    if not isinstance(signal_data, np.ndarray) or signal_data.ndim != 1 or len(signal_data) < 10:
        return None
    try:
        bg_slice = signal_data[-int(len(signal_data) * 0.1):]
        guess_B = np.mean(bg_slice) if len(bg_slice) > 0 else signal_data[-1]
        guess_I0 = signal_data[0] - guess_B
        guess_tau = 140.0
        p0 = [guess_I0, guess_tau, guess_B]
        bounds = ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        params, _ = curve_fit(decay_model_fit, t_axis, signal_data, p0=p0, bounds=bounds, maxfev=10000)
        if not np.all(np.isfinite(params)):
            return None
        fitted_curve = decay_model_fit(t_axis, *params)
        residuals = signal_data - fitted_curve
        return {'I0': params[0], 'tau': params[1], 'B': params[2], 'residuals_std': np.std(residuals)}
    except (RuntimeError, ValueError, TypeError):
        return None


def find_top_n_frequencies(residuals, fs, n=5):
    """对残差进行FFT分析，找到前N个最强的干扰频率。"""
    N = len(residuals)
    if N < n: return []
    yf = np.fft.fft(residuals)
    xf = np.fft.fftfreq(N, 1 / fs)
    yf_abs = np.abs(yf[1:N // 2])
    top_n_indices = np.argsort(yf_abs)[::-1][:n]
    top_n_peak_indices_corrected = top_n_indices + 1
    return xf[top_n_peak_indices_corrected]


def natural_sort_key(s):
    """自然顺序排序。"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]


# --- 新增的绘图函数 ---
def plot_residual_fft_comparison(fs, residuals_original, residuals_filtered, output_path, filename, signal_index):
    """
    绘制并保存滤波前后残差的FFT频谱对比图。
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # 计算原始残差的FFT
    N = len(residuals_original)
    xf = np.fft.fftfreq(N, 1 / fs)
    yf_original = np.fft.fft(residuals_original)

    # 计算滤波后残差的FFT
    yf_filtered = np.fft.fft(residuals_filtered)

    # 绘制频谱
    ax.plot(xf[:N // 2], 2.0 / N * np.abs(yf_original[0:N // 2]), label='原始残差频谱', color='skyblue', linewidth=1.5)
    ax.plot(xf[:N // 2], 2.0 / N * np.abs(yf_filtered[0:N // 2]), label='滤波后残差频谱', color='red', linewidth=1.2,
            alpha=0.9)

    ax.set_title(f'文件 {filename} - 信号 #{signal_index + 1} 残差FFT对比')
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('幅度')
    ax.set_xlim(0, fs / 2)  # 可根据需要调整显示范围，例如 ax.set_xlim(0, 50)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path / f"{filename}_signal{signal_index + 1}_residuals_fft_comp.png")
    plt.close(fig)


# --- 2. 主分析函数 ---

def compare_group_stats_with_individual_filtering(data_folder, output_folder, sequence_length, t_end, n_filters=5):
    """
    主函数：进行个性化滤波和拟合，汇总统计，并为部分信号生成FFT对比图。
    """
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    plots_folder = output_folder / "fft_comparison_plots"  # 为新图表创建专用文件夹
    output_folder.mkdir(parents=True, exist_ok=True)
    plots_folder.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(list(data_folder.glob("*.npz")), key=natural_sort_key)
    if not npz_files:
        print(f"错误：在 '{data_folder}' 目录中未找到任何 .npz 文件。")
        return

    fs = sequence_length / t_end
    print(f"--- 分析开始 (采用个性化滤波策略) ---")
    print(f"采样率 (fs) 设置为: {fs:.2f} Hz")
    print(f"将对每个独立信号移除其自己的前 {n_filters} 大干扰。")
    print("-" * 40)

    all_files_summary = []

    for npz_file in tqdm(npz_files, desc="处理并对比NPZ文件"):
        results_original = []
        results_filtered = []

        all_signals = np.load(npz_file)
        input_sequences = all_signals.get('data', all_signals.get('arr_0'))
        if input_sequences.ndim == 1:
            input_sequences = [input_sequences]

        t_axis = np.linspace(0, t_end, input_sequences[0].shape[-1])

        for i, signal_data in enumerate(input_sequences):
            orig_params = perform_nlf_fit(signal_data, t_axis)
            if not orig_params:
                continue
            results_original.append(orig_params)

            fitted_curve_orig = decay_model_fit(t_axis, orig_params['I0'], orig_params['tau'], orig_params['B'])
            residuals_initial = signal_data - fitted_curve_orig
            top_frequencies = find_top_n_frequencies(residuals_initial, fs, n=n_filters)

            signal_to_filter = signal_data
            for f0 in top_frequencies:
                if f0 > 0:
                    Q = 200.0
                    b, a = sg.iirnotch(f0, Q, fs)
                    signal_to_filter = sg.filtfilt(b, a, signal_to_filter)

            filt_params = perform_nlf_fit(signal_to_filter, t_axis)
            if filt_params:
                results_filtered.append(filt_params)

                # --- 新增：为部分信号生成残差FFT对比图 ---
                if i % 50 == 0:  # 每50个信号生成一张对比图
                    # 计算滤波后的最终残差
                    fitted_curve_filt = decay_model_fit(t_axis, filt_params['I0'], filt_params['tau'], filt_params['B'])
                    residuals_final_filtered = signal_to_filter - fitted_curve_filt

                    plot_residual_fft_comparison(
                        fs=fs,
                        residuals_original=residuals_initial,
                        residuals_filtered=residuals_final_filtered,
                        output_path=plots_folder,
                        filename=npz_file.stem,
                        signal_index=i
                    )

        if not results_original or not results_filtered:
            tqdm.write(f"文件 {npz_file.name} 处理后无有效结果，已跳过。")
            continue

        df_original = pd.DataFrame(results_original)
        df_filtered = pd.DataFrame(results_filtered)
        stats_original = df_original.describe().transpose()
        stats_filtered = df_filtered.describe().transpose()

        summary_row = {'filename': npz_file.name}
        for param in ['I0', 'tau', 'B', 'residuals_std']:
            for stat in ['mean', 'std']:
                summary_row[f'{param}_{stat}_orig'] = stats_original.loc[param, stat]
                summary_row[f'{param}_{stat}_filt'] = stats_filtered.loc[param, stat]
        all_files_summary.append(summary_row)

    if all_files_summary:
        final_report_df = pd.DataFrame(all_files_summary)
        output_path = output_folder / "individual_filtering_report_with_plots.csv"
        final_report_df.to_csv(output_path, index=False, encoding='utf-8-sig', float_format='%.6f')
        print(f"\n个性化滤波对比分析报告已成功保存至: {output_path}")
    else:
        print("\n未能生成任何有效的分析结果。")


# --- 3. 运行主函数 ---
if __name__ == "__main__":
    # !!! 重要：请根据您的实际情况修改以下路径和参数 !!!

    BASE_DATA_FOLDER = Path(r"C:\Users\Mingkai\Desktop\rawdata\processed1_15000")
    OUTPUT_FOLDER = Path(r"processed/individual_comparison_v2")

    SEQUENCE_LENGTH = 15000
    T_END = 1500

    compare_group_stats_with_individual_filtering(
        data_folder=BASE_DATA_FOLDER,
        output_folder=OUTPUT_FOLDER,
        sequence_length=SEQUENCE_LENGTH,
        t_end=T_END,
        n_filters=2
    )