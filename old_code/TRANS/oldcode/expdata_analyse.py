import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import re
import pandas as pd
from tqdm import tqdm  # 引入tqdm来显示进度条

# 设置 Matplotlib 字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# --- 指数衰减模型 ---
# (此函数保持不变)
def decay_model_fit(t, I0, tau, B):
    tau = max(tau, 1e-6)
    t = np.clip(t, 0, None)
    exp_arg = -t / tau
    exp_arg = np.clip(exp_arg, -700, 700)
    with np.errstate(over='ignore'):
        result = I0 * np.exp(exp_arg) + B
    return result


# --- NLF 拟合函数 ---
# (此函数保持不变)
def perform_nlf_fit(signal, t_axis):
    initial_guess_I0 = signal[0] if len(signal) > 0 else 1.0
    initial_guess_tau = 140.0
    initial_guess_B = np.mean(signal[-int(len(signal) * 0.1):]) if len(signal) > 100 else signal[-1] if len(
        signal) > 0 else 0.0
    p0 = [initial_guess_I0, initial_guess_tau, initial_guess_B]
    bounds = ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
    try:
        params, covariance = curve_fit(decay_model_fit, t_axis, signal, p0=p0, bounds=bounds, maxfev=10000)
        if not np.all(np.isfinite(params)):
            raise ValueError("拟合参数包含非有限值。")
        fitted_signal = decay_model_fit(t_axis, *params)
        residuals = signal - fitted_signal
        residuals_std = np.std(residuals)
        return params[0], params[1], params[2], residuals_std, True
    except (RuntimeError, ValueError, TypeError) as e:
        return np.nan, np.nan, np.nan, np.nan, False


def natural_sort_key(s):
    # (此函数保持不变)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


# --- 新增: 辅助函数，用于计算单个参数数组的统计信息 ---
def calculate_parameter_stats(values):
    """计算一个参数列表的完整统计数据，返回一个字典。"""
    s = pd.Series(values).dropna()  # 使用Pandas Series方便计算
    if s.empty:
        return {
            'mean': np.nan, 'std': np.nan, 'min': np.nan,
            '25%': np.nan, '50%': np.nan, '75%': np.nan, 'max': np.nan,
            'count': 0
        }
    stats = {
        'mean': s.mean(),
        'std': s.std(),
        'min': s.min(),
        '25%': s.quantile(0.25),
        '50%': s.quantile(0.50),  # 中位数
        '75%': s.quantile(0.75),
        'max': s.max(),
        'count': len(s)  # 成功拟合的数量
    }
    return stats


def analyze_each_npz_file(data_folder, sequence_length=12000, t_end=1200, output_csv="parameter_ranges_per_file.csv"):
    """
    修改后的主函数：
    遍历文件夹下所有NPZ文件，对每个文件单独进行拟合与统计，
    最后将所有文件的统计结果汇总保存。
    """
    npz_files = glob.glob(os.path.join(data_folder, "*.npz"))
    if not npz_files:
        print(f"在 '{data_folder}' 目录中未找到任何 .npz 文件。")
        return

    npz_files.sort(key=natural_sort_key)
    print(f"在 '{data_folder}' 中找到 {len(npz_files)} 个 .npz 文件。")

    t_axis = np.linspace(0, t_end, sequence_length, dtype=np.float32)

    # --- 修改: 创建一个主列表，用于存储每个文件的统计结果 ---
    all_files_summary_data = []
    failed_files = []

    print("开始对每个NPZ文件进行独立的拟合与分析...")
    # 使用tqdm来显示文件处理进度
    for npz_file in tqdm(npz_files, desc="处理NPZ文件"):
        file_basename = os.path.basename(npz_file)

        # --- 修改: 为当前文件初始化临时参数列表 ---
        file_i0_list = []
        file_tau_list = []
        file_b_list = []
        file_residuals_std_list = []

        try:
            data = np.load(npz_file)
            input_sequences = data['data']

            for signal in input_sequences:
                if len(signal) != sequence_length:
                    continue  # 长度不符则跳过

                I0, tau, B, residuals_std, success = perform_nlf_fit(signal, t_axis)
                if success:
                    file_i0_list.append(I0)
                    file_tau_list.append(tau)
                    file_b_list.append(B)
                    file_residuals_std_list.append(residuals_std)

            # --- 新增: 在处理完一个文件的所有信号后，立即进行统计 ---
            if not file_tau_list:  # 如果当前文件没有一个信号成功拟合
                print(f"警告: 文件 {file_basename} 未能成功拟合任何信号。")
                continue

            stats_i0 = calculate_parameter_stats(file_i0_list)
            stats_tau = calculate_parameter_stats(file_tau_list)
            stats_b = calculate_parameter_stats(file_b_list)
            stats_residuals = calculate_parameter_stats(file_residuals_std_list)

            # --- 新增: 将当前文件的统计结果整理成一个字典 ---
            summary_row = {
                'filename': file_basename,
                'successful_fits': stats_tau['count'],
                'I0_mean': stats_i0['mean'], 'I0_std': stats_i0['std'], 'I0_min': stats_i0['min'],
                'I0_max': stats_i0['max'],
                'Tau_mean': stats_tau['mean'], 'Tau_std': stats_tau['std'], 'Tau_min': stats_tau['min'],
                'Tau_max': stats_tau['max'],
                'B_mean': stats_b['mean'], 'B_std': stats_b['std'], 'B_min': stats_b['min'], 'B_max': stats_b['max'],
                'Residuals_Std_mean': stats_residuals['mean'], 'Residuals_Std_std': stats_residuals['std']
            }
            all_files_summary_data.append(summary_row)

        except Exception as e:
            print(f"读取或处理文件 '{file_basename}' 时发生严重错误: {e}")
            failed_files.append(file_basename)

    print("\n所有文件分析完成。")
    if failed_files:
        print(f"以下文件处理失败: {', '.join(failed_files)}")

    # --- 修改: 将所有文件的统计结果汇总并保存 ---
    if not all_files_summary_data:
        print("未能从任何文件中成功提取统计数据，不生成CSV文件。")
        return

    summary_df = pd.DataFrame(all_files_summary_data)

    print("\n--- 每个文件的参数统计摘要 ---")
    # 为了简洁显示，只打印部分关键列
    print(summary_df[['filename', 'successful_fits', 'Tau_mean', 'Tau_std', 'Residuals_Std_mean']].round(4))

    summary_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n已将每个文件的详细统计摘要保存至: {output_csv}")

    # --- 可视化调整: 绘制每个文件统计值(例如均值)的分布 ---
    plt.figure(figsize=(15, 10))
    plt.suptitle("各文件统计值分布图 (Distribution of Per-File Statistics)", fontsize=16)

    plt.subplot(2, 2, 1)
    summary_df['Tau_mean'].plot(kind='hist', bins=30, alpha=0.7, title='各文件 Tau 平均值分布')
    plt.xlabel('Tau 平均值')
    plt.ylabel('文件数量')

    plt.subplot(2, 2, 2)
    summary_df['Tau_std'].plot(kind='hist', bins=30, alpha=0.7, title='各文件 Tau 标准差分布')
    plt.xlabel('Tau 标准差')
    plt.ylabel('文件数量')

    plt.subplot(2, 2, 3)
    summary_df['I0_mean'].plot(kind='hist', bins=30, alpha=0.7, title='各文件 I0 平均值分布')
    plt.xlabel('I0 平均值')
    plt.ylabel('文件数量')

    plt.subplot(2, 2, 4)
    summary_df['Residuals_Std_mean'].plot(kind='hist', bins=30, alpha=0.7, title='各文件残差标准差平均值分布')
    plt.xlabel('残差标准差平均值')
    plt.ylabel('文件数量')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(os.path.dirname(output_csv), "per_file_stats_distributions.png"))
    plt.show()


# --- 主运行部分 ---
if __name__ == "__main__":
    # 根据你的文件路径进行配置
    data_folder_path = r"C:\Users\Mingkai\Desktop\rawdata\processed1_10000"  # 包含 .npz 文件的文件夹
    output_csv_filename = r"processed\parameter_analysis\per_file_summary.csv"  # 结果保存路径

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv_filename), exist_ok=True)

    # 你的数据信号长度和时间范围 (请根据实际 npz 文件内容调整)
    signal_length = 10000
    total_time_span = 1000

    # 调用修改后的主函数
    analyze_each_npz_file(
        data_folder=data_folder_path,
        sequence_length=signal_length,
        t_end=total_time_span,
        output_csv=output_csv_filename
    )