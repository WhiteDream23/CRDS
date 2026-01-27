import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import torch
import torch.nn.functional as F
import re
import pandas as pd
from tqdm import tqdm  # 引入tqdm来显示进度条
from models import TUNN, UNet1D, ConvAutoencoder, SimpleRNN
from utils import apply_savgol_filter, apply_median_filter, apply_wavelet_filter, apply_kalman_filter

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
# 所有对比方法的名称
ALL_METHODS = [
    'Raw', 'Savitzky-Golay', 'Median Filter', 'Wavelet Filter', 'Kalman Filter',
    'TUNN', 'UNet1D', 'ConvAutoencoder', 'SimpleRNN'
]

# 深度学习模型的权重文件路径 (请根据实际情况修改)
MODEL_PATHS = {
    'TUNN': 'best_tunn_denoiser_model_raw.pth',
    'UNet1D': 'best_UNet_denoiser_model.pth',
    'ConvAutoencoder': 'best_ConvAutoencoder_denoiser_model.pth',
    'SimpleRNN': 'best_SimpleRNN_denoiser_model.pth'
}
MODEL_INPUT_LENGTH = 10000 # 模型训练时使用的输入长度

# 传统滤波器的参数
SG_WINDOW_LENGTH = 201
SG_POLY_ORDER = 3
MEDIAN_FILTER_SIZE = 51

def load_all_models(device):
    """加载所有深度学习模型到指定设备"""
    models = {
        'TUNN': TUNN(MODEL_INPUT_LENGTH),
        'UNet1D': UNet1D(),
        'ConvAutoencoder': ConvAutoencoder(MODEL_INPUT_LENGTH),
        'SimpleRNN': SimpleRNN(MODEL_INPUT_LENGTH)
    }
    loaded_models = {}
    print("\n--- 正在加载深度学习模型 ---")
    for name, model in models.items():
        path = MODEL_PATHS.get(name)
        if path and os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=device))
                model.to(device)
                model.eval()
                loaded_models[name] = model
                print(f"  - 成功加载模型: {name} (来自 {path})")
            except Exception as e:
                print(f"  - 警告: 加载模型 {name} 失败: {e}")
        else:
            print(f"  - 警告: 未找到模型 {name} 的权重文件 '{path}'，将跳过此模型。")
    print("--- 模型加载完成 ---\n")
    return loaded_models


# --- 4. 修改后的主分析函数 ---

def analyze_data_with_denoising(data_folder, sequence_length=12000, t_end=1200, output_csv="parameter_analysis.csv"):
    """
    遍历NPZ文件，对每个信号应用所有去噪方法进行拟合，并汇总统计结果。
    """
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载所有DL模型
    loaded_dl_models = load_all_models(device)

    # 获取文件列表
    npz_files = sorted(glob.glob(os.path.join(data_folder, "*.npz")), key=natural_sort_key)
    if not npz_files:
        print(f"在 '{data_folder}' 目录中未找到任何 .npz 文件。")
        return

    print(f"在 '{data_folder}' 中找到 {len(npz_files)} 个 .npz 文件，开始分析...")
    t_axis = np.linspace(0, t_end, sequence_length, dtype=np.float32)

    all_files_summary_data = []

    for npz_file in tqdm(npz_files, desc="处理NPZ文件"):
        file_basename = os.path.basename(npz_file)

        # 为当前文件初始化结果存储结构
        file_results = {method: {'i0': [], 'tau': [], 'b': [], 'residuals': []} for method in ALL_METHODS}

        try:
            data = np.load(npz_file)['data']

            # 遍历文件中的每一个信号
            for signal in data:
                if len(signal) != sequence_length:
                    continue

                # 1. 原始信号 (Raw)
                I0, tau, B, res_std, success = perform_nlf_fit(signal, t_axis)
                if success:
                    file_results['Raw']['i0'].append(I0)
                    file_results['Raw']['tau'].append(tau)
                    file_results['Raw']['b'].append(B)
                    file_results['Raw']['residuals'].append(res_std)

                # 2. 传统滤波器
                denoised_signals = {
                    'Savitzky-Golay': apply_savgol_filter(signal, SG_WINDOW_LENGTH, SG_POLY_ORDER),
                    'Median Filter': apply_median_filter(signal, MEDIAN_FILTER_SIZE),
                    'Wavelet Filter': apply_wavelet_filter(signal),
                    'Kalman Filter': apply_kalman_filter(signal)
                }
                for name, denoised_sig in denoised_signals.items():
                    I0, tau, B, res_std, success = perform_nlf_fit(denoised_sig, t_axis)
                    if success:
                        file_results[name]['i0'].append(I0)
                        file_results[name]['tau'].append(tau)
                        file_results[name]['b'].append(B)
                        file_results[name]['residuals'].append(res_std)

                # 3. 深度学习模型
                with torch.no_grad():
                    # 将numpy信号转为torch tensor,并增加batch和channel维度
                    input_tensor = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0).to(device)

                    for name, model in loaded_dl_models.items():
                        # --- 关键步骤: 信号长度适配 ---
                        # 将输入信号从原始长度(12000)缩放到模型期望长度(1000)
                        resized_input = F.interpolate(input_tensor, size=MODEL_INPUT_LENGTH, mode='linear',
                                                      align_corners=False)

                        # 模型推理
                        denoised_resized = model(resized_input)

                        # 将去噪后的信号从(1000)再缩放回原始长度(12000)
                        denoised_tensor_full_length = F.interpolate(denoised_resized, size=sequence_length,
                                                                    mode='linear', align_corners=False)

                        # 转回numpy用于拟合
                        denoised_sig_dl = denoised_tensor_full_length.squeeze().cpu().numpy()

                        I0, tau, B, res_std, success = perform_nlf_fit(denoised_sig_dl, t_axis)
                        if success:
                            file_results[name]['i0'].append(I0)
                            file_results[name]['tau'].append(tau)
                            file_results[name]['b'].append(B)
                            file_results[name]['residuals'].append(res_std)

            # --- 文件内所有信号处理完毕，进行统计 ---
            summary_row = {'filename': file_basename}
            for method_name in ALL_METHODS:
                params = file_results[method_name]
                if not params['tau']:  # 如果该方法没有任何成功拟合
                    continue

                stats_i0 = calculate_parameter_stats(params['i0'])
                stats_tau = calculate_parameter_stats(params['tau'])
                stats_b = calculate_parameter_stats(params['b'])
                stats_res = calculate_parameter_stats(params['residuals'])

                # 将统计结果以 "方法名_统计项" 的格式添加到行中
                summary_row[f'{method_name}_successful_fits'] = stats_tau['count']
                for stat in ['mean', 'std', 'min', 'max']:
                    summary_row[f'{method_name}_I0_{stat}'] = stats_i0[stat]
                    summary_row[f'{method_name}_Tau_{stat}'] = stats_tau[stat]
                    summary_row[f'{method_name}_B_{stat}'] = stats_b[stat]
                    summary_row[f'{method_name}_Residuals_Std_{stat}'] = stats_res[stat]

            all_files_summary_data.append(summary_row)

        except Exception as e:
            print(f"处理文件 '{file_basename}' 时发生严重错误: {e}")

    # --- 所有文件分析完成，保存并可视化 ---
    if not all_files_summary_data:
        print("未能从任何文件中成功提取统计数据，不生成CSV文件。")
        return

    summary_df = pd.DataFrame(all_files_summary_data)
    summary_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n已将所有方法的详细统计摘要保存至: {output_csv}")

    # --- 5. 新增的可视化: 对比不同方法的 Tau 均值分布 ---
    plt.style.use('seaborn-v0_8-whitegrid')

    # 提取所有方法的 Tau 均值列
    tau_mean_cols = [f'{method}_Tau_mean' for method in ALL_METHODS if f'{method}_Tau_mean' in summary_df.columns]
    tau_mean_df = summary_df[tau_mean_cols].copy()
    tau_mean_df.columns = [col.replace('_Tau_mean', '') for col in tau_mean_cols]  # 清理列名

    plt.figure(figsize=(16, 8))
    tau_mean_df.boxplot(rot=30)
    plt.title('各去噪方法得到的 Tau 均值分布对比', fontsize=16)
    plt.ylabel('Tau 均值 (Mean Tau)')
    plt.xlabel('去噪方法 (Denoising Method)')
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(output_csv), "method_comparison_boxplots.png")
    plt.savefig(plot_path)
    print(f"已将方法对比箱形图保存至: {plot_path}")
    plt.show()

# --- 主运行部分 ---
if __name__ == "__main__":
    # 根据你的文件路径进行配置
    data_folder_path = r"C:\Users\Mingkai\Desktop\rawdata\processed2_10000"  # 包含 .npz 文件的文件夹
    output_csv_filename = r"processed\parameter_analysis\modelper_file_summary.csv"  # 结果保存路径

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv_filename), exist_ok=True)

    # 你的数据信号长度和时间范围 (请根据实际 npz 文件内容调整)
    signal_length = 10000
    total_time_span = 1000

    # 调用修改后的主函数
    analyze_data_with_denoising(
        data_folder=data_folder_path,
        sequence_length=signal_length,
        t_end=total_time_span,
        output_csv=output_csv_filename
    )