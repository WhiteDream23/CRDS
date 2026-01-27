# --- START OF FILE model_evaluate_comparison_integrated.py ---

# 设置环境变量 KMP_DUPLICATE_LIB_OK=TRUE (建议放在脚本最开头)
import os
import re
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import time
import glob

# 假设 e2emodel.py 文件在同一目录下或已安装
from old_code.e2emodel import RingdownCNN # 导入你的模型类


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# --- NLF 相关函数 ---
# def decay_model(t, I0, tau):
#     tau = max(tau, 1e-6)
#     t = np.clip(t, 0, None)
#     exp_arg = -t / tau
#     exp_arg = np.clip(exp_arg, -700, 700)
#     with np.errstate(over='ignore'): # 忽略 overflow
#         result = I0 * np.exp(exp_arg)
#     return result
#
# def process_signal_nlf(signal, t_axis, sg_window=15, sg_polyorder=2):
#     try:
#         # 1. S-G 滤波
#         filtered_signal = savgol_filter(signal, sg_window, sg_polyorder)
#         # 2. NLF 拟合
#         initial_guess_I0 = filtered_signal[0] if len(filtered_signal) > 0 else 1.0
#         initial_guess_tau = 140.0 # 合理的初始猜测值
#         p0 = [initial_guess_I0, initial_guess_tau]
#
#         # 设定参数边界，防止 I0 或 tau 为负或零
#         bounds = ([0, 1e-6], [np.inf, np.inf])
#
#         params, covariance = curve_fit(decay_model, t_axis, filtered_signal, p0=p0, bounds=bounds, maxfev=5000) # 增加 maxfev
#         nlf_tau = params[1]
#         return nlf_tau
#     except (RuntimeError, ValueError, TypeError) as e:
#         # print(f"NLF 拟合失败: {e}")
#         return np.nan
#
# def process_signal_nlf_nofilter(signal, t_axis):
#     """不带滤波的指数拟合（NLF no filter）"""
#     try:
#         initial_guess_I0 = signal[0] if len(signal) > 0 else 1.0
#         initial_guess_tau = 140.0
#         p0 = [initial_guess_I0, initial_guess_tau]
#         bounds = ([0, 1e-6], [np.inf, np.inf])
#
#         params, _ = curve_fit(decay_model, t_axis, signal, p0=p0, bounds=bounds, maxfev=5000)
#         return params[1]  # 返回 tau
#     except (RuntimeError, ValueError, TypeError):
#         return np.nan
def decay_model(t, I0, tau, B):
    # 为拟合稳定性进行参数裁剪和错误状态忽略
    tau = max(tau, 1e-6)  # tau 必须 > 0
    t = np.clip(t, 0, None)  # 时间 t 必须 >= 0
    exp_arg = -t / tau
    exp_arg = np.clip(exp_arg, -700, 700)  # 防止 exp(x) 溢出或下溢
    with np.errstate(over='ignore'):  # 忽略 overflow 警告
        result = I0 * np.exp(exp_arg) + B
    return result

def process_signal_nlf(signal, t_axis, sg_window=15, sg_polyorder=2):
    try:
        # 1. S-G 滤波
        filtered_signal = savgol_filter(signal, sg_window, sg_polyorder)
        # 2. NLF 拟合
        initial_guess_I0 = filtered_signal[0] if len(filtered_signal) > 0 else 1.0
        initial_guess_tau = 140.0 # 合理的初始猜测值
        initial_guess_B = np.mean(signal[-int(len(signal) * 0.1):]) if len(signal) > 100 else signal[-1] if len(
            signal) > 0 else 0.0

        p0 = [initial_guess_I0, initial_guess_tau, initial_guess_B]

        # 设定参数边界，防止 I0 或 tau 为负或零
        bounds = (
            [0, 1e-6, -np.inf],  # 最小值 [I0_min, tau_min, B_min]
            [np.inf, np.inf, np.inf]  # 最大值 [I0_max, tau_max, B_max]
        )
        params, covariance = curve_fit(decay_model, t_axis, signal, p0=p0, bounds=bounds, maxfev=10000)  # 增加 maxfev

        nlf_tau = params[1]
        return nlf_tau
    except (RuntimeError, ValueError, TypeError) as e:
        # print(f"NLF 拟合失败: {e}")
        return np.nan

def process_signal_nlf_nofilter(signal, t_axis):
    """不带滤波的指数拟合（NLF no filter）"""
    try:
        initial_guess_I0 = signal[0] if len(signal) > 0 else 1.0
        initial_guess_tau = 140.0 # 合理的初始猜测值
        initial_guess_B = np.mean(signal[-int(len(signal) * 0.1):]) if len(signal) > 100 else signal[-1] if len(
            signal) > 0 else 0.0

        p0 = [initial_guess_I0, initial_guess_tau, initial_guess_B]

        # 设定参数边界，防止 I0 或 tau 为负或零
        bounds = (
            [0, 1e-6, -np.inf],  # 最小值 [I0_min, tau_min, B_min]
            [np.inf, np.inf, np.inf]  # 最大值 [I0_max, tau_max, B_max]
        )
        params, covariance = curve_fit(decay_model, t_axis, signal, p0=p0, bounds=bounds, maxfev=10000)  # 增加 maxfev

        nlf_tau = params[1]
        return  nlf_tau
    except (RuntimeError, ValueError, TypeError):
        return np.nan
def tau_to_absorption_coeff(tau, light_speed=3e8):
    """根据 tau 计算吸收系数 alpha（单位 1/m）"""
    if tau is None or np.isnan(tau) or tau <= 0:
        return np.nan
    return 1.0 / (tau * light_speed)
# --- 主评估与对比函数 ---
def evaluate_and_compare_models(model_path, data_file, output_dir="comparison_results", device=None,
                                sg_window=21, sg_polyorder=3): # 设为之前性能好的参数
    """
    使用预训练模型和 S-G+NLF 对数据进行预测、评估和对比

    参数:
        model_path: 预训练模型的路径
        data_file: 处理后的数据文件路径(.npz, 包含 'data' 和 'targets')
        output_dir: 保存结果图表的目录
        device: 使用的设备 ('cpu' 或 'cuda')
        sg_window, sg_polyorder: S-G 滤波器参数
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 加载数据 ---
    print(f"加载数据从: {data_file}")
    data = np.load(data_file)
    # input_sequences = data['data']
    # taus = data['filenames']
    input_sequences = data['data']
    num_samples=len(input_sequences)
    # 定义时间轴和 tau 归一化范围 (需要与模型训练时一致)
    tau_min, tau_max = 140.0, 190.0
    sequence_length=12000
    # !! 重要 !! 确认时间轴范围
    t_end = 1200 # 假设时间范围是 0 到 120
    t_axis = np.linspace(0, t_end, sequence_length, dtype=np.float32)
    print(f"假设时间轴为 0 到 {t_end:.2f} 秒，信号长度 {sequence_length}。")

    # --- 2. 加载模型 ---
    print(f"加载模型从: {model_path}")
    try:
        # checkpoint = torch.load(model_path, map_location=device, weights_only=True) # 使用 weights_only 可能更快
        checkpoint = torch.load(model_path, map_location=device) # 移除 weights_only 以防万一
        # 确保模型类与保存时一致
        model = RingdownCNN(dropout_rate=0.2).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        # return # 如果模型加载失败，可以选择是否继续 NLF

    # --- 3. 处理数据: 模型 ---
    model_taus = []
    valid_indices_model = []
    print("开始处理信号 (模型)...")
    start_time_model = time.time()
    model.eval() # 确保是评估模式
    with torch.no_grad():
        for i in range(num_samples):
            signal = input_sequences[i]
            # a. 归一化
            signal_min = signal.min()
            signal_max = signal.max()
            if signal_max > signal_min:
                normalized_signal = (signal - signal_min) / (signal_max - signal_min)
            else:
                model_taus.append(np.nan) # 无法处理，记录 nan
                continue

            # b. 预测
            try:
                input_tensor = torch.FloatTensor(normalized_signal).unsqueeze(0).to(device)
                output_normalized = model(input_tensor)
                predicted_tau = output_normalized.item() * (tau_max - tau_min) + tau_min
                if np.isfinite(predicted_tau):
                    model_taus.append(predicted_tau)
                    valid_indices_model.append(i)
                else:
                    model_taus.append(np.nan)
            except Exception as e:
                # print(f"模型预测样本 {i} 时出错: {e}")
                model_taus.append(np.nan)

    end_time_model = time.time()
    model_success_count = np.isfinite(model_taus).sum()
    model_success_rate = model_success_count / num_samples
    if model_success_count > 0 :
        time_per_signal_model = (end_time_model - start_time_model) / num_samples
    else:
         time_per_signal_model = float('inf')
    print(f"模型处理完成，用时: {end_time_model - start_time_model:.2f} 秒，成功率: {model_success_rate*100:.2f}% ({time_per_signal_model*1000:.2f}毫秒/样本)")

    # --- 4. 处理数据: S-G + NLF ---
    nlf_taus = []
    valid_indices_nlf = []
    print("开始处理信号 (S-G+NLF)...")
    start_time_nlf = time.time()
    for i in range(num_samples):
        signal = input_sequences[i]
        # a. 归一化
        signal_min = signal.min()
        signal_max = signal.max()
        if signal_max > signal_min:
            signal = (signal - signal_min) / (signal_max - signal_min)
        else:
            nlf_taus.append(np.nan)  # 无法处理，记录 nan
            continue
        fit_tau = process_signal_nlf(signal, t_axis, sg_window, sg_polyorder)
        nlf_taus.append(fit_tau) # 无论成功与否都记录，后面再过滤
        if not np.isnan(fit_tau):
            valid_indices_nlf.append(i)
    end_time_nlf = time.time()
    nlf_success_count = len(valid_indices_nlf)
    nlf_success_rate = nlf_success_count / num_samples
    if nlf_success_count > 0:
         time_per_signal_nlf = (end_time_nlf - start_time_nlf) / num_samples
    else:
         time_per_signal_nlf = float('inf')
    print(f"NLF 处理完成，用时: {end_time_nlf - start_time_nlf:.2f} 秒，成功率: {nlf_success_rate*100:.2f}% ({time_per_signal_nlf*1000:.2f}毫秒/样本)")

    # --- 处理数据: 无滤波拟合 ---
    nlf_nofilter_taus = []
    valid_indices_nofilter = []
    print("开始处理信号 (NLF-无滤波)...")
    start_time_nofilter = time.time()
    for i in range(num_samples):
        signal = input_sequences[i]
        # a. 归一化
        if signal_max > signal_min:
            signal = (signal - signal_min) / (signal_max - signal_min)
        else:
            nlf_taus.append(np.nan)  # 无法处理，记录 nan
        fit_tau = process_signal_nlf_nofilter(signal, t_axis)
        nlf_nofilter_taus.append(fit_tau)
        if not np.isnan(fit_tau):
            valid_indices_nofilter.append(i)
    end_time_nofilter = time.time()
    nofilter_success_count = len(valid_indices_nofilter)
    nofilter_success_rate = nofilter_success_count / num_samples
    if nofilter_success_count > 0:
        time_per_signal_nofilter = (end_time_nofilter - start_time_nofilter) / num_samples
    else:
        time_per_signal_nofilter = float('inf')
    print(
        f"NLF-无滤波 处理完成，用时: {end_time_nofilter - start_time_nofilter:.2f} 秒，成功率: {nofilter_success_rate * 100:.2f}% ({time_per_signal_nofilter * 1000:.2f}毫秒/样本)")

    # --- 5. 对齐数据与计算指标 ---
    print("\n--- 性能对比评估 ---")


    model_taus_valid = np.array(model_taus)
    nlf_taus_valid = np.array(nlf_taus)
    nlf_nofilter_taus_valid = np.array(nlf_nofilter_taus)

    results = {}
    print("\n模型方法:")
    if len(model_taus_valid) > 0:
        mean_pred = np.mean(model_taus_valid)
        var_pred = np.var(model_taus_valid)
        min_pred = np.min(model_taus_valid)
        max_pred = np.max(model_taus_valid)
        std_pred = np.std(model_taus_valid)  # 计算标准差
        success_rate = len(model_taus_valid) / num_samples * 100
        print(f"  有效样本数: {len(model_taus_valid)} ({success_rate:.2f}%)")
        print(f"  平均值: {mean_pred:.6f}")
        print(f"  方差:   {var_pred:.6f}")
        print(f"  最小值: {min_pred:.6f}")
        print(f"  最大值: {max_pred:.6f}")
        print(f"  标准差: {std_pred:.6f}")  # 打印标准差
        mean_alpha_model = np.mean([tau_to_absorption_coeff(t) for t in model_taus_valid if not np.isnan(t)])
        results['模型'] = {'mean': mean_pred, 'variance': var_pred, 'min': min_pred, 'max': max_pred,
                           'std_dev': std_pred, 'success_rate': success_rate,'absorption': mean_alpha_model}
    else:
        print("  没有有效的预测结果。")
        results['模型'] = {'mean': np.nan, 'variance': np.nan, 'min': np.nan, 'max': np.nan, 'std_dev': np.nan,
                           'success_rate': 0.0}

    # 计算 NLF 统计量
    print("\nS-G+NLF 方法:")
    if len(nlf_taus_valid) > 0:
        mean_pred = np.mean(nlf_taus_valid)
        var_pred = np.var(nlf_taus_valid)
        min_pred = np.min(nlf_taus_valid)
        max_pred = np.max(nlf_taus_valid)
        std_pred = np.std(nlf_taus_valid)  # 计算标准差
        success_rate = len(nlf_taus_valid) / num_samples * 100
        print(f"  有效样本数: {len(nlf_taus_valid)} ({success_rate:.2f}%)")
        print(f"  平均值: {mean_pred:.6f}")
        print(f"  方差:   {var_pred:.6f}")
        print(f"  最小值: {min_pred:.6f}")
        print(f"  最大值: {max_pred:.6f}")
        print(f"  标准差: {std_pred:.6f}")  # 打印标准差
        mean_alpha = np.mean([tau_to_absorption_coeff(t) for t in nlf_taus_valid if not np.isnan(t)])
        results['S-G+NLF'] = {'mean': mean_pred, 'variance': var_pred, 'min': min_pred, 'max': max_pred,
                              'std_dev': std_pred, 'success_rate': success_rate,'absorption': mean_alpha}
    else:
        print("  没有有效的拟合结果。")
        results['S-G+NLF'] = {'mean': np.nan, 'variance': np.nan, 'min': np.nan, 'max': np.nan, 'std_dev': np.nan,
                              'success_rate': 0.0}
    # 计算 NLF（无滤波）统计量
    print("\nNLF-无滤波 方法:")
    if len(nlf_nofilter_taus_valid) > 0:
        mean_pred = np.mean(nlf_nofilter_taus_valid)
        var_pred = np.var(nlf_nofilter_taus_valid)
        min_pred = np.min(nlf_nofilter_taus_valid)
        max_pred = np.max(nlf_nofilter_taus_valid)
        std_pred = np.std(nlf_nofilter_taus_valid)
        success_rate = len(nlf_nofilter_taus_valid) / num_samples * 100
        print(f"  有效样本数: {len(nlf_nofilter_taus_valid)} ({success_rate:.2f}%)")
        print(f"  平均值: {mean_pred:.6f}")
        print(f"  方差:   {var_pred:.6f}")
        print(f"  最小值: {min_pred:.6f}")
        print(f"  最大值: {max_pred:.6f}")
        print(f"  标准差: {std_pred:.6f}")
        mean_nofiler_alpha = np.mean([tau_to_absorption_coeff(t) for t in nlf_nofilter_taus_valid if not np.isnan(t)])
        results['NLF无滤波'] = {'mean': mean_pred, 'variance': var_pred, 'min': min_pred, 'max': max_pred,
                                'std_dev': std_pred, 'success_rate': success_rate,'absorption': mean_nofiler_alpha}
    else:
        print("  没有有效的拟合结果。")
        results['NLF无滤波'] = {'mean': np.nan, 'variance': np.nan, 'min': np.nan, 'max': np.nan,
                            'std_dev': np.nan, 'success_rate': 0.0}

    # if results:
    #     print("\n评估完成。")
    #     # 示例：绘制简单的直方图比较分布
    #     plt.figure(figsize=(10, 6))
    #     model_valid = np.array(model_taus_valid)[np.isfinite(model_taus_valid)]
    #     nlf_valid = np.array(nlf_taus_valid)[np.isfinite(nlf_taus_valid)]
    #
    #     if len(model_valid) > 0 or len(nlf_valid) > 0:
    #         all_valid = np.concatenate((model_valid, nlf_valid))
    #         bins = np.linspace(all_valid.min(), all_valid.max(), 50) if len(all_valid) > 0 else 50
    #
    #         if len(model_valid) > 0:
    #             plt.hist(model_valid, bins=bins, alpha=0.6, label=f'模型 (Std={results["模型"]["std_dev"]:.4f})',
    #                      density=True)
    #         if len(nlf_valid) > 0:
    #             plt.hist(nlf_valid, bins=bins, alpha=0.6, label=f'S-G+NLF (Std={results["S-G+NLF"]["std_dev"]:.4f})',
    #                      density=True)
    #
    #         plt.xlabel("预测 Tau (μs)")
    #         plt.ylabel("概率密度")
    #         plt.title("预测 Tau 值分布对比 (无标签数据)")
    #         plt.legend()
    #         plt.grid(True, alpha=0.3)
    #         plt.tight_layout()
    #         # plt.savefig("tau_distribution_comparison_unlabeled.png") # 可以取消注释保存图片
    #         plt.show()

    return {
        'filename': os.path.basename(data_file),
        'model': results['模型'] if '模型' in results else None,
        'nlf': results['S-G+NLF'] if 'S-G+NLF' in results else None,
        'nlf_nofilter': results['NLF无滤波'] if 'NLF无滤波' in results else None
    }
def natural_sort_key(s):
    # 从字符串中提取数字并转换为整数，用于自然排序
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def process_all_files(model_path, data_folder, output_directory, sg_filter_window, sg_filter_polyorder):
    """处理所有文件并汇总结果"""
    npz_files = glob.glob(os.path.join(data_folder, "*.npz"))
    npz_files.sort(key=natural_sort_key)
    print(f"找到 {len(npz_files)} 个 .npz 文件")

    # 存储所有结果
    all_results = []

    # 创建输出目录
    os.makedirs(output_directory, exist_ok=True)

    # 处理每个文件
    for npz_file in npz_files:
        print(f"\n正在处理文件: {os.path.basename(npz_file)}")
        result = evaluate_and_compare_models(
            model_path=model_path,
            data_file=npz_file,
            output_dir=output_directory,
            sg_window=sg_filter_window,
            sg_polyorder=sg_filter_polyorder
        )
        all_results.append(result)

    # 保存详细结果到CSV
    save_detailed_results(all_results, output_directory)

    # 显示汇总统计
    show_summary_statistics(all_results)


def show_summary_statistics(results):
    """显示汇总统计信息"""
    print("\n=== 汇总统计 ===")

    # 提取有效数据
    model_means = [r['model']['mean'] for r in results if r['model'] is not None]
    model_stds = [r['model']['std_dev'] for r in results if r['model'] is not None]
    model_rates = [r['model']['success_rate'] for r in results if r['model'] is not None]

    nlf_means = [r['nlf']['mean'] for r in results if r['nlf'] is not None]
    nlf_stds = [r['nlf']['std_dev'] for r in results if r['nlf'] is not None]
    nlf_rates = [r['nlf']['success_rate'] for r in results if r['nlf'] is not None]

    # 打印统计信息
    print("\n模型性能汇总:")
    print(f"平均值范围: {min(model_means):.2f} - {max(model_means):.2f} (均值: {np.mean(model_means):.2f})")
    print(f"标准差范围: {min(model_stds):.2f} - {max(model_stds):.2f} (均值: {np.mean(model_stds):.2f})")
    print(f"成功率范围: {min(model_rates):.2f}% - {max(model_rates):.2f}% (均值: {np.mean(model_rates):.2f}%)")

    print("\nNLF性能汇总:")
    print(f"平均值范围: {min(nlf_means):.2f} - {max(nlf_means):.2f} (均值: {np.mean(nlf_means):.2f})")
    print(f"标准差范围: {min(nlf_stds):.2f} - {max(nlf_stds):.2f} (均值: {np.mean(nlf_stds):.2f})")
    print(f"成功率范围: {min(nlf_rates):.2f}% - {max(nlf_rates):.2f}% (均值: {np.mean(nlf_rates):.2f}%)")

def compute_baseline_alpha(alphas, method,base_ranges=[(20, 40), (60, 80)]):
    base_values = []
    for start, end in base_ranges:
        for i in range(start, end):
            if i < len(alphas) and not np.isnan(alphas[i][method]['absorption']):
                base_values.append(alphas[i][method]['absorption'])
    if base_values:
        return np.mean(base_values)
    else:
        return np.nan
def save_detailed_results(results, output_dir):
    """保存详细结果到CSV文件"""
    import pandas as pd
    #计算基线
    baseline_model_alpha=compute_baseline_alpha(results, 'model')
    baseline_nlf_alpha=compute_baseline_alpha(results, 'nlf')
    baseline_nlf_nofilter_alpha=compute_baseline_alpha(results, 'nlf_nofilter')

    # 准备数据
    data = []
    for r in results:
        row = {
            '文件名': r['filename'],
            '模型_平均值': r['model']['mean'] if r['model'] else np.nan,
            '模型_标准差': r['model']['std_dev'] if r['model'] else np.nan,
            '模型_成功率': r['model']['success_rate'] if r['model'] else np.nan,
            'NLF_平均值': r['nlf']['mean'] if r['nlf'] else np.nan,
            'NLF_标准差': r['nlf']['std_dev'] if r['nlf'] else np.nan,
            'NLF_成功率': r['nlf']['success_rate'] if r['nlf'] else np.nan,
            'NLF无滤波_平均值': r['nlf_nofilter']['mean'] if r['nlf_nofilter'] else np.nan,
            'NLF无滤波_标准差': r['nlf_nofilter']['std_dev'] if r['nlf_nofilter'] else np.nan,
            'NLF无滤波_成功率': r['nlf_nofilter']['success_rate'] if r['nlf_nofilter'] else np.nan,
            '模型_吸收系数': r['model']['absorption'] if r['model'] else np.nan,
            'NLF_吸收系数': r['nlf']['absorption'] if r['nlf'] else np.nan,
            'NLF无滤波_吸收系数': r['nlf_nofilter']['absorption'] if r['nlf_nofilter'] else np.nan,
            '模型_吸收系数2': r['model']['absorption']-baseline_model_alpha if r['model'] else np.nan,
            'NLF_吸收系数2': r['nlf']['absorption']-baseline_nlf_alpha if r['nlf'] else np.nan,
            'NLF无滤波_吸收系数2': r['nlf_nofilter']['absorption']-baseline_nlf_nofilter_alpha if r['nlf_nofilter'] else np.nan
        }

        data.append(row)

    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    output_file = os.path.join(output_dir, 'detailed_results2_absorb_weiguiyi.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存至: {output_file}")


if __name__ == "__main__":
    # 设置环境变量 (也可以在脚本开头设置)
    # import os
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # --- 配置参数 ---
    model_path = r"C:\Users\Mingkai\Desktop\xmk4090code\phy_best_ringdown_model.pth"
    # model_path = "phy_best_ringdown_model.pth" # <--- 修改为你的模型路径
    # data_file = "processed_data.npz"              # <--- 修改为你的测试数据 .npz 文件路径
    data_folder = r"C:\Users\Mingkai\Desktop\rawdata\processed2"
    output_directory = r"processed\real_data_comparison_output" # <--- 输出结果目录名

    # S-G 滤波器参数 (使用之前性能好的参数)
    sg_filter_window = 201
    sg_filter_polyorder = 3

    process_all_files(
        model_path=model_path,
        data_folder=data_folder,
        output_directory=output_directory,
        sg_filter_window=sg_filter_window,
        sg_filter_polyorder=sg_filter_polyorder
    )