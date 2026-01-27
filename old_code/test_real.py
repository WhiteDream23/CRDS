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
def decay_model(t, I0, tau):
    tau = max(tau, 1e-6)
    t = np.clip(t, 0, None)
    exp_arg = -t / tau
    exp_arg = np.clip(exp_arg, -700, 700)
    with np.errstate(over='ignore'): # 忽略 overflow
        result = I0 * np.exp(exp_arg)
    return result

def process_signal_nlf(signal, t_axis, sg_window=15, sg_polyorder=2):
    try:
        # 1. S-G 滤波
        filtered_signal = savgol_filter(signal, sg_window, sg_polyorder)

        # 2. NLF 拟合
        initial_guess_I0 = filtered_signal[0] if len(filtered_signal) > 0 else 1.0
        initial_guess_tau = 140.0 # 合理的初始猜测值
        p0 = [initial_guess_I0, initial_guess_tau]

        # 设定参数边界，防止 I0 或 tau 为负或零
        bounds = ([0, 1e-6], [np.inf, np.inf])

        params, covariance = curve_fit(decay_model, t_axis, filtered_signal, p0=p0, bounds=bounds, maxfev=5000) # 增加 maxfev
        nlf_tau = params[1]
        return nlf_tau
    except (RuntimeError, ValueError, TypeError) as e:
        # print(f"NLF 拟合失败: {e}")
        return np.nan

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


    # --- 5. 对齐数据与计算指标 ---
    print("\n--- 性能对比评估 ---")

    model_taus_valid = np.array(model_taus)
    nlf_taus_valid = np.array(nlf_taus)

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
        results['模型'] = {'mean': mean_pred, 'variance': var_pred, 'min': min_pred, 'max': max_pred,
                           'std_dev': std_pred, 'success_rate': success_rate}
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
        results['S-G+NLF'] = {'mean': mean_pred, 'variance': var_pred, 'min': min_pred, 'max': max_pred,
                              'std_dev': std_pred, 'success_rate': success_rate}
    else:
        print("  没有有效的拟合结果。")
        results['S-G+NLF'] = {'mean': np.nan, 'variance': np.nan, 'min': np.nan, 'max': np.nan, 'std_dev': np.nan,
                              'success_rate': 0.0}

    if results:
        print("\n评估完成。")
        # 示例：绘制简单的直方图比较分布
        plt.figure(figsize=(10, 6))
        model_valid = np.array(model_taus_valid)[np.isfinite(model_taus_valid)]
        nlf_valid = np.array(nlf_taus_valid)[np.isfinite(nlf_taus_valid)]

        if len(model_valid) > 0 or len(nlf_valid) > 0:
            all_valid = np.concatenate((model_valid, nlf_valid))
            bins = np.linspace(all_valid.min(), all_valid.max(), 50) if len(all_valid) > 0 else 50

            if len(model_valid) > 0:
                plt.hist(model_valid, bins=bins, alpha=0.6, label=f'模型 (Std={results["模型"]["std_dev"]:.4f})',
                         density=True)
            if len(nlf_valid) > 0:
                plt.hist(nlf_valid, bins=bins, alpha=0.6, label=f'S-G+NLF (Std={results["S-G+NLF"]["std_dev"]:.4f})',
                         density=True)

            plt.xlabel("预测 Tau (μs)")
            plt.ylabel("概率密度")
            plt.title("预测 Tau 值分布对比 (无标签数据)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            # plt.savefig("tau_distribution_comparison_unlabeled.png") # 可以取消注释保存图片
            plt.show()
def natural_sort_key(s):
    # 从字符串中提取数字并转换为整数，用于自然排序
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

if __name__ == "__main__":
    # 设置环境变量 (也可以在脚本开头设置)
    # import os
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # --- 配置参数 ---
    model_path = r"C:\Users\Mingkai\Desktop\xmk4090code\phy_best_ringdown_model.pth"
    # model_path = "phy_best_ringdown_model.pth" # <--- 修改为你的模型路径
    # data_file = "processed_data.npz"              # <--- 修改为你的测试数据 .npz 文件路径
    data_folder = r"C:\Users\Mingkai\Desktop\rawdata\processed"
    output_directory = "real_data_comparison_output" # <--- 输出结果目录名

    # S-G 滤波器参数 (使用之前性能好的参数)
    sg_filter_window = 21
    sg_filter_polyorder = 3

    # # --- 执行评估与对比 ---
    # evaluate_and_compare_models(model_path, data_file, output_dir=output_directory,
    #                             sg_window=sg_filter_window, sg_polyorder=sg_filter_polyorder)
    # 获取文件夹中的所有 .npz 文件
    npz_files = glob.glob(os.path.join(data_folder, "*.npz"))
    npz_files.sort(key=natural_sort_key)  # 使用自然排序
    print(f"找到 {len(npz_files)} 个 .npz 文件")

    # 逐个处理每个 .npz 文件
    for npz_file in npz_files:
        print(f"正在处理文件: {npz_file}")
        evaluate_and_compare_models(
            model_path=model_path,
            data_file=npz_file,
            output_dir=output_directory,
            sg_window=sg_filter_window,
            sg_polyorder=sg_filter_polyorder
        )