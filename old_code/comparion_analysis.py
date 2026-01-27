import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import os
import time # 引入 time 模块来计时

# 假设 e2emodel.py 文件在同一目录下或已安装
from e2emodel import RingdownCNN # 导入你的模型类

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义指数衰减模型 (用于 NLF)
def decay_model(t, I0, tau):
    tau = max(tau, 1e-6) # 防止 tau 过小
    return I0 * np.exp(-t / tau)

# --- S-G + NLF 处理函数 ---
def process_signal_nlf(signal, t_axis, sg_window=15, sg_polyorder=2):
    """
    使用 S-G 滤波 + NLF 处理单个信号

    返回:
        拟合得到的 tau 值，如果失败则返回 np.nan
    """
    try:
        # 1. S-G 滤波
        filtered_signal = savgol_filter(signal, sg_window, sg_polyorder)

        # 2. NLF 拟合
        initial_guess_I0 = filtered_signal[0] if len(filtered_signal) > 0 else 1.0
        initial_guess_tau = 10.0 # 合理的初始猜测值
        p0 = [initial_guess_I0, initial_guess_tau]

        # 设定参数边界，防止 I0 或 tau 为负或零
        bounds = ([0, 1e-6], [np.inf, np.inf])

        params, covariance = curve_fit(decay_model, t_axis, filtered_signal, p0=p0, bounds=bounds, maxfev=5000) # 增加 maxfev
        nlf_tau = params[1]
        return nlf_tau
    except (RuntimeError, ValueError, TypeError) as e:
        # print(f"NLF 拟合失败: {e}")
        return np.nan

# --- 模型预测处理函数 ---
def process_signal_model(signal, model, device, tau_min, tau_max):
    """
    使用加载好的模型处理单个信号

    返回:
        模型预测的 tau 值
    """
    # a. 归一化 (与训练时一致)
    signal_min = signal.min()
    signal_max = signal.max()
    if signal_max > signal_min:
        normalized_signal = (signal - signal_min) / (signal_max - signal_min)
    else:
        normalized_signal = signal # 避免除零

    # b. 转换为 Tensor 并移到设备
    input_tensor = torch.FloatTensor(normalized_signal).unsqueeze(0).to(device) # 添加 batch 维度

    # c. 模型预测 (输出是归一化的 tau)
    with torch.no_grad():
        output_normalized = model(input_tensor)

    # d. 反归一化
    predicted_tau = output_normalized.item() * (tau_max - tau_min) + tau_min
    return predicted_tau

# --- 主评估与对比函数 ---
def evaluate_and_compare(model_path, data_file, output_dir="comparison_results", device=None,
                           sg_window=15, sg_polyorder=2):
    """
    加载数据，使用模型和 NLF 方法进行处理、评估和对比

    参数:
        model_path: 预训练模型路径
        data_file: .npz 数据文件路径 (包含 'data' 和 'targets' 或 'filenames' 对应的真实 tau)
        output_dir: 保存结果图表的目录
        device: 'cpu' 或 'cuda'
        sg_window: S-G 滤波器窗口大小
        sg_polyorder: S-G 滤波器多项式阶数
    """
    if device is None:
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 加载数据 ---
    print(f"加载数据从: {data_file}")
    data = np.load(data_file)
    input_sequences = data['data'] # 形状应为 (num_samples, sequence_length)
    # 检查 'targets' 或 'filenames' 来获取真实 tau
    if 'targets' in data:
        true_taus_raw = data['targets'] # 假设是未归一化的真实 tau
        print(f"找到 'targets'，共 {len(true_taus_raw)} 个样本")
    elif 'filenames' in data: # 兼容你之前的代码
        true_taus_raw = data['filenames']  # 假设是未归一化的真实 tau
        print(f"找到 'filenames'，共 {len(true_taus_raw)} 个样本")
    else:
        print("错误：数据文件中未找到 'targets' 或 'filenames' 来获取真实 tau 值。")
        return

    num_samples, sequence_length = input_sequences.shape
    print(f"数据加载完成，样本数: {num_samples}, 信号长度: {sequence_length}")

    # 定义时间轴 (需要与数据生成时一致)
    # !! 重要 !! 确认你的时间轴范围和点数
    t_end = 120 # 假设时间范围是 0 到 120
    t_axis = np.linspace(0, t_end, sequence_length, dtype=np.float32)

    # tau 归一化范围 (需要与模型训练时一致)
    tau_min, tau_max = 10.0, 20.0

    # --- 2. 加载模型 ---
    print(f"加载模型从: {model_path}")
    try:
        # 不再使用 weights_only=True，以防需要加载 optimizer 等状态 (虽然评估不需要)
        checkpoint = torch.load(model_path, map_location=device)
        # 确保模型类与保存时一致
        model = RingdownCNN(dropout_rate=0.2).to(device) # 或者 CNNLSTMModel
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # --- 3. 处理数据 ---
    model_taus = []
    nlf_taus = []
    valid_true_taus = [] # 用于存储 NLF 成功的样本对应的真实 tau
    valid_indices = []   # 用于存储 NLF 成功的样本索引

    print("开始处理信号...")
    start_time_model = time.time()
    for i in range(num_samples):
        signal = input_sequences[i]
        pred_tau = process_signal_model(signal, model, device, tau_min, tau_max)
        model_taus.append(pred_tau)
    end_time_model = time.time()
    time_per_signal_model = (end_time_model - start_time_model) / num_samples
    print(f"模型处理完成，用时: {end_time_model - start_time_model:.2f} 秒 ({time_per_signal_model*1000:.2f}毫秒/样本)")

    print("开始 NLF 处理...")
    start_time_nlf = time.time()
    for i in range(num_samples):
        signal = input_sequences[i]
        fit_tau = process_signal_nlf(signal, t_axis, sg_window, sg_polyorder)
        # 只有 NLF 成功时才记录
        if not np.isnan(fit_tau):
            nlf_taus.append(fit_tau)
            valid_true_taus.append(true_taus_raw[i]) # 记录对应的真实 tau
            valid_indices.append(i) # 记录索引
        # 可以选择打印失败信息
        # else:
        #     print(f"样本 {i} NLF 失败")
    end_time_nlf = time.time()
    nlf_success_rate = len(nlf_taus) / num_samples
    if len(nlf_taus) > 0:
         time_per_signal_nlf = (end_time_nlf - start_time_nlf) / len(nlf_taus) # 基于成功拟合的计算时间
    else:
         time_per_signal_nlf = float('inf')
    print(f"NLF 处理完成，用时: {end_time_nlf - start_time_nlf:.2f} 秒，成功率: {nlf_success_rate*100:.2f}% ({time_per_signal_nlf*1000:.2f}毫秒/成功样本)")

    # --- 4. 对齐数据 ---
    # 使模型预测结果与 NLF 成功的结果对齐
    model_taus_aligned = [model_taus[i] for i in valid_indices]
    true_taus_aligned = valid_true_taus # 已经是对应 NLF 成功样本的真实值了

    # 转换为 numpy 数组方便计算
    model_taus_np = np.array(model_taus_aligned)
    nlf_taus_np = np.array(nlf_taus)
    true_taus_np = np.array(true_taus_aligned)

    if len(true_taus_np) == 0:
        print("NLF 全部失败，无法进行比较。")
        return

    # --- 5. 计算评估指标 ---
    print("\n--- 性能对比评估 ---")
    print(f"(基于 {len(true_taus_np)} 个 NLF 成功的样本)")

    metrics = {}
    for name, predictions in [("模型", model_taus_np), ("S-G+NLF", nlf_taus_np)]:
        rmse = np.sqrt(mean_squared_error(true_taus_np, predictions))
        mae = mean_absolute_error(true_taus_np, predictions)
        r2 = r2_score(true_taus_np, predictions)
        error_std = np.std(predictions - true_taus_np)
        pred_std = np.std(predictions)
        metrics[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Error Std': error_std, 'Pred Std': pred_std}
        print(f"\n{name} 方法:")
        print(f"  RMSE: {rmse:.6f} μs")
        print(f"  MAE:  {mae:.6f} μs")
        print(f"  R²:   {r2:.6f}")
        print(f"  误差标准差: {error_std:.6f}")
        print(f"  预测值标准差: {pred_std:.6f}")

    metrics["模型"]['Time per sample (ms)'] = time_per_signal_model * 1000
    metrics["S-G+NLF"]['Time per sample (ms)'] = time_per_signal_nlf * 1000 if nlf_success_rate > 0 else float('inf')
    metrics["S-G+NLF"]['Success Rate (%)'] = nlf_success_rate * 100

    # --- 6. 结果可视化 ---
    print("\n生成对比图表...")

    # (a) 散点图: 预测 vs 真实
    plt.figure(figsize=(12, 5.5))
    plt.subplot(1, 2, 1)
    plt.scatter(true_taus_np, model_taus_np, alpha=0.5, label='模型预测', s=10)
    plt.plot([tau_min, tau_max], [tau_min, tau_max], 'r--', label='y=x')
    plt.xlabel("真实 Tau (μs)")
    plt.ylabel("预测 Tau (μs)")
    plt.title(f"模型预测 vs 真实值 (R²={metrics['模型']['R2']:.4f})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(tau_min - 1, tau_max + 1)
    plt.ylim(tau_min - 1, tau_max + 1)

    plt.subplot(1, 2, 2)
    plt.scatter(true_taus_np, nlf_taus_np, alpha=0.5, label='S-G+NLF 预测', s=10, color='orange')
    plt.plot([tau_min, tau_max], [tau_min, tau_max], 'r--', label='y=x')
    plt.xlabel("真实 Tau (μs)")
    plt.ylabel("预测 Tau (μs)")
    plt.title(f"S-G+NLF vs 真实值 (R²={metrics['S-G+NLF']['R2']:.4f})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(tau_min - 1, tau_max + 1)
    plt.ylim(tau_min - 1, tau_max + 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_vs_true_scatter.png"), dpi=300)
    plt.close()
    #
    # # (b) 误差分布直方图
    # model_errors = model_taus_np - true_taus_np
    # nlf_errors = nlf_taus_np - true_taus_np
    #
    # plt.figure(figsize=(10, 5))
    # # 确定合适的 bin 范围和数量
    # min_err = min(model_errors.min(), nlf_errors.min())
    # max_err = max(model_errors.max(), nlf_errors.max())
    # bins = np.linspace(min_err, max_err, 50) # 分 50 个 bins
    #
    # plt.hist(model_errors, bins=bins, alpha=0.6, label=f'模型误差 (Std={metrics["模型"]["Error Std"]:.4f})', density=True)
    # plt.hist(nlf_errors, bins=bins, alpha=0.6, label=f'S-G+NLF 误差 (Std={metrics["S-G+NLF"]["Error Std"]:.4f})', density=True)
    # plt.xlabel("预测误差 (μs)")
    # plt.ylabel("概率密度")
    # plt.title("预测误差分布")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.savefig(os.path.join(output_dir, "error_distribution_histogram.png"), dpi=300)
    # plt.close()
    #
    # # (c) 性能指标表格 (可选，可以用 pandas 输出更美观的表格)
    # print("\n--- 性能总结 ---")
    # print(f"{'指标':<20} | {'模型':<20} | {'S-G+NLF':<20}")
    # print("-" * 65)
    # for key in ['RMSE', 'MAE', 'R2', 'Error Std', 'Pred Std', 'Time per sample (ms)', 'Success Rate (%)']:
    #      model_val = metrics["模型"].get(key, 'N/A')
    #      nlf_val = metrics["S-G+NLF"].get(key, 'N/A')
    #      val_format = ".4f" if isinstance(model_val, (float, np.float32, np.float64)) and key != 'R2' else ".4f" if key == 'R2' else ".2f" if key == 'Time per sample (ms)' or key == 'Success Rate (%)' else ""
    #      if key == 'R2': val_format = ".4f"
    #      if key == 'Time per sample (ms)': val_format = ".2f"
    #      if key == 'Success Rate (%)': val_format = ".2f"
    #
    #      model_str = f"{model_val:{val_format}}" if model_val != 'N/A' else "N/A"
    #      nlf_str = f"{nlf_val:{val_format}}" if nlf_val != 'N/A' else "N/A"
    #
    #      # 对齐字符串
    #      print(f"{key:<20} | {model_str:<20} | {nlf_str:<20}")
    #
    # print("-" * 65)
    # print(f"结果图表已保存至目录: {output_dir}")


if __name__ == "__main__":
    # --- 配置参数 ---
    model_path = ("phy_best_ringdown_model_10-20-0.005.pth")  # <--- 修改为你的最佳模型路径
    data_file = "test.npz"              # <--- 修改为你的测试数据文件路径
    output_directory = "comparison_results_output" # <--- 输出结果目录名
    # S-G 滤波器参数 (根据需要调整)
    sg_filter_window = 21 # 必须是奇数
    sg_filter_polyorder = 3

    # --- 执行评估与对比 ---
    evaluate_and_compare(model_path, data_file, output_dir=output_directory,
                         sg_window=sg_filter_window, sg_polyorder=sg_filter_polyorder)