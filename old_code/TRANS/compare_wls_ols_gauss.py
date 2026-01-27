"""
文件名: compare_wls_ols_gauss.py
用途: 比较普通最小二乘法(OLS)、加权最小二乘法(WLS)和高斯拟合等不同拟合方法的性能
功能描述:
    - 实现三种拟合方法(OLS、WLS、高斯拟合)
    - 对指数衰减信号进行拟合
    - 比较不同方法的拟合精度和效果
    - 生成可视化对比分析报告
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit, least_squares
from tqdm import tqdm
from pathlib import Path

# --- Matplotlib 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- 1. 拟合模型与三种拟合方法 ---

def decay_model(t, I0, tau, B):
    """指数衰减模型"""
    return I0 * np.exp(-t / tau) + B


def fit_ols(signal, t_axis):
    """普通最小二乘法 (OLS)"""
    try:
        guess_B = np.mean(signal[-int(len(signal) * 0.1):])
        guess_I0 = signal[0] - guess_B
        p0 = [guess_I0, 140.0, guess_B]
        bounds = ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        params, _ = curve_fit(decay_model, t_axis, signal, p0=p0, bounds=bounds, maxfev=10000)
        return params
    except:
        return None


def fit_wls(signal, t_axis):
    """加权最小二乘法 (WLS)"""
    try:
        guess_B = np.mean(signal[-int(len(signal) * 0.1):])
        guess_I0 = signal[0] - guess_B
        p0 = [guess_I0, 140.0, guess_B]
        bounds = ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        # 权重基于信号强度，模拟散粒噪声的理想处理方式
        sigma_wls = np.sqrt(np.maximum(signal, 1e-9))
        params, _ = curve_fit(decay_model, t_axis, signal, p0=p0, bounds=bounds, sigma=sigma_wls, absolute_sigma=True,
                              maxfev=10000)
        return params
    except:
        return None


def fit_robust(signal, t_axis):
    """稳健回归 (Robust, 使用'soft_l1'损失)"""
    try:
        def residual_func(params, t, data):
            return decay_model(t, *params) - data

        guess_B, guess_I0 = np.mean(signal[-int(len(signal) * 0.1):])
        p0, bounds = [guess_I0, 140.0, guess_B], ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        result = least_squares(residual_func, p0, args=(t_axis, signal), bounds=np.transpose(bounds), loss='soft_l1',
                               f_scale=0.01)
        return result.x if result.success else None
    except:
        return None


# --- 2. 蒙特卡洛仿真主流程 ---

def run_monte_carlo_comparison(p_true, sim_params):
    """执行蒙特卡洛仿真"""
    I0_true, tau_true, B_true = p_true
    num_trials, seq_len, t_end, noise_sigma = sim_params.values()

    t_axis = np.linspace(0, t_end, seq_len)
    ideal_signal = decay_model(t_axis, I0_true, tau_true, B_true)

    results = {'OLS': [], 'WLS': [], 'Robust': []}

    for _ in tqdm(range(num_trials), desc="执行蒙特卡洛仿真"):
        # 生成理想高斯白噪声
        noise = np.random.normal(0, noise_sigma, seq_len)
        noisy_signal = ideal_signal + noise

        # 使用三种方法进行拟合
        params_ols = fit_ols(noisy_signal, t_axis)
        params_wls = fit_wls(noisy_signal, t_axis)
        params_robust = fit_robust(noisy_signal, t_axis)

        if params_ols is not None: results['OLS'].append(params_ols[1])  # 只记录tau值
        if params_wls is not None: results['WLS'].append(params_wls[1])
        if params_robust is not None: results['Robust'].append(params_robust[1])

    return results


# --- 3. 结果分析与可视化 ---

def analyze_and_plot_results(results, tau_true, output_folder):
    """分析仿真结果并绘图"""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    summary_data = []

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    for method, taus in results.items():
        if not taus: continue

        taus = np.array(taus)
        mean_tau = np.mean(taus)
        std_tau = np.std(taus)
        bias = mean_tau - tau_true
        rmse = np.sqrt(np.mean((taus - tau_true) ** 2))

        summary_data.append({
            'Method': method,
            'Mean Tau': mean_tau,
            'Std Dev (Precision)': std_tau,
            'Bias (Accuracy)': bias,
            'RMSE': rmse,
            'Fit Success Rate (%)': 100 * len(taus) / sim_params['num_trials']
        })

        # 绘制结果的概率密度分布图
        sns.kdeplot(taus, label=f'{method} (std={std_tau:.4f})', fill=True, alpha=0.2)

    # 绘制真实值垂线
    plt.axvline(tau_true, color='black', linestyle='--', label=f'真实 Tau = {tau_true}')

    plt.title('不同拟合方法对 Tau 值的估计分布', fontsize=16)
    plt.xlabel('拟合得到的 Tau (τ) 值')
    plt.ylabel('概率密度')
    plt.legend()

    plot_path = output_folder / "fitter_comparison_distribution.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n拟合结果分布图已保存至: {plot_path}")

    # 打印统计报告
    summary_df = pd.DataFrame(summary_data).set_index('Method')
    print("\n--- 拟合方法性能对比报告 ---")
    print(summary_df.to_string(float_format="%.5f"))

    csv_path = output_folder / "fitter_comparison_report.csv"
    summary_df.to_csv(csv_path, float_format='%.5f')
    print(f"\n详细报告已保存至: {csv_path}")


# --- 4. 主程序入口 ---
if __name__ == '__main__':
    # --- 1. 定义“真实世界”和仿真参数 ---

    # "上帝"知道的真实参数
    TRUE_PARAMS = {
        'I0_true': 1.0,
        'tau_true': 140.0,
        'B_true': 0.1
    }

    # 仿真环境参数
    sim_params = {
        'num_trials': 100,  # 仿真次数，越高结果越可靠
        'sequence_length': 15000,  # 信号长度
        't_end': 1500,  # 信号总时长
        'noise_sigma': 0.003  # 高斯白噪声的标准差
    }

    OUTPUT_FOLDER = "fitter_comparison_gaussian_noise"

    # --- 2. 执行仿真和分析 ---

    # 执行蒙特卡洛仿真
    simulation_results = run_monte_carlo_comparison(TRUE_PARAMS.values(), sim_params)

    # 分析并报告结果
    analyze_and_plot_results(simulation_results, TRUE_PARAMS['tau_true'], OUTPUT_FOLDER)