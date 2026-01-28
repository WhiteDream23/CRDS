"""
文件名: noise0711_compare.py
output : processednew
用途: 对不同拟合方法的结果进行详细的噪声分析和质量控制
功能描述:
    - 实现多种拟合方法(OLS、WLS、高斯等)
    - 对拟合残差进行噪声分布分析
    - 实施质量控制(QC)检查,包括方差线性性检验
    - 生成详细的对比分析报告和可视化结果
    - 支持自然排序和复杂的数据筛选
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit,least_squares
from scipy.stats import linregress
from pathlib import Path
from tqdm import tqdm
import re
from collections import defaultdict
import matplotlib.ticker as ticker 
import matplotlib.ticker as mticker # 导入用于精细控制坐标轴的模块
from matplotlib import gridspec
import warnings

# --- Matplotlib 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- 1. 拟合模型与辅助函数 ---
def decay_model(t, I0, tau, B):
    return I0 * np.exp(-t / tau) + B

# 修正后函数
def natural_sort_key(s):
    # 核心改动：将返回的列表[]转换为元组()
    return tuple(int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s)))


# --- 2. 质量控制(QC)函数 ---
def check_linearity_of_variance(fitted_curve, residuals, r_squared_threshold=0.8, nums_box=11):
    try:
        binner = np.linspace(np.min(fitted_curve), np.max(fitted_curve), num=nums_box + 1)
        bin_centers, bin_variances = [], []
        for i in range(len(binner) - 1):
            indices = np.where((fitted_curve >= binner[i]) & (fitted_curve < binner[i + 1]))[0]
            if len(indices) > 10:
                # bin_centers.append(np.mean(fitted_curve[indices]))
                # bin_variances.append(np.var(residuals[indices], ddof=1))
                bin_centers.append((binner[i] + binner[i + 1]) / 2)
                bin_variances.append(np.var(residuals[indices]))
        if len(bin_centers) < 5: return False
        slope, _, r_value, _, _ = linregress(bin_centers, bin_variances)
        return slope > 0 and (r_value ** 2) >= r_squared_threshold
    except:
        return False

#返回了R2值
def check_linearity_of_variance2(fitted_curve, residuals, r_squared_threshold=0.8, nums_box=11):
    try:
        binner = np.linspace(np.min(fitted_curve), np.max(fitted_curve), num=nums_box + 1)
        bin_centers, bin_variances = [], []
        for i in range(len(binner) - 1):
            indices = np.where((fitted_curve >= binner[i]) & (fitted_curve < binner[i + 1]))[0]
            if len(indices) > 10:
                # bin_centers.append(np.mean(fitted_curve[indices]))
                # bin_variances.append(np.var(residuals[indices], ddof=1))
                bin_centers.append((binner[i] + binner[i + 1]) / 2)
                bin_variances.append(np.var(residuals[indices]))
        if len(bin_centers) < 5: return False
        slope, _, r_value, _, _ = linregress(bin_centers, bin_variances)
        return slope > 0 and (r_value ** 2) >= r_squared_threshold,r_value ** 2
    except:
        return False, 0.0


# --- 3. 核心拟合方法 (返回全参数) ---
def fit_wls(signal, t_axis):
    try:
        guess_B, guess_I0 = np.mean(signal[-int(len(signal) * 0.1):]), signal[0] - np.mean(
            signal[-int(len(signal) * 0.1):])
        p0, bounds = [guess_I0, 170.0, guess_B], ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        sigma_wls = np.sqrt(np.maximum(signal, 1e-9))
        sigma_wls = np.sqrt(np.log1p(np.exp(signal)))

        # sigma_wls = np.sqrt(np.maximum(signal + 0.1, 1e-9))
        params, pcov = curve_fit(decay_model, t_axis, signal, p0=p0, bounds=bounds, sigma=sigma_wls, absolute_sigma=False,
                        maxfev=10000)
        perr=np.sqrt(np.diag(pcov))
        tau_std=perr[1]
        return params, tau_std
    except:
        return None



def fit_ols(signal, t_axis):
    try:
        guess_B, guess_I0 = np.mean(signal[-int(len(signal) * 0.1):]), signal[0] - np.mean(
            signal[-int(len(signal) * 0.1):])
        p0, bounds = [guess_I0, 170.0, guess_B], ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        params, pcov = curve_fit(decay_model, t_axis, signal, p0=p0, bounds=bounds, maxfev=10000)
        perr=np.sqrt(np.diag(pcov))
        tau_std=perr[1]
        return params, tau_std
    except:
        return None

def fit_robust(signal, t_axis):
    """使用 'soft_l1' 损失函数（类似L1范数）进行稳健回归。"""
    try:
        # 定义残差函数，供 least_squares 使用
        def residual_func(params, t, data):
            return decay_model(t, *params) - data

        # 初始猜测和边界
        guess_B, guess_I0 = np.mean(signal[-int(len(signal) * 0.1):]), signal[0] - np.mean(
            signal[-int(len(signal) * 0.1):])
        p0, bounds = [guess_I0, 170.0, guess_B], ([0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])

        # 调用 least_squares 并指定损失函数为 'soft_l1'
        result = least_squares(residual_func, p0, args=(t_axis, signal), bounds=bounds, loss='soft_l1',
                               f_scale=0.003)

        if result.success:
            return result.x  # 返回优化后的参数 [I0, tau, B]
        else:
            return None
    except (RuntimeError, ValueError, TypeError):
        return None


# --- 4. 绘图函数 (无变化) ---
def plot_intra_condition_comparison(df, condition_name, output_path, analysis_type, methods):
    """
    【修改版】为单一处理条件绘制方法对比图，优化了线条可见性。
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), sharex=True)
    fig.suptitle(f"【方法对比】在条件 '{condition_name}' 下的标准差（对数坐标）", fontsize=18)

    file_indices = df.index

    # --- 为不同方法定义独特的样式 ---
    # 使用 字典(dict) 来管理样式，让代码更清晰
    styles = {
        'WLS': {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'zorder': 3},
        'OLS': {'color': 'green', 'marker': 's', 'linestyle': '--', 'zorder': 2},
        'robust': {'color': 'red', 'marker': '^', 'linestyle': ':', 'zorder': 1},
        'WLS_QC': {'color': 'cyan', 'marker': 'o', 'linestyle': '-', 'zorder': 3, 'fillstyle': 'none'},
        'OLS_QC': {'color': 'lime', 'marker': 's', 'linestyle': '--', 'zorder': 2, 'fillstyle': 'none'},
        'robust_QC': {'color': 'magenta', 'marker': '^', 'linestyle': ':', 'zorder': 1, 'fillstyle': 'none'}
    }

    # --- 上图：所有信号的拟合结果 ---
    for method in methods[:3]:
        # 使用 **styles[method] 将字典中的所有样式一次性应用
        ax1.plot(file_indices, df[f'tau_std_{method}'],
                 label=method,
                 markersize=5,
                 alpha=0.8,  # 增加透明度
                 **styles[method])

    ax1.set_title("所有信号的拟合结果")
    ax1.set_ylabel("Tau (τ) 标准差")
    ax1.legend(title="拟合方法")
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)  # 主次网格都显示
    # ★ 核心改动：使用对数坐标轴
    ax1.set_yscale('log')
    # 为对数坐标轴设置更易读的刻度格式
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.yaxis.get_major_formatter().set_scientific(False)
    ax1.yaxis.set_minor_formatter(mticker.NullFormatter())

    # --- 下图：仅通过QC的信号的拟合结果 ---
    for method in methods[2:]:
        ax2.plot(file_indices, df[f'tau_std_{method}'],
                 label=method,
                 markersize=6,
                 alpha=0.8,  # 增加透明度
                 **styles[method])

    ax2.set_title("仅通过独立质量控制(QC)的信号的拟合结果")
    ax2.set_xlabel("文件序号 (File Index)", fontsize=12)
    ax2.set_ylabel("Tau (τ) 标准差（QC）")
    ax2.legend(title="拟合方法 (QC)")
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    # ★ 核心改动：使用对数坐标轴
    ax2.set_yscale('log')
    # 为对数坐标轴设置更易读的刻度格式
    ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.yaxis.get_major_formatter().set_scientific(False)
    ax2.yaxis.set_minor_formatter(mticker.NullFormatter())

    # --- 优化X轴刻度 ---
    if len(file_indices) > 20:
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=20))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')  # 旋转刻度防止重叠

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path / f"{analysis_type}_std_comparison_{condition_name}.png", dpi=150)
    plt.close(fig)


def plot_inter_condition_stability(plot_data, output_path, analysis_config, methods):
    fig, ax = plt.subplots(figsize=(12, 8))
    styles = ['-o', '-s', '-^', '--x', '--P', '--D']
    for i, name in enumerate(methods):
        ax.plot(plot_data[name]['x'], plot_data[name]['mean_of_stds'], styles[i], label=name)
    ax.set_xlabel(analysis_config['x_axis_label'], fontsize=14);
    ax.set_ylabel("Tau (τ) 标准差的平均值 (衡量稳定性)", fontsize=14)
    ax.set_title(f"【稳定性分析】不同{analysis_config['x_axis_label']}对稳定性的影响", fontsize=16);
    ax.legend(title="方法 (---QC)", fontsize=12);
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if len(plot_data['WLS']['x']) > 0: plt.xticks(plot_data['WLS']['x'])
    plt.tight_layout();
    plt.savefig(output_path / f"{analysis_config['type']}_stability_comparison.png", dpi=300);
    plt.close(fig)



# --- 5. 主分析流程 (最终改造版) ---
def run_analysis_suite_final(base_data_path, output_path, analysis_config):
    # 从配置中解包参数
    analysis_type = analysis_config['type']
    folder_prefix = analysis_config['folder_prefix']
    r2_threshold = analysis_config['r2_threshold']
    time_per_point = analysis_config['time_per_point']
    base_len_for_freq = analysis_config.get('base_len_for_freq', None)

    output_path = Path(output_path) / f"{analysis_type}_final"
    output_path.mkdir(parents=True, exist_ok=True)

    data_folders = sorted(
        [p for p in Path(base_data_path).iterdir() if p.is_dir() and p.name.startswith(folder_prefix)],
        key=natural_sort_key)
    if not data_folders: print(f"错误: 未找到 '{folder_prefix}*' 文件夹。"); return

    print(f"\n--- 开始 '{analysis_type.upper()}' 分析 (独立QC, 动态 t_end) ---")

    master_summary_list = []
    stability_plot_data = defaultdict(lambda: {'x': [], 'mean_of_stds': []})
    methods = ['WLS', 'OLS',  'WLS_QC', 'OLS_QC','robust', 'robust_QC']
    # methods = ['OLS', 'OLS_QC','robust', 'robust_QC']

    for folder in tqdm(data_folders, desc=f"分析 {analysis_type} 文件夹"):
        condition_value = int(re.findall(r'\d+', folder.name)[-1])
        # condition_value = 8000
        # --- 核心改动：动态计算 T_END ---
        if analysis_type == 'length':
            current_t_end = condition_value * time_per_point
        elif analysis_type == 'frequency':
            current_t_end = base_len_for_freq * time_per_point
        # --------------------------------

        npz_files = sorted(list(folder.glob("*.npz")), key=natural_sort_key)
        if not npz_files: continue

        folder_results, agg_stds = [], defaultdict(list)
        for npz_file in tqdm(npz_files,'处理NPZ文件', leave=False):
            try:
                # 从文件名中提取数字, 例如 "processed_data_batch_50.npz" -> 50
                # [-1]确保我们取的是文件名中最后一个数字，通常是批次号
                file_number = int(re.findall(r'\d+', npz_file.name)[-1])

                # 检查文件号是否在要跳过的范围内 [50, 51, 52, 53]
                # if 50 <= file_number <= 53:
                #     # 如果是，打印一条提示信息并用 continue 跳过本次循环
                #     tqdm.write(f"跳过文件: {npz_file.name}")
                #     continue
            except (IndexError, ValueError):
                # 如果文件名中没有数字或无法转换，则正常处理，不跳过
                pass
            with np.load(npz_file) as ld:
                signals = ld.get('data', ld.get('arr_0'))
            if signals.ndim == 1: signals = [signals]

            # 使用动态计算出的 t_end 来创建时间轴
            t_axis = np.linspace(0, current_t_end, signals.shape[-1])
            # t_axis = np.linspace(0, current_t_end, condition_value)
            taus_per_file = defaultdict(list)
            for signal in signals:
                signal = signal[:condition_value]
                qc_passed_wls, qc_passed_ols = False, False
                qc_passed_robust = False
                tau_wls, tau_ols =None, None
                tau_robust = None

                ols_params,_ = fit_ols(signal, t_axis)
                if ols_params is not None:
                    tau_ols = ols_params[1]
                    qc_passed_ols = check_linearity_of_variance(decay_model(t_axis, *ols_params),
                                                                signal - decay_model(t_axis, *ols_params), r2_threshold)

                wls_params,_ = fit_wls(signal, t_axis)
                if wls_params is not None:
                    tau_wls = wls_params[1]
                    qc_passed_wls = check_linearity_of_variance(decay_model(t_axis, *ols_params),
                                                                signal - decay_model(t_axis, *ols_params), r2_threshold)

                robust_params = fit_robust(signal, t_axis)
                if robust_params is not None:
                    tau_robust = robust_params[1]
                    qc_passed_robust = check_linearity_of_variance(decay_model(t_axis, *ols_params),
                                                                   signal - decay_model(t_axis, *ols_params), r2_threshold)
                if tau_wls: taus_per_file['WLS'].append(tau_wls)
                if tau_ols: taus_per_file['OLS'].append(tau_ols)
                if tau_robust: taus_per_file['robust'].append(tau_robust)

                if qc_passed_wls and tau_wls: taus_per_file['WLS_QC'].append(tau_wls)
                if qc_passed_ols and tau_ols: taus_per_file['OLS_QC'].append(tau_ols)
                if qc_passed_robust and tau_robust: taus_per_file['robust_QC'].append(tau_robust)

            file_summary = {'filename': npz_file.name, 'condition': condition_value}
            for name in methods:
                taus = taus_per_file[name]
                mean_tau, std_tau = (np.mean(taus), np.std(taus)) if len(taus) > 1 else (
                np.mean(taus) if taus else np.nan, 0)
                file_summary[f'tau_mean_{name}'], file_summary[f'tau_std_{name}'] = mean_tau, std_tau
                if not np.isnan(std_tau): agg_stds[name].append(std_tau)

            folder_results.append(file_summary)

        if folder_results:
            folder_df = pd.DataFrame(folder_results).set_index('filename')
            plot_intra_condition_comparison(folder_df, folder.name, output_path, analysis_type, methods)

        for name in methods:
            stability_plot_data[name]['x'].append(condition_value)
            stability_plot_data[name]['mean_of_stds'].append(np.mean(agg_stds[name]) if agg_stds[name] else np.nan)

        master_summary_list.extend(folder_results)

    plot_inter_condition_stability(stability_plot_data, output_path, analysis_config, methods)
    if master_summary_list:
        # 修正后代码
        # 1. 先创建DataFrame，不进行排序
        report_df = pd.DataFrame(master_summary_list)

        # 2. 为 'filename' 列创建一个临时的自然排序键列
        report_df['filename_sort_key'] = report_df['filename'].apply(natural_sort_key)

        # 3. 根据 'condition' 和新的排序键列进行排序，然后删除临时列
        report_df = report_df.sort_values(by=['condition', 'filename_sort_key']).drop(columns=['filename_sort_key'])

        report_path = output_path / f"{analysis_type}_main_report_final.csv"
        report_df.to_csv(report_path, index=False, float_format='%.6f', encoding='utf-8-sig')
        tqdm.write(f"'{analysis_type}' 的主报告和所有图像已保存至: {output_path}")

def calculate_r_squared(y_true, y_pred):
    """计算 R^2 (拟合优度)"""
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def plot_qc_verification_combined_finalimg(t_axis, signal, params, method_name, qc_passed, r2_val, output_path, filename, signal_idx):
    """
    【新增】生成左右并排的组合图 (Combined Plot)：
    左图: 信号拟合曲线 + 残差 (上下子图)
    右图: 噪声方差 vs 信号强度
    特点: 字体清晰，布局紧凑，适合期刊发表
    """
    fitted_curve = decay_model(t_axis, *params)
    residuals = signal - fitted_curve
    
    # 计算方差-强度关系
    nums_box = 11
    binner = np.linspace(np.min(fitted_curve), np.max(fitted_curve), num=nums_box + 1)
    bin_centers, bin_variances = [], []
    for i in range(len(binner) - 1):
        indices = np.where((fitted_curve >= binner[i]) & (fitted_curve < binner[i + 1]))[0]
        if len(indices) > 10:
            bin_centers.append((binner[i] + binner[i + 1]) / 2)
            bin_variances.append(np.var(residuals[indices]))
    
    # 线性拟合方差图
    slope, intercept, r_val_var, _, _ = linregress(bin_centers, bin_variances) if len(bin_centers) > 2 else (0,0,0,0,0)
    var_r2 = r_val_var**2

    # --- 1. 设置绘图风格 ---
    with plt.rc_context({
        'font.family': 'serif',
        'font.sans-serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': False,
        'ytick.right': False,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'axes.labelsize': 16,
        'legend.fontsize': 18
    }):
        
        # 确定保存路径
        sub_folder = "QC_Passed" if qc_passed else "QC_Failed"
        if r2_val > 0.99 and not qc_passed: sub_folder = "QC_Deceptive_HighR2_Fail"
        save_dir = output_path / "QC_Verification_Plots" / sub_folder
        save_dir.mkdir(parents=True, exist_ok=True)

        # 创建宽幅画布: 16x7 英寸
        fig = plt.figure(figsize=(16, 6))
        
        # 使用 GridSpec 定义布局: 1行2列，左列再分为上下两行(3:1)
        gs_main = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.02, left=0.05, right=0.98, top=0.92, bottom=0.1)
        gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0], height_ratios=[3, 1], hspace=0.05)
        
        # ==========================================
        # 左侧部分 (a): 信号拟合与残差
        # ==========================================
        
        # 左上: 拟合曲线
        ax1 = fig.add_subplot(gs_left[0])
        ax1.plot(t_axis*10, signal, 'o', color='gray', markersize=4, alpha=0.4, label='Raw Data', markeredgewidth=0)
        ax1.plot(t_axis*10, fitted_curve, color='#D62728', linewidth=2.5, label='Exponential Fit')
        
        # 标记信息 (左图)
        info_text_1 = (
            f"QC Result: {'PASS' if qc_passed else 'FAIL'}\n"
            f"$R^2_{{signal}}$ = {r2_val:.5f}\n"
            f"$\\tau$ = {params[1]:.4f} $\\mu s$"
        )
        # 信息框位置：中间偏上
        ax1.text(0.4, 0.85, info_text_1, transform=ax1.transAxes, 
                 ha='center', va='top', fontsize=20, fontweight='normal',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.5))
        
        # 标记 (a) - 右上角
        # ax1.text(0.96, 0.96, '(a)', transform=ax1.transAxes, 
        #          ha='right', va='top', fontsize=20, fontweight='bold')
        # ax1.legend(loc='upper right', bbox_to_anchor=(1.02, 0.88), frameon=False)
        
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax1.set_ylabel("Signal Intensity (V)", fontsize=18)
        ax1.tick_params(labelbottom=False) 
        
        # 左下: 残差
        ax2 = fig.add_subplot(gs_left[1], sharex=ax1)
        ax2.plot(t_axis*10, residuals, '-', color='#1F77B4', linewidth=1.2, alpha=0.9)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        
        # 残差图设置 (对称和刻度)
        max_resid = np.max(np.abs(residuals))
        if max_resid == 0: max_resid = 1e-6
        limit = max_resid * 1.1
        ax2.set_ylim(-limit, limit)
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

        ax2.set_ylabel("Residuals", fontsize=18)
        ax2.set_xlabel("Data Points", fontsize=18)
        
# ==========================================
        # 右侧部分 (b): 噪声方差 vs 信号强度 (核心修改)
        # ==========================================
        ax3 = fig.add_subplot(gs_main[1])
        ax3.set_box_aspect(1) 

        # --- 修改开始：计算扩展范围以实现“与边框相交” ---
        if len(bin_centers) > 0:
            # 1. 获取数据的物理范围
            x_min_data, x_max_data = np.min(bin_centers), np.max(bin_centers)
            
            # 2. 设定 X 轴的显示范围 (Limits)，两端各扩展 10%
            # 这里的 x_start 和 x_end 将成为最终图表的 X 轴左右边界
            x_range = x_max_data - x_min_data
            if x_range == 0: x_range = 1.0
            
            x_start = x_min_data - x_range * 0.15
            x_end = x_max_data + x_range * 0.15
            
            # 3. 基于 X 轴的显示范围，计算回归线的起点和终点
            # 这样画出来的线就会精确地顶到左右边框
            line_x = np.array([x_start, x_end])
            line_y = line_x * slope + intercept
            
            # 4. 计算 Y 轴的合适范围
            # 必须同时容纳：散点数据 和 回归线在边缘的延伸值
            y_values_all = np.concatenate((bin_variances, line_y))
            y_min_val, y_max_val = np.min(y_values_all), np.max(y_values_all)
            
            y_range = y_max_val - y_min_val
            if y_range == 0: y_range = 1e-9
            
            # Y轴也扩展一点，避免点贴在边框上
            y_start = y_min_val - y_range * 0.1
            y_end = y_max_val + y_range * 0.1
            
            # (可选) 如果方差理论上不能小于0，且数据均大于0，可以锁底
            # if y_start < 0 and np.min(bin_variances) >= 0: y_start = 0 
            
        else:
            # 默认空值情况
            x_start, x_end = 0, 1
            y_start, y_end = 0, 1
            line_x = np.array([0, 1])
            line_y = np.array([0, 0])
        
        # --- 绘图 ---
        # 绘制散点
        ax3.scatter(bin_centers, bin_variances, c='black', s=80, marker='o', label='Binned Variance', zorder=3, edgecolors='white')
        
        # 绘制回归线 (使用计算出的延伸坐标)
        ax3.plot(line_x, line_y, linestyle='--', color='#D62728', linewidth=2.5, label='Linear Fit', zorder=2)
        
        # --- 关键：强制设置坐标轴范围 ---
        # 这一步确保了图表的边框正好压在 line_x 的起点和终点上
        ax3.set_xlim(x_start, x_end)
        ax3.set_ylim(y_start, y_end)

        # 标记信息
        info_text_2 = (
            f"QC Result: {'PASS' if qc_passed else 'FAIL'}\n"
            f"$R^2_{{linear}}$ = {var_r2:.4f}"
        )
        
        # 智能放置位置 (根据斜率决定放在左上还是右下)
        if slope < 0:
            loc_pos_kwargs = dict(x=0.04, y=0.96, ha='left', va='top')
        else:
            loc_pos_kwargs = dict(x=0.96, y=0.04, ha='right', va='bottom')

        ax3.text(transform=ax3.transAxes, fontsize=20, fontweight='normal',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.5),
                 s=info_text_2, **loc_pos_kwargs)

        # 设置刻度格式
        ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax3.set_ylabel("Noise Variance ($\\sigma^2$)", fontsize=18)
        ax3.set_xlabel("Signal Intensity (V)", fontsize=18)
        
        # 科学计数法
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        ax3.yaxis.get_offset_text().set_fontsize(14)
        ax3.grid(True, linestyle=':', alpha=0.4)
        
        # --- 图例与标签 ---
        fig.text(0.52, 0.91, '(a)', ha='right', va='top', fontsize=24, fontweight='bold')
        fig.text(0.65, 0.91, '(b)', ha='right', va='top', fontsize=24, fontweight='bold')
        
        # 统一图例位置
        handles1, labels1 = ax1.get_legend_handles_labels()
        fig.legend(handles1, labels1, loc='upper right', bbox_to_anchor=(0.54, 0.88), frameon=False)
        
        handles3, labels3 = ax3.get_legend_handles_labels()
        fig.legend(handles3, labels3, loc='upper right', bbox_to_anchor=(0.80, 0.88), frameon=False)

        # ==========================================
        # 保存
        # ==========================================
        sub_folder = "QC_Passed" if qc_passed else "QC_Failed"
        if r2_val > 0.99 and not qc_passed: sub_folder = "QC_Deceptive_HighR2_Fail"
        save_dir = output_path / "QC_Verification_Plots_Combined_final" / sub_folder
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_dir / f"{filename}_idx{signal_idx}_Combined.png", format='png', dpi=300, bbox_inches='tight')
        # plt.savefig(save_dir / f"{filename}_idx{signal_idx}_Combined.pdf", format='pdf', bbox_inches='tight')
        plt.close(fig)

def plot_r2_distribution(r2_values, output_path, filename, condition_value,binnum):
    """
    【新增】绘制单个文件中所有信号的 R^2 分布直方图
    """
    if not r2_values:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制直方图
    n, bins, patches = ax.hist(r2_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 添加统计信息
    mean_r2 = np.mean(r2_values)
    median_r2 = np.median(r2_values)
    std_r2 = np.std(r2_values)
    min_r2 = np.min(r2_values)
    
    stats_text = (
        f"Count: {len(r2_values)}\n"
        f"Mean: {mean_r2:.6f}\n"
        f"Median: {median_r2:.6f}\n"
        f"Std: {std_r2:.6f}\n"
        f"Min: {min_r2:.6f}"
    )
    
    # 将统计信息放在图中
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f"OLS $R^2$ Distribution - {filename} (Len={condition_value})", fontsize=14)
    ax.set_xlabel("$R^2$ Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 使用对数纵坐标，因为通常绝大多数R2都很高，低R2很少
    # ax.set_yscale('log') 

    plt.tight_layout()
    
    # 保存路径
    save_dir = output_path / f"R2_Distributions_{binnum}"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"R2_Dist_{filename}.png", dpi=150)
    plt.close(fig)


def analyze_removal_effect_simple(data, removal_ratios=None, num_trials=100, plot=True, output_path=None,filename=None):
    """
    简单分析：测试不同随机剔除比例下的标准差变化

    参数:
    -----------
    data : array-like
        你的tau值数组
    removal_ratios : array-like, optional
        剔除比例列表，如 [0.1, 0.2, 0.3, 0.4, 0.5]
        默认: np.linspace(0, 0.5, 11)  即 0%到50%，步长5%
    num_trials : int
        每个比例的重复次数，默认100
    plot : bool
        是否画图，默认True
    output_path : str or Path, optional
        图片保存路径

    返回:
    --------
    dict : {
        'n_total': 数据总数,
        'std_original': 原始标准差,
        'removal_ratios': 剔除比例数组,
        'n_removed': 剔除数量数组,
        'n_remaining': 剩余数量数组,
        'std_after': 剔除后的标准差数组,
        'std_change': 标准差变化率数组
    }
    """
    data = np.array(data)
    n_total = len(data)
    std_original = np.std(data)

    # 设置默认剔除比例
    if removal_ratios is None:
        removal_ratios = np.linspace(0, 0.9, 10)  # 0%, 5%, 10%, ..., 50%

    removal_ratios = np.array(removal_ratios)

    results = {
        'n_total': n_total,
        'std_original': std_original,
        'removal_ratios': [],
        'n_removed': [],
        'n_remaining': [],
        'std_after': [],
        'std_change': []
    }

    for ratio in removal_ratios:
        n_removed = int(n_total * ratio)
        n_remaining = n_total - n_removed

        if n_remaining < 2:
            continue

        # 复用现有的 simulate_random_removal_stability 函数
        std_after = simulate_random_removal_stability(data, n_keep=n_remaining, num_trials=num_trials)

        if not np.isnan(std_after):
            results['removal_ratios'].append(ratio)
            results['n_removed'].append(n_removed)
            results['n_remaining'].append(n_remaining)
            results['std_after'].append(std_after)
            results['std_change'].append((std_after - std_original) / std_original * 100)

    # 转为numpy数组
    for key in ['removal_ratios', 'n_removed', 'n_remaining', 'std_after', 'std_change']:
        results[key] = np.array(results[key])

    if plot:
        _plot_removal_effect_simple(results, output_path,filename,num_trials)

    return results


def _plot_removal_effect_simple(results, output_path=None,filename=None,num=None):
    """绘制简单的剔除效果分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ratios_pct = results['removal_ratios'] * 100
    std_after = results['std_after']
    std_original = results['std_original']
    std_change = results['std_change']

    # 子图1: 标准差随剔除比例变化
    ax1 = axes[0]
    ax1.plot(ratios_pct, std_after, 'bo-', linewidth=2, markersize=6, label='随机剔除后的std')
    ax1.axhline(y=std_original, color='r', linestyle='--', linewidth=2, label=f'原始std={std_original:.4f}')
    ax1.set_xlabel('剔除比例 (%)', fontsize=12)
    ax1.set_ylabel('标准差', fontsize=12)
    ax1.set_title('随机剔除对标准差的影响', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 子图2: 标准差变化率
    ax2 = axes[1]
    ax2.plot(ratios_pct, std_change, 'gs-', linewidth=2, markersize=6)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('剔除比例 (%)', fontsize=12)
    ax2.set_ylabel('标准差变化率 (%)', fontsize=12)
    ax2.set_title('标准差相对变化', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        save_dir = output_path / "R2_Distributions"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"removal_effect_simple_{filename}_{num}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    else:
        plt.show()

    plt.close()




#函数添加了有关R2的分析，没有包含随机策略的分析
def run_analysis_suite_final_2(base_data_path, output_path, analysis_config):
    # 从配置中解包参数
    analysis_type = analysis_config['type']
    folder_prefix = analysis_config['folder_prefix']
    r2_threshold = analysis_config['r2_threshold']
    time_per_point = analysis_config['time_per_point']
    base_len_for_freq = analysis_config.get('base_len_for_freq', None)

    output_path = Path(output_path) / f"{analysis_type}_final"
    output_path.mkdir(parents=True, exist_ok=True)

    data_folders = sorted(
        [p for p in Path(base_data_path).iterdir() if p.is_dir() and p.name.startswith(folder_prefix)],
        key=natural_sort_key)
    if not data_folders: print(f"错误: 未找到 '{folder_prefix}*' 文件夹。"); return

    print(f"\n--- 开始 '{analysis_type.upper()}' 分析 (独立QC, 动态 t_end) ---")

    # --- 新增：用于限制绘图数量的计数器 ---
    qc_plot_counters = {
        'pass': 0,
        'fail': 0,
        'deceptive': 0  # 高R2但QC失败
    }
    MAX_PLOTS_PER_TYPE = 30  # 每种类型最多画几张图
    all_r2_values = []
    NUM_BOX=31
    for folder in tqdm(data_folders, desc=f"分析 {analysis_type} 文件夹"):
        condition_value = int(re.findall(r'\d+', folder.name)[-1])
        # condition_value = 8000
        # --- 核心改动：动态计算 T_END ---
        if analysis_type == 'length':
            current_t_end = condition_value * time_per_point
        elif analysis_type == 'frequency':
            current_t_end = base_len_for_freq * time_per_point
        # --------------------------------

        npz_files = sorted(list(folder.glob("*.npz")), key=natural_sort_key)
        if not npz_files: continue

        for npz_file in tqdm(npz_files,'处理NPZ文件', leave=False):
            try:
                # 从文件名中提取数字, 例如 "processed_data_batch_50.npz" -> 50
                # [-1]确保我们取的是文件名中最后一个数字，通常是批次号
                file_number = int(re.findall(r'\d+', npz_file.name)[-1])

                # 检查文件号是否在要跳过的范围内 [50, 51, 52, 53]
                # if 50 <= file_number <= 53:
                #     # 如果是，打印一条提示信息并用 continue 跳过本次循环
                #     tqdm.write(f"跳过文件: {npz_file.name}")
                #     continue
            except (IndexError, ValueError):
                # 如果文件名中没有数字或无法转换，则正常处理，不跳过
                pass
            with np.load(npz_file) as ld:
                signals = ld.get('data', ld.get('arr_0'))
            if signals.ndim == 1: signals = [signals]

            # 使用动态计算出的 t_end 来创建时间轴
            t_axis = np.linspace(0, current_t_end, signals.shape[-1])
            # t_axis = np.linspace(0, current_t_end, condition_value)
            taus_per_file = defaultdict(list)
            qc_plot_counters['deceptive']=0
            qc_plot_counters['pass']=0
            qc_plot_counters['fail']=0
            # 初始化r2列表
            r2_list_ols = []
            for idx, signal in enumerate(signals): # 增加 enumerate 获取索引
                signal = signal[:condition_value]
                
                # 1. OLS 拟合与检查
                ols_params,ols_err = fit_ols(signal, t_axis)
                if ols_params is not None:
                    fitted_ols = decay_model(t_axis, *ols_params)
                    residuals_ols = signal - fitted_ols
                    
                    # 计算 R^2
                    r2_ols = calculate_r_squared(signal, fitted_ols)
                    
                    # 执行 QC
                    qc_passed_ols, r2_value_ols = check_linearity_of_variance2(fitted_ols, residuals_ols, r2_threshold,NUM_BOX)
                    r2_list_ols.append(r2_value_ols)

                    tau_ols = ols_params[1]
                    taus_per_file['OLS'].append(tau_ols)
                    all_r2_values.append(r2_value_ols)
                    

                    if qc_passed_ols: taus_per_file['OLS_QC'].append(tau_ols)

                    # --- 核心新增：绘图逻辑 (仅针对 OLS 示例，WLS同理) ---
                    # 只有当我们需要证明观点时才绘图
                    is_deceptive = (r2_ols > 0.995) and (not qc_passed_ols) # 极高R2但QC失败
                    
                    should_plot = False
                    if is_deceptive and qc_plot_counters['deceptive'] < MAX_PLOTS_PER_TYPE:
                        should_plot = True
                        qc_plot_counters['deceptive'] += 1
                    elif qc_passed_ols and qc_plot_counters['pass'] < MAX_PLOTS_PER_TYPE:
                        should_plot = True
                        qc_plot_counters['pass'] += 1
                    elif (not qc_passed_ols) and qc_plot_counters['fail'] < MAX_PLOTS_PER_TYPE:
                        should_plot = True
                        qc_plot_counters['fail'] += 1
                    
                    if should_plot:
                        pass
                        # plot_qc_verification_combined_finalimg(t_axis, signal, ols_params, 'OLS', qc_passed_ols, r2_ols, output_path, npz_file.name, idx)
            plot_r2_distribution(r2_list_ols, output_path, npz_file.name, condition_value,NUM_BOX)
            # analyze_removal_effect_simple(data=taus_per_file['OLS'],output_path= output_path,num_trials=1000,filename=npz_file.name)
         
        if all_r2_values:
            plot_r2_distribution(all_r2_values, output_path, "ALL_FILES_AGGREGATED", "ALL",NUM_BOX)
  



def simulate_random_removal_stability(all_taus, n_keep, num_trials=100):
    """
    模拟随机剔除：从原始数据中随机保留 n_keep 个数据，计算标准差。
    重复 num_trials 次取平均，以获得统计上稳定的结果。
    """
    if n_keep < 2 or len(all_taus) <= n_keep:
        return np.nan
    
    stds = []
    all_taus_np = np.array(all_taus)
    for _ in range(num_trials):
        # 无放回随机抽样
        random_subset = np.random.choice(all_taus_np, n_keep, replace=False)
        stds.append(np.std(random_subset))
    
    return np.mean(stds)

def plot_qc_vs_random_comparison(comparison_data, output_path, condition_name):
    """
    绘制 QC过滤 vs 随机剔除 的稳定性对比图。
    comparison_data: 字典，key为方法名(OLS/WLS), value为列表 [(std_qc, std_random), ...]
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    methods = comparison_data.keys()
    colors = {'OLS': 'green', 'WLS': 'blue', 'robust': 'red'}
    
    # 子图1: 散点对比图 (QC Std vs Random Std)
    ax1 = axes[0]
    for method in methods:
        data = np.array(comparison_data[method])
        if len(data) == 0: continue
        # x轴: 随机剔除后的Std, y轴: QC后的Std
        ax1.scatter(data[:, 1], data[:, 0], c=colors.get(method, 'black'), 
                    label=method, alpha=0.6, s=25)
    
    # 绘制 y=x 参考线
    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),
        np.max([ax1.get_xlim(), ax1.get_ylim()]),
    ]
    ax1.plot(lims, lims, 'k--', alpha=0.75, label='y=x (无特定优势)')
    
    ax1.set_xlabel('随机剔除相同数量后的标准差 (Std_Random)', fontsize=12)
    ax1.set_ylabel('质量控制(QC)后的标准差 (Std_QC)', fontsize=12)
    ax1.set_title(f'QC有效性验证: QC vs 随机剔除 ({condition_name})', fontsize=14)
    ax1.text(0.05, 0.95, '点在虚线下方表示 QC 有效\n(QC Std < Random Std)', 
             transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 子图2: 提升率分布图
    ax2 = axes[1]
    bins = np.linspace(-20, 80, 50)
    for method in methods:
        data = np.array(comparison_data[method])
        if len(data) == 0: continue
        # 计算提升率: (Random - QC) / Random * 100
        improvement = (data[:, 1] - data[:, 0]) / data[:, 1] * 100
        ax2.hist(improvement, bins=bins, color=colors.get(method, 'black'), 
                 alpha=0.5, label=f'{method} (Mean: {np.mean(improvement):.1f}%)')
    
    ax2.axvline(0, color='k', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('QC相对于随机剔除的稳定性提升率 (%)', fontsize=12)
    ax2.set_ylabel('文件数量 (Frequency)', fontsize=12)
    ax2.set_title('QC 带来的额外稳定性增益分布', fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path / f"QC_vs_Random_Stability_{condition_name}.png", dpi=150)
    plt.close(fig)

def plot_qc_vs_random_trajectory(qc_vs_random_data, output_path, condition_name):
    """
    【修改版】绘制 QC剔除 vs 随机剔除 vs 无筛选 的逐文件轨迹对比图。
    """
    # 提取所有方法
    methods = list(qc_vs_random_data.keys())
    if not methods: return

    # 创建子图
    fig, axes = plt.subplots(len(methods), 1, figsize=(16, 6 * len(methods)), sharex=True)
    if len(methods) == 1: axes = [axes] # 确保axes是列表

    fig.suptitle(f"【效能轨迹】QC剔除 vs 随机剔除 vs 无筛选 - {condition_name}", fontsize=18)

    # 定义颜色
    colors = {'OLS': 'green', 'WLS': 'blue', 'robust': 'red'}

    for ax, method in zip(axes, methods):
        # 准备数据
        data_list = qc_vs_random_data[method]
        df = pd.DataFrame(data_list)
        
        # 尝试自然排序
        try:
            df['sort_key'] = df['filename'].apply(natural_sort_key)
            df = df.sort_values('sort_key').reset_index(drop=True)
        except:
            pass 

        indices = df.index
        
        # --- 1. 绘制 无筛选 (Raw/Original) ---
        # 检查是否存在 std_raw 列，如果不存在则跳过或报错
        if 'std_raw' in df.columns:
            ax.plot(indices, df['std_raw'], 
                    color='black', linestyle=':', marker='.', markersize=4, alpha=0.6, linewidth=1.5,
                    label='无筛选 (No Removal)')

        # --- 2. 绘制 随机剔除 (Random Removal) ---
        ax.plot(indices, df['std_random'], 
                color='gray', linestyle='--', marker='x', markersize=6, alpha=0.7,
                label='随机剔除 (Random Removal)')
        
        # --- 3. 绘制 QC剔除 (QC Removal) ---
        ax.plot(indices, df['std_qc'], 
                color=colors.get(method, 'blue'), linestyle='-', marker='o', markersize=6, linewidth=2,
                label='质量控制 (QC Removal)')

        # 填充区域 (仅在 QC 和 随机 之间填充，避免过于混乱)
        ax.fill_between(indices, df['std_qc'], df['std_random'], 
                        where=(df['std_qc'] < df['std_random']),
                        interpolate=True, color='green', alpha=0.1, label='QC优于随机')
        
        ax.set_title(f"方法: {method}", fontsize=14)
        ax.set_ylabel("Tau (τ) 标准差 (Log Scale)", fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # 使用对数坐标
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)

        # 优化X轴
        if len(indices) > 20:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=20))
            
    axes[-1].set_xlabel("文件序号 (File Index)", fontsize=14)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path / f"QC_vs_Random_vs_Raw_Trajectory_{condition_name}.png", dpi=150)
    plt.close(fig)

#基本只用2/4
def run_analysis_suite_final_3(base_data_path, output_path, analysis_config):
    # 从配置中解包参数
    analysis_type = analysis_config['type']
    folder_prefix = analysis_config['folder_prefix']
    r2_threshold = analysis_config['r2_threshold']
    time_per_point = analysis_config['time_per_point']
    base_len_for_freq = analysis_config.get('base_len_for_freq', None)

    output_path = Path(output_path) / f"{analysis_type}_final"
    output_path.mkdir(parents=True, exist_ok=True)

    data_folders = sorted(
        [p for p in Path(base_data_path).iterdir() if p.is_dir() and p.name.startswith(folder_prefix)],
        key=natural_sort_key)
    if not data_folders: print(f"错误: 未找到 '{folder_prefix}*' 文件夹。"); return

    print(f"\n--- 开始 '{analysis_type.upper()}' 分析 (独立QC, 动态 t_end) ---")

    master_summary_list = []
    stability_plot_data = defaultdict(lambda: {'x': [], 'mean_of_stds': []})
    file_trace_data = defaultdict(lambda: {'WLS_QC': defaultdict(list), 'OLS_QC': defaultdict(list)})
    
    methods = ['WLS', 'OLS',  'WLS_QC', 'OLS_QC','robust', 'robust_QC']
    # methods = ['OLS', 'OLS_QC','robust', 'robust_QC']

    for folder in tqdm(data_folders, desc=f"分析 {analysis_type} 文件夹"):
        condition_value = int(re.findall(r'\d+', folder.name)[-1])
        # condition_value = 8000
        # --- 核心改动：动态计算 T_END ---
        if analysis_type == 'length':
            current_t_end = condition_value * time_per_point
        elif analysis_type == 'frequency':
            current_t_end = base_len_for_freq * time_per_point
        # --------------------------------
        qc_vs_random_data = defaultdict(list)

        npz_files = sorted(list(folder.glob("*.npz")), key=natural_sort_key)
        if not npz_files: continue

        folder_results, agg_stds = [], defaultdict(list)
        for npz_file in tqdm(npz_files,'处理NPZ文件', leave=False):
            try:
                # 从文件名中提取数字, 例如 "processed_data_batch_50.npz" -> 50
                # [-1]确保我们取的是文件名中最后一个数字，通常是批次号
                file_number = int(re.findall(r'\d+', npz_file.name)[-1])

                # 检查文件号是否在要跳过的范围内 [50, 51, 52, 53]
                # if 50 <= file_number <= 53:
                #     # 如果是，打印一条提示信息并用 continue 跳过本次循环
                #     tqdm.write(f"跳过文件: {npz_file.name}")
                #     continue
            except (IndexError, ValueError):
                # 如果文件名中没有数字或无法转换，则正常处理，不跳过
                pass
            with np.load(npz_file) as ld:
                signals = ld.get('data', ld.get('arr_0'))
            if signals.ndim == 1: signals = [signals]

            # 使用动态计算出的 t_end 来创建时间轴
            t_axis = np.linspace(0, current_t_end, signals.shape[-1])
            # t_axis = np.linspace(0, current_t_end, condition_value)
            taus_per_file = defaultdict(list)
            for signal in signals:
                signal = signal[:condition_value]
                qc_passed_wls, qc_passed_ols = False, False
                qc_passed_robust = False
                tau_wls, tau_ols =None, None
                tau_robust = None

                ols_params,_ = fit_ols(signal, t_axis)
                if ols_params is not None:
                    tau_ols = ols_params[1]
                    qc_passed_ols = check_linearity_of_variance(decay_model(t_axis, *ols_params),
                                                                signal - decay_model(t_axis, *ols_params), r2_threshold)

                wls_params,_ = fit_wls(signal, t_axis)
                if wls_params is not None:
                    tau_wls = wls_params[1]
                    qc_passed_wls = check_linearity_of_variance(decay_model(t_axis, *ols_params),
                                                                signal - decay_model(t_axis, *ols_params), r2_threshold)

                robust_params = fit_robust(signal, t_axis)
                if robust_params is not None:
                    tau_robust = robust_params[1]
                    qc_passed_robust = check_linearity_of_variance(decay_model(t_axis, *ols_params),
                                                                   signal - decay_model(t_axis, *ols_params), r2_threshold)
                if tau_wls: taus_per_file['WLS'].append(tau_wls)
                if tau_ols: taus_per_file['OLS'].append(tau_ols)
                if tau_robust: taus_per_file['robust'].append(tau_robust)

                if qc_passed_wls and tau_wls: taus_per_file['WLS_QC'].append(tau_wls)
                if qc_passed_ols and tau_ols: taus_per_file['OLS_QC'].append(tau_ols)
                if qc_passed_robust and tau_robust: taus_per_file['robust_QC'].append(tau_robust)

                        # --- 新增：执行 QC vs Random 对比分析 ---
            file_summary = {'filename': npz_file.name, 'condition': condition_value}
            for method in ['OLS', 'WLS', 'robust']:
                raw_key = method
                qc_key = f"{method}_QC"
                file_summary[f'tau_std_random_{method}'] = np.nan
                
                # 确保该方法有数据且进行了QC
                if raw_key in taus_per_file and qc_key in taus_per_file:
                    all_taus = taus_per_file[raw_key]
                    qc_taus = taus_per_file[qc_key]
                    
                    n_total = len(all_taus)
                    n_qc = len(qc_taus)
                    n_removed = n_total - n_qc
                    
                    # 只有当确实剔除了数据，且剩余数据量足够计算std时才进行对比
                    if n_removed > 0 and n_qc > 2:
                        # 1. 计算 QC 后的 Std
                        std_qc = np.std(qc_taus)
                        
                        std_random = simulate_random_removal_stability(all_taus, n_keep=n_qc, num_trials=50)
                        
                        if not np.isnan(std_random):
                            # 【修改点】这里改为存储字典，包含文件名，以便后续画轨迹图
                            file_summary[f'tau_std_random_{method}'] = std_random
                            
                            qc_vs_random_data[method].append({
                                'filename': npz_file.name,
                                'std_qc': std_qc,
                                'std_random': std_random
                            })
            for method in ['WLS_QC', 'OLS_QC']:
                if method in taus_per_file:
                    # 使用文件名作为唯一标识，追踪该文件在不同条件下的表现
                    file_trace_data[npz_file.name][method][condition_value].extend(taus_per_file[method])
            for name in methods:
                taus = taus_per_file[name]
                mean_tau, std_tau = (np.mean(taus), np.std(taus)) if len(taus) > 1 else (
                np.mean(taus) if taus else np.nan, 0)
                file_summary[f'tau_mean_{name}'], file_summary[f'tau_std_{name}'] = mean_tau, std_tau
                if not np.isnan(std_tau): agg_stds[name].append(std_tau)

            folder_results.append(file_summary)

        if folder_results:
            folder_df = pd.DataFrame(folder_results).set_index('filename')
            plot_intra_condition_comparison(folder_df, folder.name, output_path, analysis_type, methods)

        for name in methods:
            stability_plot_data[name]['x'].append(condition_value)
            stability_plot_data[name]['mean_of_stds'].append(np.mean(agg_stds[name]) if agg_stds[name] else np.nan)

        master_summary_list.extend(folder_results)

        if any(len(v) > 0 for v in qc_vs_random_data.values()):
            # 1. 画之前的散点统计图 (需要适配新的数据结构)
            # 为了兼容旧函数，我们需要把字典列表转回元组列表
            simple_data_for_scatter = {}
            for m, data_list in qc_vs_random_data.items():
                simple_data_for_scatter[m] = [(d['std_qc'], d['std_random']) for d in data_list]
            
            plot_qc_vs_random_comparison(
                simple_data_for_scatter, 
                output_path, 
                folder.name 
            )

            # 2. 【新增】画新的轨迹对比图
            plot_qc_vs_random_trajectory(
                qc_vs_random_data,
                output_path,
                folder.name
            )


    plot_inter_condition_stability(stability_plot_data, output_path, analysis_config, methods)

    if master_summary_list:
        # 修正后代码
        # 1. 先创建DataFrame，不进行排序
        report_df = pd.DataFrame(master_summary_list)

        # 2. 为 'filename' 列创建一个临时的自然排序键列
        report_df['filename_sort_key'] = report_df['filename'].apply(natural_sort_key)

        # 3. 根据 'condition' 和新的排序键列进行排序，然后删除临时列
        report_df = report_df.sort_values(by=['condition', 'filename_sort_key']).drop(columns=['filename_sort_key'])

        report_path = output_path / f"{analysis_type}_main_report_final.csv"
        report_df.to_csv(report_path, index=False, float_format='%.6f', encoding='utf-8-sig')
        tqdm.write(f"'{analysis_type}' 的主报告和所有图像已保存至: {output_path}")

#该函数添加了Per_File_Distributions目录下的分析
def run_analysis_suite_final_4(base_data_path, output_path, analysis_config):
    # 1. 参数解包与路径设置
    analysis_type = analysis_config['type']
    folder_prefix = analysis_config['folder_prefix']
    r2_threshold = analysis_config['r2_threshold']
    time_per_point = analysis_config['time_per_point']
    base_len_for_freq = analysis_config.get('base_len_for_freq', None)

    output_path = Path(output_path) / f"{analysis_type}_final"
    output_path.mkdir(parents=True, exist_ok=True)
    
    dist_plot_folder = output_path / "Per_File_Distributions"
    dist_plot_folder.mkdir(parents=True, exist_ok=True)

    # 2. 获取所有相关文件夹
    data_folders = sorted(
        [p for p in Path(base_data_path).iterdir() if p.is_dir() and p.name.startswith(folder_prefix)],
        key=natural_sort_key)
    if not data_folders: print(f"错误: 未找到 '{folder_prefix}*' 文件夹。"); return

    print(f"\n--- 开始 '{analysis_type.upper()}' 分析 (文件优先模式) ---")

    # 3. 【预扫描】获取所有唯一的文件名
    all_filenames = set()
    for folder in data_folders:
        for p in folder.glob("*.npz"):
            all_filenames.add(p.name)
    sorted_filenames = sorted(list(all_filenames), key=natural_sort_key)
    print(f"共发现 {len(sorted_filenames)} 个唯一文件，准备逐个分析...")

    # 4. 初始化全局容器 (用于最后生成汇总报告和汇总图)
    global_folder_results = defaultdict(list) 
    global_qc_random = defaultdict(lambda: defaultdict(list))
    global_agg_stds = defaultdict(lambda: defaultdict(list))
    master_summary_list = []
    methods = ['WLS', 'OLS', 'WLS_QC', 'OLS_QC', 'robust', 'robust_QC']

    # ==========================================
    # 核心循环：按文件遍历 (File -> Condition)
    # ==========================================
    for filename in tqdm(sorted_filenames, desc="逐文件分析进度"):
        
        # 用于存储当前文件在不同条件下的原始Tau分布 (用于画散点图)
        # 结构: file_trace_data['WLS_QC'][condition_value] = [tau1, tau2...]
        file_trace_data = defaultdict(lambda: defaultdict(list))
        
        # 遍历所有条件文件夹，寻找当前文件
        for folder in data_folders:
            npz_file = folder / filename
            if not npz_file.exists(): continue 

            # --- 解析条件参数 ---
            try:
                condition_value = int(re.findall(r'\d+', folder.name)[-1])
            except: continue

            if analysis_type == 'length':
                current_t_end = condition_value * time_per_point
            elif analysis_type == 'frequency':
                current_t_end = base_len_for_freq * time_per_point
            
            # --- 加载与拟合 ---
            try:
                with np.load(npz_file) as ld:
                    signals = ld.get('data', ld.get('arr_0'))
                if signals.ndim == 1: signals = [signals]
                
                t_axis = np.linspace(0, current_t_end, signals.shape[-1])
                taus_per_file = defaultdict(list)
                
                # 逐信号拟合
                for signal in signals:
                    signal = signal[:condition_value]
                    
                    # OLS
                    ols_params, _ = fit_ols(signal, t_axis)
                    if ols_params is not None:
                        tau = ols_params[1]
                        taus_per_file['OLS'].append(tau)
                        if check_linearity_of_variance(decay_model(t_axis, *ols_params), 
                                                     signal - decay_model(t_axis, *ols_params), r2_threshold):
                            taus_per_file['OLS_QC'].append(tau)
                    
                    # WLS
                    wls_params, _ = fit_wls(signal, t_axis)
                    if wls_params is not None:
                        tau = wls_params[1]
                        taus_per_file['WLS'].append(tau)
                        if check_linearity_of_variance(decay_model(t_axis, *ols_params), 
                                                     signal - decay_model(t_axis, *ols_params), r2_threshold):
                            taus_per_file['WLS_QC'].append(tau)

                    # Robust
                    robust_params = fit_robust(signal, t_axis)
                    if robust_params is not None:
                        tau = robust_params[1]
                        taus_per_file['robust'].append(tau)
                        if check_linearity_of_variance(decay_model(t_axis, *ols_params), 
                                                     signal - decay_model(t_axis, *ols_params), r2_threshold):
                            taus_per_file['robust_QC'].append(tau)

                # --- 收集数据 1: 用于立即画图 (单文件分布) ---
                for method in ['WLS', 'OLS']:
                    if method in taus_per_file:
                        file_trace_data[method][condition_value].extend(taus_per_file[method])

                # --- 收集数据 2: 用于全局汇总 (Summary Stats) ---
                file_summary = {'filename': filename, 'condition': condition_value}
                
                # 计算 QC vs Random 并存入 file_summary 和 global_qc_random
                for method in ['OLS', 'WLS', 'robust']:
                    raw_key = method
                    qc_key = f"{method}_QC"
                    file_summary[f'tau_std_random_{method}'] = np.nan
                    
                    if raw_key in taus_per_file and qc_key in taus_per_file:
                        all_taus = taus_per_file[raw_key]
                        qc_taus = taus_per_file[qc_key]
                        n_total = len(all_taus)
                        n_qc = len(qc_taus)
                        n_removed = n_total - n_qc
                        
                        if n_removed > 0 and n_qc > 2:
                            std_qc = np.std(qc_taus)
                            std_rand = simulate_random_removal_stability(all_taus, n_keep=n_qc, num_trials=1000)
                            std_raw = np.std(all_taus)
                            
                            if not np.isnan(std_rand):
                                file_summary[f'tau_std_random_{method}'] = std_rand
                                # 存入全局容器，用于后续按文件夹绘图
                                global_qc_random[folder.name][method].append({
                                    'filename': filename,
                                    'std_qc': std_qc,
                                    'std_random': std_rand,
                                    'std_raw': std_raw
                                })

                # 计算常规统计量
                for name in methods:
                    taus = taus_per_file[name]
                    mean_tau, std_tau = (np.mean(taus), np.std(taus)) if len(taus) > 1 else (np.mean(taus) if taus else np.nan, 0)
                    file_summary[f'tau_mean_{name}'] = mean_tau
                    file_summary[f'tau_std_{name}'] = std_tau
                    if not np.isnan(std_tau): 
                        global_agg_stds[folder.name][name].append(std_tau)
                
                global_folder_results[folder.name].append(file_summary)
                master_summary_list.append(file_summary)

            except Exception as e:
                pass

        # ==========================================
        # 【关键修正】绘图逻辑移至 for folder 循环之外
        # 确保收集了该文件在所有条件下的数据后，再画一张总图
        # ==========================================
        if file_trace_data['WLS'] or file_trace_data['OLS']: # 修正键名匹配
            plot_tau_distribution_comparison(
                file_trace_data, 
                dist_plot_folder, 
                analysis_config, 
                filename_tag=filename
            )

    # ==========================================
    # 5. 后处理：生成汇总图表 (按文件夹/条件)
    # ==========================================
    print("\n正在生成汇总报告和图表...")
    
    stability_plot_data = defaultdict(lambda: {'x': [], 'mean_of_stds': []})

    for folder in tqdm(data_folders, desc="生成汇总图"):
        folder_name = folder.name
        results = global_folder_results.get(folder_name, [])
        if not results: continue

        # 1. 画 intra_condition_comparison (单条件下的方法对比)
        folder_df = pd.DataFrame(results).set_index('filename')
        plot_intra_condition_comparison(folder_df, folder_name, output_path, analysis_type, methods)

        # 2. 画 QC vs Random 对比图 (使用全局收集的数据)
        qc_data = global_qc_random.get(folder_name, {})
        if any(len(v) > 0 for v in qc_data.values()):
            # 转换数据格式以适配绘图函数
            simple_data = {m: [(d['std_qc'], d['std_random']) for d in lst] for m, lst in qc_data.items()}
            plot_qc_vs_random_comparison(simple_data, output_path, folder_name)
            
            # 轨迹图需要包含文件名的字典列表
            plot_qc_vs_random_trajectory(qc_data, output_path, folder_name)

        # 3. 收集稳定性数据
        try:
            cond_val = int(re.findall(r'\d+', folder_name)[-1])
            for name in methods:
                stds = global_agg_stds[folder_name][name]
                mean_std = np.mean(stds) if stds else np.nan
                stability_plot_data[name]['x'].append(cond_val)
                stability_plot_data[name]['mean_of_stds'].append(mean_std)
        except: pass

    # 6. 画 inter_condition_stability (跨条件稳定性)
    plot_inter_condition_stability(stability_plot_data, output_path, analysis_config, methods)

    # 7. 保存最终 CSV
    if master_summary_list:
        report_df = pd.DataFrame(master_summary_list)
        report_df['filename_sort_key'] = report_df['filename'].apply(natural_sort_key)
        report_df = report_df.sort_values(by=['condition', 'filename_sort_key']).drop(columns=['filename_sort_key'])
        
        report_path = output_path / f"{analysis_type}_main_report_final.csv"
        report_df.to_csv(report_path, index=False, float_format='%.6f', encoding='utf-8-sig')
        print(f"主报告已保存至: {report_path}")


def plot_tau_distribution_comparison(distribution_data, output_path, analysis_config, filename_tag=None):
    """
    【修改版】绘制特定条件下的 Tau 值序列图 + 边缘直方图。
    功能：
    1. 主图：展示 Tau 值随信号索引的变化 (序列图)。
    2. 侧边图：展示 Tau 值的分布直方图，直观对比收敛宽度。
    样式：
        - WLS: 浅蓝色实线 + 实心三角形
        - OLS: 浅红色实线 + 实心圆形
    调整：
        - 点更大 (markersize=6)
        - 图更窄 (figsize width reduced) 以突显垂直分布差异
    """
    # 1. 准备数据
    raw_keys = list(distribution_data['WLS'].keys())
    if not raw_keys: 
        raw_keys = list(distribution_data['OLS'].keys())
        if not raw_keys: return
    
    try:
        conditions = sorted([int(k) for k in raw_keys])
    except ValueError:
        conditions = sorted(raw_keys)
    
    if not conditions: return
    
    # --- 锁定目标：第一个条件 ---
    target_condition = conditions[0]
    
    def get_data(method, cond):
        if cond in distribution_data[method]: return distribution_data[method][cond]
        if str(cond) in distribution_data[method]: return distribution_data[method][str(cond)]
        return []

    ols_data = np.array(get_data('OLS', target_condition))
    wls_data = np.array(get_data('WLS', target_condition))
    
    if len(ols_data) == 0 and len(wls_data) == 0: return

    # 对齐长度
    current_len = len(ols_data)
    if len(wls_data) > 0:
        current_len = min(len(ols_data), len(wls_data))
        ols_data = ols_data[:current_len]
        wls_data = wls_data[:current_len]
    
    x_indices = np.arange(1, current_len + 1)

    # --- 绘图布局：主图占 4/5，右侧直方图占 1/5 ---
    # 【修改点1】减小宽度 (14 -> 10)，使横轴视觉上变窄，突显纵向波动幅度
    fig = plt.figure(figsize=(10, 6)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
    
    ax_main = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharey=ax_main)

    # --- 1. 主图：序列波动 ---
    # 【修改点2】增大 markersize (4->6)，增加 alpha (0.6->0.75)
    
    # OLS
    if len(ols_data) > 0:
        ax_main.plot(x_indices, ols_data, 
                color='lightcoral', linestyle='-', linewidth=1, alpha=0.75,
                marker='o', markersize=6, markeredgewidth=0, 
                label=f'OLS (Cond={target_condition})')
        ax_main.axhline(np.mean(ols_data), color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    # WLS
    if len(wls_data) > 0:
        ax_main.plot(x_indices, wls_data, 
                color='skyblue', linestyle='-', linewidth=1, alpha=0.75,
                marker='^', markersize=6, markeredgewidth=0,
                label=f'WLS (Cond={target_condition})')
        ax_main.axhline(np.mean(wls_data), color='blue', linestyle='--', linewidth=1.5, alpha=0.8)

    ax_main.set_xlabel("Signal Index", fontsize=12)
    ax_main.set_ylabel("Fitted Decay Time $\\tau$ ($\\mu s$)", fontsize=12)
    ax_main.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_main.grid(True, linestyle=':', alpha=0.4)

    # --- 2. 侧边图：分布直方图 ---
    bins = 30
    # 确定直方图的 bins 范围，保证两者使用相同的刻度
    all_vals = np.concatenate([ols_data, wls_data])
    # 剔除极端值计算范围，使直方图更聚焦
    p01, p99 = np.percentile(all_vals, [1, 99])
    hist_range = (p01, p99)

    if len(ols_data) > 0:
        ax_hist.hist(ols_data, bins=bins, range=hist_range, orientation='horizontal', 
                     color='lightcoral', alpha=0.6, density=True, label='OLS Dist') # alpha 0.5->0.6
        # 画高斯拟合曲线或KDE曲线会让对比更明显，这里简单画个均值线
        ax_hist.axhline(np.mean(ols_data), color='red', linestyle='--', linewidth=1.5)

    if len(wls_data) > 0:
        ax_hist.hist(wls_data, bins=bins, range=hist_range, orientation='horizontal', 
                     color='skyblue', alpha=0.6, density=True, label='WLS Dist') # alpha 0.5->0.6
        ax_hist.axhline(np.mean(wls_data), color='blue', linestyle='--', linewidth=1.5)

    ax_hist.axis('off') # 隐藏直方图的坐标轴，只看形状
    
    # --- 3. 自动调整 Y 轴范围 (聚焦核心区域) ---
    # 计算稳健的 Y 轴范围 (Mean +/- 3*Std)
    if len(all_vals) > 0:
        median_val = np.median(all_vals)
        # 使用 MAD (Median Absolute Deviation) 或 Std 来确定范围
        std_val = np.std(all_vals)
        
        # 限制显示范围，让图表更紧凑
        y_min = median_val - 3.5 * std_val
        y_max = median_val + 3.5 * std_val
        ax_main.set_ylim(y_min, y_max)

    # 标题
    title_suffix = f" - File: {filename_tag}" if filename_tag else ""
    fig.suptitle(f"Tau Sequence & Distribution Comparison (Cond: {target_condition}){title_suffix}", fontsize=14, y=0.95)

    # 保存
    if filename_tag:
        save_name = f"Tau_Seq_Dist_{analysis_config['type']}_{filename_tag}.png"
    else:
        save_name = f"Tau_Seq_Dist_Comparison_AllFiles.png"
   
    plt.savefig(output_path / save_name, dpi=150, bbox_inches='tight')
    plt.close(fig)

# --- 6. 运行主程序 ---
if __name__ == "__main__":
    # --- 基础配置 ---
    BASE_RAWDATA_PATH = Path(r"C:\Users\Mingkai\Desktop\rawdata")
    BASE_RAWDATA_PATH = Path(r"F:\rawdatanew\12processed_data")
    # MAIN_OUTPUT_FOLDER = Path(r"processednew/fit_comparison_data2_changeols_wls")
    MAIN_OUTPUT_FOLDER = Path(r"processednew/essay_pictures/fit_comarison_data12_QC")
    # --- 关键物理参数 ---
    TIME_PER_POINT = 0.1
    # 对于频率分析，原始信号长度固定为18000点
    BASE_LEN_FOR_FREQ_ANALYSIS = 18000
    # QC的R²阈值
    QC_R_SQUARED_THRESHOLD = 0.8

    # --- 分析任务配置列表 ---
    analysis_configs = [
        {
            'type': 'length',
            'folder_prefix': 'processed_len',
            'x_axis_label': '数据点数',
            'r2_threshold': QC_R_SQUARED_THRESHOLD,
            'time_per_point': TIME_PER_POINT,
        },
        {
            'type': 'frequency',
            'folder_prefix': 'resampled_freq_AA',
            'x_axis_label': '采样频率/步长',
            'r2_threshold': QC_R_SQUARED_THRESHOLD,
            'time_per_point': TIME_PER_POINT,
            'base_len_for_freq': BASE_LEN_FOR_FREQ_ANALYSIS  # 此项仅用于频率分析
        }
    ]

    for config in analysis_configs:
        run_analysis_suite_final_2(
            base_data_path=BASE_RAWDATA_PATH,
            output_path=MAIN_OUTPUT_FOLDER,
            analysis_config=config
        )


