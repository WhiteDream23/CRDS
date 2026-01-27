"""
文件名: singlevsnoise.py
output : processednew
用途: 对比单一频率信号与噪声,进行信噪比(SNR)分析和光谱对比 (期刊绘图版)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pathlib import Path
import re

# --- 0. 期刊绘图风格设置 ---
def set_journal_style():
    config = {
"font.family": "serif",  # 修改为 serif，匹配 Times New Roman
        "font.serif": ['Times New Roman', 'SimSun', 'STSong'], # 加入宋体 (SimSun)
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix", # 数学公式字体
        "font.size": 16,
        "axes.linewidth": 1.0,
        "xtick.direction": "in", # 刻度朝内
        "ytick.direction": "in",
        "xtick.top": False,      # 不显示顶部刻度
        "ytick.right": False,    # 不显示右侧刻度
        "legend.frameon": False, # 图例无边框
    }
    rcParams.update(config)
def plot_spectrum_with_inset(report_df, c, freq_map_func, peak_range, baseline_ranges, methods_to_compare, output_folder):
    """
    绘制光谱图（含局部放大）。
    逻辑修改：基于【全部数据】计算SNR，但仅绘制【最后110个点】。
    """
    set_journal_style()
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # --- 步骤1：数据准备 (全局) ---
    # 1.1 计算频率并排序 (使用完整数据)
    report_df['frequency'] = report_df['filename'].apply(freq_map_func)
    full_sorted_df = report_df.sort_values('frequency').reset_index(drop=True)
    
    print(f"全局数据量: {len(full_sorted_df)} 点")

    # --- 步骤2：确定全局基线掩膜 (用于计算SNR) ---
    baseline_masks_full = []
    for start_freq, end_freq in baseline_ranges:
        mask = (full_sorted_df['frequency'] >= start_freq) & (full_sorted_df['frequency'] <= end_freq)
        baseline_masks_full.append(mask)
    
    if baseline_masks_full:
        is_baseline_full = pd.concat(baseline_masks_full, axis=1).any(axis=1)
    else:
        is_baseline_full = pd.Series([True]*20 + [False]*(len(full_sorted_df)-20), index=full_sorted_df.index)

    # --- 步骤3：全局计算 Alpha 和 SNR ---
    # 我们先计算所有数据的 Alpha，然后再截取用于绘图
    full_spectra_alpha = {} # 存储完整长度的 alpha
    snr_results = {}
    
    colors = {'OLS': '#1f77b4', 'WLS': '#d62728', 'robust': '#2ca02c', 
              'OLS_QC': '#9467bd', 'WLS_QC': '#d62728', 'robust_QC': '#8c564b'}

    for method in methods_to_compare:
        tau_col = f'tau_mean_{method}'
        if tau_col not in full_sorted_df.columns: continue

        # 3.1 计算全局 Tau0 (基线均值)
        tau_data_full = full_sorted_df[tau_col]
        baseline_data = tau_data_full[is_baseline_full]
        
        if baseline_data.empty:
            print(f"警告: 方法 {method} 在指定基线范围内无数据，使用前10点兜底。")
            tau0 = tau_data_full.head(10).mean()
            noise_std = tau_data_full.head(10).std() # 仅作兜底
        else:
            tau0 = baseline_data.mean()
        
        # 3.2 计算全局吸收系数 Alpha
        tau_sec = tau_data_full * 1e-6
        tau0_sec = tau0 * 1e-6
        alpha_full = (1 / c) * (1 / tau_sec - 1 / tau0_sec)
        full_spectra_alpha[method] = alpha_full

        # 3.3 计算全局 SNR
        # Signal = 全局最大值
        # Noise = 全局基线区域的标准差
        signal = alpha_full.max()
        
        # 重新在 alpha 域计算噪声 (基线部分的 std)
        if not baseline_data.empty:
            noise = alpha_full[is_baseline_full].std()
        else:
            noise = alpha_full.head(10).std()

        snr = signal / noise if noise > 0 else 0
        snr_results[method] = snr
        print(f"方法 {method}: 全局SNR = {snr:.2f} (Signal={signal:.2e}, Noise={noise:.2e})")

    # --- 步骤4：数据截取 (用于绘图) ---
    # 只保留最后 100 个点
    if len(full_sorted_df) > 100:
        plot_df = full_sorted_df.tail(100).copy()
        # 同时也截取 alpha 数据
        plot_spectra = {k: v.tail(100).reset_index(drop=True) for k, v in full_spectra_alpha.items()}
    else:
        plot_df = full_sorted_df.copy()
        plot_spectra = {k: v.copy().reset_index(drop=True) for k, v in full_spectra_alpha.items()}
    
    plot_df = plot_df.reset_index(drop=True)
    print(f"绘图范围: 最后 {len(plot_df)} 点")

    # --- 步骤5：绘制主图与插图 ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # 5.1 绘制主光谱曲线
    if 'wave_number' in plot_df.columns:
        x = plot_df['wave_number'].values
    else:
        x = np.arange(1, len(plot_df) + 1)
    
    legend_map = {
        'OLS': '传统提取方法',
        'WLS': 'WNLS',
        'OLS_QC': '',
        'WLS_QC': '高精度提取方法',
        'robust': 'Robust',
        'robust_QC': 'Robust + NMQC'
    }
    for method in methods_to_compare:
        if method not in plot_spectra: continue
        
        y = plot_spectra[method] *1e8
        color = colors.get(method, 'black')
        display_name = legend_map.get(method, method) # 如果找不到映射，就用原名
        label_text = f"{display_name} (SNR={snr_results[method]:.1f})"
        
        ax.plot(x, y, label=label_text, color=color, linewidth=1.5, alpha=0.9)

    # 5.2 设置主图标签
    ax.set_xlabel(r'$\nu$ ($cm^{-1}$)', fontsize=18)
    ax.set_ylabel(r"Absorption Coefficient $\alpha$ ($10^{-8}$cm$^{-1}$)", fontsize=18)
    ax.legend(loc='upper left', prop={'family': 'SimSun', 'size': 15})
    
    # 5.3 添加局部放大图 (Inset)
    # 寻找绘图数据中的基线部分用于放大
    # 注意：这里需要在 plot_df (截取后的数据) 中重新找基线
    plot_baseline_masks = []
    for start_freq, end_freq in baseline_ranges:
        mask = (plot_df['frequency'] >= start_freq) & (plot_df['frequency'] <= end_freq)
        plot_baseline_masks.append(mask)
    
    if plot_baseline_masks:
        is_baseline_plot = pd.concat(plot_baseline_masks, axis=1).any(axis=1)
        baseline_df_plot = plot_df[is_baseline_plot]
    else:
        baseline_df_plot = pd.DataFrame()

    # 设置小图位置 (左侧中部)
    axins = inset_axes(ax, width="30%", height="30%", loc='center left', 
                       bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax.transAxes)
    
    # 确定放大区域
    # 优先使用截取数据中定义的基线区域，如果没有，则取前30%
    if not baseline_df_plot.empty:
        # 找到基线在 x 轴(1-110)上的索引范围
        valid_indices = baseline_df_plot.index
        zoom_idx_start = valid_indices.min()
        zoom_idx_end = valid_indices.max()
        
        # 如果基线区域太大，只取一段
        if (zoom_idx_end - zoom_idx_start) > 40:
            zoom_idx_end = zoom_idx_start + 40
    else:
        # 兜底：取前 30 个点
        zoom_idx_start = 0
        zoom_idx_end = min(30, len(plot_df)-1)

    zoom_x = x[zoom_idx_start+10:zoom_idx_end+1]
    
    # 绘制小图
    for method in methods_to_compare:
        if method not in plot_spectra: continue
        y_full_segment = plot_spectra[method] *1e8
        y_zoom = y_full_segment.iloc[zoom_idx_start:zoom_idx_end+1]
        
        axins.plot(x, y_full_segment, color=colors.get(method, 'black'), linewidth=1.2)
        
    # 设置小图范围
    axins.set_xlim(zoom_x.min(), zoom_x.max())
    
    # 自动计算小图Y轴范围
    y_zoom_vals = []
    for m in methods_to_compare:
        if m in plot_spectra:
            y_zoom_vals.extend(plot_spectra[m].iloc[zoom_idx_start:zoom_idx_end+1]*1e8)
    
    if y_zoom_vals:
        y_mean = np.mean(y_zoom_vals)
        y_std = np.std(y_zoom_vals)
        axins.set_ylim(y_mean - 4*y_std, y_mean + 4*y_std)

    axins.set_title("Baseline Noise", fontsize=14)
    axins.tick_params(labelsize=12)
    axins.grid(True, linestyle=':', alpha=0.3)
    axins.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4)) 
    axins.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')

    # 5.4 保存图片
    save_path = output_folder / "spectral_snr_last110_journal.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {save_path}")

# --- 2. 主程序入口 ---
if __name__ == '__main__':
    # !!! 参数配置区域 !!!

    # 1. 数据路径
    REPORT_CSV_PATH = Path(r"E:\pythonProject_crds\processednew\essay_pictures\fit_comarison_data12_QC\length_final\length_main_report_final_picture.csv")

    # 2. 输出文件夹
    OUTPUT_FOLDER = Path(r"E:\pythonProject_crds\processednew\essay_pictures\fit_comarison_data12_QC\length_final")

    # 3. CRDS物理参数
    SPEED_OF_LIGHT = 2.99792458e10  # cm/s

    # 4. 频率映射
    def map_filename_to_frequency(filename):
        try:
            return float(re.findall(r'\d+', filename)[-1])
        except:
            return np.nan

    # 5. 区域定义 (用于计算SNR，不再用于画图背景)
    # 请确保这些范围在最后100个点的频率范围内，否则SNR计算会回退到默认逻辑
    PEAK_RANGE = (70, 115) 
    BASELINE_RANGES = [
        (1, 69),
        (116, 130),
    ]

    # 6. 方法选择
    METHODS_TO_COMPARE = ['WLS_QC', 'OLS'] # 建议只选两个对比，图面更整洁

    # --- 运行 ---
    try:
        full_report_df = pd.read_csv(REPORT_CSV_PATH)
        # 筛选特定条件
        target_condition_df = full_report_df[full_report_df['condition'] == 18000].copy()

        if target_condition_df.empty:
            print("错误：在报告中找不到指定condition的数据。")
        else:
            plot_spectrum_with_inset(
                report_df=target_condition_df,
                c=SPEED_OF_LIGHT,
                freq_map_func=map_filename_to_frequency,
                peak_range=PEAK_RANGE,
                baseline_ranges=BASELINE_RANGES,
                methods_to_compare=METHODS_TO_COMPARE,
                output_folder=OUTPUT_FOLDER
            )
    except FileNotFoundError:
        print(f"错误：报告文件不存在 -> {REPORT_CSV_PATH}")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()