import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import platform
from pathlib import Path

# ================= 配置区域 =================
CSV_FILE_PATH = Path(r"E:\pythonProject_crds\processednew\essay_pictures\fit_comarison_data7_QC\length_final\length_main_report_final_picture.csv")

# 更加专业的配色方案 (Science/Nature 风格)
COLORS = {
    'OLS': '#4E79A7',  # 稳重的深蓝色
    'WLS': '#E15759',  # 醒目的珊瑚红
    'FILL': '#E15759'  # 填充色
}

COLUMNS_TO_PLOT = [
    {
        'col': 'tau_std_OLS_QC', 
        'label': 'NLS + NMQC', 
        'color': COLORS['OLS'],
        'marker': 'D',
        'linestyle': '--',
        'linewidth': 1.5
    },
    {
        'col': 'tau_std_WLS_QC', 
        'label': 'WNLS + NMQC', # 强调这是提出的方法
        'color': COLORS['WLS'],
        'marker': 'o',
        'linestyle': '-',
        'linewidth': 2.0 # 主方法线条稍微加粗
    },
]

# X_LABEL = 'Measurement Batch Index' # 英文标签更通用，也可改回中文
# Y_LABEL = r'Std. Dev. of Decay Time $\sigma_{\tau}$ ($\mu s$)'
X_LABEL = r'$\nu_i$'# 英文标签更通用，也可改回中文
Y_LABEL = r'Std. Dev. of Decay Time $\sigma_{\tau}$ ($\mu s$)'
# ================= 核心：极简学术风格设置 =================
def set_science_style():
    """配置类似 Science/Nature 的极简风格"""
    
    # 字体设置：优先使用 Arial (无衬线) 或 Times New Roman (衬线)
    sys = platform.system()
    if sys == "Windows":
        font_list = ['Arial', 'SimHei', 'Microsoft YaHei']
    else:
        font_list = ['Arial', 'Helvetica', 'DejaVu Sans']
        
    config = {
        "font.family": "serif",
        "font.sans-serif": ['Times New Roman'],
        "mathtext.fontset": "stixsans", # 公式字体与无衬线体匹配
        
        "figure.figsize": (10, 6),   # 稍宽一点，适合放局部放大图
        "figure.dpi": 300,
        
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "axes.linewidth": 1.2,       # 坐标轴线宽
        "axes.spines.top": True,    # 【关键】去除顶部边框
        "axes.spines.right": True,  # 【关键】去除右侧边框
        
        "xtick.direction": 'out',    # 刻度朝外，更现代
        "ytick.direction": 'out',
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        
        "legend.fontsize": 16,
        "legend.frameon": False,     # 无边框图例
        "legend.loc": "upper right",
        
        "grid.linestyle": ':',       # 点状网格
        "grid.alpha": 0.4,
        "grid.color": "gray",
    }
    rcParams.update(config)

def plot_elegant_chart(show_inset=True):
    set_science_style()

    # 1. 读取数据
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except Exception as e:
        print(f"无法读取文件，生成模拟数据... ({e})")
        # 生成模拟收敛数据
        x_sim = np.arange(1, 151)
        # OLS 波动大，收敛慢
        y_ols = 0.5 + 0.5 * np.exp(-x_sim/50) + np.random.normal(0, 0.05, 150)
        # WLS 波动小，收敛快且低
        y_wls = 0.2 + 0.1 * np.exp(-x_sim/50) + np.random.normal(0, 0.01, 150)
        df = pd.DataFrame({'tau_std_OLS_QC': y_ols, 'tau_std_WLS_QC': y_wls})

    # --- 修改：只截取最后 100 行数据，并重置索引 ---
    if len(df) > 100:
        df = df.tail(100).reset_index(drop=True)
    # --------------------------------

    # 2. 创建画布
    fig, ax = plt.subplots()
    
    # x 轴重置为 1 到 N (N<=100)
    x = np.arange(1, len(df) + 1)
    
    # 3. 绘制主图线条
    # 稀疏标记策略

    for item in COLUMNS_TO_PLOT:
        col = item['col']
        if col not in df.columns: continue
        
        # 绘制线条 (无标记)
        ax.plot(x, df[col], 
                label=item['label'], 
                color=item['color'], 
                linestyle=item['linestyle'],
                linewidth=item['linewidth'],
                marker=item['marker'],       # 添加标记
                markersize=6,                # 标记大小
                markeredgecolor='white',     # 标记边缘颜色
                markeredgewidth=0.8,         # 标记边缘宽度
                alpha=0.9,
                zorder=10)
        

    # 4. 【亮点】添加填充区域 (Highlight Improvement)
    # if 'tau_std_OLS_QC' in df and 'tau_std_WLS_QC' in df:
    #     ax.fill_between(x, df['tau_std_OLS_QC'], df['tau_std_WLS_QC'], 
    #                     color=COLORS['FILL'], alpha=0.1, label='Performance Gain')

    # 5. 【亮点】添加局部放大图 (Inset Zoom) - 可选
    if show_inset:
        # 自动寻找最后 25% 的数据区域进行放大
        zoom_idx_start = int(len(df) * 0.75)
        
        # 创建嵌入坐标轴 [x, y, width, height] (相对于父坐标轴)
        axins = inset_axes(ax, width="35%", height="30%", loc='center right', 
                           bbox_to_anchor=(0, 0.1, 1, 1), bbox_transform=ax.transAxes)
        
        # 在小图中重画曲线
        for item in COLUMNS_TO_PLOT:
            col = item['col']
            if col not in df.columns: continue
            axins.plot(x, df[col], 
                       color=item['color'], 
                       linestyle=item['linestyle'],
                       linewidth=1.5) # 小图线条稍细
            # 小图也画点，但更稀疏
            axins.plot(x[::mark_interval], df[col][::mark_interval], 
                       linestyle='None', marker=item['marker'], color=item['color'], 
                       markersize=5, markeredgecolor='white', markeredgewidth=0.5)

        # 设置小图的显示范围 (使用重置后的 x 轴数值)
        axins.set_xlim(x[zoom_idx_start], x[-1])
        
        # 自动计算小图 Y 轴范围
        y_slice_ols = df['tau_std_OLS_QC'].iloc[zoom_idx_start:]
        y_slice_wls = df['tau_std_WLS_QC'].iloc[zoom_idx_start:]
        y_min_zoom = min(y_slice_ols.min(), y_slice_wls.min())
        y_max_zoom = max(y_slice_ols.max(), y_slice_wls.max())
        margin_zoom = (y_max_zoom - y_min_zoom) * 0.2
        axins.set_ylim(y_min_zoom - margin_zoom, y_max_zoom + margin_zoom)
        
        # 美化小图
        axins.tick_params(labelsize=10)
        axins.grid(True, linestyle=':', alpha=0.3)
        
        # 添加连接线 (Mark Inset)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')

    # 6. 装饰主图
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    fixed_ticks = [0, 20, 40, 60, 80, 100]
    ax.set_xticks(fixed_ticks)
    ax.grid(True, axis='y') # 只保留横向网格
    
    # 坐标范围
    y_vals = df[[c['col'] for c in COLUMNS_TO_PLOT if c['col'] in df]].values.flatten()
    ax.set_ylim(0, y_vals.max() * 1.4)
    ax.set_xlim(0, len(df) * 1.02)

    # 图例
    ax.legend(loc='upper right', frameon=False, ncol=1, fontsize=18)

    # 7. 保存
    plt.tight_layout()
    suffix = "_Zoom" if show_inset else ""
    save_path_png = CSV_FILE_PATH.parent / f'Elegant_Comparison_Last100{suffix}.png'
    save_path_pdf = CSV_FILE_PATH.parent / f'Elegant_Comparison_Last100{suffix}.pdf'
    
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    print(f"图表已保存至: {save_path_png} 和 {save_path_pdf}")

if __name__ == "__main__":
    plot_elegant_chart(False)