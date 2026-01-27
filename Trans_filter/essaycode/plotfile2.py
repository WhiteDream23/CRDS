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
    'NLS': '#4E79A7',    # 稳重的深蓝色
    'NLS+RANDOM': '#E15759',    # 醒目的珊瑚红
    'NLS+QC': '#59A14F', # 清新的森林绿 (新增)
    'FILL': '#BAB0AC'    # 填充色 (灰色，用于背景范围)
}

# 【修改点】在这里定义你要对比的三列数据
COLUMNS_TO_PLOT = [
    {
        'col': 'tau_std_OLS', 
        'label': r'Dataset$_{\mathrm{raw}}$',
        'color': COLORS['NLS'],
        'marker': 'D', # 菱形
        'linestyle': '--',
        'linewidth': 1.5
    },
    {
        'col': 'tau_std_random_OLS',  # 假设第三列名为 robust
        'label': r'Dataset$_{\mathrm{random}}$', 
        'color': COLORS['NLS+RANDOM'],
        'marker': '^', # 三角形
        'linestyle': 'dotted',
        'linewidth': 1.5
    },
    {
        'col': 'tau_std_OLS_QC', 
        'label': r'Dataset$_{\mathrm{NMQC}}$',  
        'color': COLORS['NLS+QC'],
        'marker': 'o', # 圆形
        'linestyle': '-',
        'linewidth': 2.0 # 主方法线条稍微加粗
    },
]

X_LABEL = r'$\nu_i$'# 英文标签更通用，也可改回中文
Y_LABEL = r'Std. Dev. of Decay Time $\sigma_{\tau}$ ($\mu s$)'

# ================= 核心：极简学术风格设置 =================
def set_science_style():
    """配置类似 Science/Nature 的极简风格"""
    
    sys = platform.system()
    if sys == "Windows":
        font_list = ['Arial', 'SimHei', 'Microsoft YaHei']
    else:
        font_list = ['Arial', 'Helvetica', 'DejaVu Sans']
        
    config = {
        "font.family": "serif",
        "font.sans-serif": ['Times New Roman'],
        "mathtext.fontset": "stixsans", 
        
        "figure.figsize": (10, 6),   
        "figure.dpi": 300,
        
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "axes.linewidth": 1.2,       
        "axes.spines.top": True,    
        "axes.spines.right": True,  
        
        "xtick.direction": 'out',    
        "ytick.direction": 'out',
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        
        "legend.fontsize": 16,       # 稍微调小一点以适应3个图例
        "legend.frameon": False,     
        "legend.loc": "upper right",
        
        "grid.linestyle": ':',       
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
        # 生成模拟收敛数据 (3列)
        x_sim = np.arange(1, 151)
        y_ols = 0.5 + 0.5 * np.exp(-x_sim/50) + np.random.normal(0, 0.05, 150)
        y_robust = 0.3 + 0.3 * np.exp(-x_sim/50) + np.random.normal(0, 0.03, 150) # 中间性能
        y_wls = 0.2 + 0.1 * np.exp(-x_sim/50) + np.random.normal(0, 0.01, 150)
        df = pd.DataFrame({
            'tau_std_OLS_QC': y_ols, 
            'tau_std_robust_QC': y_robust,
            'tau_std_WLS_QC': y_wls
        })

    # --- 只截取最后 100 行数据 ---
    if len(df) > 100:
        df = df.tail(100).reset_index(drop=True)
    
    # 2. 创建画布
    fig, ax = plt.subplots()
    
    # x 轴重置为 1 到 N
    x = np.arange(1, len(df) + 1)
    
    # 3. 绘制主图线条

    # 用于计算填充范围
    y_data_list = []

    for item in COLUMNS_TO_PLOT:
        col = item['col']
        if col not in df.columns: 
            print(f"警告: 列名 {col} 不存在于数据中，跳过绘制。")
            continue
        
        y_data = df[col]
        y_data_list.append(y_data)

        
        # 绘制标记
        ax.plot(x, y_data, 
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

    # 4. 【修改】添加填充区域 (填充最大值和最小值之间，显示性能差异范围)
    if len(y_data_list) >= 2:
        # 计算所有绘制曲线的逐点最大值和最小值
        y_min_all = np.min(y_data_list, axis=0)
        y_max_all = np.max(y_data_list, axis=0)
        
        # ax.fill_between(x, y_min_all, y_max_all, 
        #                 color=COLORS['FILL'], alpha=0.1, label='Performance Spread')

    # ================= 新增：标注第35和74组数据 =================
    highlight_indices = [35, 51]
    
    # 确保有数据才绘制
    if y_data_list:
        # 获取整体数据的最大值，用于确定标签高度
        all_y_values = np.concatenate(y_data_list)
        y_max_global = all_y_values.max()
        
        for idx in highlight_indices:
            if 1 <= idx <= len(df):
                x_pt = idx
                
                # 1. 绘制垂直竖线 (贯穿全图，灰色虚线，表示截面对比)
                ax.axvline(x=x_pt, color='#666666', linestyle='--', linewidth=1.0, alpha=0.6, zorder=5)
                
                # 2. 在顶部添加标签 (带半透明白色背景，防止文字与网格线混淆)
                ax.text(x_pt, y_max_global * 1.02, f'#{idx}', 
                        ha='left', va='bottom',
                        fontsize=12, fontweight='bold', color='#333333',
                        # bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5),
                        zorder=25)
                
                # 3. 高亮该位置的所有数据点 (强调三组数据的对比)
                # for item in COLUMNS_TO_PLOT:
                #     col = item['col']
                #     if col in df.columns:
                #         y_val = df[col].iloc[idx-1] # df索引从0开始
                        
                #         # 在交叉点绘制高亮圈 (颜色与曲线一致，但加粗空心)
                #         ax.plot(x_pt, y_val, 'o', 
                #                 ms=9,              # 标记大小
                #                 mfc='none',        # 空心填充
                #                 mec=item['color'], # 边框颜色跟随曲线
                #                 mew=2.0,           # 边框加粗
                #                 zorder=20)

    # 5. 添加局部放大图
    if show_inset:
        zoom_idx_start = int(len(df) * 0.75)
        
        axins = inset_axes(ax, width="35%", height="30%", loc='center right', 
                           bbox_to_anchor=(0, 0.1, 1, 1), bbox_transform=ax.transAxes)
        
        # 收集放大区域的数据用于计算 Y 轴范围
        zoom_y_values = []

        for item in COLUMNS_TO_PLOT:
            col = item['col']
            if col not in df.columns: continue
            
            # 小图绘制
            axins.plot(x, df[col], 
                       color=item['color'], 
                       linestyle=item['linestyle'],
                       linewidth=1.5,
                       marker=item['marker'],    # 小图也加标记
                       markersize=5,             # 小图标记稍微小一点
                       markeredgecolor='white',
                       markeredgewidth=0.5)
            
            # 收集数据
            zoom_y_values.append(df[col].iloc[zoom_idx_start:])

        # 设置小图 X 轴范围
        axins.set_xlim(x[zoom_idx_start], x[-1])
        
        # 【修改】动态计算小图 Y 轴范围 (基于所有曲线)
        if zoom_y_values:
            # 计算所有曲线在放大区域的全局最小值和最大值
            y_min_zoom = min([s.min() for s in zoom_y_values])
            y_max_zoom = max([s.max() for s in zoom_y_values])
            
            margin_zoom = (y_max_zoom - y_min_zoom) * 0.2
            # 防止 max 和 min 相等导致报错
            if margin_zoom == 0: margin_zoom = 0.01 
            
            axins.set_ylim(y_min_zoom - margin_zoom, y_max_zoom + margin_zoom)
        
        # 美化小图
        axins.tick_params(labelsize=10)
        axins.grid(True, linestyle=':', alpha=0.3)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')

    # 6. 装饰主图
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    fixed_ticks = [0, 20, 40, 60, 80, 100]
    ax.set_xticks(fixed_ticks)
    ax.grid(True, axis='y')
    
    # 坐标范围
    if y_data_list:
        all_y = np.concatenate(y_data_list)
        ax.set_ylim(0, all_y.max() * 1.15) # 稍微留多一点顶部空间给图例
    ax.set_xlim(0, len(df) * 1.02)

    # 图例
    ax.legend(loc='upper right', frameon=False, ncol=1, fontsize=18)

    # 7. 保存
    plt.tight_layout()
    suffix = "_Zoom" if show_inset else ""
    save_path_png = CSV_FILE_PATH.parent / f'Elegant_Comparison_3Methods{suffix}.png'
    save_path_pdf = CSV_FILE_PATH.parent / f'Elegant_Comparison_3Methods{suffix}.pdf'
    
    plt.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
    print(f"图表已保存至: {save_path_png}")# filepath: e:\pythonProject_crds\Trans_filter\essaycode\plotfile2.py
    print(f"图表已保存至: {save_path_pdf}")

if __name__ == "__main__":
    plot_elegant_chart(False)