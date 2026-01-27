import pandas as pd
import numpy as np


def analyze_tau_columns(csv_file_path):
    """
    读取CSV文件，对列名中含有“Tau_mean”的列进行多项统计分析：
    1. 计算所有数据的最大值。
    2. 计算特定行(0-40, 60-90)的平均值和标准差。
    3. 计算最大值与特定行平均值的差值。
    """
    print(f"正在分析文件: {csv_file_path}\n")

    # 1. 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"错误: 文件未找到，请确认路径 '{csv_file_path}' 是否正确。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 2. 筛选出包含 "Tau_mean" 的列名
    target_columns = [col for col in df.columns if '平均值' in col]

    if not target_columns:
        print("错误: 在文件中未找到任何列名包含 '平均值' 的列。")
        return

    print(f"找到目标列: {target_columns}\n")

    # 3. 选择指定的行范围
    rows_part1 = df.iloc[19:35]  # 选取第 0 到 40 行
    rows_part2 = df.iloc[60:70]  # 选取第 60 到 90 行
    selected_rows_df = pd.concat([rows_part1, rows_part2])

    print("--- 计算结果 ---")
    # 4. 遍历目标列并进行所有计算
    for col in target_columns:
        # 从选定的行中提取数据系列，并排除空值
        selected_data = selected_rows_df[col].dropna()

        # 从整个DataFrame中提取当前列的所有数据，并排除空值
        full_column_data = df[col].dropna()

        if selected_data.empty or full_column_data.empty:
            print(f"对于列 '{col}': 没有足够的数据进行计算。\n" + "-" * 25)
            continue

        # --- 进行各项计算 ---
        # a. 计算选中行的标准差
        std_dev_selected = selected_data.std()

        # b. 新增: 计算选中行的平均值
        mean_selected = selected_data.mean()

        # c. 新增: 计算所有数据中的最大值
        max_value_full = full_column_data.max()

        # d. 新增: 计算差值
        diff_max_mean = max_value_full - mean_selected

        # e. 新增: 计算信噪比
        snr = diff_max_mean/ std_dev_selected if std_dev_selected != 0 else np.nan

        # --- 打印所有结果 ---
        print(f"对于列 '{col}':")
        print(f"  - 所有数据的最大值:              {max_value_full:.6f}")
        print(f"  - 选中行 (0-40, 60-90) 的平均值: {mean_selected:.6f}")
        print(f"  - 选中行 (0-40, 60-90) 的标准差: {std_dev_selected:.6f}")
        print(f"  - (最大值) - (选中行平均值) 的差值: {diff_max_mean:.6f}")
        print(f"  - 信噪比 (SNR):                 {snr:.6f}")
        print("-" * 25)  # 分隔符，让输出更清晰


# --- 主运行部分 ---
if __name__ == "__main__":
    # ##############################################################
    # ##  请在这里修改为您要分析的CSV文件的实际路径            ##
    # ##############################################################
    # 示例路径，请根据你的文件位置进行修改
    # 例如: "processed/parameter_analysis/raw_vs_filtered_summary.csv"

    csv_path = r"E:\pythonProject_crds\processed\real_data_comparison_output\detailed_results2_absorb_weiguiyi.csv"

    # 调用函数执行分析 (函数名已更新)
    analyze_tau_columns(csv_path)