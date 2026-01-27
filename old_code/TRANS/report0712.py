"""
文件名: report0712.py
用途: 分析质量控制报告,按多个区间对数据进行统计分析
功能描述:
    - 读取质量控制报告CSV文件
    - 支持按行号范围选择数据子集
    - 计算每个条件下多个tau_mean列的统计量(均值、标准差)
    - 生成分组统计分析报告
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple


def analyze_multiple_subsets_std(csv_path: Path, intervals: List[Tuple[int, int]]):
    """
    读取CSV报告，对每个条件下指定的一个或多个行区间的数据，计算tau_mean的标准差。

    参数:
        csv_path (Path): CSV文件的路径。
        intervals (List[Tuple[int, int]]): 一个区间列表, e.g., [(10, 20), (50, 60)]。
    """
    if not csv_path.is_file():
        print(f"错误: 找不到CSV文件: {csv_path}")
        print("请确认路径和文件名是否正确。\n")
        return

    # 将区间格式化为字符串，用于打印输出
    intervals_str = ", ".join([f"{start}-{end}" for start, end in intervals])
    print(f"--- 正在分析文件: {csv_path.name} ---")
    print(f"计算每个条件下, 第 {intervals_str} 行 'tau_mean' 的标准差")
    print("-" * (40 + len(csv_path.name)))

    try:
        df = pd.read_csv(csv_path)
        conditions = sorted(df['condition'].unique())
        tau_mean_cols = [col for col in df.columns if 'tau_mean' in col]
        results = []

        for cond in conditions:
            condition_data = df[df['condition'] == cond].reset_index(drop=True)

            # --- 核心改动：处理多个区间 ---

            # 检查数据总行数是否满足最大区间的需求
            if not intervals:
                print("错误：未提供任何分析区间。")
                return
            max_required_row = max(end for start, end in intervals)
            if len(condition_data) < max_required_row:
                print(f"  - 条件 {cond}: 总行数 ({len(condition_data)}) 不足 {max_required_row}，已跳过。")
                continue

            # 提取每个区间的数据，并存入一个列表
            list_of_subsets = []
            for start, end in intervals:
                # 检查单个区间的合法性
                if start > end:
                    print(f"  - 警告: 区间 ({start}, {end}) 无效，已跳过。")
                    continue
                # 1-based to 0-based index: start-1. iloc的end是不包含的，所以正好
                subset_slice = condition_data.iloc[start - 1: end]
                list_of_subsets.append(subset_slice)

            # 如果没有有效的子集，则跳过
            if not list_of_subsets:
                continue

            # 将所有提取出的子集合并成一个大的DataFrame
            subset = pd.concat(list_of_subsets)
            # --- 核心改动结束 ---

            std_results = {'condition': cond}
            for col in tau_mean_cols:
                std_dev = subset[col].dropna().std()
                std_results[col.replace('mean', 'subset_std')] = std_dev
                std_mean = subset[col].dropna().mean()
                std_results[col.replace('mean', 'subset_mean')] = std_mean

            results.append(std_results)

        if not results:
            print("未能计算出任何结果。请检查CSV文件内容和行范围设置。")
            return

        result_df = pd.DataFrame(results).set_index('condition')
        print(result_df.to_string(float_format="%.6f"))
        print("\n")

    except Exception as e:
        print(f"处理文件时发生错误: {e}\n")


if __name__ == '__main__':
    # --- 用户配置 ---
    # 指定包含最终分析报告的根文件夹
    BASE_OUTPUT_FOLDER = Path("processed/fit_comparison_final2")

    # ★★★ 在这里定义您想分析的一个或多个行区间 ★★★
    # 区间格式: [(起始行号1, 结束行号1), (起始行号2, 结束行号2), ...]

    # 示例 1: 只分析 40-60行 (与之前脚本行为一致)
    # INTERVALS_TO_ANALYZE = [(40, 60)]

    # 示例 2: 分析 10-20行 和 50-60行 (多区间)
    # 若要使用此设置，请取消下面一行的注释，并注释掉上面一行
    #0-10/25-40/60-100
    INTERVALS_TO_ANALYZE = [(0, 10), (25, 40), (58, 97)]
    INTERVALS_TO_ANALYZE = [(58, 97)]
    # --- 脚本执行 ---
    # 定义两个CSV文件的具体路径
    # 注意：请根据您实际的文件夹名称进行调整
    csv_length_path = BASE_OUTPUT_FOLDER / "length_final" / "length_main_report_final.csv"
    csv_frequency_path = BASE_OUTPUT_FOLDER / "frequency_final" / "frequency_main_report_final.csv"

    # 执行分析
    analyze_multiple_subsets_std(csv_length_path, INTERVALS_TO_ANALYZE)
    analyze_multiple_subsets_std(csv_frequency_path, INTERVALS_TO_ANALYZE)