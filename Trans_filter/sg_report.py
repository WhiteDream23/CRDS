"""
文件名: sg_report.py
用途: 按行号对质量控制报告进行分组统计分析
功能描述:
    - 读取质量控制比较报告CSV文件
    - 根据用户明确指定的行号进行分组
    - 计算每组数据的统计量(均值、标准差等)
    - 对比"审查前"和"审查后"的数据差异
    - 生成详细的分组统计分析报告
"""

import pandas as pd
from pathlib import Path

def analyze_report_by_row_indices(report_csv_path: Path, groups_to_analyze: dict):
    """
    根据用户明确指定的行号（0-based index），对报告进行分组统计分析。

    Args:
        report_csv_path (Path): 指向 `quality_control_comparison_report.csv` 的路径。
        groups_to_analyze (dict): 一个字典，定义了要分析的组。
                                  键(key)是组的名称。
                                  值(value)是一个包含行号（整数）的列表。
    """
    try:
        df = pd.read_csv(report_csv_path)
        print(f"成功读取报告: {report_csv_path.name}，共包含 {len(df)} 行数据 (行号从 0 到 {len(df)-1})。")
    except FileNotFoundError:
        print(f"错误：报告文件不存在，请检查路径: {report_csv_path}")
        return
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    print("\n" + "="*50)

    # 遍历用户定义的每一个分析组
    for group_name, row_indices in groups_to_analyze.items():
        print(f"\n正在分析组: '{group_name}'")
        print(f"指定统计的行号为: {row_indices}")

        # 使用 .iloc[] 方法，通过整数索引精确地选择行
        try:
            group_df = df.iloc[row_indices]
        except IndexError:
            print(f" -> 错误：您提供的行号列表中，有索引超出了文件范围（0 到 {len(df)-1}）。已跳过此组。")
            print("="*50)
            continue
        except Exception as e:
            print(f" -> 选择行时出错: {e}。已跳过此组。")
            print("="*50)
            continue

        if len(group_df) < 2:
            print(f" -> 选定的行数不足2行，无法计算标准差。仅显示均值。")
        else:
            print(f" -> 已成功选定 {len(group_df)} 行数据，开始进行统计...")

        # 对筛选出的数据组，计算tau均值的批次间标准差
        # a) “审查前”的情况
        tau_means_before_qc = group_df['tau_mean_before_qc']
        mean_of_means_before = tau_means_before_qc.mean()
        std_of_means_before = tau_means_before_qc.std() if len(group_df) >= 2 else 0.0

        # b) “审查后”的情况
        tau_means_after_qc = group_df['tau_mean_after_qc']
        mean_of_means_after = tau_means_after_qc.mean()
        std_of_means_after = tau_means_after_qc.std() if len(group_df) >= 2 else 0.0

        # 打印该组的分析报告
        print("\n  --- 该组tau值稳定性分析报告 ---")
        print("  【审查前 (Before QC)】")
        print(f"    - tau均值的平均值: {mean_of_means_before:.4f}")
        print(f"    - tau均值的标准差: {std_of_means_before:.4f}")

        print("\n  【审查后 (After QC)】")
        print(f"    - tau均值的平均值: {mean_of_means_after:.4f}")
        print(f"    - tau均值的标准差: {std_of_means_after:.4f}")
        print("  ---------------------------------")
        print("="*50)
        print("\n所有组分析完成。")


if __name__ == "__main__":
    # --- 请修改以下参数 ---

    # 1. 指向您之前生成的CSV报告文件
    REPORT_CSV_PATH = Path("processed/qc_report_final2/quality_control_comparison_reportwls.csv")

    # --- 新增步骤：自动计算要统计的行号 ---
    try:
        # 步骤1：先读取文件以获取总行数
        report_df = pd.read_csv(REPORT_CSV_PATH)
        total_rows = len(report_df)
        print(f"报告文件共有 {total_rows} 行数据。")

        # 步骤2：生成最后四十行的索引列表
        # 为防止文件行数不足40，做一个简单的保护
        start_index = max(0, total_rows - 40)
        last_40_indices = list(range(start_index, total_rows))

        print(f"将自动统计最后40行，行号索引从 {start_index} 到 {total_rows - 1}。")

        # 步骤3：在这里定义您想分析的“组”
        # 我们将自动生成的行号列表放入字典中
        # manual_indices = list(range(24, 41))
        # last_40_indices.extend(manual_indices)
        GROUPS_TO_ANALYZE = {
            f"最后 {len(last_40_indices)} 行数据": last_40_indices,
            #"手动指定的行数据 (24-39行)": manual_indices,
            # 您仍然可以添加其他您想分析的组，例如：
            # "前5行数据": [0, 1, 2, 3, 4]
        }

        # 运行分析
        analyze_report_by_row_indices(
            report_csv_path=REPORT_CSV_PATH,
            groups_to_analyze=GROUPS_TO_ANALYZE
        )

    except FileNotFoundError:
        print(f"错误：报告文件未找到，请确认路径是否正确: {REPORT_CSV_PATH}")
    except Exception as e:
        print(f"分析过程中发生错误: {e}")