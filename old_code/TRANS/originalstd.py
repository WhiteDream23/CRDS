"""
文件名: originalstd.py
用途: 从CSV数据文件中计算指定列的标准差,支持条件筛选
功能描述:
    - 读取原始CSV数据文件
    - 根据用户定义的条件对数据进行筛选
    - 计算指定列的标准差和其他统计量
    - 生成标准差分析报告
"""

import pandas as pd
import numpy as np

# --- 1. 用户配置 ---

# 1.1: 输入您的 CSV 文件名
file_name = '.\processed\original.csv'

# 1.2: 为您的数据列命名 (由于您的文件没有表头，我们需要手动指定名称)
#      根据您之前提供的数据，一共有 8 列
column_names = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8']

# 1.3: 指定您要计算标准差的目标列
target_column = 'col_5'  # 例如，我们想计算第 2 列 ('col_2') 的标准差


# --- 2. 定义筛选条件 (选择您需要的行) ---

# 这里是核心部分。您需要定义一个条件来筛选出您想要的行。
# 下面提供了三个常用示例，请根据您的需求选择一个或进行修改。
#
# 使用方法：取消您想用的那一行代码的注释 (删除前面的 # 号)。

def filter_data(df):
    """
    在此函数中定义筛选逻辑。
    :param df: 包含所有数据的 DataFrame
    :return: 筛选后的 DataFrame
    """

    # --- 示例 A: 根据某一列的数值范围进行筛选 ---
    #    例如，选择 'col_1' (第一列) 的值大于 -2.4 的所有行
    # condition = df['col_1'] > -2.4

    # --- 示例 B: 根据行的索引（行号）进行筛选 ---
    #    例如，只选择第 3 行到第 5 行（索引是从 0 开始的）
    condition1 = (df.index >= 61) & (df.index <= 100)
    condition2 = (df.index >= 24) & (df.index <= 40)
    condition2=0
    # --- 示例 C: 根据最后一列是否为 0 进行筛选 ---
    #    例如，选择 'col_8' (第八列) 的值不等于 0 的所有行
    # condition = df['col_8'] != 0
    condition =condition1 | condition2
    # 将您选择的条件应用到 DataFrame 上
    filtered_df = df[condition1]
    return filtered_df


# --- 3. 执行计算 (通常无需修改这部分) ---

try:
    # 使用 pandas 读取 CSV 文件
    # header=None 表示我们的文件没有标题行
    # names=column_names 表示我们手动为列指定名称
    df = pd.read_csv(file_name, header=None, names=column_names)

    print("成功读取文件，原始数据共有 {} 行。".format(len(df)))
    print("-" * 40)

    # 调用筛选函数获取特定行
    filtered_df = filter_data(df)

    # 检查筛选后是否还有数据
    if filtered_df.empty:
        print("警告：根据您设定的条件，没有筛选到任何数据行。")
    else:
        # 从筛选后的数据中，提取出我们想计算的目标列
        target_data = filtered_df[target_column]
        target_data*= 1e6
        # 使用 .std() 方法计算标准差
        # ddof=1（默认值）计算的是样本标准差，这是统计学中最常用的。
        # 如果需要计算总体标准差，请使用 std_dev = target_data.std(ddof=0)
        std_dev = target_data.std()

        # 打印结果
        print(f"筛选条件已应用。共有 {len(filtered_df)} 行数据满足条件。")
        print(f"目标计算列: '{target_column}'")
        print(f"这些数据的标准差 (Sample Standard Deviation) 是: {std_dev:.6f}")  # 格式化输出，保留6位小数

        # (可选) 打印筛选出的数据的前5行，方便核对
        # print("\n筛选出的数据 (前5行):")
        # print(filtered_df.head())

except FileNotFoundError:
    print(f"错误：找不到文件 '{file_name}'。请确认文件名是否正确，以及文件和脚本是否在同一个文件夹下。")
except KeyError as e:
    print(f"错误：找不到指定的列 {e}。")
    print(f"请检查 'target_column' 的设置是否正确。可用的列名有: {column_names}")
except Exception as e:
    print(f"发生了未知错误: {e}")