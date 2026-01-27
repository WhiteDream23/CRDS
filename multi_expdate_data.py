import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# --- 以下部分与您的原代码相同 ---

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def natural_sort_key(s):
    # 从字符串中提取数字并转换为整数，用于自然排序
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def process_txt_files(folder_path, output_folder="processed_data", column_index=2, start=0, max_length=800,
                      batch_size=250):
    """
    处理文件夹中的所有txt文件，提取指定列数据并保存，每batch_size个文件保存到一个npz文件

    参数:
        folder_path: 包含txt文件的文件夹路径
        output_folder: 输出文件夹路径
        column_index: 要提取的列索引（从0开始）
        max_length: 每个文件提取的最大数据点数
        batch_size: 每个npz文件包含的txt文件数量
    """
    # 获取所有txt文件
    txt_files = glob(os.path.join(folder_path, "*.txt"))
    if not txt_files:
        print(f"警告: 在 '{folder_path}' 中未找到任何 .txt 文件。")
        return

    txt_files.sort(key=natural_sort_key)
    print(f"在 '{folder_path}' 中找到 {len(txt_files)} 个 .txt 文件")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹: {output_folder}")

    all_series = []
    filenames = []
    batch_count = 0

    for i, file_path in enumerate(txt_files):
        try:
            # 使用pandas读取文件（空格/制表符分隔）
            df = pd.read_csv(file_path, sep='\s+', header=None, engine='python')

            # 提取指定列
            if column_index < len(df.columns):
                series = df.iloc[:, column_index].values

                # 截取指定长度
                if len(series) >= start + max_length:
                    series = series[start:start + max_length]
                    all_series.append(series)
                    filenames.append(os.path.basename(file_path))
                    # 为了避免刷屏，可以注释掉下面这行
                    # print(f"处理文件: {os.path.basename(file_path)}, 数据长度: {len(series)}")
                else:
                    print(f"警告: {os.path.basename(file_path)} 数据点不足 {start + max_length} 个，已跳过")
            else:
                print(f"错误: {os.path.basename(file_path)} 列索引 {column_index} 超出范围")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

        # 每batch_size个文件保存一次，或者在处理完最后一个文件时保存
        is_last_file = (i + 1) == len(txt_files)
        if all_series and ((i + 1) % batch_size == 0 or is_last_file):
            batch_count += 1
            output_file = os.path.join(output_folder, f"processed_data_batch_{batch_count}.npz")
            data_array = np.array(all_series)
            np.savez(output_file, data=data_array, filenames=filenames)
            print(f"-> 已保存 {len(all_series)} 个数据序列到 {output_file}")

            # 清空当前批次数据
            all_series = []
            filenames = []

    print(f"文件夹 '{folder_path}' 处理完成。")


# --- 主要改动部分 ---

if __name__ == "__main__":
    # --- 1. 用户配置 ---
    # 源数据文件夹路径
    source_folder_path = r"F:\rawdatanew\11\data11"

    # 所有输出文件夹的根路径
    base_output_path = r"F:\rawdatanew\11processed_data"

    # 要提取的列的索引 (0代表第一列)
    column_to_extract = 2

    # 数据截取的起始位置
    start_index = 0

    # 每个 .npz 文件包含的源文件数量
    files_per_batch = 300

    # --- 2. 循环处理不同长度 ---
    # 使用 range(起始, 结束+1, 步长) 来生成所需长度序列
    # for length in range(2000, 18001, 2000):
    #     print(f"\n{'=' * 60}")
    #     print(f"开始处理: 数据长度 = {length}")
    #     print(f"{'=' * 60}")
    #
    #     # 1. 根据当前长度创建特定的输出文件夹名称
    #     #    例如: processed_len_2000, processed_len_4000, ...
    #     current_output_folder_name = f"processed_len2_{length}"
    #     current_output_folder_path = os.path.join(base_output_path, current_output_folder_name)
    #
    #     # 2. 调用核心函数，传入当前循环的参数
    #     process_txt_files(
    #         folder_path=source_folder_path,
    #         output_folder=current_output_folder_path,
    #         column_index=column_to_extract,
    #         start=start_index,
    #         max_length=length,
    #         batch_size=files_per_batch
    #     )
    current_output_folder_name = f"processed_len18000"
    current_output_folder_path = os.path.join(base_output_path, current_output_folder_name)

    # 2. 调用核心函数，传入当前循环的参数
    process_txt_files(
        folder_path=source_folder_path,
        output_folder=current_output_folder_path,
        column_index=column_to_extract,
        start=start_index,
        max_length=18000,
        batch_size=files_per_batch
    )

    print(f"\n{'*' * 60}")
    print("所有任务已完成！")
    print(f"{'*' * 60}")