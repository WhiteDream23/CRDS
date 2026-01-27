import os
import re
import numpy as np
import pandas as pd
from glob import glob


def natural_sort_key(s):
    """
    辅助函数，用于对包含数字的文件名进行自然排序。
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def process_and_sample_files(folder_path, output_folder, column_index, initial_data_length, sampling_step, batch_size):
    """
    处理并降采样文件夹中的所有 .txt 文件。

    参数:
        folder_path (str): 包含 .txt 文件的源文件夹路径。
        output_folder (str): 输出文件夹路径。
        column_index (int): 要提取的列索引（从0开始）。
        initial_data_length (int): 从每个文件中提取的初始数据点数量（例如 18000）。
        sampling_step (int): 采样步长/频率（例如 2 表示每隔一个点取一个）。
        batch_size (int): 每个 .npz 文件包含的源文件数量。
    """
    # 获取所有 .txt 文件并进行排序
    txt_files = glob(os.path.join(folder_path, "*.txt"))
    if not txt_files:
        print(f"警告: 在 '{folder_path}' 中未找到任何 .txt 文件。")
        return

    txt_files.sort(key=natural_sort_key)
    print(f"在 '{folder_path}' 中找到 {len(txt_files)} 个 .txt 文件。")

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹: {output_folder}")

    all_sampled_series = []
    filenames = []
    batch_count = 0
    processed_count = 0

    for i, file_path in enumerate(txt_files):
        try:
            # 使用 pandas 读取文件
            df = pd.read_csv(file_path, sep='\s+', header=None, engine='python')

            # 检查列索引是否有效
            if column_index < len(df.columns):
                series = df.iloc[:, column_index].values

                # 检查数据长度是否足够
                if len(series) >= initial_data_length:
                    # 1. 先截取前 18000 个点
                    initial_series = series[:initial_data_length]

                    # 2. 按指定的步长进行降采样
                    sampled_series = initial_series[::sampling_step]

                    all_sampled_series.append(sampled_series)
                    filenames.append(os.path.basename(file_path))
                    processed_count += 1
                else:
                    print(f"警告: {os.path.basename(file_path)} 数据点不足 {initial_data_length} 个，已跳过。")
            else:
                print(f"错误: {os.path.basename(file_path)} 列索引 {column_index} 超出范围。")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

        # 每 batch_size 个文件保存一次，或在处理完最后一个文件时保存
        is_last_file = (i + 1) == len(txt_files)
        if all_sampled_series and ((i + 1) % batch_size == 0 or is_last_file):
            batch_count += 1
            output_file = os.path.join(output_folder, f"resampled_data_batch_{batch_count}.npz")
            data_array = np.array(all_sampled_series)
            np.savez(output_file, data=data_array, filenames=filenames)
            print(f"-> 已保存 {len(all_sampled_series)} 个数据序列到 {output_file} (总处理数: {processed_count})")

            # 清空当前批次数据
            all_sampled_series = []
            filenames = []

    print(f"文件夹 '{folder_path}' 处理完成。共处理了 {processed_count} 个符合条件的文件。")


if __name__ == "__main__":
    # --- 1. 用户配置 ---
    # 源数据文件夹路径
    source_folder_path = r"F:\rawdatanew\9\data9"

    # 所有输出文件夹的根路径
    base_output_path = r"F:\rawdatanew\9processed_data"

    # 要提取的列的索引 (0代表第一列)
    column_to_extract = 2

    # 从每个文件中提取的初始数据点数量
    points_to_process = 18000

    # 每个 .npz 文件包含的源文件数量
    files_per_batch = 300

    # --- 2. 循环处理不同采样频率 ---
    # 使用 range(2, 10) 来生成采样频率 2, 3, ..., 9
    for freq in range(2, 10):
        output_data_length = points_to_process // freq

        print(f"\n{'=' * 60}")
        print(f"开始处理: 采样频率 = {freq} (生成数据长度: {output_data_length})")
        print(f"{'=' * 60}")

        # 1. 根据当前频率创建特定的输出文件夹名称
        #    例如: resampled_freq_2, resampled_freq_3, ...
        current_output_folder_name = f"7resampled_len_freq_{freq}"
        current_output_folder_path = os.path.join(base_output_path, current_output_folder_name)

        # 2. 调用核心函数，传入当前循环的参数
        process_and_sample_files(
            folder_path=source_folder_path,
            output_folder=current_output_folder_path,
            column_index=column_to_extract,
            initial_data_length=points_to_process,
            sampling_step=freq,
            batch_size=files_per_batch
        )

    print(f"\n{'*' * 60}")
    print("所有降采样任务已完成！")
    print(f"{'*' * 60}")