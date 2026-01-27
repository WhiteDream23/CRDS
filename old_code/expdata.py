import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def natural_sort_key(s):
    # 从字符串中提取数字并转换为整数，用于自然排序
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]
def process_txt_files(folder_path, output_folder="processed_data", column_index=2, start=0,max_length=800, batch_size=250):
    """
    处理文件夹中的所有txt文件，提取指定列数据并保存，每250个文件保存到一个npz文件

    参数:
        folder_path: 包含txt文件的文件夹路径
        output_folder: 输出文件夹路径
        column_index: 要提取的列索引（从0开始）
        max_length: 每个文件提取的最大数据点数
        batch_size: 每个npz文件包含的txt文件数量
    """
    # 获取所有txt文件
    txt_files = glob(os.path.join(folder_path, "*.txt"))
    txt_files.sort(key=natural_sort_key)
    print(f"找到{len(txt_files)}个TXT文件")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_series = []
    filenames = []
    batch_count = 0

    for i, file_path in enumerate(txt_files):
        try:
            # 使用pandas读取文件（空格/制表符分隔）
            df = pd.read_csv(file_path, sep='\s+', header=None)

            # 提取指定列
            if column_index < len(df.columns):
                series = df.iloc[:, column_index].values

                # 截取指定长度
                if len(series) >= max_length:
                    series = series[start:start+max_length]
                    all_series.append(series)
                    filenames.append(os.path.basename(file_path))
                    print(f"处理文件: {os.path.basename(file_path)}, 数据长度: {len(series)}")
                else:
                    print(f"警告: {os.path.basename(file_path)} 数据点不足 {max_length} 个，已跳过")
            else:
                print(f"错误: {os.path.basename(file_path)} 列索引超出范围")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

        # 每batch_size个文件保存一次
        if (i + 1) % batch_size == 0 or (i + 1) == len(txt_files):
            if all_series:
                batch_count += 1
                output_file = os.path.join(output_folder, f"processed_data_batch_{batch_count}.npz")
                data_array = np.array(all_series)
                np.savez(output_file, data=data_array, filenames=filenames)
                print(f"已保存 {len(all_series)} 个数据序列到 {output_file}")

                # 清空当前批次数据
                all_series = []
                filenames = []

    print("所有文件处理完成")

if __name__ == "__main__":
    # 请替换为文件夹路径
    folder_path = r"C:\Users\Mingkai\Desktop\rawdata\1\data"
    output_folder = r"C:\Users\Mingkai\Desktop\rawdata\processed1_15000"
    process_txt_files(folder_path, output_folder, column_index=2, start=0,max_length=15000, batch_size=150)