import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def process_txt_files(folder_path, output_file="processed_data_large.npz", column_index=2, max_length=800):
    """
    处理文件夹中的所有txt文件，提取指定列数据并保存

    参数:
        folder_path: 包含txt文件的文件夹路径
        output_file: 输出文件名
        column_index: 要提取的列索引（从0开始）
        max_length: 每个文件提取的最大数据点数
    """
    # 获取所有txt文件
    txt_files = glob(os.path.join(folder_path, "*.txt"))
    print(f"找到{len(txt_files)}个TXT文件")

    all_series = []
    filenames = []

    for file_path in txt_files:
        try:
            # 使用pandas读取文件（空格/制表符分隔）
            df = pd.read_csv(file_path, sep='\s+', header=None)

            # 提取指定列
            if column_index < len(df.columns):
                series = df.iloc[:, column_index].values

                # 截取指定长度
                if len(series) >= max_length:
                    series = series[:max_length]
                    all_series.append(series)
                    filenames.append(os.path.basename(file_path))
                    print(f"处理文件: {os.path.basename(file_path)}, 数据长度: {len(series)}")
                else:
                    print(f"警告: {os.path.basename(file_path)} 数据点不足 {max_length} 个，已跳过")
            else:
                print(f"错误: {os.path.basename(file_path)} 列索引超出范围")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    # 将所有序列保存为numpy数组
    if all_series:
        data_array = np.array(all_series)
        np.savez(output_file, data=data_array, filenames=filenames)
        print(f"已处理 {len(all_series)} 个数据序列并保存到 {output_file}")

        # 绘制前几个序列的图表以进行可视化检查
        plt.figure(figsize=(12, 6))
        for i in range(min(10, len(all_series))):
            plt.plot(all_series[i], label=f'序列 {i + 1}')
        plt.legend()
        plt.title('前几个数据序列可视化')
        plt.savefig('data_visualization.png')
        plt.show()

        return data_array, filenames
    else:
        print("没有找到可用的数据序列")
        return None, None

def generatestdtau(output_file="stdtau.npz"):
    all_signals = []
    all_taus = []
    for _ in range(20000):
        tau=np.random.uniform(12,13 ,size=1).astype(np.float32)
        t = np.linspace(0, 120, 1200,dtype=np.float32)  # 时长设置为10倍maxτ
        # 生成衰减信号
        signal = np.exp(-t / tau, dtype=np.float32)
        # 添加高斯噪声
        noiserange=np.random.uniform(0.005, 0.005,size=1).astype(np.float32)
        noise = np.random.normal(0, noiserange, 1200).astype(np.float32)
        noisy_signal = signal + noise
        all_signals.append(noisy_signal)
        all_taus.append(tau)
    np.savez(output_file, data=all_signals, targets=all_taus)
    return

if __name__ == "__main__":
    #请替换为文件夹路径
    # folder_path = r"C:\Users\Mingkai\Desktop\rawdata\test"
    # process_txt_files(folder_path, column_index=2,max_length=12000)
    generatestdtau(output_file="test.npz")