"""
文件名: noise_allan_0714.py
用途: 使用Allen偏差方法分析时间序列数据的噪声特性
功能描述:
    - 计算数据集中每段信号的Allen偏差
    - 对噪声进行频域和时域分析
    - 生成Allen偏差的log-log图表
    - 用于评估数据的稳定性和噪声水平
"""

import numpy as np
import allantools
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_allan_variance(residuals, fs, plot_path, filename):
    """
    基于 residuals（等间隔时间采样）计算 Allan 偏差，并保存 log‑log 图
    residuals: 1D numpy array
    fs: 采样频率 (Hz)
    """
    # fractional frequency data: normalize residuals
    y = residuals - np.mean(residuals)
    # 若 residuals 为幅度变化，可以当作 fractional frequency 或直接用于 adev
    taus, adevs, errors, ns = allantools.oadev(y, rate=fs)
    # 保存图
    plt.figure()
    plt.loglog(taus, adevs, marker='o')
    plt.xlabel(r'$\tau$ (s)')
    plt.ylabel('Allan deviation')
    plt.title(f'Allan Deviation ‑ {filename}')
    plt.grid(True, which='both')
    plt.savefig(plot_path / f"{filename}_allan.png")
    plt.close()
    return taus, adevs

def analyze_dataset_noise(data_folder, output_folder, sequence_length, t_end, chunk_size):
    """
    分析数据集中的噪声，计算每段的 Allan 偏差，并保存结果。
    """
    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_folder.glob("*.npz"))
    if not npz_files:
        print(f"错误：在 '{data_folder}' 目录中未找到任何 .npz 文件。")
        return

    for npz_file in npz_files:
        data = np.load(npz_file)
        signal_data = data.get('data', data.get('arr_0'))
        if signal_data.ndim == 1:
            signal_data = [signal_data]

        fs = len(signal_data[0]) / t_end  # 采样频率
        filename_prefix = npz_file.stem

        for i, signal in enumerate(signal_data):

            # 计算残差（假设残差为原始信号减去其均值）
            residuals = signal - np.mean(signal)

            # 分析 Allan 偏差
            taus, adevs = analyze_allan_variance(residuals, fs, output_folder, f"{filename_prefix}_segment_{i}_chunk_{j}")

if __name__ == "__main__":
    BASE_DATA_FOLDER = Path(r"C:\Users\Mingkai\Desktop\rawdata\processed1_15000")
    OUTPUT_FOLDER = Path(r"processed\parameter_analysis_4data1_15000denoise_anllan0714")

    analyze_dataset_noise(
        data_folder=BASE_DATA_FOLDER,
        output_folder=OUTPUT_FOLDER,
        sequence_length=15000,
        t_end=1500,
        chunk_size=500  # 您指定的每段点数
    )