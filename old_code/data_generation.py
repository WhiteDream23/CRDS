import numpy as np
import os
def generate_ringdown_signal(tau, maxtau=0,noise_mean=0, noise_std=0.002, num_points=800):
    # 生成时间轴
    t = np.linspace(0, 120, num_points,dtype=np.float32)  # 时长设置为10倍maxτ
    # 生成衰减信号
    signal = np.exp(-t / tau, dtype=np.float32)
    # 添加高斯噪声
    noise = np.random.normal(noise_mean, noise_std, num_points).astype(np.float32)
    noisy_signal = signal + noise
    return noisy_signal, tau


def create_and_save_dataset(temp_foldername="temp",num_samples=1000, num_signals_per_tau=1000, noise_min=0.002,noise_max=0.008, chunk_size=1000,tau_min=5,tau_max=25,num_point=800):
    """
    分批生成并保存数据，避免内存爆炸
    """
    # 预计算总样本数
    total_samples = num_samples * num_signals_per_tau

    # 创建临时目录存储分批数据
    os.makedirs(temp_foldername, exist_ok=True)

    # 分批生成
    for chunk_idx in range(0, total_samples, chunk_size):
        chunk_signals = []
        chunk_taus = []

        # 生成当前分块的数据
        # tau = np.random.uniform(tau_min, tau_max, size=1).astype(np.float32)
        for _ in range(chunk_size):
            if len(chunk_signals) >= total_samples:
                break
            tau = np.random.uniform(tau_min, tau_max,size=1).astype(np.float32)
            noise=np.random.uniform(noise_min, noise_max,size=1)
            signal, tau_val = generate_ringdown_signal(tau,maxtau=tau_max,noise_std=noise,num_points=num_point)
            chunk_signals.append(signal)
            chunk_taus.append(tau_val)

        # 保存当前分块到临时文件
        np.savez_compressed(
            f"{temp_foldername}/chunk_{chunk_idx}.npz",
            signals=chunk_signals,
            taus=chunk_taus
        )
        del chunk_signals, chunk_taus  # 立即释放内存

# 示例调用（参数调整为更合理的值）
create_and_save_dataset(
    temp_foldername="temp_chunks_10-20_1200_0416_0.005",
    num_samples=1000,
    num_signals_per_tau=1000,
    num_point=1200,
    noise_min=0.005,
    noise_max=0.005,
    tau_min=10,
    tau_max=20,
    chunk_size=1000
)
