import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import t as student_t,norm
import math

# --- Matplotlib 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 1. 您分析和仿真工作中用到的所有辅助函数 ---
class AugmentedDecayDataset(Dataset):
    def __init__(self, num_samples, sequence_length, decay_range, amplitude_range,
                 baseline_range, noise_std_clean, noise_std_noisy, augmentations=None):
        # ... (类的__init__部分与您提供的一致)
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.decay_range = decay_range
        self.tau_min = decay_range[0]
        self.tau_max = decay_range[1]
        self.amplitude_range = amplitude_range
        self.baseline_range = baseline_range
        self.noise_std_clean = noise_std_clean
        self.noise_std_noisy = noise_std_noisy
        self.augmentations = augmentations if augmentations is not None else {}
        self.data = self._generate_data()
        print(f"Generated Augmented Dataset with {num_samples} samples. Augmentations: {self.augmentations}")

    def _generate_data(self):
        # ... (类的_generate_data部分与您提供的一致)
        samples = []
        time_points = np.linspace(0, self.sequence_length / 10, self.sequence_length, dtype=np.float32)

        for _ in tqdm(range(self.num_samples), desc="Generating Augmented Data"):
            tau = np.random.uniform(*self.decay_range)
            amplitude = np.random.uniform(*self.amplitude_range)
            baseline_val = np.random.uniform(*self.baseline_range)
            # ... (其他增强)
            baseline = baseline_val
            t_axis_augmented = time_points

            ideal_decay_curve = amplitude * np.exp(-t_axis_augmented / tau) + baseline

            if self.augmentations.get('realistic_shot_noise', False):
                df = self.augmentations.get('shot_noise_df', 5)
                # 注意：这里我们假设'clean'是高斯噪声，'noisy'是逼真噪声来进行对比
                base_gamma = np.random.uniform(*self.noise_std_noisy)
                noise_level = 1.5*base_gamma
                total_noise = generate_realistic_noise(
                    clean_signal=ideal_decay_curve,
                    noise_level=noise_level,
                    degrees_of_freedom=df
                )
                # gaussian_noise, realistic_noise = setup_fair_comparison(ideal_decay_curve, self.sequence_length,
                #                                                         base_sigma, df)

                clean_decay_curve = ideal_decay_curve + np.random.normal(0, base_gamma,
                                                                         self.sequence_length)
                noisy_decay_curve = clean_decay_curve + total_noise
            else:
                clean_decay_curve = ideal_decay_curve + np.random.normal(0, np.random.uniform(*self.noise_std_clean),
                                                                         self.sequence_length)
                noisy_decay_curve = clean_decay_curve + np.random.normal(0, np.random.uniform(*self.noise_std_noisy),
                                                                         self.sequence_length)

            # ... (其他增强)

            if self.tau_max != self.tau_min:
                tau_norm = (tau - self.tau_min) / (self.tau_max - self.tau_min)
            else:
                tau_norm = tau

            samples.append({
                'noisy_spectrum': torch.FloatTensor(noisy_decay_curve).unsqueeze(0),
                'clean_spectrum': torch.FloatTensor(clean_decay_curve).unsqueeze(0),
                'raw_spectrum': torch.FloatTensor(ideal_decay_curve).unsqueeze(0),
                'true_concentration': torch.FloatTensor([tau]),
                "tau_norm": torch.FloatTensor([tau_norm])
            })
        return samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

def generate_realistic_noise(clean_signal, noise_level, degrees_of_freedom):
    """
    生成一个具有重尾分布特性的、信号依赖的散粒噪声。

    Args:
        clean_signal (np.array): 不含噪声的理想信号曲线。
        noise_level (float): 噪声的总体强度系数。
        degrees_of_freedom (int): 学生t分布的自由度，越小尾部越重。推荐值为 4-7。

    Returns:
        np.array: 生成的仿真噪声。
    """
    if degrees_of_freedom <= 2:
        raise ValueError("自由度必须大于2以保证方差有限。")

    n_points = len(clean_signal)
    # 从学生t分布生成基础“重尾”随机数
    heavy_tailed_randoms = student_t.rvs(df=degrees_of_freedom, size=n_points)

    # 将其标准化，使其标准差为1
    std_t = np.sqrt(degrees_of_freedom / (degrees_of_freedom - 2))
    normalized_randoms = heavy_tailed_randoms / std_t

    # 施加信号依赖性（噪声标准差与信号强度的平方根成正比）
    signal_dependency = np.sqrt(np.maximum(clean_signal, 0))

    # 结合总体强度系数，生成最终噪声
    final_noise = normalized_randoms * signal_dependency * noise_level

    return final_noise



def pink_noise(N):
    """生成 1/f 噪声"""
    uneven = N % 2
    X = np.random.randn(N//2 + 1 + uneven) + 1j * np.random.randn(N//2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)  # 1/f
    y = (np.fft.irfft(X/S)).real
    if uneven:
        y = y[:-1]
    return y / np.max(np.abs(y))


def setup_fair_comparison(ideal_curve, seq_len, base_sigma, df):
    """
    生成总能量（方差）相等的高斯噪声和逼真噪声。
    """
    # 1. 生成基准高斯噪声，并计算其方差作为我们的目标
    gaussian_noise = np.random.normal(0, base_sigma, seq_len)
    target_variance = base_sigma**1.5

    # 2. 生成一个“单位强度”的逼真噪声作为缩放的原材料
    # 注意：这里的 noise_level 设置为 1.0
    base_realistic_noise = generate_realistic_noise(
        clean_signal=ideal_curve,
        noise_level=1.0,
        degrees_of_freedom=df
    )

    # 3. 计算“原材料”的当前方差
    # 为防止除以零，增加一个微小的数
    base_variance = np.var(base_realistic_noise)
    if base_variance < 1e-15:
        # 如果方差过小，说明生成有问题，直接返回高斯噪声以免后续出错
        return gaussian_noise, gaussian_noise

        # 4. 计算缩放因子
    # 因为 Var(k*X) = k^2 * Var(X)，所以 k = sqrt(Var_target / Var_current)
    scaling_factor = np.sqrt(target_variance / base_variance)

    # 5. 直接对“原材料”进行缩放，得到最终能量匹配的噪声
    realistic_noise_matched = base_realistic_noise * scaling_factor

    return gaussian_noise, realistic_noise_matched


def plot_variance_vs_intensity_subplot(ax, signal, noise, label, color):
    try:
        binner = np.linspace(np.min(signal), np.max(signal), num=20)
        bin_centers = (binner[:-1] + binner[1:]) / 2
        bin_variances = [np.var(noise[np.where((signal >= b_start) & (signal < b_end))]) if len(
            np.where((signal >= b_start) & (signal < b_end))[0]) > 10 else np.nan for b_start, b_end in
                         zip(binner[:-1], binner[1:])]
        ax.plot(bin_centers, bin_variances, 'o-', label=label, color=color)
        ax.set_xlabel("理想信号强度")
        ax.set_ylabel("噪声方差")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    except Exception as e:
        ax.text(0.5, 0.5, f'绘图出错:\n{e}', ha='center', va='center')


# 图1：主对比图
def plot_main_comparison(sample, sample_index, output_folder):
    noisy_spec, clean_spec, raw_spec = [sample[key].squeeze().numpy() for key in
                                        ['noisy_spectrum', 'clean_spectrum', 'raw_spectrum']]
    noisy_noise, clean_noise = noisy_spec - raw_spec, clean_spec - raw_spec
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'样本 #{sample_index} 主对比图', fontsize=20)

    # --- 左上角：时域信号对比 ---
    axes[0, 0].plot(raw_spec, 'k--', label='Raw (无噪)')
    axes[0, 0].plot(clean_spec, label='Clean Spectrum', color='deepskyblue', alpha=0.8)
    axes[0, 0].plot(noisy_spec, label='Noisy Spectrum', color='red', alpha=0.7)
    axes[0, 0].set_title('1. 信号时域对比');
    axes[0, 0].legend();
    axes[0, 0].grid(True)

    # --- 右上角：残差(噪声)对比 (核心修改点) ---
    # 1. 在这里计算两种残差的标准差
    std_clean = np.std(clean_noise)
    std_noisy = np.std(noisy_noise)

    # 2. 将计算出的标准差添加到图例(label)中
    axes[0, 1].plot(clean_noise, label=f'"Clean" Noise (σ={std_clean:.5f})', color='deepskyblue', alpha=0.8)
    axes[0, 1].plot(noisy_noise, label=f'"Noisy" Noise (σ={std_noisy:.5f})', color='red', alpha=0.6)
    axes[0, 1].axhline(0, color='black', linestyle='--');
    axes[0, 1].set_title('2. 残差（噪声）对比');
    axes[0, 1].legend();
    axes[0, 1].grid(True)

    # --- 左下角和右下角：噪声特性分析 ---
    axes[1, 0].set_title('3. "Clean" 噪声特性');
    plot_variance_vs_intensity_subplot(axes[1, 0], raw_spec, clean_noise, 'Clean Noise 方差', 'deepskyblue')
    axes[1, 1].set_title('4. "Noisy" 噪声特性');
    plot_variance_vs_intensity_subplot(axes[1, 1], raw_spec, noisy_noise, 'Noisy Noise 方差', 'red')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = output_folder / f"sample_{sample_index}_main_comparison.png"
    plt.savefig(output_path);
    plt.close(fig)
    print(f"已保存样本 {sample_index} 的主对比图至: {output_path}")


# 图2：总残差直方图
def plot_residual_histograms_comparison(clean_noise, noisy_noise, sample_index, output_folder):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'样本 #{sample_index} 总残差直方图对比', fontsize=16)
    for ax, data, label, color in zip(axes, [clean_noise, noisy_noise], ['Clean Noise', 'Noisy Noise'],
                                      ['deepskyblue', 'red']):
        ax.hist(data, bins=100, density=True, color=color, alpha=0.7, label='实际分布')
        mu, std = norm.fit(data)
        x = np.linspace(*ax.get_xlim(), 100)
        ax.plot(x, norm.pdf(x, mu, std), 'k--', label='高斯拟合')
        ax.set_title(f'{label} (μ={mu:.4f}, σ={std:.4f})')
        ax.set_xlabel('残差值');
        ax.set_ylabel('概率密度');
        ax.legend();
        ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    output_path = output_folder / f"sample_{sample_index}_histograms.png"
    plt.savefig(output_path);
    plt.close(fig)
    print(f"已保存样本 {sample_index} 的直方图对比图至: {output_path}")


# 图3：分段残差直方图
def plot_segmented_histograms(residuals, chunk_size, title_prefix, sample_index, output_folder):
    num_chunks = math.ceil(len(residuals) / chunk_size)
    ncols = 4
    nrows = math.ceil(num_chunks / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4), sharex=True, sharey=True)
    fig.suptitle(f'样本 #{sample_index} {title_prefix} - 分段残差直方图', fontsize=16)
    axes = np.array(axes).flatten()
    for i in range(num_chunks):
        ax = axes[i]
        segment = residuals[i * chunk_size:(i + 1) * chunk_size]
        ax.hist(segment, bins=50, density=True, color='gray', alpha=0.8)
        mu, std = norm.fit(segment)
        x = np.linspace(*ax.get_xlim(), 100)
        ax.plot(x, norm.pdf(x, mu, std), 'r-')
        ax.set_title(f'段 {i + 1} (σ={std:.4f})')
        if i >= (nrows - 1) * ncols: ax.set_xlabel('残差值')
        if i % ncols == 0: ax.set_ylabel('概率密度')
    for j in range(num_chunks, len(axes)): axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    output_path = output_folder / f"sample_{sample_index}_{title_prefix}_segmented_hist.png"
    plt.savefig(output_path);
    plt.close(fig)
    print(f"已保存样本 {sample_index} 的分段直方图至: {output_path}")


# 主控制器
def visualize_dataset_sample(sample, sample_index, chunk_size, output_folder):
    """为单个样本调用所有绘图函数，生成全套分析图表。"""
    noisy_spec, clean_spec, raw_spec = [sample[key].squeeze().numpy() for key in
                                        ['noisy_spectrum', 'clean_spectrum', 'raw_spectrum']]
    noisy_noise, clean_noise = noisy_spec - raw_spec, clean_spec - raw_spec

    # 生成图1：主对比图
    plot_main_comparison(sample, sample_index, output_folder)
    # 生成图2：总残差直方图
    plot_residual_histograms_comparison(clean_noise, noisy_noise, sample_index, output_folder)
    # 生成图3：两套分段残差图
    plot_segmented_histograms(clean_noise, chunk_size, 'Clean Noise', sample_index, output_folder)
    plot_segmented_histograms(noisy_noise, chunk_size, 'Noisy Noise', sample_index, output_folder)


# --- 2. 您提供的数据集类定义 ---
# (这是您提供的类，我只在__init__中加入了对torch和random库的引用)


# --- 3. 新增的可视化函数 ---
# --- 4. 主程序入口 ---
if __name__ == '__main__':
    # --- 参数设置 ---
    output_dir = Path("dataset_visualization2")
    output_dir.mkdir(exist_ok=True)

    num_samples_to_visualize = 10  # 您想看几个样本的对比图
    train_augmentations = {'baseline_drift': False, 'time_jitter': False, 'spike_noise': False,'pink_noise':False,'realistic_shot_noise': True,
                           'shot_noise_df': 5}
    chunk_size_for_hist = 1500
    dataset = AugmentedDecayDataset(
        num_samples=100,  # Use a separate validation set
        sequence_length=15000,
        decay_range=(140.0, 140.0),
        amplitude_range=(1.1, 1.3),
        baseline_range=(0.015, 0.025),
        noise_std_clean=(0.0001, 0.0005),
        noise_std_noisy=(0.0031, 0.0032),
        # noise_std_noisy=(0.0058, 0.006),
        augmentations=train_augmentations
        # decay_range=(140.0, 140.0),
        # amplitude_range=(0.8, 1.7),
        # baseline_range=(0.01, 0.035),
        # noise_std_clean=(0.0001, 0.0005),
        # noise_std_noisy=(0.003, 0.008)
        # decay_range=(110.0, 170.0),
        # amplitude_range=(1.2, 1.2),
        # baseline_range=(0.02, 0.02),
        # noise_std_clean=(0.0001, 0.0005),
        # noise_std_noisy=(0.003, 0.003)
    )

    # --- 随机挑选样本并生成对比图 ---
    if len(dataset) < num_samples_to_visualize:
        num_samples_to_visualize = len(dataset)

    selected_indices = random.sample(range(len(dataset)), num_samples_to_visualize)

    for idx in selected_indices:
        sample_data = dataset[idx]
        visualize_dataset_sample(sample_data, idx, chunk_size_for_hist, output_dir)