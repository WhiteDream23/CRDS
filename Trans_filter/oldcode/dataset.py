import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# --- 拟合参数统计 ---
#
# 参数: I0
#   平均值: 1.392375
#   标准差: 0.091220
#   最小值: 0.838129
#   最大值: 1.631312
#   25%分位数: 1.358308
#   75%分位数: 1.454680
#
# 参数: Tau
#   平均值: 141.937299
#   标准差: 2.260737
#   最小值: 115.479326
#   最大值: 167.210479
#   25%分位数: 142.151541
#   75%分位数: 142.608097
#
# 参数: B
#   平均值: 0.024150
#   标准差: 0.000280
#   最小值: 0.014986
#   最大值: 0.030051
#   25%分位数: 0.024035
#   75%分位数: 0.024264
#
# 参数: Residuals_Std
#   平均值: 0.003474
#   标准差: 0.000538
#   最小值: 0.003128
#   最大值: 0.027086
#   25%分位数: 0.003390
#   75%分位数: 0.003496

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.stats import t as student_t # 导入学生t分布
from tqdm import tqdm
import random
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
    target_variance = base_sigma**2

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

class AugmentedDecayDataset(Dataset):
    """
    在原始DecayDataset基础上，加入了多种数据增强方法，包括新的逼真噪声模型。
    """

    def __init__(self, num_samples, sequence_length, decay_range, amplitude_range,
                 baseline_range, noise_std_clean, noise_std_noisy, tau_generation_config,augmentations=None):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.decay_range = decay_range
        self.tau_min = decay_range[0]
        self.tau_max = decay_range[1]
        self.amplitude_range = amplitude_range
        self.baseline_range = baseline_range
        self.noise_std_clean = noise_std_clean
        self.noise_std_noisy = noise_std_noisy
        self.tau_generation_config = tau_generation_config
        self.augmentations = augmentations if augmentations is not None else {}
        self.data = self._generate_data()
        print(f"Generated Augmented Dataset with {num_samples} samples. Augmentations: {self.augmentations}")

    def _generate_data(self):
        samples = []
        time_points = np.linspace(0, self.sequence_length / 10, self.sequence_length, dtype=np.float32)

        for _ in tqdm(range(self.num_samples), desc="Generating Augmented Data"):
            gen_mode = self.tau_generation_config['mode']

            if gen_mode == 'uniform':
                tau_range = self.tau_generation_config['range']
                tau = np.random.uniform(*tau_range)
            elif gen_mode == 'clustered':
                zones = self.tau_generation_config['zones']
                chosen_zone = random.choice(zones)
                tau = np.random.uniform(*chosen_zone)
            elif gen_mode == 'interleaved':  # 新增的交错模式
                ratio = self.tau_generation_config.get('ratio', 0.7)  # 困难样本的比例，默认为70%
                if np.random.rand() < ratio:
                    # 生成一个困难样本
                    zones = self.tau_generation_config['zones']
                    chosen_zone = random.choice(zones)
                    tau = np.random.uniform(*chosen_zone)
                else:
                    # 生成一个通识样本
                    general_range = self.tau_generation_config['general_range']
                    tau = np.random.uniform(*general_range)
            else:
                raise ValueError(...)

            amplitude = np.random.uniform(*self.amplitude_range)
            baseline_val = np.random.uniform(*self.baseline_range)

            # --- 1. 基线增强 ---
            if self.augmentations.get('baseline_drift', False):
                drift_slope = np.random.uniform(-5E-6, 5E-6)
                baseline = baseline_val + drift_slope * time_points
            else:
                baseline = baseline_val

            # --- 2. 时间轴增强 ---
            if self.augmentations.get('time_jitter', False):
                t_axis_augmented = time_points.copy()
                block_size = random.randint(20, 50)
                for start in range(0, self.sequence_length, block_size):
                    jitter_val = np.random.normal(0, 0.005)
                    end = min(start + block_size, self.sequence_length)
                    t_axis_augmented[start:end] += jitter_val
            else:
                t_axis_augmented = time_points

            ideal_decay_curve = amplitude * np.exp(-t_axis_augmented / tau) + baseline

            # --- 3. 噪声增强 (核心修改点) ---
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
                total_noise+=np.random.normal(0, base_gamma,self.sequence_length)
                clean_decay_curve = ideal_decay_curve + np.random.normal(0, base_gamma,
                                                                         self.sequence_length)
                # clean_decay_curve = ideal_decay_curve + np.random.normal(0,
                #                                                          np.random.uniform(*self.noise_std_clean),
                #                                                          self.sequence_length)
                noisy_decay_curve = ideal_decay_curve+total_noise

            else:
                # --- B) 保持原来的高斯噪声模型 ---
                clean_decay_curve = ideal_decay_curve + np.random.normal(0,
                                                                         np.random.uniform(*self.noise_std_clean),
                                                                         self.sequence_length)
                noisy_decay_curve = clean_decay_curve + np.random.normal(0,
                                                                         np.random.uniform(*self.noise_std_noisy),
                                                                         self.sequence_length)

            # --- 4. 其他叠加噪声增强 (Spike & Pink Noise) ---
            if self.augmentations.get('spike_noise', False):
                # (这部分代码与您原来的一样，保持不变)
                num_spikes = random.randint(0, 10)
                for _ in range(num_spikes):
                    spike_idx = random.randint(0, self.sequence_length - 5)
                    spike_amp = random.uniform(0.02, 0.08)
                    spike_width = random.randint(1, 5)
                    for w in range(spike_width):
                        if spike_idx + w < self.sequence_length:
                            noisy_decay_curve[spike_idx + w] += spike_amp * np.exp(-w / 2)
                            clean_decay_curve[spike_idx + w] += spike_amp * np.exp(-w / 2)

            if self.augmentations.get('pink_noise', False):
                # (这部分代码与您原来的一样，保持不变)
                pn = pink_noise(self.sequence_length) * np.random.uniform(0.0001, 0.001)
                clean_decay_curve += pn
                noisy_decay_curve += pn

            # --- 5. 数据整理 ---
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

class DecayDataset(Dataset):
    def __init__(self, num_samples, sequence_length, decay_range=(10.0, 30.0),
                 amplitude_range=(0.8, 1.7), baseline_range=(0.01, 0.035),
                 noise_std_clean=(0.0001, 0.0005), noise_std_noisy=(0.003, 0.008)):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.decay_range = decay_range
        self.amplitude_range = amplitude_range
        self.baseline_range = baseline_range
        self.noise_std_clean = noise_std_clean
        self.noise_std_noisy = noise_std_noisy
        self.data = self._generate_data()

    def _generate_data(self):
        samples = []
        time_points = np.linspace(0, self.sequence_length/10, self.sequence_length, dtype=np.float32)
        for _ in range(self.num_samples):
            tau = np.random.uniform(*self.decay_range)
            amplitude = np.random.uniform(*self.amplitude_range)
            baseline = np.random.uniform(*self.baseline_range)

            ideal_decay_curve = amplitude * np.exp(-time_points / tau) + baseline
            noise_clean = np.random.normal(0, np.random.uniform(*self.noise_std_clean), self.sequence_length)
            clean_decay_curve = ideal_decay_curve + noise_clean

            noise_noisy = np.random.normal(0, np.random.uniform(*self.noise_std_noisy), self.sequence_length)
            freq = np.random.uniform(0.1, 2.0)
            interference = 0.02 * np.sin(2 * np.pi * freq * time_points)
            impulses = np.zeros_like(time_points)
            idx = np.random.choice(self.sequence_length, size=int(0.01 * self.sequence_length), replace=False)
            impulses[idx] = np.random.normal(0.1, 0.05, size=len(idx))

            noisy_decay_curve = clean_decay_curve + noise_noisy + interference + impulses

            samples.append({
                'noisy_spectrum': torch.FloatTensor(noisy_decay_curve).unsqueeze(0),
                'clean_spectrum': torch.FloatTensor(clean_decay_curve).unsqueeze(0),
                'true_concentration': torch.FloatTensor([tau])
            })
        return samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class DecayDataset(Dataset):
    def __init__(self, num_samples, sequence_length, decay_range=(10.0, 30.0),
                 amplitude_range=(0.8, 1.2), baseline_range=(0.01, 0.1),
                 noise_std_clean=(0.0001, 0.0005), noise_std_noisy=(0.005, 0.02)):
        """
        Generates simulated decay time data.

        Args:
            num_samples (int): Number of data samples to generate.
            sequence_length (int): Number of time points in each decay curve.
            decay_range (tuple): (min, max) range for the decay constant (tau).
            amplitude_range (tuple): (min, max) range for the initial amplitude.
            baseline_range (tuple): (min, max) range for the constant baseline.
            noise_std_clean (tuple): (min, max) range for standard deviation of noise
                                     added to the "clean" signal for realistic base.
            noise_std_noisy (tuple): (min, max) range for standard deviation of noise
                                     added to the "noisy" signal.
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.decay_range = decay_range
        self.amplitude_range = amplitude_range
        self.baseline_range = baseline_range
        self.noise_std_clean = noise_std_clean
        self.noise_std_noisy = noise_std_noisy

        self.data = self._generate_data()

    def _generate_data(self):
        samples = []
        time_points = np.linspace(0, self.sequence_length/10, self.sequence_length, dtype=np.float32)

        for _ in range(self.num_samples):
            tau = np.random.uniform(*self.decay_range)
            amplitude = np.random.uniform(*self.amplitude_range)
            baseline = np.random.uniform(*self.baseline_range)

            ideal_decay_curve = amplitude * np.exp(-time_points / tau) + baseline

            noise_clean = np.random.normal(0, np.random.uniform(*self.noise_std_clean), self.sequence_length)
            clean_decay_curve = ideal_decay_curve + noise_clean

            noise_noisy = np.random.normal(0, np.random.uniform(*self.noise_std_noisy), self.sequence_length)
            noisy_decay_curve = clean_decay_curve + noise_noisy

            samples.append({
                'noisy_spectrum': torch.FloatTensor(noisy_decay_curve).unsqueeze(0),
                'clean_spectrum': torch.FloatTensor(clean_decay_curve).unsqueeze(0),
                'true_concentration': torch.FloatTensor([tau])  # Keep true_concentration for potential plotting/context
            })
        return samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    # Example usage
    dataset = DecayDataset(num_samples=100, sequence_length=1050)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Noisy spectrum shape: {sample['noisy_spectrum'].shape}")
    print(f"Clean spectrum shape: {sample['clean_spectrum'].shape}")
    print(f"True concentration (tau): {sample['true_concentration'].item()}")

    import matplotlib.pyplot as plt

    time_points = np.linspace(0, 10, 1050)
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, sample['clean_spectrum'].squeeze().numpy(), label='Clean Decay')
    plt.plot(time_points, sample['noisy_spectrum'].squeeze().numpy(), label='Noisy Decay', alpha=0.7)
    plt.title(f"Simulated Decay Curve (Tau: {sample['true_concentration'].item():.3f})")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()