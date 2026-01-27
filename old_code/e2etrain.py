import random
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
import os
from old_code.e2emodel import RingdownCNN
# 添加性能评估指标函数
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
class ChunkedRingdownDataset(Dataset):
    """处理分块存储的衰减信号数据集"""

    def __init__(self, chunks_dir="temp_chunks", transform=None, shuffle_chunks=True, normalize_targets=True,tau_min=5.0,tau_max=25.0,cache_size=10):
        """
        初始化数据集

        参数:
        chunks_dir: 存储数据分块的目录
        transform: 可选的数据转换函数
        shuffle_chunks: 是否打乱分块顺序
        """
        self.cache = {}
        self.cache_size = cache_size
        self.chunks_dir = chunks_dir
        self.transform = transform
        self.normalize_targets = normalize_targets
        self.tau_min, self.tau_max = tau_min, tau_max  # 预知的τ范围
        # 获取所有分块文件
        self.chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('.npz')]
        if shuffle_chunks:
            random.shuffle(self.chunk_files)

        # 计算每个分块中的样本数
        self.chunk_sizes = {}
        self.total_samples = 0

        for chunk_file in self.chunk_files:
            try:
                data = np.load(os.path.join(chunks_dir, chunk_file))
                chunk_size = len(data['signals'])
                self.chunk_sizes[chunk_file] = chunk_size
                self.total_samples += chunk_size
            except Exception as e:
                print(f"读取 {chunk_file} 时出错: {e}")

        # 建立索引映射
        self.chunk_indices = []
        for chunk_file, size in self.chunk_sizes.items():
            for i in range(size):
                self.chunk_indices.append((chunk_file, i))

        print(f"数据集初始化完成，共 {len(self.chunk_files)} 个分块，{self.total_samples} 个样本")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        if idx >= len(self.chunk_indices):
            raise IndexError("索引超出范围")

        # 获取对应的分块文件和内部索引
        chunk_file, internal_idx = self.chunk_indices[idx]
        # 检查缓存中是否有此文件
        if chunk_file not in self.cache:
            # 如果缓存已满，移除最早加入的文件
            if len(self.cache) >= self.cache_size:
                oldest_file = next(iter(self.cache))
                del self.cache[oldest_file]
                
            # 加载并缓存文件
            self.cache[chunk_file] = np.load(os.path.join(self.chunks_dir, chunk_file))
          # 从缓存获取数据
        data = self.cache[chunk_file]
        # 加载数据
        signal = data['signals'][internal_idx]
        tau = data['taus'][internal_idx]

        # 应用转换（如果有）
        if self.transform:
            signal = self.transform(signal)
        if self.normalize_targets:
            # 归一化tau到[0,1]
            tau = (tau - self.tau_min) / (self.tau_max - self.tau_min)

        return signal, tau

class InMemoryRingdownDataset(Dataset):
    """将所有数据加载到内存的衰减信号数据集"""

    def __init__(self, chunks_dir="temp_chunks", transform=None, shuffle=True, normalize_targets=True, tau_min=5.0, tau_max=25.0):
        """初始化数据集并将所有数据加载到内存中"""
        self.transform = transform
        self.normalize_targets = normalize_targets
        self.tau_min, self.tau_max = tau_min, tau_max
        
        # 获取所有分块文件
        chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('.npz')]
        
        print(f"开始加载所有数据到内存，共{len(chunk_files)}个文件...")
        
        # 预先分配空间用于存储所有数据
        self.all_signals = []
        self.all_taus = []
        
        # 加载所有数据
        for i, chunk_file in enumerate(chunk_files):
            try:
                data = np.load(os.path.join(chunks_dir, chunk_file))
                self.all_signals.extend(data['signals'])
                self.all_taus.extend(data['taus'])
                if i % 10 == 0:
                    print(f"已加载 {i}/{len(chunk_files)} 个文件...")
            except Exception as e:
                print(f"读取 {chunk_file} 时出错: {e}")
        
        # 转换为numpy数组以提高访问效率
        self.all_signals = np.array(self.all_signals)
        self.all_taus = np.array(self.all_taus)
        
        # 如果需要随机打乱数据
        if shuffle:
            indices = np.random.permutation(len(self.all_signals))
            self.all_signals = self.all_signals[indices]
            self.all_taus = self.all_taus[indices]
        
        print(f"数据集加载完成，共 {len(self.all_signals)} 个样本")
        print(f"内存占用约: {self.all_signals.nbytes / (1024**3):.2f} GB")

    def __len__(self):
        return len(self.all_signals)

    def __getitem__(self, idx):
        """从内存中直接获取样本"""
        # 获取数据
        signal = self.all_signals[idx]
        tau = self.all_taus[idx]
        
        # 应用转换
        if self.transform:
            signal = self.transform(signal)
            
        # 目标归一化
        if self.normalize_targets:
            tau = (tau - self.tau_min) / (self.tau_max - self.tau_min)
            
        return signal, tau
# signals归一化
class SignalTransform:
    def __init__(self):
        pass

    def __call__(self, signal):
        # 归一化信号到[-1, 1]或[0, 1]范围
        signal_min = signal.min()
        signal_max = signal.max()
        normalized_signal = (signal - signal_min) / (signal_max - signal_min)
        return normalized_signal
def create_dataloaders(chunks_dir="temp_chunks6", batch_size=32, train_ratio=0.8,use_in_memory=True,mintau=5.0,maxtau=25.0):
    """创建训练和验证数据加载器"""
    if use_in_memory:
        numworkers=0
        dataset = InMemoryRingdownDataset(chunks_dir=chunks_dir, transform=SignalTransform(),tau_max=maxtau,tau_min=mintau)
    else:
        numworkers=0
        dataset = ChunkedRingdownDataset(chunks_dir=chunks_dir, transform=SignalTransform(),tau_max=maxtau,tau_min=mintau,cache_size=100)
    analyze_dataset_quality(dataset)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=numworkers,
        pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=numworkers,
        pin_memory=True, drop_last=False
    )

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    print(f"训练集loader大小: {len(train_loader)}, 验证集loader大小: {len(val_loader)}")
    return train_loader, val_loader


def analyze_dataset_quality(dataset,sample_size=1000):
    """分析数据集质量，仅使用随机采样的子集"""
    total_size = len(dataset)

    # 如果数据集小于采样大小，使用全部数据
    if total_size <= sample_size:
        indices = range(total_size)
    else:
        # 随机采样
        indices = random.sample(range(total_size), sample_size)

    signals = []
    taus = []

    print(f"分析数据集质量 (采样 {min(sample_size, total_size)} / {total_size} 样本)...")

    # 获取采样数据
    for idx in indices:
        signal, tau = dataset[idx]
        signals.append(signal)
        taus.append(tau)

    # 统计特征
    tau_stats = {
        'mean': np.mean(taus),
        'std': np.std(taus),
        'min': np.min(taus),
        'max': np.max(taus)
    }

    signal_stats = {
        'mean_amplitude': np.mean([np.max(s) - np.min(s) for s in signals]),
        'mean_noise_level': np.mean([np.std(s) for s in signals])
    }

    print("Tau Distribution:", tau_stats)
    print("Signal Characteristics:", signal_stats)


def plot_random_samples(inputs, num_samples=5, figsize=(15, 10)):
    """
    从输入张量中随机选择并绘制几个样本

    参数:
    - inputs: 形状为 [batch_size, channels, data_points] 的PyTorch张量
    - num_samples: 要绘制的随机样本数量
    - figsize: 图形大小
    """
    # 确保不超出可用样本数量
    batch_size = inputs.shape[0]
    num_samples = min(num_samples, batch_size)

    # 随机选择样本索引
    sample_indices = random.sample(range(batch_size), num_samples)

    # 创建图形
    plt.figure(figsize=figsize)

    # 绘制每个随机样本
    for i, idx in enumerate(sample_indices):
        # 提取样本数据 (去除通道维度，因为只有1个通道)
        sample_data = inputs[idx, 0, :].cpu().numpy()

        # 创建子图
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(sample_data, linewidth=1.5)
        plt.title(f'样本 #{idx}')
        plt.grid(True)

        # 只在最后一个子图上显示x轴标签
        if i == num_samples - 1:
            plt.xlabel('数据点')

        plt.ylabel('强度')

    plt.tight_layout()
    #plt.savefig('random_samples2.png')
    plt.show()

# 添加额外评估指标
def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 收集预测和真实值
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算指标
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    # 如果使用了归一化的τ，将其转换回原始范围进行评估
    if hasattr(val_loader.dataset.dataset, 'normalize_targets') and val_loader.dataset.dataset.normalize_targets:
        tau_min, tau_max = val_loader.dataset.dataset.tau_min, val_loader.dataset.dataset.tau_max
        all_preds = [p * (tau_max - tau_min) + tau_min for p in all_preds]
        all_targets = [t * (tau_max - tau_min) + tau_min for t in all_targets]
        
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return mse, mae, r2, all_preds, all_targets
def test_cnn_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        chunks_dir="temp_chunks_525_1200_0415_0.08",
        batch_size=128,
        train_ratio=0.8,
        mintau=5,
        maxtau=25,
        use_in_memory=True
    )

    model=RingdownCNN(dropout_rate=0.2).to(device)
    # model = CNNLSTMModel(
    #     cnn_output_channels=128,  # 需要与 CNNLSTMModel 中 CNN 最后一层输出通道一致
    #     lstm_hidden_size=64,      # LSTM 隐藏层大小，可以调整
    #     num_lstm_layers=2,        # LSTM 层数，可以调整
    #     dropout_rate=0.3          # Dropout 率，可以调整
    # ).to(device)
    criterion = nn.MSELoss()

    # 优化器 - 使用AdamW并添加权重衰减
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-4
    )
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # 监控损失下降
        factor=0.3,        # 每次将学习率减少
        patience=2,        # 等待2个epoch
        min_lr=1e-6,       # 最小学习率
        threshold=1e-4
    )
    # 训练
    num_epochs = 30
    # 添加最佳模型跟踪
    best_val_loss = float('inf')
    best_model_path = 'best_ringdown_model.pth'
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batchnum=0
        for inputs, targets in train_loader:
            batchnum+=1
            inputs, targets = inputs.to(device), targets.to(device)
            #inputs = inputs.unsqueeze(1)

            # print(f"模型输入形状:{inputs.shape}")
            # print(f"模型标签形状:{targets.shape}")
            targets = targets.squeeze()
            # 绘制5个随机样本
            #plot_random_samples(inputs, num_samples=5)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if batchnum % 100 == 0:
                print(f"Batch {batchnum}/{len(train_loader)}, Train Loss: {loss.item():.6f}")
            running_loss += loss.item()
        train_loss = running_loss / batchnum
        #train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        batchnum = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                batchnum += 1
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.squeeze()
                # print(f"val模型输入形状:{inputs.shape}")
                # print(f"val模型标签形状:{targets.shape}")
                #inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if batchnum % 100  == 0:
                    print(f"Batch {batchnum}/{len(val_loader)}, Val Loss: {loss.item():.6f}")
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
                # 验证阶段结束后，检查是否为最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'scheduler_state_dict': scheduler.state_dict(),
            }, best_model_path)
            print(f"保存新的最佳模型 (Epoch {epoch+1}), 验证损失: {val_loss:.6f}")
        # 更新学习率
        scheduler.step(val_loss)
        #current_lr = scheduler.get_last_lr()
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率的更可靠方式
        print(f"Epoch {epoch + 1}, Learning Rate: {current_lr}")
        val_losses.append(val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        # 在每个epoch结束后添加
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            mse, mae, r2, preds, targets = evaluate_model(model, val_loader, device)
            
            # 绘制真实值vs预测值散点图
            plt.figure(figsize=(8, 8))
            plt.scatter(targets, preds, alpha=0.5)
            plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
            plt.title(f'预测值 vs 真实值 (Epoch {epoch+1})')
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.savefig(f'pred_vs_true_epoch_{epoch+1}.png')
            plt.close()

    # 可视化损失
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('CNN_model_training_results.png', dpi=300)
    plt.close()
    # 训练结束后加载最佳模型
    print(f"加载最佳模型，最佳验证损失: {best_val_loss:.6f}")
    checkpoint = torch.load(best_model_path,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 使用最佳模型进行最终评估
    print("使用最佳模型进行最终评估...")
    final_mse, final_mae, final_r2, final_preds, final_targets = evaluate_model(model, val_loader, device)
    
    # 绘制最终预测结果
    plt.figure(figsize=(10, 8))
    plt.scatter(final_targets, final_preds, alpha=0.5, color='blue')
    plt.plot([min(final_targets), max(final_targets)], [min(final_targets), max(final_targets)], 'r--')
    plt.title(f'最佳模型预测结果 (Epoch {checkpoint["epoch"]})\nMSE: {final_mse:.6f}, MAE: {final_mae:.6f}, R²: {final_r2:.6f}')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.grid(True, alpha=0.3)
    plt.savefig('best_model_predictions.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    test_cnn_model()