import os

from scipy.optimize import curve_fit

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models import TUNN , ConvAutoencoder, UNet1D, SimpleRNN # Import only TUNN now
from dataset import DecayDataset
from utils import evaluate_denoising, apply_savgol_filter, apply_median_filter
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42) # 设置一个固定的种子
# 加权MSE损失：前段对τ敏感区域加权
def weighted_mse_loss(output, target, weight=None):
    if weight is None:
        weight = torch.ones_like(output)
        seq_len = output.shape[-1]
        cutoff = int(0.3 * seq_len)  # 前30%时间段
        weight[..., :cutoff] = 3.0
    return torch.mean(weight * (output - target) ** 2)

# τ拟合函数
def fit_tau(signal, time_axis):
    try:
        popt, _ = curve_fit(lambda t, A, tau, B: A*np.exp(-t/tau) + B,
                            time_axis, signal, p0=(1.0, 20.0, 0.01), maxfev=10000)
        return popt[1]  # 返回tau
    except:
        return np.nan

# τ辅助损失函数
def compute_tau_loss_batch(output_batch, true_tau_batch, time_pts):
    losses = []
    for i in range(output_batch.size(0)):
        signal = output_batch[i, 0].detach().cpu().numpy()
        try:
            pred_tau = fit_tau(signal, time_pts)
            tau_loss = (pred_tau - true_tau_batch[i].item()) ** 2
            losses.append(tau_loss)
        except:
            continue
    if not losses:
        return torch.tensor(0.0, device=output_batch.device)
    return torch.tensor(losses, device=output_batch.device).mean()

def train_model(model, train_loader, val_loader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Only MSE for denoising

    #学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    time_pts = np.linspace(0, INPUT_LENGTH / 10, INPUT_LENGTH)
    best_val_rmse = float('inf')  # Track best RMSE for denoising
    history = {
        'train_loss': [], 'val_loss': [],
        'val_denoising_snr': [], 'val_denoising_rmse': [], 'val_denoising_mae': []
    }

    print(f"Training TUNN on {device}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        batch_idx = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            noisy_spec = batch['noisy_spectrum'].to(device)
            clean_spec = batch['clean_spectrum'].to(device)

            optimizer.zero_grad()

            denoised_spec = model(noisy_spec)  # Only TUNN output
            loss = criterion(denoised_spec, clean_spec)
            # loss_main = weighted_mse_loss(denoised_spec, clean_spec)
            loss_raw=loss
            true_tau = batch['true_concentration'].to(device).squeeze(1)
            loss_tau = compute_tau_loss_batch(denoised_spec, true_tau, time_pts)
            loss +=  0.1 * loss_tau  # τ损失加权项
            # loss = weighted_mse_loss(denoised_spec, clean_spec)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            if batch_idx % 20 == 0:  # 每20个batch打印一次，避免刷屏
                print(
                    f"  Epoch {epoch + 1}, Batch {batch_idx}: Loss = {loss.item():.6f},Raw Loss ={loss_raw.item():.6f} Tau Loss = {loss_tau.item():.6f}")
            batch_idx += 1

        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_noisy_specs_list = []
        val_clean_specs_list = []
        val_denoised_specs_list = []
        val_loss_epoch = 0

        with torch.no_grad():
            batch_idx = 0
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                noisy_spec = batch['noisy_spectrum'].to(device)
                clean_spec = batch['clean_spectrum'].to(device)

                denoised_spec = model(noisy_spec)

                loss = criterion(denoised_spec, clean_spec)
                loss_raw = loss
                # loss_main = weighted_mse_loss(denoised_spec, clean_spec)
                true_tau = batch['true_concentration'].to(device).squeeze(1)
                loss_tau = compute_tau_loss_batch(denoised_spec, true_tau, time_pts)
                loss += 0.1 * loss_tau  # τ损失加权项
                # loss = weighted_mse_loss(denoised_spec, clean_spec)
                val_loss_epoch += loss.item()

                if batch_idx % 20 == 0:  # 每20个batch打印一次，避免刷屏
                    print(
                        f"  Epoch {epoch + 1}, Batch {batch_idx}: Loss = {loss.item():.6f},Raw Loss ={loss_raw.item():.6f} Tau Loss = {loss_tau.item():.6f}")
                batch_idx += 1
                # 收集验证集数据
                val_noisy_specs_list.extend(noisy_spec.cpu().numpy())
                val_clean_specs_list.extend(clean_spec.cpu().numpy())
                val_denoised_specs_list.extend(denoised_spec.cpu().numpy())

        avg_val_loss = val_loss_epoch / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        # 更新学习率
        scheduler.step(avg_val_loss)
        val_noisy_specs_arr = np.array(val_noisy_specs_list)
        val_clean_specs_arr = np.array(val_clean_specs_list)
        val_denoised_specs_arr = np.array(val_denoised_specs_list)

        denoising_metrics = evaluate_denoising(val_noisy_specs_arr, val_clean_specs_arr, val_denoised_specs_arr)
        history['val_denoising_snr'].append(denoising_metrics['avg_denoised_snr_dB'])
        history['val_denoising_rmse'].append(denoising_metrics['avg_denoised_rmse'])
        history['val_denoising_mae'].append(denoising_metrics['avg_denoised_mae'])

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"Val Denoised SNR: {denoising_metrics['avg_denoised_snr_dB']:.2f}dB | "
              f"Val Denoised RMSE: {denoising_metrics['avg_denoised_rmse']:.6f}")

        # Save best model based on TUNN's RMSE (lower is better)
        if denoising_metrics['avg_denoised_rmse'] < best_val_rmse:
            best_val_rmse = denoising_metrics['avg_denoised_rmse']
            torch.save(model.state_dict(), 'best_UNET_denoiser_model.pth')
            print(f"Saved best model with Val RMSE: {best_val_rmse:.6f}")

    print("TUNN Training finished.")
    return history, val_noisy_specs_arr, val_clean_specs_arr, val_denoised_specs_arr


if __name__ == '__main__':
    # Hyperparameters
    INPUT_LENGTH = 1000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30  # Reduced for faster demo

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Generation (simulating decay time data)
    print("Generating training data...")
    train_dataset = DecayDataset(
        num_samples=30000,
        sequence_length=INPUT_LENGTH,
        decay_range=(10.0, 30.0),
        amplitude_range=(0.8, 1.7),
        baseline_range=(0.01, 0.035),
        noise_std_clean=(0.0001, 0.0005),
        noise_std_noisy=(0.003, 0.008)
    )
    val_dataset = DecayDataset(
        num_samples=3000,  # Use a separate validation set
        sequence_length=INPUT_LENGTH,
        decay_range=(10.0, 30.0),
        amplitude_range=(0.8, 1.7),
        baseline_range=(0.01, 0.035),
        noise_std_clean=(0.0001, 0.0005),
        noise_std_noisy=(0.003, 0.008)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True,drop_last= True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Generated {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # # Initialize TUNN model
    # tunn_model = TUNN(INPUT_LENGTH).to(device)
    #
    # # # 编译模型以获得更好的性能 (PyTorch 2.0+)
    # # if hasattr(torch, 'compile'):
    # #     tunn_model = torch.compile(tunn_model)
    # #     print("Model compiled for better performance.")
    #
    # # Start TUNN training
    # history, noisy_specs_val, clean_specs_val, denoised_specs_tunn = \
    #     train_model(tunn_model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, device)
    #
    # print("TUNN training completed.")

    # model_to_train = UNet1D().to(device)  # 例如，选择UNet1D
    # model_name = "UNet1D"
    # model_to_train = ConvAutoencoder(INPUT_LENGTH).to(device)
    # model_name = "ConvAutoencoder"
    model_to_train = SimpleRNN(INPUT_LENGTH).to(device)
    model_name = "SimpleRNN"

    history, noisy_specs_val, clean_specs_val, denoised_specs_tunn = \
        train_model(model_to_train, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, device)

    print("TUNN training completed.")

    # Plotting: Training history (Losses and Denoising Metrics)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history['val_denoising_snr'], label='Val Denoised SNR', color='blue')
    plt.title('Validation Denoised SNR History')
    plt.xlabel('Epoch')
    plt.ylabel('SNR (dB)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history['val_denoising_rmse'], label='Val Denoised RMSE', color='orange')
    plt.title('Validation Denoised RMSE History')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('methods_analyse.png', dpi=300)
    #plt.show()