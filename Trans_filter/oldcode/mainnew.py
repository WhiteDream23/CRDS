import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from models import TUNN , ConvAutoencoder, UNet1D, SimpleRNN,TauPredictorModel # Import only TUNN now
from dataset import DecayDataset,DecayDataset2,GroupedDecayDataset
from utils import evaluate_denoising, apply_savgol_filter, apply_median_filter
# --- 固定随机种子以保证可复现性 ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

if __name__ == '__main__':
    # Hyperparameters
    INPUT_LENGTH = 10000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 80  # Reduced for faster demo

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Generation (simulating decay time data)
    print("Generating training data...")
    train_dataset = DecayDataset(
        num_samples=30000,
        sequence_length=INPUT_LENGTH,
        decay_range=(110.0, 170.0),
        amplitude_range=(0.8, 1.7),
        baseline_range=(0.01, 0.035),
        noise_std_clean=(0.0001, 0.0005),
        noise_std_noisy=(0.003, 0.008)
    )
    val_dataset = DecayDataset(
        num_samples=3000,  # Use a separate validation set
        sequence_length=INPUT_LENGTH,
        decay_range=(110.0, 170.0),
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

    model_to_train = TauPredictorModel(INPUT_LENGTH).to(device)
    model_name = "TauPredictorModel"

    train_model(model_to_train, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, device)

    print("TauPredictorModel training completed.")