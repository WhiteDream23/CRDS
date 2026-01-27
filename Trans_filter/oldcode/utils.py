import numpy as np
import scipy.signal
import scipy.ndimage # For median filter
import pywt
from pykalman import KalmanFilter
def calculate_snr(signal, noise):
    """
    Calculates Signal-to-Noise Ratio (SNR) in dB.
    signal and noise should be 1D numpy arrays.
    """
    # Ensure inputs are numpy arrays for consistent behavior
    signal = np.asarray(signal)
    noise = np.asarray(noise)

    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    if noise_power == 0:
        return np.inf # Handle case with no noise
    return 10 * np.log10(signal_power / noise_power)

def calculate_rmse(predictions, targets):
    """Calculates Root Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean((np.asarray(predictions) - np.asarray(targets))**2))

def calculate_mae(predictions, targets):
    """Calculates Mean Absolute Error (MAE)."""
    return np.mean(np.abs(np.asarray(predictions) - np.asarray(targets)))

def evaluate_denoising(noisy_specs, clean_specs, denoised_specs):
    """
    Evaluates denoising performance across a batch of samples.
    Inputs are numpy arrays (batch_size, 1, sequence_length) or (batch_size, sequence_length)
    """
    # Flatten inputs to (batch_size, sequence_length) for processing
    noisy_specs = noisy_specs.reshape(noisy_specs.shape[0], -1)
    clean_specs = clean_specs.reshape(clean_specs.shape[0], -1)
    denoised_specs = denoised_specs.reshape(denoised_specs.shape[0], -1)

    original_snrs = []
    denoised_snrs = []
    denoised_rmse = []
    denoised_mae = []

    for i in range(len(noisy_specs)):
        noisy_s = noisy_specs[i]
        clean_s = clean_specs[i]
        denoised_s = denoised_specs[i]

        original_noise = noisy_s - clean_s
        denoised_noise = denoised_s - clean_s

        original_snrs.append(calculate_snr(clean_s, original_noise))
        denoised_snrs.append(calculate_snr(clean_s, denoised_noise))
        denoised_rmse.append(calculate_rmse(denoised_s, clean_s))
        denoised_mae.append(calculate_mae(denoised_s, clean_s))

    return {
        'avg_original_snr_dB': np.mean(original_snrs),
        'avg_denoised_snr_dB': np.mean(denoised_snrs),
        'avg_denoised_rmse': np.mean(denoised_rmse),
        'avg_denoised_mae': np.mean(denoised_mae)
    }

# --- Traditional Filtering Implementations ---

def apply_savgol_filter(data, window_length, polyorder):
    """Applies Savitzky-Golay filter."""
    # window_length must be odd and polyorder must be less than window_length
    if window_length % 2 == 0:
        window_length += 1 # Ensure odd
    if polyorder >= window_length:
        polyorder = window_length - 1 # Ensure polyorder < window_length
        if polyorder < 1: polyorder = 1 # Minimum polyorder

    return scipy.signal.savgol_filter(data, window_length, polyorder)

def apply_median_filter(data, size):
    """Applies Median filter."""
    return scipy.ndimage.median_filter(data, size=size)

# --- New filter: Wavelet Denoising ---
def apply_wavelet_filter(signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(signal, wavelet, mode='symmetric')
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, 'soft') if i >= level else c
                       for i, c in enumerate(coeffs)]
    return pywt.waverec(denoised_coeffs, wavelet, mode='symmetric')[:len(signal)]

# --- New filter: Simple Kalman Filter ---
# Assuming signal model x_t = x_{t-1} + process_noise
def apply_kalman_filter(signal):
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=signal[0],
        initial_state_covariance=1,
        observation_covariance=0.001,
        transition_covariance=0.0001
    )
    state_means, _ = kf.filter(signal)
    return state_means.flatten()

# Kalman filter is complex for a generic signal and often requires state-space modeling.
# For a direct comparison, Savgol and Median are more straightforward to implement.
# If you need a more specific Kalman filter for your decay data, you'd define its state
# and observation models (e.g., as a linear or extended Kalman filter).