import librosa
import numpy as np
from pathlib import Path
import os
import scipy.signal as signal
import pywt

def crawl_dataset_simple(data_dir):
    dataset_map = {'yes': [], 'no': []}
    for label in dataset_map.keys():
        # Ensure the path is absolute and uses correct separators
        folder_path = os.path.abspath(os.path.join(data_dir, label))
        
        if os.path.exists(folder_path):
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.wav', '.mp3'))]
            dataset_map[label] = files
            print(f"--- Found {len(files)} files in category: {label}") # Verification
        else:
            print(f"!!! Error: Folder not found at {folder_path}")
            
    return dataset_map

def crawl_dataset(data_dir):
    """
    Uses pathlib for robust recursive searching across Windows/Linux.
    """
    dataset_map = {'yes': [], 'no': []}
    root_path = Path(data_dir)
    
    if not root_path.exists():
        print(f"!!! Error: Root path does not exist: {root_path}")
        return dataset_map

    for label in dataset_map.keys():
        label_path = root_path / label
        if label_path.exists():
            # rglob("**/*") finds all files in the directory and ALL subdirectories
            files = [
                str(f) for f in label_path.rglob("*") 
                if f.suffix.lower() in ['.wav', '.mp3', '.flac']
            ]
            dataset_map[label] = files
            print(f"--- Found {len(files)} files in category: {label}")
        else:
            print(f"!!! Error: Folder not found: {label_path}")
            
    return dataset_map

def normalize_audio(y):
    """
    Normalizes the audio to be within -1 to 1.
    """
    if y is None or len(y) == 0:
        return y

    y = np.nan_to_num(y)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y

def high_pass_filter(y, sr, cutoff=1000):
    """
    Applies a high-pass Butterworth filter.
    """
    sos = signal.butter(10, cutoff, 'hp', fs=sr, output='sos')
    return signal.sosfilt(sos, y)

def band_pass_filter(y, sr, low_cutoff=500, high_cutoff=4000):
    """
    Applies a band-pass Butterworth filter.
    """
    sos = signal.butter(10, [low_cutoff, high_cutoff], 'bp', fs=sr, output='sos')
    return signal.sosfilt(sos, y)

def wavelet_denoise(y, wavelet='db4', level=1):
    """
    Applies Wavelet Denoising using Soft Thresholding.
    """
    # Safety check for silent/near-silent signals
    if np.all(y == 0) or np.std(y) < 1e-9:
        return np.zeros_like(y)

    try:
        # Decompose
        coeff = pywt.wavedec(y, wavelet, mode="per")
        # Calculate threshold (universal threshold)
        sigma = (1/0.6745) * np.median(np.abs(coeff[-1] - np.median(coeff[-1])))
        uthresh = sigma * np.sqrt(2 * np.log(len(y)))
        # Thresholding
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
        # Reconstruct
        rec = pywt.waverec(coeff, wavelet, mode='per')

        # Ensure length matches input (sometimes reconstruction adds a sample)
        if len(rec) > len(y):
            rec = rec[:len(y)]

        return np.nan_to_num(rec)
    except Exception as e:
        print(f"Warning: Wavelet denoise failed ({e}), returning original.")
        return y

def preprocess_audio(y, sr, filter_config=None):
    """
    Applies a sequence of filters specified in filter_config.
    filter_config: List of strings (e.g., ['highpass', 'wavelet'])
    """
    if not filter_config:
        return normalize_audio(np.nan_to_num(y)) # Ensure safety even without filters

    y_processed = y.copy()

    for filter_name in filter_config:
        if filter_name == 'preemphasis':
            y_processed = librosa.effects.preemphasis(y_processed)
        elif filter_name == 'highpass':
            y_processed = high_pass_filter(y_processed, sr)
        elif filter_name == 'bandpass':
            y_processed = band_pass_filter(y_processed, sr)
        elif filter_name == 'wavelet':
            y_processed = wavelet_denoise(y_processed)
        else:
            print(f"Warning: Unknown filter '{filter_name}' skipped.")
    
    return normalize_audio(np.nan_to_num(y_processed))

def get_audio_chunks(file_path, sr, chunk_duration):
    """
    Loads a file and splits it into 1-second chunks.
    """
    y, _ = librosa.load(file_path, sr=sr)
    chunk_samples = int(chunk_duration * sr)
    
    # Split audio into chunks of fixed length.
    # IMPORTANT: Chunks smaller than chunk_samples are IGNORED.
    chunks = [y[i:i + chunk_samples] for i in range(0, len(y), chunk_samples) 
              if len(y[i:i + chunk_samples]) == chunk_samples]
    return chunks