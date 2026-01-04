import librosa
import numpy as np


def extract_mfcc_features(y, sr, n_mfcc, n_fft=2048, hop_length=512, method="mean"):
    """
    Extracts MFCCs and aggregates them across the time dimension of the chunk.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    if method == "mean":
        return np.mean(mfccs, axis=1)
    elif method == "max":
        return np.max(mfccs, axis=1)
    elif method == "min":
        return np.min(mfccs, axis=1)
    elif method == "median":
        return np.median(mfccs, axis=1)
    elif method == "std":
        return np.std(mfccs, axis=1)
    elif method == "max_mean":
        # Avoid division by zero
        mean_val = np.mean(mfccs, axis=1)
        mean_val[mean_val == 0] = 1e-10
        return np.max(mfccs, axis=1) / mean_val
    else:
        raise ValueError(f"Unknown aggregation method: {method}")