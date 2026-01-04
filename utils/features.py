import librosa
import numpy as np


def extract_mfcc_features(y, sr, n_mfcc, method="mean"):
    """
    Extracts MFCCs and aggregates them across the time dimension of the chunk.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    if method == "mean":
        return np.mean(mfccs, axis=1)
    elif method == "max":
        return np.max(mfccs, axis=1)
    elif method == "median":
        return np.median(mfccs, axis=1)
    elif method == "std":
        return np.std(mfccs, axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")