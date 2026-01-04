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

def extract_all_mfcc_stats(y, sr, n_mfcc, n_fft=2048, hop_length=512):
    """
    Extracts MFCCs and returns a flattened vector of ALL stats for EACH coefficient.
    Stats: mean, max, min, median, std, max_mean
    Output size: n_mfcc * 6
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Calculate stats across time (axis 1)
    means = np.mean(mfccs, axis=1)
    maxs = np.max(mfccs, axis=1)
    mins = np.min(mfccs, axis=1)
    medians = np.median(mfccs, axis=1)
    stds = np.std(mfccs, axis=1)

    # Avoid div by zero for max_mean
    safe_means = means.copy()
    safe_means[safe_means == 0] = 1e-10
    max_means = maxs / safe_means

    # Stack and flatten: [mean_0, max_0, ..., mean_1, max_1, ...]
    # We stack columns to keep coefficients grouped together if we want, or just simple hstack
    # Let's return a simple concatenated vector
    # Order: [all_means, all_maxs, all_mins, all_medians, all_stds, all_max_means]
    # This keeps features of similar type together, which is often fine for Trees.
    # Alternatively: Group by coeff. Let's do Group by Coeff: [c0_mean, c0_max..., c1_mean...]

    features = []
    for i in range(n_mfcc):
        features.extend([
            means[i], maxs[i], mins[i], medians[i], stds[i], max_means[i]
        ])

    return np.array(features), ["mean", "max", "min", "median", "std", "max_mean"]