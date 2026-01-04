import librosa
import numpy as np


def _compute_stats(feature_matrix):
    """
    Helper to compute statistics over the time axis (axis 1).
    Returns (flattened_stats, stat_names)
    feature_matrix: shape (n_features, n_time_steps)
    """
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(1, -1)

    means = np.mean(feature_matrix, axis=1)
    maxs = np.max(feature_matrix, axis=1)
    mins = np.min(feature_matrix, axis=1)
    medians = np.median(feature_matrix, axis=1)
    stds = np.std(feature_matrix, axis=1)

    # Avoid div by zero for max_mean
    safe_means = means.copy()
    safe_means[safe_means == 0] = 1e-10
    max_means = maxs / safe_means

    stats_vec = []
    # Interleave stats per feature coefficient
    for i in range(feature_matrix.shape[0]):
        stats_vec.extend([
            means[i], maxs[i], mins[i], medians[i], stds[i], max_means[i]
        ])

    return np.array(stats_vec), ["mean", "max", "min", "median", "std", "max_mean"]

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
    Wraps _compute_stats for backward compatibility and simplicity.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    stats, names = _compute_stats(mfccs)
    return stats, names

def extract_extended_features(y, sr, n_mfcc, n_fft=2048, hop_length=512,
                              use_deltas=False, use_centroid=False,
                              use_flux=False, use_zcr=False):
    """
    Extracts a super-vector of features based on flags.
    Always includes base MFCCs.
    """
    features = []
    feature_names = []

    # 1. Base MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    stats, stat_names = _compute_stats(mfccs)
    features.extend(stats)
    for i in range(n_mfcc):
        for s in stat_names:
            feature_names.append(f"MFCC_{i}_{s}")

    # 2. Deltas (Delta & Delta-Delta)
    if use_deltas:
        # Delta
        delta1 = librosa.feature.delta(mfccs)
        d1_stats, _ = _compute_stats(delta1)
        features.extend(d1_stats)
        for i in range(n_mfcc):
            for s in stat_names:
                feature_names.append(f"Delta_{i}_{s}")

        # Delta-Delta
        delta2 = librosa.feature.delta(mfccs, order=2)
        d2_stats, _ = _compute_stats(delta2)
        features.extend(d2_stats)
        for i in range(n_mfcc):
            for s in stat_names:
                feature_names.append(f"Delta2_{i}_{s}")

    # 3. Spectral Centroid
    if use_centroid:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        c_stats, _ = _compute_stats(cent)
        features.extend(c_stats)
        for s in stat_names:
            feature_names.append(f"Centroid_{s}")

    # 4. Spectral Flux (Onset Strength as proxy for rate of change)
    if use_flux:
        # Calculate manually for rigorous definition: L2 norm of diff of magnitude spectrum
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        flux = np.sqrt(np.sum((S[:, 1:] - S[:, :-1])**2, axis=0))
        # Flux has 1 less time step, _compute_stats handles 1D array
        f_stats, _ = _compute_stats(flux)
        features.extend(f_stats)
        for s in stat_names:
            feature_names.append(f"Flux_{s}")

    # 5. Zero Crossing Rate
    if use_zcr:
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
        z_stats, _ = _compute_stats(zcr)
        features.extend(z_stats)
        for s in stat_names:
            feature_names.append(f"ZCR_{s}")

    return np.array(features), feature_names