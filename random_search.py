import os
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from utils.audio_loader import crawl_dataset, get_audio_chunks, preprocess_audio
from utils.features import extract_mfcc_features, extract_pcen_features
from utils.stats_engine import calculate_det_metrics

# --- Configuration ---
SEARCH_ITERATIONS = 20
OUTPUT_DIR = "results"
DATASET_DIR = "dataset"
# SR = 16000 # SR is now dynamic
CHUNK_DUR = 1.0

# --- Search Space ---
PARAM_SPACE = {
    'filter_config': [
        ['preemphasis'],
        ['highpass'],
        ['bandpass'],
        ['wavelet'],
        ['bandpass', 'wavelet'],
        ['wavelet', 'preemphasis'],
        ['highpass', 'wavelet'],
        None
    ],
    'n_mfcc': [13, 20, 40],
    'n_fft': [1024, 2048],
    'hop_length': [256, 512, 1024],
    'agg_method': ['mean', 'max', 'min', 'median', 'std', 'max_mean'],
    'target_sr': [4000, 8000, 16000],
    'feature_type': ['mfcc', 'pcen'],
    'noise_reduction': [True, False]
}

def get_random_config():
    return {k: random.choice(v) for k, v in PARAM_SPACE.items()}

def save_det_curve(far, frr, out_path, title):
    plt.figure(figsize=(6, 6))
    plt.plot(far, frr, lw=2, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.title(title)
    plt.xlabel("False Alarm Rate")
    plt.ylabel("Miss Rate")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()

def run_search():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize CSV Log
    csv_file = os.path.join(OUTPUT_DIR, "hyperparameter_search_results.csv")
    csv_headers = ["run_id", "filter_config", "n_mfcc", "n_fft", "hop_length", "agg_method",
                   "target_sr", "feature_type", "noise_reduction",
                   "mfcc_coeff_idx", "eer", "min_dcf", "auc"]

    # Write header if file doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    dataset = crawl_dataset(DATASET_DIR)

    if not dataset['yes'] and not dataset['no']:
        print("Dataset not found or empty. Please generate synthetic data first.")
        return

    for i in range(SEARCH_ITERATIONS):
        config = get_random_config()
        run_id = f"run_{i:03d}"
        print(f"\n--- Starting {run_id} ---")
        print(f"Config: {config}")

        # Prepare storage
        # coeff_scores[coeff_index]['yes'/'no'] = list of values
        coeff_scores = {k: {'yes': [], 'no': []} for k in range(config['n_mfcc'])}

        # 1. Process Dataset

        # Determine actual filter config (add noise reduction if requested)
        current_filters = (config['filter_config'] or []).copy()
        if config['noise_reduction']:
             if 'spectral_gating' not in current_filters:
                 current_filters.append('spectral_gating')

        target_sr = config['target_sr']

        for label in ['yes', 'no']:
            for file_path in dataset[label]:
                try:
                    chunks = get_audio_chunks(file_path, target_sr, CHUNK_DUR)
                    for chunk in chunks:
                        # Preprocess
                        y_proc = preprocess_audio(chunk, target_sr, filter_config=current_filters)

                        # Feature Extract
                        if config['feature_type'] == 'pcen':
                            feats = extract_pcen_features(
                                y_proc, target_sr,
                                n_mfcc=config['n_mfcc'],
                                n_fft=config['n_fft'],
                                hop_length=config['hop_length'],
                                method=config['agg_method']
                            )
                        else:
                            feats = extract_mfcc_features(
                                y_proc, target_sr,
                                n_mfcc=config['n_mfcc'],
                                n_fft=config['n_fft'],
                                hop_length=config['hop_length'],
                                method=config['agg_method']
                            )

                        # Store scores for each coefficient
                        for idx, val in enumerate(feats):
                            coeff_scores[idx][label].append(val)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        # 2. Calculate Metrics & Log
        for coeff_idx in range(config['n_mfcc']):
            yes_vals = np.array(coeff_scores[coeff_idx]['yes'])
            no_vals = np.array(coeff_scores[coeff_idx]['no'])

            if len(yes_vals) == 0 or len(no_vals) == 0:
                continue

            far, frr, thresholds = calculate_det_metrics(yes_vals, no_vals)

            # Simple EER calculation (approximate where FAR ~= FRR)
            diffs = np.abs(np.array(far) - np.array(frr))
            min_diff_idx = np.argmin(diffs)
            eer = (far[min_diff_idx] + frr[min_diff_idx]) / 2

            # Simple AUC (Area Under Curve) - lower is better for DET
            auc = np.trapz(frr, far) # This might be negative if x is decreasing, usually we want 1-ROC AUC, but for DET lower is better.

            # Min DCF (dummy calculation for now, assuming equal weights)
            min_dcf = np.min(np.array(far) + np.array(frr)) / 2

            # Log to CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    run_id,
                    str(config['filter_config']),
                    config['n_mfcc'],
                    config['n_fft'],
                    config['hop_length'],
                    config['agg_method'],
                    config['target_sr'],
                    config['feature_type'],
                    config['noise_reduction'],
                    coeff_idx,
                    f"{eer:.4f}",
                    f"{min_dcf:.4f}",
                    f"{abs(auc):.4f}"
                ])

            # Save DET curve for the best performing coefficient (e.g. lowest EER)
            # For now, let's just save the first coefficient of each run to avoid spamming 40 images per run
            if coeff_idx == 0:
                det_path = os.path.join(OUTPUT_DIR, f"{run_id}_coeff0_det.png")
                save_det_curve(far, frr, det_path, f"DET: {run_id} (Coeff 0)")

    print(f"\nSearch complete. Results saved to {csv_file}")

if __name__ == "__main__":
    run_search()