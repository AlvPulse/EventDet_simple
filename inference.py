import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa
import argparse
from utils.audio_loader import preprocess_audio, get_audio_chunks
from utils.features import extract_all_mfcc_stats

# Constants
SR = 16000
CHUNK_DUR = 1.0

def load_artifacts(model_path="results/best_models.pkl"):
    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

def analyze_file(file_path, artifacts):
    config = artifacts['config']
    baseline_threshold = artifacts['baseline_threshold']
    model_dt = artifacts['model_dt']
    model_rf = artifacts['model_rf']

    # 1. Load full audio for plotting later
    y_full, _ = librosa.load(file_path, sr=SR)
    duration = librosa.get_duration(y=y_full, sr=SR)

    # 2. Chunk and Process
    # We use the same chunking logic as training
    chunks = get_audio_chunks(file_path, SR, CHUNK_DUR)

    results = []

    for i, chunk in enumerate(chunks):
        start_time = i * CHUNK_DUR
        end_time = start_time + CHUNK_DUR

        # Preprocess
        y_proc = preprocess_audio(chunk, SR, filter_config=config['filter_config'])

        # Extract Features (Rich)
        feats, stat_names = extract_all_mfcc_stats(
            y_proc, SR,
            n_mfcc=config['n_mfcc'],
            n_fft=config['n_fft'],
            hop_length=config['hop_length']
        )

        # Extract Baseline Feature
        baseline_agg = config['baseline_feature'].split('(')[1].split(')')[0]
        baseline_coeff = int(config['baseline_feature'].split(' ')[0].split('_')[1])
        stat_idx = stat_names.index(baseline_agg)
        feat_idx = (baseline_coeff * len(stat_names)) + stat_idx
        baseline_val = feats[feat_idx]

        # --- Predictions ---

        # 1. Baseline
        pred_base = 1 if baseline_val > baseline_threshold else 0

        # 2. Decision Tree
        pred_dt = model_dt.predict([feats])[0]

        # 3. Random Forest
        pred_rf = model_rf.predict([feats])[0]

        results.append({
            'start': start_time,
            'end': end_time,
            'base': pred_base,
            'dt': pred_dt,
            'rf': pred_rf
        })

    # --- Whole File Decision ---
    # "Yes" if ANY chunk is positive
    has_event_base = any(r['base'] == 1 for r in results)
    has_event_dt = any(r['dt'] == 1 for r in results)
    has_event_rf = any(r['rf'] == 1 for r in results)

    print(f"\n--- File Analysis: {os.path.basename(file_path)} ---")
    print(f"Duration: {duration:.2f}s ({len(chunks)} chunks)")
    print(f"Overall Decision:")
    print(f"  Baseline (Thresh={baseline_threshold:.2f}): {'YES' if has_event_base else 'NO'}")
    print(f"  Decision Tree:   {'YES' if has_event_dt else 'NO'}")
    print(f"  Random Forest:   {'YES' if has_event_rf else 'NO'}")

    return y_full, results, (has_event_base, has_event_dt, has_event_rf)

def visualize_results(y, sr, results, out_path):
    time_axis = np.linspace(0, len(y)/sr, len(y))

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Waveform
    axes[0].plot(time_axis, y, color='black', alpha=0.6)
    axes[0].set_title("Waveform")
    axes[0].set_ylabel("Amplitude")

    # Plot 2: Baseline Detections
    axes[1].set_title("Baseline Detector")
    axes[1].set_yticks([])
    for r in results:
        color = 'green' if r['base'] == 1 else 'lightgray'
        axes[1].axvspan(r['start'], r['end'], color=color, alpha=0.5)
        if r['base'] == 1:
            axes[1].text(r['start'], 0.5, "EVENT", rotation=90, verticalalignment='center')

    # Plot 3: Decision Tree
    axes[2].set_title("Decision Tree")
    axes[2].set_yticks([])
    for r in results:
        color = 'blue' if r['dt'] == 1 else 'lightgray'
        axes[2].axvspan(r['start'], r['end'], color=color, alpha=0.5)
        if r['dt'] == 1:
            axes[2].text(r['start'], 0.5, "EVENT", rotation=90, verticalalignment='center')

    # Plot 4: Random Forest
    axes[3].set_title("Random Forest")
    axes[3].set_yticks([])
    axes[3].set_xlabel("Time (s)")
    for r in results:
        color = 'orange' if r['rf'] == 1 else 'lightgray'
        axes[3].axvspan(r['start'], r['end'], color=color, alpha=0.5)
        if r['rf'] == 1:
            axes[3].text(r['start'], 0.5, "EVENT", rotation=90, verticalalignment='center')

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    # For demonstration, pick a file automatically if not provided
    # ideally utilize argparse, but hardcoded for the plan step verification
    test_file = "dataset/yes/yes_000.wav"

    if os.path.exists(test_file):
        artifacts = load_artifacts()
        y, results, decisions = analyze_file(test_file, artifacts)
        visualize_results(y, SR, results, "results/inference_output.png")
    else:
        print("Test file not found. Run synthetic_data.py first.")