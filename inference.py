import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa
from utils.audio_loader import preprocess_audio, get_audio_chunks
from utils.features import extract_extended_features

# Constants
SR = 16000
CHUNK_DUR = 1.0

def load_production_model(model_path="models/production_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_production.py first.")

    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

def analyze_file(file_path, artifacts):
    model = artifacts['model']
    config = artifacts['preprocessing_config']
    flags = artifacts['feature_flags']

    # 1. Load full audio for plotting later
    y_full, _ = librosa.load(file_path, sr=SR)
    duration = librosa.get_duration(y=y_full, sr=SR)

    # 2. Chunk and Process
    chunks = get_audio_chunks(file_path, SR, CHUNK_DUR)
    results = []

    for i, chunk in enumerate(chunks):
        start_time = i * CHUNK_DUR
        end_time = start_time + CHUNK_DUR

        # Preprocess
        y_proc = preprocess_audio(chunk, SR, filter_config=config['filter_config'])

        # Extract Features using the EXACT flags from training
        feats, _ = extract_extended_features(
            y_proc, SR,
            n_mfcc=config['n_mfcc'],
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            **flags
        )

        # Predict
        # Reshape for single sample prediction
        pred = model.predict([feats])[0]
        prob = model.predict_proba([feats])[0][1]

        results.append({
            'start': start_time,
            'end': end_time,
            'prediction': pred,
            'probability': prob
        })

    # --- Whole File Decision ---
    has_event = any(r['prediction'] == 1 for r in results)

    print(f"\n--- File Analysis: {os.path.basename(file_path)} ---")
    print(f"Duration: {duration:.2f}s ({len(chunks)} chunks)")
    print(f"Overall Decision: {'YES' if has_event else 'NO'}")

    return y_full, results

def visualize_results(y, sr, results, out_path):
    time_axis = np.linspace(0, len(y)/sr, len(y))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot 1: Waveform
    axes[0].plot(time_axis, y, color='black', alpha=0.6)
    axes[0].set_title("Waveform")
    axes[0].set_ylabel("Amplitude")

    # Plot 2: Production Model Detections
    axes[1].set_title("Event Detection (Random Forest)")
    axes[1].set_yticks([])
    axes[1].set_xlabel("Time (s)")

    for r in results:
        color = 'green' if r['prediction'] == 1 else 'lightgray'
        # Draw span
        axes[1].axvspan(r['start'], r['end'], color=color, alpha=0.5)
        # Add probability text
        if r['prediction'] == 1:
            axes[1].text(r['start'] + 0.1, 0.5, f"EVENT\n{r['probability']:.2f}",
                         rotation=0, verticalalignment='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    # Test on a synthetic file
    test_file = "dataset/yes/yes_000.wav"

    if os.path.exists(test_file):
        try:
            artifacts = load_production_model()
            y, results = analyze_file(test_file, artifacts)
            visualize_results(y, SR, results, "results/inference_output_production.png")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Test file not found.")