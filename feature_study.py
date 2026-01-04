import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import det_curve
from utils.audio_loader import crawl_dataset, get_audio_chunks, preprocess_audio
from utils.features import extract_extended_features
from utils.analysis import get_best_config

# Constants
DATASET_DIR = "dataset"
RESULTS_DIR = "results"
SR = 16000
CHUNK_DUR = 1.0

def load_full_dataset(config):
    dataset = crawl_dataset(DATASET_DIR)
    X = []
    y = []

    # We will extract *everything* once (All flags=True)
    # Then slice columns dynamically during training to save time

    print("Extracting ALL features for study...")
    all_feature_names = []

    for label_idx, label in enumerate(['no', 'yes']):
        for file_path in dataset[label]:
            try:
                chunks = get_audio_chunks(file_path, SR, CHUNK_DUR)
                for chunk in chunks:
                    y_proc = preprocess_audio(chunk, SR, filter_config=config['filter_config'])

                    feats, names = extract_extended_features(
                        y_proc, SR,
                        n_mfcc=config['n_mfcc'],
                        n_fft=config['n_fft'],
                        hop_length=config['hop_length'],
                        use_deltas=True,
                        use_centroid=True,
                        use_flux=True,
                        use_zcr=True
                    )

                    X.append(feats)
                    y.append(label_idx)

                    if not all_feature_names:
                        all_feature_names = names

            except Exception as e:
                print(f"Error {file_path}: {e}")

    return np.array(X), np.array(y), all_feature_names

def get_indices_for_scenario(all_names, scenario):
    """
    Returns indices of features that match the scenario requirements.
    Base is always included (MFCC_).
    """
    indices = []
    for i, name in enumerate(all_names):
        # Base MFCCs
        if name.startswith("MFCC_"):
            indices.append(i)
            continue

        # Add others based on scenario
        if scenario == "MFCC + Deltas" and (name.startswith("Delta_") or name.startswith("Delta2_")):
            indices.append(i)
        elif scenario == "MFCC + Spectral Centroid" and name.startswith("Centroid_"):
            indices.append(i)
        elif scenario == "MFCC + Spectral Flux" and name.startswith("Flux_"):
            indices.append(i)
        elif scenario == "MFCC + ZCR" and name.startswith("ZCR_"):
            indices.append(i)
        elif scenario == "All Features":
            # Add everything else
            if not name.startswith("MFCC_"):
                indices.append(i)

    return indices

def run_feature_study():
    config = get_best_config("results/hyperparameter_search_results.csv")
    if not config: return

    X_all, y_all, all_names = load_full_dataset(config)
    print(f"Dataset shape: {X_all.shape}")

    # Split
    indices = np.arange(len(y_all))
    train_idx, test_idx, y_train, y_test = train_test_split(indices, y_all, test_size=0.2, random_state=42, stratify=y_all)

    scenarios = [
        "Baseline (MFCC Only)",
        "MFCC + Deltas",
        "MFCC + Spectral Centroid",
        "MFCC + Spectral Flux",
        "MFCC + ZCR",
        "All Features"
    ]

    results = []

    print("\n--- Starting Feature Study ---")

    # Train "All Features" model for feature importance later
    rf_all = None

    for scenario in scenarios:
        print(f"Testing: {scenario}...")

        # 1. Select Features
        feat_indices = get_indices_for_scenario(all_names, scenario)
        X_train_sub = X_all[train_idx][:, feat_indices]
        X_test_sub = X_all[test_idx][:, feat_indices]

        # 2. Train RF
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train_sub, y_train)

        # 3. Evaluate
        probs = rf.predict_proba(X_test_sub)[:, 1]
        fpr, fnr, _ = det_curve(y_test, probs)
        min_dcf = np.min(fpr + fnr) / 2

        results.append({
            'Scenario': scenario,
            'Min DCF': min_dcf,
            'Num Features': len(feat_indices)
        })

        if scenario == "All Features":
            rf_all = rf

    # --- Reporting ---
    df_res = pd.DataFrame(results)
    print("\n--- Feature Study Results ---")
    print(df_res)
    df_res.to_csv(os.path.join(RESULTS_DIR, "feature_study_results.csv"), index=False)

    # --- Visualization: Performance ---
    plt.figure(figsize=(10, 6))
    plt.barh(df_res['Scenario'], df_res['Min DCF'], color='skyblue')
    plt.xlabel('Min DCF (Lower is Better)')
    plt.title('Impact of Feature Engineering on RF Performance')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_study_dcf.png"))
    plt.close()

    # --- Visualization: Feature Importance (All) ---
    importances = rf_all.feature_importances_
    # Note: rf_all was trained on *sliced* X, which corresponds to indices for "All Features".
    # Since "All Features" includes EVERYTHING in X_all, the indices match all_names.
    # Let's verify: In get_indices_for_scenario("All Features"), it includes MFCC_ and everything else.
    # So indices match 0..N of all_names.

    sorted_idx = np.argsort(importances)[::-1]
    top_n = 15

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[sorted_idx[:top_n]], align='center')
    plt.yticks(range(top_n), [all_names[i] for i in sorted_idx[:top_n]])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Features (Combined Model)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_study_importance.png"))
    plt.close()

if __name__ == "__main__":
    run_feature_study()