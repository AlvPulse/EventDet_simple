import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, det_curve
from utils.audio_loader import crawl_dataset, get_audio_chunks, preprocess_audio
from utils.features import extract_all_mfcc_stats
from utils.analysis import get_best_config
from utils.stats_engine import calculate_det_metrics

# Constants
DATASET_DIR = "dataset"
RESULTS_DIR = "results"
SR = 16000
CHUNK_DUR = 1.0

def load_data(config):
    dataset = crawl_dataset(DATASET_DIR)
    X_rich = []       # For ML models (Decision Tree, Random Forest)
    X_baseline = []   # For Single Energy Detector (raw value of best coeff)
    y = []
    feature_names = []

    print("Loading and processing data...")
    for label_idx, label in enumerate(['no', 'yes']): # 0 for no, 1 for yes
        for file_path in dataset[label]:
            try:
                chunks = get_audio_chunks(file_path, SR, CHUNK_DUR)
                for chunk in chunks:
                    # Preprocess with BEST config
                    y_proc = preprocess_audio(chunk, SR, filter_config=config['filter_config'])

                    # 1. Extract Rich Features (ML)
                    feats, stat_names = extract_all_mfcc_stats(
                        y_proc, SR,
                        n_mfcc=config['n_mfcc'],
                        n_fft=config['n_fft'],
                        hop_length=config['hop_length']
                    )

                    # 2. Extract Baseline Feature
                    # We need to manually identify which feature corresponds to the baseline "mfcc_coeff_idx" and "agg_method"
                    # But wait, extract_all_mfcc_stats returns a flattened vector.
                    # Let's verify the index.
                    # Structure: [c0_mean, c0_max... c1_mean...]
                    # stat_names order: ["mean", "max", "min", "median", "std", "max_mean"] (Length 6)
                    # Index = (coeff_idx * 6) + stat_names.index(agg_method)

                    # NOTE: We need the config's specific baseline aggregation method and index
                    baseline_agg = config['baseline_feature'].split('(')[1].split(')')[0] # e.g. "max" from "Coeff_0 (max)"
                    baseline_coeff = int(config['baseline_feature'].split(' ')[0].split('_')[1])

                    stat_idx = stat_names.index(baseline_agg)
                    feat_idx = (baseline_coeff * len(stat_names)) + stat_idx
                    baseline_val = feats[feat_idx]

                    X_rich.append(feats)
                    X_baseline.append(baseline_val)
                    y.append(label_idx)

                    # Generate feature names only once
                    if not feature_names:
                        for i in range(config['n_mfcc']):
                            for stat in stat_names:
                                feature_names.append(f"MFCC_{i}_{stat}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return np.array(X_rich), np.array(X_baseline), np.array(y), feature_names

def find_min_dcf_threshold(scores, labels):
    """
    Finds the threshold that minimizes DCF.
    DCF = (FRR + FAR) / 2  (assuming equal priors and costs)
    """
    # Combine scores and search thresholds
    # Note: Our baseline might be "higher is better" or "lower is better" depending on the metric.
    # Energy is usually higher -> Yes.

    yes_scores = scores[labels == 1]
    no_scores = scores[labels == 0]

    if len(yes_scores) == 0 or len(no_scores) == 0:
        return 0.0

    # Use existing det metrics calculation
    far, frr, thresholds = calculate_det_metrics(yes_scores, no_scores, num_steps=200)

    dcf = (np.array(far) + np.array(frr)) / 2
    min_idx = np.argmin(dcf)
    best_threshold = thresholds[min_idx]
    min_dcf_val = dcf[min_idx]

    return best_threshold, min_dcf_val

def train_and_save():
    # 1. Get Best Config
    config = get_best_config("results/hyperparameter_search_results.csv")
    if not config:
        print("Could not load best config. Run random_search.py first.")
        return

    print(f"Using Fixed Preprocessing: {config}")

    # 2. Load Data
    X_rich, X_baseline, y, feature_names = load_data(config)
    print(f"Data loaded: {X_rich.shape} samples")

    # 3. Train/Test Split
    # We split indices so we can split both X_rich and X_baseline identically
    indices = np.arange(len(y))
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(indices, y, test_size=0.2, random_state=42, stratify=y)

    X_train_rich = X_rich[X_train_idx]
    X_test_rich = X_rich[X_test_idx]

    X_train_base = X_baseline[X_train_idx]
    X_test_base = X_baseline[X_test_idx]

    # --- A. Baseline Model (Threshold Calibration) ---
    print("Calibrating Baseline Model...")
    baseline_threshold, baseline_dcf = find_min_dcf_threshold(X_train_base, y_train)
    print(f"Baseline Threshold (Min DCF): {baseline_threshold:.4f} (Train DCF: {baseline_dcf:.4f})")

    # --- B. Decision Tree ---
    print("Training Decision Tree...")
    dt_params = {
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'criterion': ['gini']
    }
    dt = DecisionTreeClassifier(random_state=42)
    dt_grid = GridSearchCV(dt, dt_params, cv=5, scoring='roc_auc', n_jobs=-1)
    dt_grid.fit(X_train_rich, y_train)
    best_dt = dt_grid.best_estimator_
    print(f"Best DT Params: {dt_grid.best_params_}")

    # --- C. Random Forest ---
    print("Training Random Forest...")
    rf_params = {
        'n_estimators': [20, 50],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
    rf_grid.fit(X_train_rich, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"Best RF Params: {rf_grid.best_params_}")

    # --- Evaluation (on Test Set) ---
    # 1. Baseline
    # Evaluate using the threshold found on train set
    # Predictions: 1 if score > threshold
    base_preds = (X_test_base > baseline_threshold).astype(int)
    # Calc Metrics manually for test set to confirm
    far_base = np.mean(base_preds[y_test == 0] == 1) # False Pos / Negatives
    frr_base = np.mean(base_preds[y_test == 1] == 0) # False Neg / Positives
    dcf_base = (far_base + frr_base) / 2

    # 2. Decision Tree
    dt_probs = best_dt.predict_proba(X_test_rich)[:, 1]
    fpr_dt, fnr_dt, _ = det_curve(y_test, dt_probs)
    dcf_dt = np.min(fpr_dt + fnr_dt) / 2

    # 3. Random Forest
    rf_probs = best_rf.predict_proba(X_test_rich)[:, 1]
    fpr_rf, fnr_rf, _ = det_curve(y_test, rf_probs)
    dcf_rf = np.min(fpr_rf + fnr_rf) / 2

    print(f"\n--- Test Set Min DCF Comparison ---")
    print(f"Baseline:      {dcf_base:.4f}")
    print(f"Decision Tree: {dcf_dt:.4f}")
    print(f"Random Forest: {dcf_rf:.4f}")

    # --- Save Everything ---
    artifacts = {
        'config': config,
        'baseline_threshold': baseline_threshold,
        'model_dt': best_dt,
        'model_rf': best_rf,
        'feature_names': feature_names
    }

    save_path = os.path.join(RESULTS_DIR, "best_models.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"\nModels and config saved to {save_path}")

if __name__ == "__main__":
    train_and_save()