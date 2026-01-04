import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, det_curve
from utils.audio_loader import crawl_dataset, get_audio_chunks, preprocess_audio
from utils.features import extract_all_mfcc_stats
from utils.analysis import get_best_config

# Constants
DATASET_DIR = "dataset"
RESULTS_DIR = "results"
SR = 16000
CHUNK_DUR = 1.0

def load_data(config):
    dataset = crawl_dataset(DATASET_DIR)
    X = []
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

                    # Extract rich features
                    feats, stat_names = extract_all_mfcc_stats(
                        y_proc, SR,
                        n_mfcc=config['n_mfcc'],
                        n_fft=config['n_fft'],
                        hop_length=config['hop_length']
                    )

                    X.append(feats)
                    y.append(label_idx)

                    # Generate feature names only once
                    if not feature_names:
                        for i in range(config['n_mfcc']):
                            for stat in stat_names:
                                feature_names.append(f"MFCC_{i}_{stat}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y), feature_names

def optimize_decision_tree():
    # 1. Get Best Config
    config = get_best_config("results/hyperparameter_search_results.csv")
    if not config:
        print("Could not load best config. Run random_search.py first.")
        return

    print(f"Using Fixed Preprocessing: {config}")

    # 2. Load Data
    X, y, feature_names = load_data(config)
    print(f"Data loaded: {X.shape} samples")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Decision Tree Optimization (Grid Search)
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    dt = DecisionTreeClassifier(random_state=42)
    grid = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

    print("Tuning Decision Tree...")
    grid.fit(X_train, y_train)

    best_tree = grid.best_estimator_
    print(f"Best Tree Params: {grid.best_params_}")

    # 5. Evaluation
    y_pred_proba = best_tree.predict_proba(X_test)[:, 1]

    # Calculate ROC/AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Calculate DET
    # DET uses False Alarm Rate (FPR) vs Miss Rate (1 - TPR)
    # sklearn det_curve returns fpr, fnr, thresholds
    det_fpr, det_fnr, _ = det_curve(y_test, y_pred_proba)

    # Estimate EER (where FAR = FRR)
    # Using simple intersection
    diffs = np.abs(det_fpr - det_fnr)
    min_diff_idx = np.argmin(diffs)
    eer = (det_fpr[min_diff_idx] + det_fnr[min_diff_idx]) / 2

    # Min DCF (Assuming equal priors/costs for now as simple metric)
    min_dcf = np.min(det_fpr + det_fnr) / 2

    # 6. Reporting & Visualization
    report = (
        f"--- Decision Tree Results ---\n"
        f"Used Filters: {config['filter_config']}\n"
        f"MFCCs: {config['n_mfcc']}, FFT: {config['n_fft']}\n"
        f"Best Tree Params: {grid.best_params_}\n"
        f"\n--- Performance Comparison ---\n"
        f"Metric        | Baseline (Single Coeff) | Decision Tree\n"
        f"--------------|-------------------------|--------------\n"
        f"AUC           | {config['baseline_auc']:.4f}                  | {roc_auc:.4f}\n"
        f"EER           | {config['baseline_eer']:.4f}                  | {eer:.4f}\n"
        f"Min DCF       | {config['baseline_min_dcf']:.4f}                  | {min_dcf:.4f}\n"
    )

    print("\n" + report)
    with open(os.path.join(RESULTS_DIR, "decision_tree_results.txt"), "w") as f:
        f.write(report)

    # Plot Comparison (DET)
    plt.figure(figsize=(10, 6))
    plt.plot(det_fpr, det_fnr, label=f'Decision Tree (AUC={roc_auc:.2f})', lw=2)
    # Note: We can't easily replot the Baseline curve without the raw data from that specific run,
    # but we can plot the point for EER if we wanted. For now, just showing the Tree's DET.
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Miss Rate')
    plt.title('DET Curve: Decision Tree Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "decision_tree_det.png"))
    plt.close()

    # Feature Importance
    importances = best_tree.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_n = 10
    print(f"\nTop {top_n} Features:")
    for i in range(top_n):
        if i < len(indices):
            print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")

    # Visualize Tree
    plt.figure(figsize=(20, 10))
    plot_tree(best_tree, feature_names=feature_names, filled=True, max_depth=3, fontsize=10)
    plt.title("Decision Tree (Top 3 Levels)")
    plt.savefig(os.path.join(RESULTS_DIR, "decision_tree_viz.png"))
    plt.close()

if __name__ == "__main__":
    optimize_decision_tree()