import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import det_curve
from utils.audio_loader import crawl_dataset, get_audio_chunks, preprocess_audio
from utils.features import extract_extended_features
from utils.analysis import get_best_config

# Constants
DATASET_DIR = "dataset"
MODELS_DIR = "models"
# SR = 16000 # Now dynamic
CHUNK_DUR = 1.0

def train_production_model():
    # 1. Load Preprocessing Config
    config = get_best_config("results/hyperparameter_search_results.csv")
    if not config: return

    target_sr = config['target_sr']
    print(f"Using Best Preprocessing: {config['filter_config']}")
    print(f"Target SR: {target_sr}, Feature Type: {config['feature_type']}")

    # Add spectral gating if it was selected in random search
    production_filters = (config['filter_config'] or []).copy()
    if config['noise_reduction']:
         if 'spectral_gating' not in production_filters:
             production_filters.append('spectral_gating')

    # 2. Extract All Features (Production Config)
    # We decided to use "All Features" for the final model
    feature_flags = {
        'use_deltas': True,
        'use_centroid': True,
        'use_flux': True,
        'use_zcr': True,
        'feature_type': config['feature_type']
    }

    print("Extracting full feature set for production training...")
    dataset = crawl_dataset(DATASET_DIR)
    X = []
    y = []

    for label_idx, label in enumerate(['no', 'yes']):
        for file_path in dataset[label]:
            try:
                chunks = get_audio_chunks(file_path, target_sr, CHUNK_DUR)
                for chunk in chunks:
                    y_proc = preprocess_audio(chunk, target_sr, filter_config=production_filters)

                    # Extract with all flags
                    feats, _ = extract_extended_features(
                        y_proc, target_sr,
                        n_mfcc=config['n_mfcc'],
                        n_fft=config['n_fft'],
                        hop_length=config['hop_length'],
                        **feature_flags
                    )

                    X.append(feats)
                    y.append(label_idx)
            except Exception as e:
                print(f"Error {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"Training Data: {X.shape}")

    # 3. Train Model
    # We train on the full dataset or a robust split. Let's do a split to log metrics for the user,
    # but ideally for "production" you might want to retrain on everything.
    # For now, let's keep the split to ensure we have a validity check.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 4. Validate
    probs = rf.predict_proba(X_test)[:, 1]
    fpr, fnr, _ = det_curve(y_test, probs)
    min_dcf = np.min(fpr + fnr) / 2
    print(f"Production Model Validation - Min DCF: {min_dcf:.4f}")

    # 5. Save Artifacts
    # We save everything needed to reproduce feature extraction during inference
    artifacts = {
        'model': rf,
        'preprocessing_config': config,
        'feature_flags': feature_flags,
        'model_type': 'RandomForest_AllFeatures'
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "production_model.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(artifacts, f)

    print(f"Production model saved to {save_path}")

if __name__ == "__main__":
    train_production_model()