import pandas as pd
import ast

def get_best_config(csv_path):
    try:
        df = pd.read_csv(csv_path)
        # Find row with minimum EER (or Min DCF/AUC)
        best_row = df.loc[df['eer'].idxmin()]

        print("Best Configuration found:")
        print(best_row)

        # Parse filter config safely
        filter_config = best_row['filter_config']
        if pd.isna(filter_config) or filter_config == 'None':
            filter_config = None
        else:
            try:
                filter_config = ast.literal_eval(filter_config)
            except:
                filter_config = None # Fallback

        config = {
            'filter_config': filter_config,
            'n_fft': int(best_row['n_fft']),
            'hop_length': int(best_row['hop_length']),
            'n_mfcc': int(best_row['n_mfcc']),
            'target_sr': int(best_row.get('target_sr', 16000)),
            'feature_type': best_row.get('feature_type', 'mfcc'),
            'noise_reduction': bool(best_row.get('noise_reduction', False)),
            # Baseline metrics for comparison
            'baseline_eer': float(best_row['eer']),
            'baseline_auc': float(best_row['auc']),
            'baseline_min_dcf': float(best_row['min_dcf']),
            'baseline_feature': f"Coeff_{best_row['mfcc_coeff_idx']} ({best_row['agg_method']})"
        }
        return config
    except Exception as e:
        print(f"Error reading results: {e}")
        return None

if __name__ == "__main__":
    get_best_config("results/hyperparameter_search_results.csv")