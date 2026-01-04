import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.audio_loader import crawl_dataset, get_audio_chunks, preprocess_audio, crawl_dataset_simple
from utils.features import extract_mfcc_features
from utils.stats_engine import calculate_det_metrics, save_statistical_report

def run_experiment():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    dataset = crawl_dataset(config['paths']['data_dir'])
    #dataset = crawl_dataset_simple(config['paths']['data_dir'])
    
    sr = config['preprocessing']['sample_rate']
    chunk_dur = config['preprocessing']['chunk_duration']
    agg_method = config['features']['mfcc']['aggregation_method']

    for denoise in config['preprocessing']['denoise']:
        for n_mfcc in config['features']['mfcc']['n_mfcc_variants']:
            
            run_id = f"MFCC_{n_mfcc}_Denoise_{denoise}_Agg_{agg_method}"
            print(f"\nProcessing Experiment: {run_id}")

            # Prepare storage for each coefficient's scores
            coeff_scores = {i: {'yes': [], 'no': []} for i in range(n_mfcc)}

            for label in ['yes', 'no']:
                print(f"  Processing {label} chunks...")
                for file_path in dataset[label]:
                    # Split file into 1s chunks
                    chunks = get_audio_chunks(file_path, sr, chunk_dur)
                    
                    for chunk in chunks:
                        y = preprocess_audio(chunk, sr, denoise_flag=denoise)
                        # Extract aggregated vector (length n_mfcc)
                        feature_vec = extract_mfcc_features(y, sr, n_mfcc, method=agg_method)
                        
                        for i in range(n_mfcc):
                            coeff_scores[i][label].append(feature_vec[i])

            # Generation of Individual DET Curves
            for i in range(n_mfcc):
                feature_name = f"CoeffIdx_{i}"
                out_folder = os.path.join(config['paths']['output_dir'], run_id, feature_name)
                os.makedirs(out_folder, exist_ok=True)

                yes_vals = np.array(coeff_scores[i]['yes'])
                no_vals = np.array(coeff_scores[i]['no'])

                if len(no_vals) == 0 or len(yes_vals) == 0:
                    continue

                far, frr, _ = calculate_det_metrics(yes_vals, no_vals)
                
                plt.figure(figsize=(8, 6))
                plt.plot(far, frr, lw=2, color='darkorange')
                plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
                plt.title(f"DET Curve: {feature_name}\n({run_id})")
                plt.xlabel("False Alarm Rate")
                plt.ylabel("Miss Rate")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(out_folder, "det_curve.png"))
                plt.close()

                save_statistical_report(out_folder, f"Coeff_{i}", yes_vals, no_vals)

    print("\nAll experiments complete. Results saved to:", config['paths']['output_dir'])

if __name__ == "__main__":
    run_experiment()