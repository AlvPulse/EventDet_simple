import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_det_metrics(yes_scores, no_scores, num_steps=100):
    """
    Calculates False Alarm Rate (FAR) and False Rejection Rate (FRR) 
    across a range of thresholds.
    """
    # Combine scores to find the global range for thresholds
    all_scores = np.concatenate([yes_scores, no_scores])
    thresholds = np.linspace(np.min(all_scores), np.max(all_scores), num_steps)
    
    far_list = []
    frr_list = []
    
    for t in thresholds:
        # FAR: Percentage of 'no' (background) samples that exceed the threshold
        far = np.mean(no_scores > t)
        # FRR: Percentage of 'yes' (event) samples that fail to reach the threshold
        frr = np.mean(yes_scores <= t)
        
        far_list.append(far)
        frr_list.append(frr)
        
    return far_list, frr_list, thresholds

def save_statistical_report(output_path, name, yes_scores, no_scores):
    """
    Saves a text file with basic stats (mean, std) for the specific hyperparameter run.
    """
    report = (
        f"Experiment: {name}\n"
        f"--- YES (Event) --- \n Mean: {np.mean(yes_scores):.4f}, Std: {np.std(yes_scores):.4f}\n"
        f"--- NO (Background) --- \n Mean: {np.mean(no_scores):.4f}, Std: {np.std(no_scores):.4f}\n"
    )
    with open(os.path.join(output_path, f"{name}_stats.txt"), "w") as f:
        f.write(report)