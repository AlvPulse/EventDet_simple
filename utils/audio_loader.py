import librosa
import numpy as np
from pathlib import Path
import os

def crawl_dataset_simple(data_dir):
    dataset_map = {'yes': [], 'no': []}
    for label in dataset_map.keys():
        # Ensure the path is absolute and uses correct separators
        folder_path = os.path.abspath(os.path.join(data_dir, label))
        
        if os.path.exists(folder_path):
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.wav', '.mp3'))]
            dataset_map[label] = files
            print(f"--- Found {len(files)} files in category: {label}") # Verification
        else:
            print(f"!!! Error: Folder not found at {folder_path}")
            
    return dataset_map

def crawl_dataset(data_dir):
    """
    Uses pathlib for robust recursive searching across Windows/Linux.
    """
    dataset_map = {'yes': [], 'no': []}
    root_path = Path(data_dir)
    
    if not root_path.exists():
        print(f"!!! Error: Root path does not exist: {root_path}")
        return dataset_map

    for label in dataset_map.keys():
        label_path = root_path / label
        if label_path.exists():
            # rglob("**/*") finds all files in the directory and ALL subdirectories
            files = [
                str(f) for f in label_path.rglob("*") 
                if f.suffix.lower() in ['.wav', '.mp3', '.flac']
            ]
            dataset_map[label] = files
            print(f"--- Found {len(files)} files in category: {label}")
        else:
            print(f"!!! Error: Folder not found: {label_path}")
            
    return dataset_map

def preprocess_audio(y, sr, denoise_flag=False):
    """
    Handles denoising logic. 
    You can expand this to include Spectral Subtraction or Bandpass filters.
    """
    if not denoise_flag:
        return y
    
    # 1. Simple Pre-emphasis filter (boosts high frequencies, reduces low-end noise)
    y_processed = librosa.effects.preemphasis(y)
    
    # 2. Add more complex denoising here (e.g., noise gate or spectral gating)
    # y_processed = some_denoise_function(y_processed)
    
    return y_processed

def get_audio_chunks(file_path, sr, chunk_duration):
    """
    Loads a file and splits it into 1-second chunks.
    """
    y, _ = librosa.load(file_path, sr=sr)
    chunk_samples = int(chunk_duration * sr)
    
    # Split audio into chunks of fixed length
    chunks = [y[i:i + chunk_samples] for i in range(0, len(y), chunk_samples) 
              if len(y[i:i + chunk_samples]) == chunk_samples]
    return chunks