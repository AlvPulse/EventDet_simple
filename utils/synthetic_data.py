import numpy as np
import os
import soundfile as sf
import shutil
from scipy import signal

def ensure_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def generate_synthetic_data(base_dir="dataset", sr=16000, num_files=20, duration=3.0):
    yes_dir = os.path.join(base_dir, "yes")
    no_dir = os.path.join(base_dir, "no")

    ensure_dir(yes_dir)
    ensure_dir(no_dir)

    print(f"Generating {num_files} 'yes' samples and {num_files} 'no' samples...")

    for i in range(num_files):
        # 1. Generate Noise (Background)
        t = np.linspace(0, duration, int(sr * duration))
        # White noise
        noise = np.random.normal(0, 0.02, len(t))

        # 2. 'No' Sample: Just Noise
        no_sample = noise
        sf.write(os.path.join(no_dir, f"no_{i:03d}.wav"), no_sample, sr)

        # 3. 'Yes' Sample: Noise + Transient Event
        # Create a chirp/pulse event
        event_start = np.random.randint(int(0.1 * sr), int((duration - 0.5) * sr))
        event_duration = 0.3 # 300ms
        event_t = np.linspace(0, event_duration, int(event_duration * sr))

        # Chirp signal: 1000Hz to 2000Hz
        chirp = signal.chirp(event_t, f0=1000, t1=event_duration, f1=4000, method='linear')
        chirp *= np.hanning(len(chirp)) * 0.5 # Window and scale

        yes_sample = noise.copy()
        yes_sample[event_start:event_start + len(chirp)] += chirp

        sf.write(os.path.join(yes_dir, f"yes_{i:03d}.wav"), yes_sample, sr)

    print("Dataset generation complete.")

if __name__ == "__main__":
    generate_synthetic_data()