import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import collections

def analyze_actions():
    dataset_dir = Path("/home/billy/25-1kp/vla/ROS_action/basket_dataset")
    h5_files = list(dataset_dir.glob("*left*.h5"))
    
    total_samples = 0
    counts = collections.defaultdict(int)
    
    # Mapping logic used in dataset
    # 0: Stop, 1: Forward, 2: Left, 3: Right, 4: Diag FL, 5: Diag FR
    
    print(f"Analyzing {len(h5_files)} files...")
    
    for f_path in tqdm(h5_files):
        with h5py.File(f_path, 'r') as f:
            actions = f['actions'][:] # (frames, 2+)
            for a in actions:
                x, y = a[0], a[1]
                if abs(x) < 0.5 and abs(y) < 0.5: label = 0
                elif x > 0.5 and abs(y) < 0.1: label = 1
                elif abs(x) < 0.1 and y > 0.5: label = 2
                elif abs(x) < 0.1 and y < -0.5: label = 3
                elif x > 0.5 and y > 0.5: label = 4
                elif x > 0.5 and y < -0.5: label = 5
                else: label = 0 # Default
                
                counts[label] += 1
                total_samples += 1

    print("\nAction Distribution:")
    labels = {0: "Stop", 1: "Forward", 2: "Left", 3: "Right", 4: "Diag FL", 5: "Diag FR"}
    for i in range(6):
        pct = (counts[i] / total_samples) * 100 if total_samples > 0 else 0
        print(f"  Class {i} ({labels[i]:<8}): {counts[i]:>6} ({pct:>5.2f}%)")

if __name__ == "__main__":
    analyze_actions()
