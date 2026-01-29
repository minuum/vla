import os
import glob
import h5py
import numpy as np

DATA_DIR = "/home/billy/25-1kp/vla/ROS_action/basket_dataset"

def inspect_actions():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
    print(f"Found {len(files)} files. Sampling actions from dataset...")
    
    all_linear_x = []
    all_linear_y = []
    
    # Sample 50 random files to get a good distribution
    np.random.seed(42)
    sample_indices = np.random.choice(len(files), min(50, len(files)), replace=False)
    
    for idx in sample_indices:
        filepath = files[idx]
        try:
            with h5py.File(filepath, 'r') as f:
                if 'actions' in f:
                    actions = f['actions'][:]
                    # actions shape is typically (T, 3) -> x, y, z(angular) or just 2
                    # The collector saves partial actions? Let's check shape
                    # Collector saves: [linear_x, linear_y, angular_z]
                    
                    if actions.shape[1] >= 2:
                        all_linear_x.extend(actions[:, 0])
                        all_linear_y.extend(actions[:, 1])
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # Convert to numpy for analysis
    arr_x = np.array(all_linear_x)
    arr_y = np.array(all_linear_y)
    
    print("\n=== Action value analysis (Linear X) ===")
    unique_x, counts_x = np.unique(np.round(arr_x, 4), return_counts=True)
    for val, count in zip(unique_x, counts_x):
        print(f"Value: {val:6.4f} | Count: {count}")
        
    print("\n=== Action value analysis (Linear Y) ===")
    unique_y, counts_y = np.unique(np.round(arr_y, 4), return_counts=True)
    for val, count in zip(unique_y, counts_y):
        print(f"Value: {val:6.4f} | Count: {count}")

if __name__ == "__main__":
    inspect_actions()
