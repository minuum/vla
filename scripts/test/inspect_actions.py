import h5py
import numpy as np

file_path = "/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2/test/episode_20260129_011917_basket_1box_hori_left_core_medium.h5"

with h5py.File(file_path, 'r') as f:
    actions = f['actions'][:]
    print(f"File: {file_path}")
    print(f"Actions shape: {actions.shape}")
    print(f"Max action: {np.max(actions, axis=0)}")
    print(f"Min action: {np.min(actions, axis=0)}")
    print(f"Mean action: {np.mean(actions, axis=0)}")
    print("\nFirst 10 actions:")
    print(actions[:10])
