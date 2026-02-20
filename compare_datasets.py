import h5py
import numpy as np
from PIL import Image
import os

def extract_and_analyze(h5_path, output_prefix):
    print(f"Analyzing {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        images = f['images'][:] # (N, H, W, 3)
        n = images.shape[0]
        
        # Pick 2 frames: early and middle
        indices = [0, n // 2]
        results = []
        
        for i, idx in enumerate(indices):
            img = images[idx]
            # Save image
            out_name = f"{output_prefix}_frame_{idx}.png"
            Image.fromarray(img).save(out_name)
            
            # Calculate mean RGB
            mean_rgb = np.mean(img, axis=(0, 1))
            results.append({
                "path": out_name,
                "mean_rgb": mean_rgb.tolist(),
                "std_rgb": np.std(img, axis=(0, 1)).tolist()
            })
            print(f"  Frame {idx} saved to {out_name}. Mean RGB: {mean_rgb}")
            
    return results

# Paths
v1_path = "ROS_action/basket_dataset/episode_20260129_010041_basket_1box_hori_left_core_medium.h5"
v2_path = "ROS_action/basket_dataset_v2/episode_20260129_010041_basket_1box_hori_left_core_medium.h5"

if not os.path.exists("analysis"):
    os.makedirs("analysis")

res_v1 = extract_and_analyze(v1_path, "analysis/v1")
res_v2 = extract_and_analyze(v2_path, "analysis/v2")

print("\n--- Summary Comparison ---")
for i in range(2):
    m1 = res_v1[i]['mean_rgb']
    m2 = res_v2[i]['mean_rgb']
    diff = [m2[j] - m1[j] for j in range(3)]
    print(f"Frame Pair {i}:")
    print(f"  V1 RGB: {m1}")
    print(f"  V2 RGB: {m2}")
    print(f"  Diff (V2 - V1): {diff}")
