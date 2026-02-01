import h5py
import numpy as np
import sys
from pathlib import Path

# Add python path
sys.path.append("/home/billy/25-1kp/vla/RoboVLMs_upstream")

# Import normalize_action directly
from robovlms.data.data_utils import normalize_action

def check_clipping():
    # 20251204 파일 중 하나 선택
    data_dir = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")
    h5_files = sorted([f for f in data_dir.glob("*.h5") if "20251204" in f.name])
    
    if not h5_files:
        print("No HDF5 files found.")
        return

    target_file = h5_files[0]
    print(f"📂 Checking file: {target_file.name}")
    
    with h5py.File(target_file, "r") as f:
        actions = f["actions"][:] # [T, 3] -> linear_x, linear_y, angular_z
        
    print(f"\nOriginal Actions Shape: {actions.shape}")
    
    lin_x = actions[:, 0]
    lin_y = actions[:, 1]
    
    print(f"Original linear_x Unique Values (First 20): {np.unique(lin_x)[:20]}")
    print(f"Original linear_y Unique Values (First 20): {np.unique(lin_y)[:20]}")
    
    # Simulate Dataset Normalization (Min=-1, Max=1)
    norm_min = -1.0
    norm_max = 1.0
    
    norm_actions = normalize_action(actions, action_min=norm_min, action_max=norm_max)
    
    # Check clipping
    # normalize_action does: clip(min, max) -> scale to [-1, 1]
    # So if original was > 1, it became 1, then scaled to 1.
    # If original was 1.15, clip -> 1.0, norm -> 1.0
    
    # We compare original vs. denormalized (using only denorm logic)
    # But checking if normalized value is exactly 1.0 or -1.0 is easier (if original was strictly > 1)
    
    norm_x = norm_actions[:, 0]
    norm_y = norm_actions[:, 1]
    
    print(f"\nNormalized linear_x Range: [{norm_x.min():.4f}, {norm_x.max():.4f}]")
    print(f"Normalized linear_y Range: [{norm_y.min():.4f}, {norm_y.max():.4f}]")
    
    # Identify clipped indices based on original values
    clipped_x_indices = np.where((lin_x < norm_min) | (lin_x > norm_max))[0]
    clipped_y_indices = np.where((lin_y < norm_min) | (lin_y > norm_max))[0]
    
    print(f"\n⚠️ Clipping Analysis (Threshold: {norm_min} ~ {norm_max})")
    print(f"  linear_x Clipped Count: {len(clipped_x_indices)} / {len(lin_x)} ({len(clipped_x_indices)/len(lin_x)*100:.2f}%)")
    if len(clipped_x_indices) > 0:
        print(f"  Sample Clipped X: {lin_x[clipped_x_indices[:5]]} -> {norm_x[clipped_x_indices[:5]]}")
        
    print(f"  linear_y Clipped Count: {len(clipped_y_indices)} / {len(lin_y)} ({len(clipped_y_indices)/len(lin_y)*100:.2f}%)")
    if len(clipped_y_indices) > 0:
        print(f"  Sample Clipped Y: {lin_y[clipped_y_indices[:5]]} -> {norm_y[clipped_y_indices[:5]]}")

    # Check scaling correctness for non-clipped values
    # For value 0.5: 2 * (0.5 - (-1)) / (1 - (-1)) - 1 = 2 * 1.5 / 2 - 1 = 1.5 - 1 = 0.5
    # For value 0.0: 2 * (0 - (-1)) / 2 - 1 = 1 - 1 = 0.0
    # MinMax scaling does not change values if min=-1, max=1 !! (It's identity transform for range [-1, 1])
    # Let's verify this.
    
    print("\nScaling Function Verification (Identity check for [-1, 1] range):")
    sample_val = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
    norm_sample = normalize_action(sample_val, action_min=-1.0, action_max=1.0)
    print(f"  Input: {sample_val}")
    print(f"  Output: {norm_sample}")
    
    if np.allclose(sample_val, norm_sample):
        print("  ✅ Scaling is Identity for [-1, 1] range.")
    else:
        print("  ❌ Scaling is NOT Identity.")

if __name__ == "__main__":
    check_clipping()
