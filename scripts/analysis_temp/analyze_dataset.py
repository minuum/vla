import os
import glob
import h5py
import json
from collections import defaultdict

dataset_dir = "ROS_action/mobile_vla_dataset"
h5_files = glob.glob(os.path.join(dataset_dir, "*.h5"))

stats = defaultdict(lambda: {"count": 0, "total_frames": 0, "files": []})

print(f"Found {len(h5_files)} H5 files.")

for file_path in h5_files:
    filename = os.path.basename(file_path)
    
    # Expected format: episode_TIMESTAMP_TRAJECTORY_TYPE.h5
    # e.g. episode_20251119_170029_1box_hori_left_core_medium.h5
    parts = filename.replace(".h5", "").split("_")
    if len(parts) >= 4:
        # parts[0]=episode, parts[1]=date, parts[2]=time
        # trajectory type is from parts[3] onwards
        traj_type = "_".join(parts[3:])
    else:
        traj_type = "unknown"
        
    try:
        with h5py.File(file_path, 'r') as f:
            # Count frames
            if 'action' in f:
                num_frames = len(f['action'])
            elif 'actions' in f:
                num_frames = len(f['actions'])
            elif 'observations' in f and 'images' in f['observations']: 
                 num_frames = len(f['observations']['images'])
            elif 'images' in f:
                 num_frames = len(f['images'])
            else:
                num_frames = 0
                # print(f"Warning: Could not determine frame count for {filename}")

            stats[traj_type]["count"] += 1
            stats[traj_type]["total_frames"] += num_frames
            stats[traj_type]["files"].append(filename)
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")

print("\n" + "="*80)
print(f"{'Trajectory Type':<40} | {'Count':<5} | {'Avg Frames':<10} | {'Total Frames'}")
print("-" * 80)

for traj_type, data in sorted(stats.items()):
    avg_frames = data["total_frames"] / data["count"] if data["count"] > 0 else 0
    print(f"{traj_type:<40} | {data['count']:<5} | {avg_frames:<10.1f} | {data['total_frames']}")
print("="*80)


