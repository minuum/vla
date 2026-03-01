import os
import shutil

def clean_checkpoints(base_dir):
    print(f"--- Scanning {base_dir} ---")
    for root, dirs, files in os.walk(base_dir):
        ckpts = [f for f in files if f.endswith('.ckpt')]
        if len(ckpts) <= 1:
            continue
            
        print(f"Cleaning {root} ({len(ckpts)} checkpoints found)")
        # Paths with full path
        ckpt_paths = [os.path.join(root, f) for f in ckpts]
        # Sort by modification time (newest first)
        ckpt_paths.sort(key=os.path.getmtime, reverse=True)
        
        # Keep list:
        # 1. Any file with 'last' in name
        # 2. The single newest file (if 'last' isn't already the newest)
        to_keep = []
        for p in ckpt_paths:
            if 'last' in os.path.basename(p).lower():
                to_keep.append(p)
        
        # Also keep the actual newest one if it's not already in to_keep
        if ckpt_paths[0] not in to_keep:
            to_keep.append(ckpt_paths[0])
            
        # Delete the rest
        for p in ckpt_paths:
            if p not in to_keep:
                try:
                    size_gb = os.path.getsize(p) / (1024**3)
                    print(f"  [DELETE] {os.path.basename(p)} ({size_gb:.2f} GB)")
                    os.remove(p)
                except Exception as e:
                    print(f"  [ERROR] Failed to delete {p}: {e}")

if __name__ == "__main__":
    runs_path = "/home/billy/25-1kp/vla/runs"
    if os.path.exists(runs_path):
        clean_checkpoints(runs_path)
    
    # Check for other heavy folders
    print("\n--- Summary After Run ---")
    os.system("df -h /")
