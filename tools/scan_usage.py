import os
import collections

def get_file_info(directory):
    file_info = {}
    for root, _, files in os.walk(directory):
        for name in files:
            try:
                path = os.path.join(root, name)
                size = os.path.getsize(path)
                if size > 100 * 1024 * 1024:  # Only track files > 100MB
                    file_info[path] = size
            except OSError:
                continue
    return file_info

def check_duplicates_and_checkpoints():
    print("--- Scanning for Heavy Checkpoints (Top 10 folders) ---")
    
    # Check runs folders for checkpoint accumulation
    runs_dirs = [
        "/home/billy/25-1kp/vla/runs",
        "/home/billy/25-1kp/vla/RoboVLMs_upstream/runs"
    ]
    
    for runs_dir in runs_dirs:
        if not os.path.exists(runs_dir):
            continue
            
        print(f"\nScanning {runs_dir}...")
        for root, dirs, files in os.walk(runs_dir):
            ckpt_files = [f for f in files if f.endswith(('.pt', '.pth', '.ckpt', '.safetensors', '.bin'))]
            if len(ckpt_files) > 0:
                total_size = sum(os.path.getsize(os.path.join(root, f)) for f in ckpt_files)
                if total_size > 1024 * 1024 * 1024: # > 1GB
                    print(f"Folder: {root}")
                    print(f"  - Count: {len(ckpt_files)} checkpoints")
                    print(f"  - Total Size: {total_size / (1024**3):.2f} GB")
                    # List first 3 and last 3 to give an idea
                    sorted_ckpts = sorted(ckpt_files)
                    if len(sorted_ckpts) > 5:
                        print(f"  - Examples: {', '.join(sorted_ckpts[:3])} ... {', '.join(sorted_ckpts[-3:])}")

    print("\n--- Checking for Potential Large Duplicates (.vlms) ---")
    dir1 = "/home/billy/25-1kp/vla/.vlms"
    dir2 = "/home/billy/25-1kp/vla/RoboVLMs_upstream/.vlms"
    
    if os.path.exists(dir1) and os.path.exists(dir2):
        files1 = {f: os.path.getsize(os.path.join(dir1, f)) for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))}
        files2 = {f: os.path.getsize(os.path.join(dir2, f)) for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))}
        
        common = set(files1.keys()) & set(files2.keys())
        for f in common:
            if files1[f] == files2[f] and files1[f] > 100 * 1024 * 1024:
                 print(f"Potential Duplicate: {f} ({files1[f] / (1024**3):.2f} GB)")
                 print(f"  - {os.path.join(dir1, f)}")
                 print(f"  - {os.path.join(dir2, f)}")

if __name__ == "__main__":
    check_duplicates_and_checkpoints()
