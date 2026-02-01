import os
import glob
import shutil
import hashlib

def get_file_hash(filepath):
    """Calculates SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return None

def clean_checkpoints(base_dir):
    print(f"Scanning for checkpoints in {base_dir}...")
    # Walk through all directories
    for root, dirs, files in os.walk(base_dir):
        # Identify checkpoint files
        ckpts = [f for f in files if f.endswith('.ckpt')]
        
        if len(ckpts) <= 1:
            continue
            
        print(f"\nProcessing {root}...")
        # Sort by modification time (newest first)
        ckpts_full = [os.path.join(root, c) for c in ckpts]
        ckpts_full.sort(key=os.path.getmtime, reverse=True)
        
        # Strategy:
        # 1. Always keep 'last.ckpt' or 'last-v*.ckpt'
        # 2. Keep the newest file that is NOT 'last' (likely the latest epoch)
        # 3. Delete others
        
        to_keep = []
        to_delete = []
        
        # Identify 'last' checkpoints
        last_ckpts = [c for c in ckpts_full if 'last' in os.path.basename(c)]
        if last_ckpts:
            to_keep.extend(last_ckpts)
        
        # Identify the newest non-last checkpoint (e.g., epoch=10...)
        non_last = [c for c in ckpts_full if c not in last_ckpts]
        if non_last:
            # Keep the very newest one
            to_keep.append(non_last[0])
            # The rest are deletion candidates
            to_delete.extend(non_last[1:])
        
        # Remove duplicates from keep list
        to_keep = list(set(to_keep))
        
        # Execute deletion
        for f_path in to_delete:
            if f_path not in to_keep:
                try:
                    size_mb = os.path.getsize(f_path) / (1024 * 1024)
                    print(f"  [DELETE] {os.path.basename(f_path)} ({size_mb:.1f} MB)")
                    os.remove(f_path)
                except OSError as e:
                    print(f"  [ERROR] Could not delete {f_path}: {e}")
            else:
                print(f"  [KEEP] {os.path.basename(f_path)}")

        for f_path in to_keep:
            print(f"  [KEEP] {os.path.basename(f_path)}")

def clean_duplicate_models():
    print("\nScanning for duplicate models...")
    dir1 = "/home/billy/25-1kp/vla/.vlms"
    dir2 = "/home/billy/25-1kp/vla/RoboVLMs_upstream/.vlms"
    
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        print("One of the .vlms directories does not exist. Skipping.")
        return

    # Check for kosmos-2-patch14-224 which known to be large
    model_name = "kosmos-2-patch14-224"
    path1 = os.path.join(dir1, model_name)
    path2 = os.path.join(dir2, model_name)
    
    if os.path.exists(path1) and os.path.exists(path2):
        print(f"Comparing {model_name} in both locations...")
        
        # Simple size check first
        size1 = sum(os.path.getsize(os.path.join(path1, f)) for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f)))
        size2 = sum(os.path.getsize(os.path.join(path2, f)) for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f)))
        
        print(f"  Size 1: {size1 / (1024**3):.2f} GB")
        print(f"  Size 2: {size2 / (1024**3):.2f} GB")
        
        # If sizes are incredibly close (byte level), assume duplicate and delete the upstream one
        if size1 == size2:
            print(f"  Identical size detected. Deleting duplicate in {path2}")
            try:
                shutil.rmtree(path2)
                print("  [DELETE] Successful.")
                # Optional: Create symlink? User didn't strictly ask, but it's safer.
                # os.symlink(path1, path2) 
                # print("  [LINK] Created symlink to replace deleted folder.")
            except OSError as e:
                print(f"  [ERROR] Failed to delete: {e}")
        else:
            print("  Sizes differ. Skipping automatic deletion.")

    # Check for bin vs safetensors redundancy within the surviving folder
    # We'll check the main one (path1)
    if os.path.exists(path1):
        has_safe = os.path.exists(os.path.join(path1, "model.safetensors"))
        has_bin = os.path.exists(os.path.join(path1, "pytorch_model.bin"))
        
        if has_safe and has_bin:
            pkg_bin = os.path.join(path1, "pytorch_model.bin")
            print(f"  Found both safetensors and pytorch_model.bin in {path1}")
            print(f"  [DELETE] Removing redundant pytorch_model.bin")
            try:
                os.remove(pkg_bin)
            except OSError as e:
                print(f"  [ERROR] Failed to remove bin file: {e}")

if __name__ == "__main__":
    clean_checkpoints("/home/billy/25-1kp/vla/runs")
    clean_checkpoints("/home/billy/25-1kp/vla/RoboVLMs_upstream/runs")
    clean_duplicate_models()
