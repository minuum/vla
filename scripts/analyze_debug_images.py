import os
import glob
from PIL import Image
import numpy as np

IMG_DIR = "debug_dataset_images"

def params(arr):
    return f"Mean={np.mean(arr):.1f}, Std={np.std(arr):.1f}, Min={np.min(arr)}, Max={np.max(arr)}"

def analyze():
    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    print(f"Analyzing {len(files)} images in {IMG_DIR}...\n")
    
    issues = []
    
    for f in files:
        try:
            img = Image.open(f)
            arr = np.array(img)
            
            # Check for black/empty
            if np.mean(arr) < 1:
                issues.append(f"{os.path.basename(f)}: BLACK IMAGE (Mean < 1)")
            # Check for low variance
            elif np.std(arr) < 2:
                issues.append(f"{os.path.basename(f)}: LOW VARIANCE (Std < 2) - Flat color?")
            
            print(f"{os.path.basename(f)}: {params(arr)}")
        except Exception as e:
            print(f"Error reading {f}: {e}")
            issues.append(f"{os.path.basename(f)}: Read Error")

    print("\n\n=== ANOMALY REPORT ===")
    if issues:
        for i in issues:
            print(i)
    else:
        print("No obvious anomalies detected (no pure black or flat images).")

if __name__ == "__main__":
    analyze()
