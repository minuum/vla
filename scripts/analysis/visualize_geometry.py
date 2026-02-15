import h5py
import numpy as np
import cv2
import glob
import os

def check_geometry():
    dataset_path = "/home/billy/25-1kp/vla/ROS_action/basket_dataset"
    
    # 1. Find on 'Left' episode
    left_files = glob.glob(os.path.join(dataset_path, "*left*.h5"))
    if not left_files:
        print("[WARN] No LEFT files found")
        return

    # 2. Find one 'Right' episode
    right_files = glob.glob(os.path.join(dataset_path, "*right*.h5"))
    if not right_files:
        print("[WARN] No RIGHT files found")
        return

    def analyze_frame(h5_path, label):
        with h5py.File(h5_path, 'r') as f:
            images = f['images'][:]
            # Pick a middle frame where obstacle should be visible
            mid_idx = len(images) // 2
            img = images[mid_idx] # RGB
            
            # Simple gray check? Or just save it.
            # Convert to BGR for cv2
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            out_name = f"verify_{label}.jpg"
            cv2.imwrite(out_name, img_bgr)
            print(f"[{label.upper()}] Saved middle frame to {out_name}")
            print(f"  - File: {os.path.basename(h5_path)}")
            print(f"  - Expected Obstacle Position: {'RIGHT' if 'left' in label else 'LEFT'}")
            
            # Simple heuristic: Divide image in half
            # Gray basket is likely distinguishable. 
            # Let's just output the file for now as the user asked for a "check".

    print("--- Visual Geometry Check ---")
    analyze_frame(left_files[0], "left_episode")
    analyze_frame(right_files[0], "right_episode")
    print("Please open 'verify_left_episode.jpg' and 'verify_right_episode.jpg' to confirm obstacle position.")

if __name__ == "__main__":
    check_geometry()
