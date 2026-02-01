import h5py
import numpy as np
import cv2
import os
from PIL import Image

FILEPATH = "/home/billy/25-1kp/vla/ROS_action/basket_dataset/episode_20260129_010041_basket_1box_hori_left_core_medium.h5"
OUTPUT_DIR = "/home/billy/25-1kp/vla/debug_episode_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_episode():
    with h5py.File(FILEPATH, 'r') as f:
        images = f['images'][:]
        actions = f['actions'][:]
        
        print(f"Episode: {os.path.basename(FILEPATH)}")
        print(f"Total Frames: {len(images)}")
        print(f"Instruction: Navigate to the brown pot on the left (Inferred from filename)")
        print("-" * 50)
        print(f"{'Frame':<8} | {'Linear X':<10} | {'Linear Y':<10} | {'Action Type'}")
        print("-" * 50)
        
        for i in range(len(images)):
            # Save image
            img = images[i]
            img_path = os.path.join(OUTPUT_DIR, f"frame_{i:02d}.jpg")
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, img_bgr)
            
            # Action analysis
            lx = actions[i][0]
            ly = actions[i][1]
            
            action_desc = "Stop"
            if lx > 0.5: action_desc = "Forward (W)"
            elif lx < -0.5: action_desc = "Backward (S)"
            
            if ly > 0.5: action_desc += " + Left (A)"
            elif ly < -0.5: action_desc += " + Right (D)"
            
            print(f"{i:02d}       | {lx:10.4f} | {ly:10.4f} | {action_desc}")

    print("-" * 50)
    print(f"Images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_episode()
