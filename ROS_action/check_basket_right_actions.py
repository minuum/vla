
import os
import glob
import h5py
import numpy as np
from collections import defaultdict

DATASET_DIR = "/home/soda/vla/ROS_action/mobile_vla_dataset"

def get_action_code(linear_x, angular_z):
    threshold = 0.01
    code = ""
    
    if linear_x > threshold:
        code += "W"
    elif linear_x < -threshold:
        code += "S"
    else:
        code += "_"
        
    if angular_z > threshold:
        code += "A"
    elif angular_z < -threshold:
        code += "D"
    else:
        code += "_"
        
    if code == "W_": return "W"
    if code == "S_": return "S"
    if code == "__": return "X"
    if code == "_A": return "A"
    if code == "_D": return "D"
    if code == "WA": return "Q"
    if code == "WD": return "E"
    return code

def analyze_basket_1box_right():
    print(f"📂 Analyzing basket_1box_right files...")
    h5_files = glob.glob(os.path.join(DATASET_DIR, "*basket*1box*right*.h5"))
    
    patterns = defaultdict(list)
    
    for fpath in sorted(h5_files):
        filename = os.path.basename(fpath)
        
        try:
            with h5py.File(fpath, 'r') as f:
                num_frames = f.attrs.get('num_frames', 0)
                if 'images' in f:
                    num_frames = f['images'].shape[0]
                
                if 'actions' not in f:
                    print(f"⚠️  {filename}: No actions data")
                    continue
                    
                actions = f['actions'][:]
                
                chk_str = []
                for i in range(len(actions)):
                    lin_x = actions[i][0]
                    ang_z = actions[i][1]
                    chk_str.append(get_action_code(lin_x, ang_z))
                
                pattern_key = "-".join(chk_str)
                patterns[pattern_key].append(filename)
                
                print(f"✅ {filename}")
                print(f"   Frames: {num_frames}, Pattern: {pattern_key}")
                print(f"   Actions shape: {actions.shape}")
                print("")
                
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    print("-" * 60)
    print(f"📊 Summary: {len(h5_files)} files analyzed")
    print(f"🎯 Unique patterns: {len(patterns)}")
    
    if len(patterns) == 1:
        print("✅ All files have the SAME action pattern!")
    else:
        print("⚠️  Files have DIFFERENT action patterns:")
        for i, (pattern, files) in enumerate(patterns.items(), 1):
            print(f"  Pattern {i}: {len(files)} files")
            print(f"    {pattern}")

if __name__ == "__main__":
    analyze_basket_1box_right()
