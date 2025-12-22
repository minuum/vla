import h5py
import numpy as np
import os

# Configuration
DATASET_DIR = "/home/soda/vla/ROS_action/mobile_vla_dataset"
# Target pattern from user request
TARGET_PATTERN = ['W', 'W', 'A', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'W', 'W', 'E', 'E', 'E', 'W', 'W']

def get_action_char(lx, ly, az):
    th = 0.5
    if abs(lx) < 0.1 and abs(ly) < 0.1 and abs(az) < 0.1:
        return 'S'
    if lx > th:
        if ly > th: return 'Q'
        elif ly < -th: return 'E'
        elif abs(ly) < th: return 'W'
    elif abs(lx) < th:
        if ly > th: return 'A'
        elif ly < -th: return 'D'
    return '?'

def check_file(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'action' not in f: return False, [], 0
            actions = f['action'][:]
            if len(actions) != 18: return False, [], 0
            
            seq = []
            for i in range(1, 18):
                seq.append(get_action_char(*actions[i]))
            
            # Calculate similarity (simple matching count)
            score = 0
            if len(seq) == len(TARGET_PATTERN):
                for i in range(len(seq)):
                    if seq[i] == TARGET_PATTERN[i]:
                        score += 1
            
            return True, seq, score
    except:
        return False, [], 0

def main():
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.h5')]
    results = []
    
    print(f"Scanning {len(files)} files...")
    
    for filename in files:
        filepath = os.path.join(DATASET_DIR, filename)
        valid, seq, score = check_file(filepath)
        if valid:
            results.append((filename, seq, score))
    
    # Sort by score descending
    results.sort(key=lambda x: x[2], reverse=True)
    
    print("-" * 30)
    print("Top 10 closest matches:")
    for i in range(min(10, len(results))):
        filename, seq, score = results[i]
        print(f"{filename}: Score {score}/17")
        print(f"  Seq: {seq}")
        print(f"  Tgt: {TARGET_PATTERN}")

if __name__ == "__main__":
    main()
