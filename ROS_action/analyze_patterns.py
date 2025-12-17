import h5py
import numpy as np
import os
import csv

# Configuration
DATASET_DIR = "/home/soda/vla/ROS_action/mobile_vla_dataset"
TARGET_PATTERN = ['W', 'W', 'A', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'W', 'W', 'E', 'E', 'E', 'W', 'W']

# Action mapping based on CSV analysis
# W: lx=1.15, ly=0.0, az=0.0
# A: lx=0.0, ly=1.15, az=0.0 (Wait, CSV shows ly=1.15 for frame 4. Is 'A' Left or Slide Left? User said 'A' is Left, but CSV shows ly. Let's re-read code or assume CSV is truth for "A" in this context.)
# Actually, looking at CSV frame 4: action_x=0.0, action_y=1.15, action_z=0.0.
# In standard ROS, y is lateral. So this is a slide left.
# User's pattern says 'A'. In WASD, A is usually Left.
# Let's map based on the CSV values:
# W: lx > 0.5, ly < 0.5
# A: lx < 0.5, ly > 0.5 (Slide Left)
# Q: lx > 0.5, ly > 0.5 (Forward Left)
# E: lx > 0.5, ly < -0.5 (Forward Right) - Wait, target pattern has E.
# Let's look at the CSV again.
# Frames 1-3: 1.15, 0, 0 -> W
# Frame 4: 0, 1.15, 0 -> A (Slide Left)
# Frames 5-12: 1.15, 1.15, 0 -> Q (Forward Left)
# Frames 13-17: 1.15, 0, 0 -> W
# The CSV sequence is W W W A Q Q Q Q Q Q Q Q W W W W W.
# The user's target pattern is W W A Q Q Q Q Q Q Q W W E E E W W.
# Wait, the CSV has W W W A ... but user target has W W A ...
# Also user target has E E E. The CSV does not have E.
# The user said "Similar to this file... [1 2 3 ... 17] [W W A ...]"
# And "Analyze this file... to see data characteristics... then select based on target pattern".
# So I should use the CSV to define what W, A, Q, E *look like* in terms of values, then find files matching the *user's* target pattern.

def get_action_char(lx, ly, az):
    # Thresholds
    th = 0.5
    
    if abs(lx) < 0.1 and abs(ly) < 0.1 and abs(az) < 0.1:
        return 'S' # Stop
    
    if lx > th:
        if ly > th:
            return 'Q' # Forward Left
        elif ly < -th:
            return 'E' # Forward Right
        elif abs(ly) < th:
            return 'W' # Forward
    elif abs(lx) < th:
        if ly > th:
            return 'A' # Slide Left
        elif ly < -th:
            return 'D' # Slide Right
            
    return '?' # Unknown

def check_file(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'action' not in f:
                return False, "No action data", []
            
            actions = f['action'][:]
            if len(actions) != 18:
                return False, f"Length {len(actions)}", []
            
            # Convert frames 1-17 to chars
            sequence = []
            for i in range(1, 18):
                char = get_action_char(actions[i][0], actions[i][1], actions[i][2])
                sequence.append(char)
            
            # Compare with target
            if sequence == TARGET_PATTERN:
                return True, "Match", sequence
            else:
                return False, "Mismatch", sequence
    except Exception as e:
        return False, f"Error: {e}", []

def main():
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.h5')]
    matches = []
    
    print(f"Scanning {len(files)} files for pattern: {TARGET_PATTERN}")
    
    for filename in files:
        filepath = os.path.join(DATASET_DIR, filename)
        is_match, msg, seq = check_file(filepath)
        
        if is_match:
            matches.append(filename)
            # print(f"MATCH: {filename}")
        # else:
            # print(f"NO: {filename} -> {seq}")

    print("-" * 30)
    print(f"Total Matches: {len(matches)}")
    if matches:
        print("Matching files:")
        for m in matches:
            print(m)
    else:
        print("No matches found.")

if __name__ == "__main__":
    main()
