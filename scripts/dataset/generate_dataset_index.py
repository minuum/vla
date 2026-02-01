import os
import glob
import pandas as pd

# Define paths
DATA_DIR = "/home/billy/25-1kp/vla/ROS_action/basket_dataset"
OUTPUT_CSV = "/home/billy/25-1kp/vla/ROS_action/basket_dataset/dataset_index.csv"

# Instructions
INSTRUCTION_LEFT = "Navigate to the brown pot on the left"
INSTRUCTION_RIGHT = "Navigate to the brown pot on the right"

def generate_index():
    print(f"Scanning {DATA_DIR}...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
    
    data = []
    stats = {"left": 0, "right": 0, "unknown": 0}
    
    for f in files:
        fname = os.path.basename(f).lower()
        
        # Determine instruction based on filename
        if "left" in fname:
            instruction = INSTRUCTION_LEFT
            stats["left"] += 1
        elif "right" in fname:
            instruction = INSTRUCTION_RIGHT
            stats["right"] += 1
        else:
            instruction = "Navigate to the brown pot" # Default/Fallback
            stats["unknown"] += 1
            print(f"Warning: Could not determine direction for {fname}")
            
        # Store absolute path and instruction
        data.append({
            "episode_path": f,
            "instruction": instruction
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nGenerated {OUTPUT_CSV}")
    print(f"Total episodes: {len(df)}")
    print(f"Stats: Left={stats['left']}, Right={stats['right']}, Unknown={stats['unknown']}")
    
    # Verify a few
    print("\nSample entries:")
    print(df.head())

if __name__ == "__main__":
    generate_index()
