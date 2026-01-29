import os
import h5py
import glob
from PIL import Image
import numpy as np

SEARCH_DIR = "ROS_action"
OUTPUT_DIR = "debug_dataset_images"
# User asked for 1st, 50th, 100th... which maps to indices 0, 49, 99...
INDICES = [0, 49, 99, 149, 199, 249] 

def extract_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Find all h5 files
    print(f"Searching for .h5 files in {SEARCH_DIR}...")
    files = glob.glob(os.path.join(SEARCH_DIR, "**/*.h5"), recursive=True)
    
    # Filter and sort
    # We look for "left" or "right" in the filename
    left_files = sorted([f for f in files if "left" in os.path.basename(f).lower()])
    right_files = sorted([f for f in files if "right" in os.path.basename(f).lower()])
    
    print(f"Found {len(left_files)} Left files")
    print(f"Found {len(right_files)} Right files")

    for label, file_list in [("left", left_files), ("right", right_files)]:
        print(f"\nProcessing {label.upper()} datasets...")
        for idx in INDICES:
            if idx < len(file_list):
                filepath = file_list[idx]
                filename = os.path.basename(filepath)
                print(f"  Target #{idx+1} (Index {idx}): {filename}")
                
                try:
                    with h5py.File(filepath, 'r') as f:
                        if 'images' not in f:
                            print(f"    Error: No 'images' key in {filename}")
                            continue
                         
                        # Handle potential dataset vs group issues
                        # Based on check, f['images'] is the dataset
                        if isinstance(f['images'], h5py.Dataset):
                            images = f['images']
                        else:
                            # If it's a group, try to find a camera inside
                            # For now, assume dataset as per check
                            print(f"    Warning: 'images' is not a dataset? Type: {type(f['images'])}")
                            continue

                        num_frames = images.shape[0]
                        if num_frames == 0:
                            print("    Empty images dataset")
                            continue

                        # Select frames: start, middle, end
                        frame_indices = [0, num_frames // 2, num_frames - 1]
                        frame_indices = sorted(list(set(frame_indices)))
                        
                        for frame_idx in frame_indices:
                            img_data = images[frame_idx]
                            # H5 images might be RGB or BGR, usually RGB in robotics if from ROS/OpenCV converted
                            # However, cv2 reads BGR, so if simple write, it might be BGR. 
                            # But PIL expects RGB.
                            # If they look weird (blue skin), we know they are BGR.
                            # Usually we convert to RGB. Let's assume RGB for now or just save as is.
                            
                            img = Image.fromarray(img_data.astype(np.uint8))
                            
                            # Clean filename for saving
                            clean_fname = filename.replace('.h5', '')
                            save_name = f"{label}_idx{idx+1}_{clean_fname}_frame{frame_idx}.jpg"
                            save_path = os.path.join(OUTPUT_DIR, save_name)
                            
                            img.save(save_path)
                            print(f"    Saved frame {frame_idx} to {save_name}")
                            
                except Exception as e:
                    print(f"    Error processing {filename}: {e}")
            else:
                print(f"  Target #{idx+1} (Index {idx}) is out of range (Total {len(file_list)})")

if __name__ == "__main__":
    extract_images()
