import h5py
import numpy as np
import requests
import base64
from PIL import Image
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def analyze_per_frame_performance(num_episodes=20):
    api_server = "http://localhost:8000"
    api_key = "test_key_1234"
    dataset_dir = Path("/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2/test")
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    test_files = sorted(list(dataset_dir.glob("*.h5")))[:num_episodes]
    
    # Store results per frame index (0 to 17)
    frame_stats = {i: {"total": 0, "dir_match": 0, "err_x": [], "err_y": []} for i in range(18)}
    
    print(f"Analyzing {len(test_files)} episodes frame-by-frame...")
    
    for file_path in tqdm(test_files):
        requests.post(f"{api_server}/reset", headers=headers)
        with h5py.File(file_path, 'r') as f:
            images = f['images'][:]
            actions = f['actions'][:]
            instruction = "Navigate to the brown pot on the left" if "left" in file_path.name else "Navigate to the brown pot on the right"
            
            for i in range(len(images)):
                img_pil = Image.fromarray(images[i])
                img_b64 = image_to_base64(img_pil)
                true_act = actions[i][:2]
                
                try:
                    resp = requests.post(
                        f"{api_server}/predict",
                        json={"image": img_b64, "instruction": instruction},
                        headers=headers, timeout=10
                    ).json()
                    
                    pred_act = np.array(resp['action'])
                    
                    # Direction Match
                    true_dir = np.sign(true_act[1]) if abs(true_act[1]) > 0.1 else 0
                    pred_dir = np.sign(pred_act[1]) if abs(pred_act[1]) > 0.1 else 0
                    
                    frame_stats[i]["total"] += 1
                    if true_dir == pred_dir:
                        frame_stats[i]["dir_match"] += 1
                    
                    frame_stats[i]["err_x"].append(abs(pred_act[0] - true_act[0]))
                    frame_stats[i]["err_y"].append(abs(pred_act[1] - true_act[1]))
                except:
                    continue

    print("\nFrame-wise Analysis Results:")
    print("Frame | Dir Agreement | Avg X Err | Avg Y Err")
    print("-" * 45)
    for i in range(18):
        s = frame_stats[i]
        acc = (s["dir_match"] / s["total"] * 100) if s["total"] > 0 else 0
        ex = np.mean(s["err_x"]) if s["err_x"] else 0
        ey = np.mean(s["err_y"]) if s["err_y"] else 0
        print(f"{i:5} | {acc:13.1f}% | {ex:9.3f} | {ey:9.3f}")

if __name__ == "__main__":
    analyze_per_frame_performance()
