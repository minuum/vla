import requests
import h5py
import numpy as np
import base64
import torch
import time
from tqdm import tqdm
from PIL import Image
import io
import os

# Configuration
api_server = "http://localhost:8000"
api_key = os.getenv("VLA_API_KEY", "vla-mobile-fixed-key-20260205")
headers = {"X-API-Key": api_key}

# Target file for validation (Difficult Turn Case)
test_file = "/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/episode_20251203_143945_1box_hori_left_core_medium.h5"

def run_stress_test(temperature=0.7, use_zero_enforcement=False, turn_bias=0.5):
    print(f"\n🔥 Running Hard-Turn Validation: Temp={temperature}, ZeroEnf={use_zero_enforcement}, Bias={turn_bias}")
    
    with h5py.File(test_file, 'r') as f:
        images = f['images'][:]
        actions = f['actions'][:]
        instruction = "Navigate to the gray basket on the left"
        
        # Reset API
        requests.get(f"{api_server}/reset", headers=headers)
        
        pm_count = 0
        dm_count = 0
        total_steps = len(images) - 8  # 8 is window size
        
        # Simple window tracking
        for i in range(8, len(images)):
            img_array = images[i]
            img = Image.fromarray(img_array.astype(np.uint8))
            
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            response = requests.post(
                f"{api_server}/predict",
                json={
                    "image": img_b64,
                    "instruction": instruction,
                    "temperature": temperature,
                    "use_zero_enforcement": use_zero_enforcement,
                    "turn_bias": turn_bias
                },
                headers=headers,
                timeout=20
            )
            
            if response.status_code == 200:
                pred_action = np.array(response.json()["action"])
                true_action = actions[i][:2]
                
                # PM: simple match (discrete class equivalent)
                # DM: direction match (x and y signs match or both are zero)
                if np.allclose(pred_action, true_action, atol=0.1):
                    pm_count += 1
                
                # Direction Match (Simplified for 2D)
                pred_sign = np.sign(pred_action)
                true_sign = np.sign(true_action)
                if np.array_equal(pred_sign, true_sign):
                    dm_count += 1
            else:
                print(f"Error at step {i}: {response.text}")

    pm_rate = (pm_count / total_steps) * 100
    dm_rate = (dm_count / total_steps) * 100
    print(f"📈 Result -> PM: {pm_rate:.2f}%, DM: {dm_rate:.2f}%")
    return pm_rate, dm_rate

if __name__ == "__main__":
    # Test with Optimized settings
    run_stress_test(temperature=0.7, use_zero_enforcement=False, turn_bias=0.5)
    # Test without Bias (for comparison)
    run_stress_test(temperature=0.7, use_zero_enforcement=False, turn_bias=0.0)
