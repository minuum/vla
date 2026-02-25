import requests
import base64
import numpy as np
import h5py
import random
from PIL import Image
from io import BytesIO
from pathlib import Path
from tqdm import tqdm

def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def run_evaluation(num_episodes=30, temperature=1.0, use_zero_enforcement=False):
    api_server = "http://localhost:8000"
    api_key = "test_key_1234"
    dataset_dir = Path("/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2")
    
    all_files = list(dataset_dir.glob("*.h5"))
    test_files = random.sample(all_files, min(num_episodes, len(all_files)))
    
    total_frames = 0
    perfect_matches = 0
    direction_matches = 0
    
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    for file_path in tqdm(test_files, desc=f"Temp={temperature}, ZeroEnf={use_zero_enforcement}", leave=False):
        try:
            requests.post(f"{api_server}/reset", headers=headers, timeout=5)
        except:
            pass
            
        with h5py.File(file_path, 'r') as f:
            images = f['images'][:]
            actions = f['actions'][:]
            total_len = len(images)
            
            # Choose a random frame
            window_size = 8
            if total_len < window_size:
                continue
            idx = random.randint(window_size - 1, total_len - 1)
            
            instruction = "Navigate to the brown pot on the left" if "left" in file_path.name else "Navigate to the brown pot on the right"
            true_action = actions[idx][:2]
            
            try:
                response = None
                for i in range(idx - window_size + 1, idx + 1):
                    img_pil = Image.fromarray(images[i])
                    img_b64 = image_to_base64(img_pil)
                    
                    response = requests.post(
                        f"{api_server}/predict",
                        json={
                            "image": img_b64,
                            "instruction": instruction,
                            "temperature": temperature,
                            "use_zero_enforcement": use_zero_enforcement
                        },
                        headers=headers,
                        timeout=20
                    )
                
                if response and response.status_code == 200:
                    data = response.json()
                    pred_action = np.array(data['action'])
                    
                    is_pm = np.allclose(pred_action, true_action, atol=0.01)
                    if is_pm:
                        perfect_matches += 1
                    else:
                        is_dm = (np.sign(true_action[1]) == np.sign(pred_action[1])) or (abs(true_action[1]) < 0.1 and abs(pred_action[1]) < 0.1)
                        if is_dm:
                            direction_matches += 1
                    
                    total_frames += 1
            except Exception as e:
                pass

    pm_rate = (perfect_matches / total_frames * 100) if total_frames > 0 else 0
    dm_rate = ((perfect_matches + direction_matches) / total_frames * 100) if total_frames > 0 else 0
    return pm_rate, dm_rate, total_frames

def main():
    print("🚀 API Accuracy Squeezer (V3 LoRA)")
    print("-" * 50)
    
    configs = [
        {"temp": 1.0, "zero": True, "label": "Baseline (Current)"},
        {"temp": 1.0, "zero": False, "label": "No Zero Enforcement"},
        {"temp": 0.5, "zero": False, "label": "Low Temp (High Certainty)"},
        {"temp": 1.5, "zero": False, "label": "High Temp (Soft decision)"}
    ]
    
    results = []
    for cfg in configs:
        pm, dm, count = run_evaluation(num_episodes=20, temperature=cfg["temp"], use_zero_enforcement=cfg["zero"])
        results.append({
            "label": cfg["label"],
            "temp": cfg["temp"],
            "zero": cfg["zero"],
            "pm": pm,
            "dm": dm
        })
        print(f"✅ {cfg['label']}: PM={pm:.2f}%, DM={dm:.2f}%")

    print("\n" + "="*70)
    print("🥇 Final Comparison Report")
    print("="*70)
    print(f"{'Configuration':<30} | {'PM Rate':<10} | {'DM Rate':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<30} | {r['pm']:>8.2f}% | {r['dm']:>8.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
