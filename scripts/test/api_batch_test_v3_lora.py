#!/usr/bin/env python3
"""
API 서버를 통한 Basket Navigation 대량 검증 스크립트 (V3 LoRA Classification 모델 대상)
"""

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

def test_api_batch(num_episodes=50):
    api_server = "http://localhost:8000"
    api_key = "test_key_1234"
    dataset_dir = Path("/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2")
    
    all_files = list(dataset_dir.glob("*.h5"))
    test_files = random.sample(all_files, min(num_episodes, len(all_files)))
    
    print(f"📡 API 서버 일괄 테스트 시작: {len(test_files)} 에피소드")
    
    total_frames = 0
    perfect_matches = 0
    direction_matches = 0
    
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    for file_path in tqdm(test_files, desc="Processing"):
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
                # Send context frames to build history
                for i in range(idx - window_size + 1, idx + 1):
                    img_pil = Image.fromarray(images[i])
                    img_b64 = image_to_base64(img_pil)
                    
                    response = requests.post(
                        f"{api_server}/predict",
                        json={
                            "image": img_b64,
                            "instruction": instruction,
                            "strategy": "receding_horizon"
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
                        status = "✅ PM MATCH"
                    else:
                        is_dm = (np.sign(true_action[1]) == np.sign(pred_action[1])) or (abs(true_action[1]) < 0.1 and abs(pred_action[1]) < 0.1)
                        if is_dm:
                            direction_matches += 1
                            status = "⚠️ DM MATCH"
                        else:
                            status = "❌ MISMATCH"
                    
                    target = "Left" if "left" in file_path.name else "Right"
                    print(f"  [{target}] True: {true_action} | Pred: {pred_action} | {status}")
                    
                    total_frames += 1
                else:
                    msg = response.status_code if response else "No response"
                    print(f"❌ API 에러 ({file_path.name}): {msg}")
            except Exception as e:
                print(f"❌ 통신 에러 ({file_path.name}): {e}")

    print("\n" + "="*70)
    print("📊 API 기반 Basket Navigation V3 LoRA 검증 결과")
    print("="*70)
    print(f"✅ 테스트 프레임 수: {total_frames}")
    if total_frames > 0:
        print(f"✅ PM (Perfect Match): {(perfect_matches / total_frames * 100):.2f}%")
        print(f"✅ DM (Direction Match): {((perfect_matches + direction_matches) / total_frames * 100):.2f}%")
    print("="*70)

if __name__ == "__main__":
    test_api_batch(50)
