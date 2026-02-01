#!/usr/bin/env python3
"""
API 서버를 통한 Basket Navigation 대량 검증 스크립트
실제 서버의 Snap-to-Grid (1.15 기준)가 로봇 제어에 적합한지 확인
"""

import requests
import base64
import numpy as np
import h5py
import random
from PIL import Image
from io import BytesIO
import time
from pathlib import Path
from tqdm import tqdm

def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_api_batch(num_episodes=15):
    api_server = "http://localhost:8000"
    api_key = "Cf7VGtw3-BykHjmsPa12V3QL41qk87Aywjg_8P8GKU4"
    dataset_dir = Path("/home/billy/25-1kp/vla/ROS_action/basket_dataset")
    
    all_files = list(dataset_dir.glob("*.h5"))
    test_files = random.sample(all_files, min(num_episodes, len(all_files)))
    
    print(f"📡 API 서버 일괄 테스트 시작: {len(test_files)} 에피소드")
    
    total_frames = 0
    perfect_matches = 0
    
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    results = []

    for file_path in tqdm(test_files, desc="Processing"):
        with h5py.File(file_path, 'r') as f:
            images = f['images'][:]
            actions = f['actions'][:]
            total_len = len(images)
            
            # 에피소드당 1개 지점만 테스트 (API 부하 고려)
            if total_len < 1: continue
            idx = random.randint(0, total_len - 1)
            
            img_pil = Image.fromarray(images[idx])
            img_b64 = image_to_base64(img_pil)
            true_action = actions[idx][:2]
            
            instruction = "Navigate to the brown pot on the left" if "left" in file_path.name else "Navigate to the brown pot on the right"
            
            try:
                response = requests.post(
                    f"{api_server}/predict",
                    json={
                        "image": img_b64,
                        "instruction": instruction,
                        "snap_to_grid": True
                    },
                    headers=headers,
                    timeout=20
                )
                
                if response.status_code == 200:
                    data = response.json()
                    pred_action = np.array(data['action'])
                    
                    is_match = np.allclose(pred_action, true_action, atol=0.01)
                    if is_match:
                        perfect_matches += 1
                    
                    total_frames += 1
                else:
                    print(f"❌ API 에러 ({file_path.name}): {response.status_code}")
            except Exception as e:
                print(f"❌ 통신 에러 ({file_path.name}): {e}")

    print("\n" + "="*70)
    print("📊 API 기반 Basket Navigation 검증 결과")
    print("="*70)
    print(f"✅ 테스트 에피소드 수: {len(test_files)}")
    print(f"✅ Perfect Match Rate: {(perfect_matches / total_frames * 100):.2f}% (Snap to 1.15)")
    print("="*70)

if __name__ == "__main__":
    test_api_batch(15)
