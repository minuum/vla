#!/usr/bin/env python3
"""
API 서버를 통한 상세 분석 스크립트
표 형식 분석을 위해 각 에피소드별 예측값 및 정답 비교 데이터를 수집
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
import json

def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_api_detail(num_episodes=20):
    api_server = "http://localhost:8000"
    api_key = "test_key_1234"
    dataset_dir = Path("/home/billy/25-1kp/vla/ROS_action/basket_dataset")
    
    all_files = list(dataset_dir.glob("*.h5"))
    test_files = random.sample(all_files, min(num_episodes, len(all_files)))
    
    print(f"📡 상세 분석 시작: {len(test_files)} 에피소드")
    
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    detailed_results = []

    for file_path in tqdm(test_files, desc="Analyzing"):
        with h5py.File(file_path, 'r') as f:
            images = f['images'][:]
            actions = f['actions'][:]
            total_len = len(images)
            
            if total_len < 1: continue
            
            # 중간 지점 샘플링
            idx = total_len // 2
            
            img_pil = Image.fromarray(images[idx])
            img_b64 = image_to_base64(img_pil)
            true_action = actions[idx][:2]
            
            instruction = "Navigate to the brown pot on the left" if "left" in file_path.name else "Navigate to the brown pot on the right"
            
            try:
                # 1. Snap 적용 버전
                res_snap = requests.post(
                    f"{api_server}/predict",
                    json={"image": img_b64, "instruction": instruction, "snap_to_grid": True},
                    headers=headers, timeout=20
                ).json()
                
                # 2. Raw 버전 (서버 내부 로직 거치지 않은 순수 모델 출력 유회를 위해 snap_to_grid=False 요청)
                # Note: API 서버 코드를 보면 snap_to_grid=False면 Raw Action 반환
                res_raw = requests.post(
                    f"{api_server}/predict",
                    json={"image": img_b64, "instruction": instruction, "snap_to_grid": False},
                    headers=headers, timeout=20
                ).json()
                
                pred_snap = res_snap['action']
                pred_raw = res_raw['action']
                
                status = "✅ MATCH" if np.allclose(pred_snap, true_action, atol=0.01) else "❌ MISMATCH"
                
                detailed_results.append({
                    "episode": file_path.name,
                    "frame": idx,
                    "target": "Left" if "left" in file_path.name else "Right",
                    "truth": true_action.tolist(),
                    "raw": [round(x, 4) for x in pred_raw],
                    "snapped": pred_snap,
                    "status": status
                })
                
            except Exception as e:
                print(f"❌ Error ({file_path.name}): {e}")

    # 결과 출력 (JSON으로 저장하여 AI가 파싱)
    with open("detailed_analysis.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n✅ 분석 데이터 저장 완료: detailed_analysis.json")

if __name__ == "__main__":
    test_api_detail(20)
