#!/usr/bin/env python3
"""
API 서버를 통한 에피소드 전체(프레임별) 검증 스크립트
히스토리 버퍼를 유지하며 모든 프레임의 정확도를 측정합니다.
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

def test_full_episodes(num_episodes=5):
    api_server = "http://localhost:8000"
    api_key = "test_key_1234" # 위에서 설정한 키와 맞춤
    dataset_dir = Path("/home/billy/25-1kp/vla/ROS_action/basket_dataset")
    
    all_files = list(dataset_dir.glob("*.h5"))
    test_files = random.sample(all_files, min(num_episodes, len(all_files)))
    
    print(f"🎬 전체 에피소드 검증 시작: {len(test_files)} 에피소드")
    
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    overall_total = 0
    overall_matches = 0
    
    episode_reports = []

    for file_path in test_files:
        print(f"\n📂 에피소드 분석 중: {file_path.name}")
        
        # 1. 히스토리 리셋
        requests.post(f"{api_server}/reset", headers=headers)
        
        with h5py.File(file_path, 'r') as f:
            images = f['images'][:]
            actions = f['actions'][:]
            n_frames = len(images)
            
            # 지시어 결정
            instruction = "Navigate to the brown pot on the left" if "left" in file_path.name else "Navigate to the brown pot on the right"
            
            ep_matches = 0
            ep_details = []

            for i in range(n_frames):
                img_pil = Image.fromarray(images[i])
                img_b64 = image_to_base64(img_pil)
                true_action = actions[i][:2]
                
                try:
                    response = requests.post(
                        f"{api_server}/predict",
                        json={"image": img_b64, "instruction": instruction, "snap_to_grid": True},
                        headers=headers, timeout=10
                    ).json()
                    
                    pred_action = response['action']
                    is_match = np.allclose(pred_action, true_action, atol=0.01)
                    
                    if is_match:
                        ep_matches += 1
                        overall_matches += 1
                    overall_total += 1
                    
                    ep_details.append({
                        "frame": i + 1,
                        "truth": true_action.tolist(),
                        "pred": pred_action,
                        "match": "✅" if is_match else "❌"
                    })
                except Exception as e:
                    print(f"  Fail Frame {i+1}: {e}")

            accuracy = (ep_matches / n_frames * 100) if n_frames > 0 else 0
            print(f"  🎯 에피소드 정확도: {accuracy:.2f}% ({ep_matches}/{n_frames})")
            
            episode_reports.append({
                "name": file_path.name,
                "accuracy": f"{accuracy:.2f}%",
                "frames": ep_details
            })

    # 최종 리포트 출력
    print("\n" + "="*80)
    print(f"🏆 최종 요약 (전체 프레임 기준)")
    print("="*80)
    print(f"  총 검증 프레임: {overall_total}")
    print(f"  전체 정확도: {(overall_matches / overall_total * 100):.2f}%")
    print("="*80)
    
    # 상세 표 출력 (AI가 보기 편하게)
    for ep in episode_reports:
        print(f"\n[Episode: {ep['name']} | Accuracy: {ep['accuracy']}]")
        print(f"{'FR':<3} | {'Truth [X, Y]':<15} | {'Pred [X, Y]':<15} | {'Result'}")
        print("-" * 50)
        for f in ep['frames']:
            truth_str = f"[{f['truth'][0]:.2f}, {f['truth'][1]:.2f}]"
            pred_str = f"[{f['pred'][0]:.2f}, {f['pred'][1]:.2f}]"
            print(f"{f['frame']:<3} | {truth_str:<15} | {pred_str:<15} | {f['match']}")

if __name__ == "__main__":
    test_full_episodes(5) # 5개 에피소드 전체 검증 (약 90프레임)
