#!/usr/bin/env python3
"""
API 서버를 통한 에피소드 심층 분석 스크립트 (Grounded Prompting 버전)
훈련 시와 동일하게 프레임 단위 액션 정보를 접두어로 붙여서 모델의 최대 성능을 측정합니다.
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
import sys

def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_drilldown_grounded(num_episodes=10):
    api_server = "http://localhost:8000"
    api_key = "test_key_1234"
    dataset_dir = Path("/home/billy/25-1kp/vla/ROS_action/basket_dataset")
    
    all_files = list(dataset_dir.glob("*.h5"))
    test_files = random.sample(all_files, min(num_episodes, len(all_files)))
    
    print(f"🔍 Grounded 에피소드 심층 분석 시작 (Target: {len(test_files)} episodes)")
    
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    global_stats = {
        "total_frames": 0,
        "perfect_matches": 0,
        "direction_matches": 0,
        "x_errors": [],
        "y_errors": []
    }
    
    episode_results = []

    for file_path in test_files:
        print(f"\nProcessing: {file_path.name}")
        requests.post(f"{api_server}/reset", headers=headers)
        
        with h5py.File(file_path, 'r') as f:
            images = f['images'][:]
            actions = f['actions'][:]
            n_frames = len(images)
            
            base_instruction = "Navigate to the brown pot on the left" if "left" in file_path.name else "Navigate to the brown pot on the right"
            
            ep_stats = {
                "perfect": 0,
                "dir": 0,
                "x_err": [],
                "y_err": []
            }

            for i in tqdm(range(n_frames), desc="  Frames", leave=False):
                img_pil = Image.fromarray(images[i])
                img_b64 = image_to_base64(img_pil)
                true_act = actions[i][:2]
                
                # Grounding Suffix (훈련 코드와 동일한 로직)
                lin_y = true_act[1]
                if lin_y > 0.5:
                    suffix = ", sliding left"
                elif lin_y < -0.5:
                    suffix = ", sliding right"
                else:
                    suffix = ", moving forward"
                
                instruction = f"{base_instruction}{suffix}"
                
                try:
                    resp = requests.post(
                        f"{api_server}/predict",
                        json={"image": img_b64, "instruction": instruction, "snap_to_grid": True},
                        headers=headers, timeout=10
                    ).json()
                    
                    pred_act = np.array(resp['action'])
                    
                    # 1. Perfect Match
                    is_perfect = np.allclose(pred_act, true_act, atol=0.01)
                    if is_perfect: ep_stats["perfect"] += 1
                    
                    # 2. Direction Match (Sign)
                    true_dir = np.sign(true_act[1]) if abs(true_act[1]) > 0.1 else 0
                    pred_dir = np.sign(pred_act[1]) if abs(pred_act[1]) > 0.1 else 0
                    if true_dir == pred_dir: ep_stats["dir"] += 1
                    
                    ep_stats["x_err"].append(pred_act[0] - true_act[0])
                    ep_stats["y_err"].append(pred_act[1] - true_act[1])
                    
                except Exception as e:
                    pass

            # Summary counts
            accuracy = (ep_stats["perfect"] / n_frames * 100) if n_frames > 0 else 0
            dir_acc = (ep_stats["dir"] / n_frames * 100) if n_frames > 0 else 0
            rmse_x = np.sqrt(np.mean(np.array(ep_stats["x_err"])**2)) if ep_stats["x_err"] else 0
            
            episode_results.append({
                "name": file_path.name,
                "perfect_acc": accuracy,
                "dir_acc": dir_acc,
                "rmse_x": rmse_x,
                "frames": n_frames
            })
            
            global_stats["total_frames"] += n_frames
            global_stats["perfect_matches"] += ep_stats["perfect"]
            global_stats["direction_matches"] += ep_stats["dir"]
            global_stats["x_errors"].extend(ep_stats["x_err"])
            global_stats["y_errors"].extend(ep_stats["y_err"])

    # Output table
    print("\n" + "="*85)
    print(f"{'Episode Name':<45} | {'Perfect':<8} | {'Dir':<8} | {'RMSE X':<8}")
    print("-" * 85)
    for res in episode_results:
        print(f"{res['name'][:43]:<45} | {res['perfect_acc']:>6.1f}% | {res['dir_acc']:>6.1f}% | {res['rmse_x']:>7.3f}")
    
    print("="*85)
    total_perf = (global_stats["perfect_matches"] / global_stats["total_frames"] * 100)
    total_dir = (global_stats["direction_matches"] / global_stats["total_frames"] * 100)
    
    print(f"📊 GROUNDED SUMMARY REPORT")
    print(f"  - Total Evaluated Frames: {global_stats['total_frames']}")
    print(f"  - Overall Perfect Match: {total_perf:.2f}%")
    print(f"  - Directional Agreement: {total_dir:.2f}%")
    print("="*85)

if __name__ == "__main__":
    test_drilldown_grounded(10)
