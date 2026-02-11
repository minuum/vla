#!/usr/bin/env python3
import os
import h5py
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io
import base64
from collections import defaultdict

API_BASE = "http://localhost:8000"
API_KEY = "vla-mobile-fixed-key-20260205"
HEADERS = {"X-API-Key": API_KEY}
DATASET_DIR = "/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2/test"
NUM_EPISODES = 20
TOLERANCE = 0.05

def main():
    print(f"🚀 초반 구간(Initial Phase) 프레임별 정밀 분석 시작")
    dataset_path = Path(DATASET_DIR)
    h5_files = sorted(list(dataset_path.glob("*.h5")))[:NUM_EPISODES]
    
    frame_stats = defaultdict(lambda: {"total": 0, "perfect": 0, "dir_match": 0, "avg_error": 0.0})
    
    for h5_path in tqdm(h5_files, desc="EP 분석 중"):
        # Reset episode history at the start of each episode
        try:
            requests.post(f"{API_BASE}/reset", headers=HEADERS, timeout=5)
        except:
            pass
            
        with h5py.File(h5_path, 'r') as f:
            images = f['images'][:]
            actions = f['actions'][:]
            
        # 각 에피소드의 처음 10프레임 혹은 실제 길이만큼 분석
        limit = min(len(images), 10)
        
        for i in range(limit):
            img_pil = Image.fromarray(images[i])
            buf = io.BytesIO()
            img_pil.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            try:
                response = requests.post(f"{API_BASE}/predict", headers=HEADERS,
                                         json={"image": img_b64, "instruction": "Navigate to the basket"},
                                         timeout=10)
                response.raise_for_status()
                pred_action = np.array(response.json()['action']).flatten()[:2]
            except:
                continue
                
            gt_action = actions[i][:2]
            
            error = np.linalg.norm(gt_action - pred_action)
            perfect = error < TOLERANCE
            dir_match = (np.sign(gt_action[0]) == np.sign(pred_action[0])) and \
                        (np.sign(gt_action[1]) == np.sign(pred_action[1]))
            
            frame_stats[i]["total"] += 1
            if perfect: frame_stats[i]["perfect"] += 1
            if dir_match: frame_stats[i]["dir_match"] += 1
            frame_stats[i]["avg_error"] += error

    print("\n" + "="*70)
    print(f"{'Index':<6} | {'Status':<10} | {'PM Rate':<10} | {'DA Rate':<10} | {'Avg Error':<10}")
    print("-" * 70)
    
    for i in range(10):
        stats = frame_stats[i]
        if stats["total"] == 0: continue
        
        pm = stats["perfect"] / stats["total"] * 100
        da = stats["dir_match"] / stats["total"] * 100
        avg_err = stats["avg_error"] / stats["total"]
        status = "Stop" if i == 0 else "Accelerating" if i < 4 else "Cruising"
        
        print(f"#{i:<4} | {status:<10} | {pm:>8.1f}% | {da:>8.1f}% | {avg_err:>9.4f}")
    
    print("="*70)

if __name__ == "__main__":
    main()
