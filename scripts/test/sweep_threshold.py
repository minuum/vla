#!/usr/bin/env python3
"""
Threshold Optimization: 최적의 Snapping 임계값 찾기
0.1부터 0.8까지 sweep하며 어떤 값이 PM/DA를 가장 극대화하는지 확인
"""
import requests
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import base64
from PIL import Image
import io

API_BASE = "http://localhost:8000"
API_KEY = "vla-mobile-fixed-key-20260205"
HEADERS = {"X-API-Key": API_KEY}
DATASET_DIR = "/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2/test"

def run_sweep():
    dataset_path = Path(DATASET_DIR)
    h5_files = sorted(list(dataset_path.glob("*.h5")))[:5] # 빠른 검증을 위해 5개만
    
    # 1단계: Raw 데이터 수집 (서버 부하 방지)
    print("🚀 Raw 데이터 수집 중...")
    all_data = []
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as f:
            images = f['images'][:]
            actions = f['actions'][:]
            
        for i in range(len(images)):
            img_pil = Image.fromarray(images[i])
            buf = io.BytesIO()
            img_pil.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            resp = requests.post(f"{API_BASE}/predict", headers=HEADERS, 
                               json={"image": img_b64, "instruction": "test", "snap_to_grid": False}, timeout=5)
            pred_raw = np.array(resp.json()['action']).flatten()[:2]
            gt = actions[i][:2]
            all_data.append((gt, pred_raw))

    # 2단계: 다양한 Threshold 시뮬레이션
    target_values = [-1.15, 0.0, 1.15]
    print("\n🔍 Threshold Sweep 시작...")
    print(f"{'Threshold':<10} | {'Perfect Match':<15} | {'Dir Agreement':<15}")
    print("-" * 45)
    
    best_pm = 0
    best_thr = 0

    for thr in np.arange(0.1, 0.9, 0.1):
        pm_count = 0
        da_count = 0
        
        for gt, pred in all_data:
            # Snap logic simulation
            snapped = []
            for v in pred:
                if abs(v) > thr:
                    snapped.append(1.15 if v > 0 else -1.15)
                else:
                    snapped.append(0.0)
            snapped = np.array(snapped)
            
            # Metric
            if np.allclose(gt, snapped, atol=0.01): pm_count += 1
            if (np.sign(gt[0]) == np.sign(snapped[0])) and (np.sign(gt[1]) == np.sign(snapped[1])):
                da_count += 1
        
        pm_pct = pm_count / len(all_data) * 100
        da_pct = da_count / len(all_data) * 100
        print(f"{thr:<10.1f} | {pm_pct:<15.2f} | {da_pct:<15.2f}")
        
        if pm_pct > best_pm:
            best_pm = pm_pct
            best_thr = thr

    print("-" * 45)
    print(f"✅ 최적의 Threshold: {best_thr:.1f} (PM: {best_pm:.2f}%)")

if __name__ == "__main__":
    run_sweep()
