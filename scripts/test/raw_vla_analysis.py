#!/usr/bin/env python3
"""
Raw Action Analysis: 근사한 값 vs 아예 다른 값 판별
- snap_to_grid=False로 추론하여 생생한 회귀값 분석
- 1.15, 0, -1.15를 타겟으로 하여 오차 분석
"""

import os
import sys
import json
import h5py
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
import base64
from PIL import Image
import io

# API 설정 (기존 서버 사용)
API_BASE = "http://localhost:8000"
API_KEY = "vla-mobile-fixed-key-20260205"
HEADERS = {"X-API-Key": API_KEY}

# 데이터 설정
DATASET_DIR = "/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2/test"
NUM_EPISODES = 10 

def analyze_raw():
    dataset_path = Path(DATASET_DIR)
    h5_files = sorted(list(dataset_path.glob("*.h5")))[:NUM_EPISODES]
    
    # 통계용
    stats = {
        'total_frames': 0,
        'dist_categories': {
            'exact_match': 0, # 오차 < 0.05
            'near_match': 0,  # 오차 < 0.25 (근사함)
            'far_away': 0,    # 오차 >= 0.25 (아예 다름)
            'wrong_direction': 0 # 부호가 다름
        },
        'phase_stats': {
            'initial': {'exact': 0, 'near': 0, 'far': 0, 'wrong': 0, 'count': 0},
            'middle': {'exact': 0, 'near': 0, 'far': 0, 'wrong': 0, 'count': 0},
            'final': {'exact': 0, 'near': 0, 'far': 0, 'wrong': 0, 'count': 0}
        }
    }

    for h5_path in h5_files:
        print(f"🎬 Processing {h5_path.name}")
        with h5py.File(h5_path, 'r') as f:
            images = f['images'][:]
            actions = f['actions'][:]
            language = "Navigate to the basket" # Default
            
        num_frames = len(images)
        
        for i in tqdm(range(num_frames), leave=False):
            # 이미지 준비
            img_pil = Image.fromarray(images[i])
            buf = io.BytesIO()
            img_pil.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # API 호출 (Raw 값 요청)
            try:
                response = requests.post(
                    f"{API_BASE}/predict",
                    headers=HEADERS,
                    json={
                        "image": img_b64,
                        "instruction": language,
                        "snap_to_grid": False # 👈 중요: Raw 회귀값 요청
                    },
                    timeout=5
                )
                response.raise_for_status()
                pred = np.array(response.json()['action']).flatten()[:2]
            except Exception as e:
                print(f"Error at {h5_path.name} frame {i}: {e}")
                continue

            gt = actions[i][:2]
            
            # 오차 분석 (Linear X 기준 - 전진/정지 핵심)
            gt_x = gt[0]
            pred_x = pred[0]
            
            error = abs(gt_x - pred_x)
            same_dir = (np.sign(gt_x) == np.sign(pred_x)) or (abs(gt_x) < 0.05 and abs(pred_x) < 0.05)
            
            # 카테고리 분류
            cat = ''
            if not same_dir:
                cat = 'wrong_direction'
            elif error < 0.05:
                cat = 'exact_match'
            elif error < 0.25:
                cat = 'near_match'
            else:
                cat = 'far_away'
            
            # 통계 업데이트
            stats['total_frames'] += 1
            stats['dist_categories'][cat] += 1
            
            # 구간 판별
            phase = 'middle'
            if i < 5: phase = 'initial'
            elif i >= num_frames - 5: phase = 'final'
            
            ps = stats['phase_stats'][phase]
            ps['count'] += 1
            if cat == 'wrong_direction': ps['wrong'] += 1
            elif cat == 'exact_match': ps['exact'] += 1
            elif cat == 'near_match': ps['near'] += 1
            else: ps['far'] += 1

    # 결과 출력
    print("\n" + "="*50)
    print("📊 VLA Raw Action 분석 결과 (Linear X 기준)")
    print("="*50)
    total = stats['total_frames']
    print(f"전체 프레임: {total}")
    print("-" * 50)
    
    for cat, count in stats['dist_categories'].items():
        pct = count / total * 100
        desc = {
            'exact_match': "완벽 일치 (±0.05)",
            'near_match': "근사한 값 (±0.25)",
            'far_away': "아예 다른 값 (>0.25)",
            'wrong_direction': "반대 방향/정지 오류"
        }[cat]
        print(f"{desc:25s}: {count:4d} ({pct:5.1f}%)")
        
    print("\n📍 구간별 상세 (Phase Analysis)")
    print("-" * 50)
    for phase in ['initial', 'middle', 'final']:
        ps = stats['phase_stats'][phase]
        if ps['count'] == 0: continue
        c = ps['count']
        print(f"[{phase.upper():7s}] ({c:3d} frames)")
        print(f"  - 정답권(Exact+Near): {(ps['exact']+ps['near'])/c*100:5.1f}%")
        print(f"  - 아예 다름(Far):     {ps['far']/c*100:5.1f}%")
        print(f"  - 방향 틀림(Wrong):   {ps['wrong']/c*100:5.1f}%")
    print("="*50)

if __name__ == "__main__":
    analyze_raw()
