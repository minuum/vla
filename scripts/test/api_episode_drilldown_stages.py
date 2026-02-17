#!/usr/bin/env python3
"""
API 서버를 통한 에피소드 심층 분석 스크립트 (Stages: 초기, 중기, 후기)
에피소드 진행 단계별로 정확도를 분석합니다.
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
from datetime import datetime

def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_drilldown_stages(num_episodes=10, save_log=True):
    api_server = "http://localhost:8000"
    api_key = "test_key_1234"
    # Using the standard basket_dataset
    dataset_dir = Path("/home/billy/25-1kp/vla/ROS_action/basket_dataset")
    
    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"api_episode_stages_{timestamp}.log"
    
    def log_print(msg, file_only=False):
        """Print to console and log file"""
        if not file_only:
            print(msg)
        if save_log:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
    
    all_files = list(dataset_dir.glob("*.h5"))
    test_files = random.sample(all_files, min(num_episodes, len(all_files)))
    
    log_print(f"🔍 에피소드 단계별(Stages) 심층 분석 시작 (Target: {len(test_files)} episodes)")
    log_print(f"📝 Log file: {log_file}")
    log_print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("")
    
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    # Stage definitions
    STAGES = {
        "초기 (Initial)": list(range(0, 5)),   # Frames 0-4
        "중기 (Middle)":  list(range(5, 14)),  # Frames 5-13
        "후기 (Final)":   list(range(14, 18))  # Frames 14-17 (assuming 18 total)
    }
    
    stage_stats = {
        name: {"total": 0, "matches": 0, "dir_matches": 0} for name in STAGES
    }
    
    for file_path in test_files:
        log_print(f"\nProcessing: {file_path.name}")
        requests.post(f"{api_server}/reset", headers=headers)
        
        with h5py.File(file_path, 'r') as f:
            images = f['images'][:]
            actions = f['actions'][:]
            n_frames = len(images)
            
            instruction = "Navigate to the brown pot on the left" if "left" in file_path.name else "Navigate to the brown pot on the right"
            
            for i in range(n_frames):
                img_pil = Image.fromarray(images[i])
                img_b64 = image_to_base64(img_pil)
                true_act = actions[i][:2]
                
                # Determine stage
                current_stage = None
                for name, frames in STAGES.items():
                    if i in frames:
                        current_stage = name
                        break
                
                if current_stage is None: continue
                
                try:
                    resp_raw = requests.post(
                        f"{api_server}/predict",
                        json={"image": img_b64, "instruction": instruction, "snap_to_grid": True},
                        headers=headers, timeout=10
                    )
                    
                    if resp_raw.status_code != 200:
                        log_print(f"  ❌ Error at Frame {i}: {resp_raw.text}")
                        continue
                        
                    resp = resp_raw.json()
                    pred_act = np.array(resp['action'])
                    
                    # 1. Perfect Match
                    is_perfect = np.allclose(pred_act, true_act, atol=0.01)
                    
                    # 2. Direction Match
                    true_dir = np.sign(true_act[1]) if abs(true_act[1]) > 0.1 else 0
                    pred_dir = np.sign(pred_act[1]) if abs(pred_act[1]) > 0.1 else 0
                    is_dir_match = (true_dir == pred_dir)
                    
                    # Update stage stats
                    stage_stats[current_stage]["total"] += 1
                    if is_perfect: stage_stats[current_stage]["matches"] += 1
                    if is_dir_match: stage_stats[current_stage]["dir_matches"] += 1
                    
                except Exception as e:
                    log_print(f"  ⚠️ Frame {i} error: {e}")
                    pass

    # Final Summary Table
    log_print("\n" + "="*80)
    log_print(f"{'Stage':<20} | {'Frames':<10} | {'Perfect Match':<15} | {'Dir Agreement':<15}")
    log_print("-" * 80)
    
    total_frames = 0
    total_perfect = 0
    total_dir = 0
    
    for name, stats in stage_stats.items():
        if stats["total"] == 0: continue
        perf_acc = (stats["matches"] / stats["total"] * 100)
        dir_acc = (stats["dir_matches"] / stats["total"] * 100)
        
        log_print(f"{name:<20} | {stats['total']:<10} | {perf_acc:>13.1f}% | {dir_acc:>13.1f}%")
        
        total_frames += stats["total"]
        total_perfect += stats["matches"]
        total_dir += stats["dir_matches"]
        
    log_print("-" * 80)
    if total_frames > 0:
        overall_perf = (total_perfect / total_frames * 100)
        overall_dir = (total_dir / total_frames * 100)
        log_print(f"{'OVERALL':<20} | {total_frames:<10} | {overall_perf:>13.1f}% | {overall_dir:>13.1f}%")
    log_print("="*80)
    
    log_print(f"\n⏰ Test finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='API Episode Stages Test')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to test')
    args = parser.parse_args()
    
    test_drilldown_stages(num_episodes=args.num_episodes)
