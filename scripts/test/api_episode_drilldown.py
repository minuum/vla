#!/usr/bin/env python3
"""
API 서버를 통한 에피소드 심층 분석 스크립트 (Directional Agreement 중심)
단순 일치율(Perfect Match)을 넘어 방향성 일관성과 오차를 분석합니다.
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

def get_direction_label(y_val):
    if y_val > 0.1: return "Left"
    elif y_val < -0.1: return "Right"
    else: return "Straight"

def test_drilldown(num_episodes=5, save_log=True):
    api_server = "http://localhost:8000"
    api_key = "test_key_1234"
    dataset_dir = Path("/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2/test")
    
    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"api_episode_drilldown_{timestamp}.log"
    
    def log_print(msg, file_only=False):
        """Print to console and log file"""
        if not file_only:
            print(msg)
        if save_log:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
    
    all_files = list(dataset_dir.glob("*.h5"))
    test_files = random.sample(all_files, min(num_episodes, len(all_files)))
    
    log_print(f"🔍 에피소드 심층 분석 시작 (Target: {len(test_files)} episodes)")
    log_print(f"📝 Log file: {log_file}")
    log_print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("")
    
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
        log_print(f"\nProcessing: {file_path.name}")
        requests.post(f"{api_server}/reset", headers=headers)
        
        with h5py.File(file_path, 'r') as f:
            images = f['images'][:]
            actions = f['actions'][:]
            n_frames = len(images)
            
            instruction = "Navigate to the brown pot on the left" if "left" in file_path.name else "Navigate to the brown pot on the right"
            
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
                
                try:
                    resp = requests.post(
                        f"{api_server}/predict",
                        json={"image": img_b64, "instruction": instruction, "snap_to_grid": True},
                        headers=headers, timeout=10
                    ).json()
                    
                    pred_act = np.array(resp['action'])
                    raw_act = np.array(resp.get('raw_action', [0, 0]))
                    
                    # 1. Perfect Match
                    is_perfect = np.allclose(pred_act, true_act, atol=0.01)
                    if is_perfect: ep_stats["perfect"] += 1
                    
                    # 2. Direction Match (Sign)
                    true_dir = np.sign(true_act[1]) if abs(true_act[1]) > 0.1 else 0
                    pred_dir = np.sign(pred_act[1]) if abs(pred_act[1]) > 0.1 else 0
                    if true_dir == pred_dir: ep_stats["dir"] += 1
                    
                    # 3. Errors (Raw baseline if possible, but using pred for consistency)
                    ep_stats["x_err"].append(pred_act[0] - true_act[0])
                    ep_stats["y_err"].append(pred_act[1] - true_act[1])
                    
                except Exception as e:
                    log_print(f"  ⚠️ Frame {i} error: {e}", file_only=True)
                    pass

            # Episode summary
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
            
            log_print(f"  ✓ {n_frames} frames | Perfect: {accuracy:.1f}% | Dir: {dir_acc:.1f}% | RMSE: {rmse_x:.3f}")
            
            # Global accumulation
            global_stats["total_frames"] += n_frames
            global_stats["perfect_matches"] += ep_stats["perfect"]
            global_stats["direction_matches"] += ep_stats["dir"]
            global_stats["x_errors"].extend(ep_stats["x_err"])
            global_stats["y_errors"].extend(ep_stats["y_err"])

    # Final Output
    log_print("\n" + "="*85)
    log_print(f"{'Episode Name':<45} | {'Perfect':<8} | {'Dir':<8} | {'RMSE X':<8}")
    log_print("-" * 85)
    for res in episode_results:
        log_print(f"{res['name'][:43]:<45} | {res['perfect_acc']:>6.1f}% | {res['dir_acc']:>6.1f}% | {res['rmse_x']:>7.3f}")
    
    log_print("="*85)
    total_perf = (global_stats["perfect_matches"] / global_stats["total_frames"] * 100)
    total_dir = (global_stats["direction_matches"] / global_stats["total_frames"] * 100)
    total_rmse_x = np.sqrt(np.mean(np.array(global_stats["x_errors"])**2))
    
    log_print(f"📊 SUMMARY REPORT")
    log_print(f"  - Total Evaluated Frames: {global_stats['total_frames']}")
    log_print(f"  - Overall Perfect Match: {total_perf:.2f}%")
    log_print(f"  - Directional Agreement: {total_dir:.2f}%  <-- 실질적 주행 가능성 지표")
    log_print(f"  - Linear X RMSE: {total_rmse_x:.4f}")
    log_print("="*85)
    log_print(f"\n⏰ Test finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save JSON summary
    if save_log:
        json_file = log_dir / f"api_episode_drilldown_{timestamp}.json"
        summary = {
            "timestamp": timestamp,
            "test_config": {
                "num_episodes": len(test_files),
                "dataset_dir": str(dataset_dir),
                "api_server": api_server
            },
            "episode_results": episode_results,
            "global_stats": {
                "total_frames": global_stats["total_frames"],
                "perfect_match_rate": total_perf,
                "directional_agreement": total_dir,
                "rmse_x": total_rmse_x,
                "rmse_y": np.sqrt(np.mean(np.array(global_stats["y_errors"])**2))
            }
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log_print(f"💾 JSON report saved: {json_file}")
    
    return {
        "total_perf": total_perf,
        "total_dir": total_dir,
        "total_rmse_x": total_rmse_x
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='API Episode Drilldown Test')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to test')
    parser.add_argument('--no_log', action='store_true', help='Disable log file saving')
    args = parser.parse_args()
    
    test_drilldown(num_episodes=args.num_episodes, save_log=not args.no_log)
