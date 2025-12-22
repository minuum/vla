#!/usr/bin/env python3
import requests
import base64
import numpy as np
import matplotlib.pyplot as plt
import h5py
import io
import time
from PIL import Image
import os

# --- 설정 ---
API_URL = "http://localhost:8000"
API_KEY = os.getenv("VLA_API_KEY", "qkGswLr0qTJQALrbYrrQ9rB-eVGrzD7IqLNTmswE0lU")
H5_PATH = "ROS_action/mobile_vla_dataset/episode_20251203_042905_1box_hori_left_core_medium.h5"
FRAME_IDX = 0  # 시작 프레임
NUM_RUNS = 10  # 모델당 실행 횟수 (분산 확인용)
DT = 0.2  # 시각화를 위한 시간 간격 (초)

def get_image_from_h5(path, idx):
    with h5py.File(path, 'r') as f:
        img = f['images'][idx]
        inst_raw = f['language_instruction'][()]
        if isinstance(inst_raw, np.ndarray):
            inst_raw = inst_raw[0]
        inst = inst_raw.decode() if hasattr(inst_raw, 'decode') else inst_raw
    return img, inst

def encode_image(img_np):
    img_pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def run_inference(model_name, img_b64, instruction):
    # 1. 모델 스위치
    requests.post(
        f"{API_URL}/model/switch",
        json={"model_name": model_name},
        headers={"X-API-Key": API_KEY}
    )
    time.sleep(2) # 로딩 대기
    
    trajectories = []
    
    print(f"Collecting trajectories for {model_name}...")
    for i in range(NUM_RUNS):
        res = requests.post(
            f"{API_URL}/predict",
            json={"image": img_b64, "instruction": instruction},
            headers={"X-API-Key": API_KEY},
            timeout=10
        )
        if res.status_code == 200:
            data = res.json()
            if 'full_chunk' in data and data['full_chunk']:
                chunk = np.array(data['full_chunk']) # (N, 2)
                
                # Trajectory Integration (Omni-model assumption for visualization)
                # x, y = 0, 0 starting point
                traj_x = [0]
                traj_y = [0]
                curr_x, curr_y = 0, 0
                
                for step in chunk:
                    vx, vy = step[0], step[1]
                    # Integrate
                    curr_x += vx * DT
                    curr_y += vy * DT
                    traj_x.append(curr_x)
                    traj_y.append(curr_y)
                
                trajectories.append((traj_x, traj_y))
                print(f"  Run {i+1}: Chunk size {len(chunk)}")
            else:
                print(f"  Run {i+1}: No full_chunk data")
        else:
            print(f"  Run {i+1}: Failed {res.status_code}")
            
    return trajectories

def main():
    # 1. 데이터 로드
    print(f"Loading image from {H5_PATH}...")
    img, inst = get_image_from_h5(H5_PATH, FRAME_IDX)
    img_b64 = encode_image(img)
    print(f"Instruction: {inst}")
    
    # 2. 추론 실행
    models = ["chunk5_epoch6", "chunk10_epoch8"]
    all_trajs = {}
    
    for model in models:
        all_trajs[model] = run_inference(model, img_b64, inst)
        
    # 3. 시각화
    print("Visualizing...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        trajs = all_trajs[model]
        
        # Draw Robot (Origin)
        ax.plot(0, 0, 'ko', markersize=10, label='Robot Start')
        ax.arrow(0, 0, 0.2, 0, head_width=0.05, head_length=0.05, fc='k', ec='k') # Initial heading (x-axis)
        
        # Draw Obstacle (Example position)
        # 장애물이 "Navigate around"라고 했으므로 전방에 있다고 가정
        circle = plt.Circle((1.5, 0), 0.2, color='r', alpha=0.5, label='Obstacle')
        ax.add_patch(circle)
        
        # Draw Target (Left)
        # "Left bottle" -> y > 0
        target = plt.Circle((2.0, 1.0), 0.2, color='g', alpha=0.5, label='Target (Left)')
        ax.add_patch(target)
        
        # Draw Trajectories
        for tx, ty in trajs:
            ax.plot(tx, ty, 'b-', alpha=0.3, linewidth=2)
            ax.plot(tx[-1], ty[-1], 'b.', markersize=5) # End point
            
        ax.set_title(f"Model: {model} (Runs: {NUM_RUNS})")
        ax.set_xlabel("X (Forward) [m]")
        ax.set_ylabel("Y (Left/Right) [m]")
        ax.grid(True)
        ax.axis('equal')
        ax.set_xlim(-0.5, 3.0)
        ax.set_ylim(-1.5, 2.0)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig("docs/model_comparison/trajectory_comparison_chunk5_vs_10.png")
    print("Saved to docs/model_comparison/trajectory_comparison_chunk5_vs_10.png")

if __name__ == "__main__":
    main()
