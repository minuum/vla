import os
import sys
import torch
import json
import base64
from PIL import Image
import numpy as np

# 경로 설정
project_root = "/home/billy/25-1kp/vla"
sys.path.insert(0, project_root)
# Mobile_VLA 폴더의 inference_server 모듈을 임포트하기 위해 추가
sys.path.insert(0, os.path.join(project_root, "Mobile_VLA"))

from inference_server import MobileVLAInference

def run_test():
    # 1. 모델 설정 (최신 체크포인트 사용)
    checkpoint_path = os.path.join(project_root, "runs/unified_regression_win12/kosmos/mobile_vla_exp17_win8_k1/2026-02-10/exp17_win8_k1/epoch=epoch=09-val_loss=val_loss=0.0013.ckpt")
    config_path = os.path.join(project_root, "Mobile_VLA/configs/mobile_vla_exp17_win8_k1.json")
    image_path = "/home/billy/25-1kp/vla/analysis/image.png"
    
    print(f"--- Inference Test ---")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Image: {image_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # 2. 모델 로드
    model = MobileVLAInference(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 3. 이미지 로드 및 Base64 변환
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    
    # 4. 테스트 명령어 리스트
    instructions = [
        "Navigate to the brown pot",
        "Navigate to the target location"
    ]
    
    # 5. 추론 수행
    # 첫 프레임은 Zero Enforcement가 적용되므로 두 번 돌려서 LSTM 업데이트 후의 값을 확인
    for instr in instructions:
        print(f"\n[Instruction]: {instr}")
        
        # Reset once for each instruction test
        model.reset()
        
        # Step 1: Priming (First Frame Safety)
        action_1, lat_1 = model.predict(img_base64, instr)
        print(f"Step 1 (Safety): {action_1} (Latency: {lat_1:.2f}ms)")
        
        # Step 2: Actual Prediction
        action_2, lat_2 = model.predict(img_base64, instr)
        print(f"Step 2 (Actual): {action_2} (Latency: {lat_2:.2f}ms)")

if __name__ == "__main__":
    run_test()
