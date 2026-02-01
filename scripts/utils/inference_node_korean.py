#!/usr/bin/env python3
"""
Mobile VLA Inference Node (Standalone)
현재 학습된 모델(한국어 instruction 기반)을 사용한 추론 노드

Usage:
    python3 scripts/inference_node_korean.py --scenario 1  # Left
    python3 scripts/inference_node_korean.py --scenario 2  # Right
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add path
sys.path.insert(0, str(Path(__file__).parent.parent))
from Mobile_VLA.instruction_mapping import get_instruction_for_robot_id
from Mobile_VLA.inference_pipeline import MobileVLAInferencePipeline


def run_inference_node(scenario_id: str, checkpoint_path: str, config_path: str, test_image_path: str = None):
    """
    추론 노드 실행
    
    Args:
        scenario_id: '1' (left) or '2' (right)
        checkpoint_path: 체크포인트 경로
        config_path: Config JSON 경로
        test_image_path: 테스트 이미지 경로 (None이면 더미 이미지 사용)
    """
    print("=" * 60)
    print("Mobile VLA Inference Node (Korean Instruction)")
    print("=" * 60)
    
    # 1. Instruction 가져오기 (학습 시 사용된 한국어)
    instruction = get_instruction_for_robot_id(scenario_id)
    print(f"\n[Scenario {scenario_id}]")
    print(f"Instruction: {instruction}")
    
    # 2. 파이프라인 초기화
    print(f"\n[Loading Model]")
    print(f"Checkpoint: {Path(checkpoint_path).name}")
    
    pipeline = MobileVLAInferencePipeline(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 3. 이미지 준비
    if test_image_path and Path(test_image_path).exists():
        image = Image.open(test_image_path).convert('RGB')
        print(f"Test Image: {test_image_path}")
    else:
        # 더미 이미지 (빨간색 = 테스트용)
        image = Image.new('RGB', (224, 224), color='red')
        print(f"Test Image: Dummy (224x224 red)")
    
    # 4. 추론 실행
    print(f"\n[Running Inference]")
    start_time = time.time()
    
    result = pipeline.predict(image, instruction)
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # 5. 결과 출력
    action = result['action']
    action_normalized = result['action_normalized']
    
    print(f"\n[Results]")
    print(f"Inference Time: {inference_time:.1f} ms")
    print(f"Normalized Output (Model): {action_normalized}")
    print(f"Final Action (Corrected):  {action}")
    print(f"  - linear_x:  {action[0]:.4f} m/s")
    print(f"  - linear_y:  {action[1]:.4f} rad/s")
    
    # 6. 검증
    print(f"\n[Validation]")
    if scenario_id == '1':  # Left
        # 왼쪽이므로 linear_y가 양수(+)여야 함
        if action[1] > 0:
            print(f"✓ PASS: Left turn detected (linear_y = {action[1]:.4f} > 0)")
        else:
            print(f"✗ FAIL: Expected left turn but got linear_y = {action[1]:.4f}")
    elif scenario_id == '2':  # Right
        # 오른쪽이므로 linear_y가 음수(-)여야 함
        if action[1] < 0:
            print(f"✓ PASS: Right turn detected (linear_y = {action[1]:.4f} < 0)")
        else:
            print(f"✗ FAIL: Expected right turn but got linear_y = {action[1]:.4f}")
    
    # Gain correction 확인
    if abs(action[0]) > 1.0 or abs(action[1]) > 1.0:
        print(f"✓ PASS: Gain correction working (values exceed 1.0)")
    else:
        print(f"⚠ INFO: Action values within [-1, 1] range")
    
    print("=" * 60)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Mobile VLA Inference Node (Korean)")
    parser.add_argument('--scenario', type=str, required=True, choices=['1', '2'],
                       help="Scenario ID: '1' (left) or '2' (right)")
    parser.add_argument('--checkpoint', type=str, 
                       default="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt",
                       help="Path to checkpoint")
    parser.add_argument('--config', type=str,
                       default="Mobile_VLA/configs/mobile_vla_chunk5_20251217.json",
                       help="Path to config JSON")
    parser.add_argument('--image', type=str, default=None,
                       help="Path to test image (optional)")
    
    args = parser.parse_args()
    
    run_inference_node(
        scenario_id=args.scenario,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        test_image_path=args.image
    )


if __name__ == "__main__":
    main()
