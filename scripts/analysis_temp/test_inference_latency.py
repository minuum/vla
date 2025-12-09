#!/usr/bin/env python3
"""
Inference Latency 측정 스크립트
목적: Best checkpoint의 실제 추론 속도 측정
"""

import torch
import time
import numpy as np
from pathlib import Path
import argparse
import sys

# RoboVLMs 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "RoboVLMs_upstream"))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer


def measure_inference_latency(checkpoint_path, num_iterations=100):
    """
    Inference latency 측정
    
    Args:
        checkpoint_path: 체크포인트 경로
        num_iterations: 측정 반복 횟수
    """
    print("="*60)
    print("Inference Latency 측정")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Iterations: {num_iterations}")
    print()
    
    # 1. 모델 로드
    print("[1/4] 모델 로딩...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = MobileVLATrainer.load_from_checkpoint(checkpoint_path)
        model = model.to(device)
        model.eval()
        print(f"  ✅ 모델 로드 완료 (Device: {device})")
    except Exception as e:
        print(f"  ❌ 모델 로드 실패: {e}")
        return
    
    # 2. 더미 입력 생성
    print("\n[2/4] 더미 입력 생성...")
    window_size = 8
    image_size = 224
    batch_size = 1
    
    dummy_images = torch.randn(batch_size, window_size, 3, image_size, image_size).to(device)
    dummy_text = torch.zeros(batch_size, 256, dtype=torch.long).to(device)
    dummy_attention_mask = torch.ones(batch_size, 256, dtype=torch.long).to(device)
    
    print(f"  Images shape: {dummy_images.shape}")
    print(f"  Text shape: {dummy_text.shape}")
    
    # 3. Warm-up (GPU 준비)
    print("\n[3/4] Warm-up (10 iterations)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model.model(dummy_images, dummy_text, dummy_attention_mask)
    print("  ✅ Warm-up 완료")
    
    # 4. 실제 측정
    print(f"\n[4/4] Latency 측정 ({num_iterations} iterations)...")
    
    vlm_times = []
    action_head_times = []
    total_times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            # Total time
            t_total_start = time.time()
            
            # VLM forward
            t_vlm_start = time.time()
            context = model.model.encode_images(dummy_images)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_vlm_end = time.time()
            
            # Action Head forward
            t_action_start = time.time()
            actions = model.model.act_head(context)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_action_end = time.time()
            
            t_total_end = time.time()
            
            vlm_times.append((t_vlm_end - t_vlm_start) * 1000)  # ms
            action_head_times.append((t_action_end - t_action_start) * 1000)
            total_times.append((t_total_end - t_total_start) * 1000)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_iterations}")
    
    # 5. 결과 출력
    print("\n" + "="*60)
    print("측정 결과")
    print("="*60)
    
    print(f"\n📊 VLM Forward:")
    print(f"  Mean: {np.mean(vlm_times):.2f} ms")
    print(f"  Std:  {np.std(vlm_times):.2f} ms")
    print(f"  Min:  {np.min(vlm_times):.2f} ms")
    print(f"  Max:  {np.max(vlm_times):.2f} ms")
    
    print(f"\n📊 Action Head Forward:")
    print(f"  Mean: {np.mean(action_head_times):.2f} ms")
    print(f"  Std:  {np.std(action_head_times):.2f} ms")
    print(f"  Min:  {np.min(action_head_times):.2f} ms")
    print(f"  Max:  {np.max(action_head_times):.2f} ms")
    
    print(f"\n📊 Total Inference:")
    print(f"  Mean: {np.mean(total_times):.2f} ms")
    print(f"  Std:  {np.std(total_times):.2f} ms")
    print(f"  Min:  {np.min(total_times):.2f} ms")
    print(f"  Max:  {np.max(total_times):.2f} ms")
    
    # 6. 목표 달성 여부
    print("\n" + "="*60)
    print("목표 달성 여부")
    print("="*60)
    
    target_latency = 200.0  # ms
    mean_total = np.mean(total_times)
    
    if mean_total < target_latency:
        print(f"  ✅ 목표 달성! ({mean_total:.2f} ms < {target_latency} ms)")
        print(f"  → 0.4초 간격 추론에 충분합니다!")
    else:
        print(f"  ⚠️  목표 미달 ({mean_total:.2f} ms >= {target_latency} ms)")
        print(f"  → 추론 간격 조정 필요")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Latency 측정")
    parser.add_argument("--checkpoint", type=str, 
                       default="RoboVLMs_upstream/runs/mobile_vla_lora_20251203/kosmos/mobile_vla_finetune/2025-12-03/mobile_vla_lora_20251203/epoch_epoch=09-val_loss=val_loss=0.013.ckpt",
                       help="Checkpoint 경로")
    parser.add_argument("--iterations", type=int, default=100,
                       help="측정 반복 횟수")
    
    args = parser.parse_args()
    
    measure_inference_latency(args.checkpoint, args.iterations)
