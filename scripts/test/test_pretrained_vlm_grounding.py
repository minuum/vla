#!/usr/bin/env python3
"""
Pretrained VLM Instruction Grounding Test
Best checkpoint (Epoch 3, val_loss=0.093) 검증
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RoboVLMs_upstream'))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.utils.config_utils import load_config


def print_section(title, char="="):
    print(f"\n{char * 70}")
    print(f" {title}")
    print(f"{char * 70}")


def run_test():
    print_section("Pretrained VLM Instruction Grounding Test")
    
    # Config 및 Checkpoint
    config_path = "Mobile_VLA/configs/mobile_vla_pretrained.json"
    checkpoint_path = "runs/mobile_vla_pretrained/kosmos/mobile_vla_transfer_learning/2026-01-10/mobile_vla_pretrained_vlm/epoch_epoch=03-val_loss=val_loss=0.093.ckpt"
    
    print(f"\n[Config] {config_path}")
    print(f"[Checkpoint] {checkpoint_path}")
    print(f"[Checkpoint Size] {os.path.getsize(checkpoint_path) / (1024**3):.2f} GB")
    
    # 모델 로드
    print("\n[모델 로드 중...]")
    configs = load_config(config_path)
    
    # Checkpoint에서 로드
    model = MobileVLATrainer.load_from_checkpoint(checkpoint_path, configs=configs)
    model.eval()
    print("✅ 모델 로드 완료")
    
    # 파라미터 확인
    print_section("파라미터 상태")
    total = sum(p.numel() for name, p in model.named_parameters() if 'model' in name)
    frozen = sum(p.numel() for name, p in model.named_parameters() if 'model' in name and not p.requires_grad)
    trainable_ah = sum(p.numel() for name, p in model.named_parameters() if 'act_head' in name and p.requires_grad)
    
    print(f"  Total VLM: {frozen:,} params (Frozen)")
    print(f"  Action Head: {trainable_ah:,} params (Trained)")
    
    # 테스트 이미지 (동일한 이미지)
    print_section("Test 1: 동일 이미지, 다른 Instruction")
    
    # 실제 데이터에서 이미지 가져오기
    import h5py
    dataset_dir = "/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset"
    h5_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.h5')])
    
    if not h5_files:
        print("❌ 데이터셋을 찾을 수 없습니다")
        return
    
    # 첫 번째 파일에서 이미지 가져오기
    h5_path = os.path.join(dataset_dir, h5_files[0])
    print(f"\n[데이터셋] {h5_files[0]}")
    
    with h5py.File(h5_path, 'r') as f:
        rgb_static = f['images'][0]  # 첫 프레임
    
    # PIL Image로 변환
    image = Image.fromarray(rgb_static)
    
    # Instruction 준비
    instruction_left = "Navigate around the obstacle on the left side and move forward"
    instruction_right = "Navigate around the obstacle on the right side and move forward"
    
    print(f"\n  Instruction 1: {instruction_left}")
    print(f"  Instruction 2: {instruction_right}")
    
    # Inference
    with torch.no_grad():
        # LEFT
        input_dict_left = {
            'rgb': [image],
            'lang': [instruction_left]
        }
        output_left = model.inference_step(input_dict_left)
        action_left = output_left['action'].cpu().numpy()
        
        # RIGHT
        input_dict_right = {
            'rgb': [image],
            'lang': [instruction_right]
        }
        output_right = model.inference_step(input_dict_right)
        action_right = output_right['action'].cpu().numpy()
    
    # 결과 비교
    print(f"\n[결과]")
    print(f"  LEFT action:  {action_left}")
    print(f"  RIGHT action: {action_right}")
    
    diff = np.abs(action_left - action_right).mean()
    print(f"\n  평균 차이: {diff:.6f}")
    
    # 판정
    print_section("판정")
    
    threshold = 0.01
    if diff > threshold:
        print(f"\n  ✅ SUCCESS: Instruction Grounding 성공!")
        print(f"  차이 ({diff:.6f}) > 임계값 ({threshold})")
        print(f"\n  → Pretrained VLM이 LEFT/RIGHT instruction을 구분합니다!")
    else:
        print(f"\n  ❌ FAILED: Instruction Grounding 실패")
        print(f"  차이 ({diff:.6f}) ≤ 임계값 ({threshold})")
        print(f"\n  → VLM이 instruction을 무시하고 있습니다")
    
    # Test 2: 다른 이미지, 동일 Instruction
    print_section("Test 2: 다른 이미지, 동일 Instruction (대조군)")
    
    if len(h5_files) > 1:
        h5_path2 = os.path.join(dataset_dir, h5_files[1])
        with h5py.File(h5_path2, 'r') as f:
            rgb_static2 = f['images'][0]
        
        image2 = Image.fromarray(rgb_static2)
        
        with torch.no_grad():
            input_dict1 = {'rgb': [image], 'lang': [instruction_left]}
            input_dict2 = {'rgb': [image2], 'lang': [instruction_left]}
            
            action1 = model.inference_step(input_dict1)['action'].cpu().numpy()
            action2 = model.inference_step(input_dict2)['action'].cpu().numpy()
        
        diff_image = np.abs(action1 - action2).mean()
        print(f"\n  이미지 차이에 따른 action 차이: {diff_image:.6f}")
        print(f"  (Vision 정상 작동 확인용)")
    
    print_section("완료")
    print(f"\n  🎯 Pretrained VLM Instruction Grounding Test 완료!")
    
    # 결과 요약 저장
    with open('test_result_pretrained_vlm.txt', 'w') as f:
        f.write(f"Pretrained VLM Instruction Grounding Test Result\n")
        f.write(f"=" * 70 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: 3 (Best, val_loss=0.093)\n\n")
        f.write(f"LEFT action:  {action_left}\n")
        f.write(f"RIGHT action: {action_right}\n")
        f.write(f"Difference: {diff:.6f}\n\n")
        f.write(f"Result: {'SUCCESS ✅' if diff > threshold else 'FAILED ❌'}\n")
    
    print(f"\n  결과 저장: test_result_pretrained_vlm.txt")


if __name__ == "__main__":
    run_test()
